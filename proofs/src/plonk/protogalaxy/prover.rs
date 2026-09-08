use std::{hash::Hash, marker::PhantomData, time::Instant};

use ff::{FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use rand_core::{CryptoRng, RngCore};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    plonk::{
        compute_h_poly, compute_nu_poly, compute_queries, evaluation::LookupAux, logup,
        partially_evaluate_identities, permutation, trash, write_evals_to_transcript, Circuit,
        Error, ProvingKey,
    },
    plonk::{
        linearization::prover::compute_linearization_poly,
        protogalaxy::utils::{
            batch_traces_fft, eval_lagrange_on_beta, fold_traces, FoldingLogupTrace,
            FoldingProverTrace,
        },
        prover::compute_raw_trace,
        traces::ProverTrace,
    },
    poly::{
        commitment::PolynomialCommitmentScheme, Coeff, EvaluationDomain, ExtendedLagrangeCoeff,
        LagrangeCoeff, Polynomial, PolynomialLabel, ProverQuery,
    },
    transcript::{Hashable, Sampleable, Transcript},
    utils::arithmetic::eval_polynomial,
};

/// Protogalaxy prover: folds `NB_FOLDED` instances into a single accumulator.
#[derive(Debug)]
pub struct ProtogalaxyProver<
    F: WithSmallOrderMulGroup<3>,
    CS: PolynomialCommitmentScheme<F>,
    const NB_FOLDED: usize,
> {
    _phantom: PhantomData<(F, CS)>,
}

impl<F: WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>, const NB_FOLDED: usize>
    ProtogalaxyProver<F, CS, NB_FOLDED>
{
    /// Fold `NB_FOLDED` circuit instances into one accumulator.
    ///
    /// Each circuit in `circuits` is proved using `compute_trace` and then all
    /// traces are folded via the protogalaxy protocol. A final corrected plonk
    /// proof is written to `transcript`.
    pub fn fold<C, T>(
        params: &CS::Parameters,
        pk: ProvingKey<F, CS>,
        circuits: Vec<C>,
        instances: &[&[&[F]]],
        mut rng: impl CryptoRng + RngCore,
        transcript: &mut T,
    ) -> Result<(), Error>
    where
        C: Circuit<F>,
        T: Transcript,
        CS::Commitment: Hashable<T::Hash>,
        F: WithSmallOrderMulGroup<3>
            + Sampleable<T::Hash>
            + Hashable<T::Hash>
            + Hash
            + Ord
            + FromUniformBytes<64>,
    {
        assert_eq!(circuits.len(), NB_FOLDED);
        assert_eq!(instances.len(), NB_FOLDED);
        assert!(NB_FOLDED.is_power_of_two());

        let domain = pk.vk.get_domain();

        // ── Step 1: compute k standard PLONK traces (advice stays Lagrange) ──
        let t0 = Instant::now();
        let folding_traces: Vec<FoldingProverTrace<F>> = circuits
            .into_iter()
            .zip(instances.iter())
            .map(|(circuit, inst)| -> Result<FoldingProverTrace<F>, Error> {
                let raw = compute_raw_trace(
                    params,
                    &pk,
                    &circuit,
                    #[cfg(feature = "committed-instances")] 0,
                    inst,
                    transcript,
                    &mut rng,
                )?;
                // Bring every auxiliary polynomial into Lagrange form once, here,
                // rather than once per extended evaluation point inside step 6.
                let to_lag = |q: Polynomial<F, Coeff>| domain.coeff_to_lagrange(q);
                Ok(FoldingProverTrace {
                    advice_polys: raw.advice_lagrange,
                    instance_polys: raw.instance_values,
                    lookups: raw.lookups.into_iter().map(|l| FoldingLogupTrace {
                        multiplicities: to_lag(l.multiplicities),
                        helper_polys: l.helper_polys.into_iter().map(to_lag).collect(),
                        aggregator_poly: to_lag(l.aggregator_poly),
                    }).collect(),
                    trash_polys: raw.trashcans.into_iter().map(|t| to_lag(t.trash_poly)).collect(),
                    perm_polys: raw.permutations.sets.into_iter()
                        .map(|s| to_lag(s.permutation_product_poly)).collect(),
                    beta: raw.beta,
                    gamma: raw.gamma,
                    theta: raw.theta,
                    trash_challenge: raw.trash_challenge,
                    y: raw.y,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let traces_refs: Vec<&FoldingProverTrace<F>> = folding_traces.iter().collect();
        eprintln!("[protogalaxy] step1 compute_traces:     {:>8.2?}", t0.elapsed());

        // ── Step 2: protogalaxy challenges ────────────────────────────────────
        let beta_pg: F = transcript.squeeze_challenge();

        // ── Step 3: precompute Lagrange-form proving-key data ─────────────────
        let l0_lag = domain.extended_to_lagrange(pk.l0.clone());
        let l_last_lag = domain.extended_to_lagrange(pk.l_last.clone());
        let l_active_lag = domain.extended_to_lagrange(pk.l_active_row.clone());
        let perm_pk_lag: Vec<Polynomial<F, LagrangeCoeff>> = pk
            .permutation
            .polys
            .iter()
            .map(|p| domain.coeff_to_lagrange(p.clone()))
            .collect();

        let beta_coeffs = eval_lagrange_on_beta(domain, &beta_pg);

        // ── Step 4: set up the dk-domain ──────────────────────────────────────
        let folding_degree = pk.vk.cs.degree() as u32;
        let dk_domain = EvaluationDomain::new(folding_degree, NB_FOLDED.trailing_zeros());

        // ── Step 5: batch traces over dk-domain ───────────────────────────────
        let t1 = Instant::now();
        let lifted = batch_traces_fft(&dk_domain, &traces_refs);
        eprintln!("[protogalaxy] step5 batch_traces:        {:>8.2?}", t1.elapsed());

        // ── Step 6: evaluate circuit identity at each extended dk-point ───────
        let t2 = Instant::now();
        // g_poly[t] = G(x_t) = ⟨nu_t, beta_coeffs⟩
        let nu_ctx = NuContext {
            pk: &pk,
            domain,
            l0: &l0_lag,
            l_last: &l_last_lag,
            l_active: &l_active_lag,
            perm_pk: &perm_pk_lag,
        };

        // Parallel over the extended points: each is independent, and the
        // per-point work no longer owns the whole machine.
        let g_poly: Vec<F> = lifted
            .par_iter()
            .map(|trace_t| {
                let nu_t = nu_ctx.numerator(trace_t);
                nu_t.values.iter().zip(beta_coeffs.iter()).map(|(v, b)| *v * b).sum()
            })
            .collect();

        if std::env::var("PG_CHECK_STEP6").is_ok() {
            let t = Instant::now();
            let reference: Vec<F> = lifted
                .iter()
                .map(|trace_t| {
                    let nu_t = nu_ctx.numerator_reference(trace_t);
                    nu_t.values.iter().zip(beta_coeffs.iter()).map(|(v, b)| *v * b).sum()
                })
                .collect();
            assert_eq!(
                g_poly.len(),
                reference.len(),
                "step6: g_poly length differs"
            );
            for (t, (fast, refv)) in g_poly.iter().zip(reference.iter()).enumerate() {
                assert!(
                    fast == refv,
                    "step6: g_poly differs from reference at point {t}"
                );
            }
            eprintln!(
                "[protogalaxy] step6 REFERENCE (check): {:>8.2?}  -- g_poly identical",
                t.elapsed()
            );
        }

        eprintln!("[protogalaxy] step6 eval_G_poly:         {:>8.2?}", t2.elapsed());

        // ── Step 7: compute K = G / Z_dk and commit ───────────────────────────
        let t3 = Instant::now();
        let g_ext: Polynomial<F, ExtendedLagrangeCoeff> = Polynomial {
            values: g_poly,
            _marker: PhantomData,
        };
        let k_ext = dk_domain.divide_by_vanishing_poly(g_ext);
        let k_coeff_vals = dk_domain.extended_to_coeff(k_ext);
        let k_poly: Polynomial<F, Coeff> = Polynomial {
            values: k_coeff_vals,
            _marker: PhantomData,
        };

        let k_commitment = CS::commit(params, &k_poly, PolynomialLabel::Custom("protogalaxy_K".into()));
        transcript.write(&k_commitment)?;

        let gamma: F = transcript.squeeze_challenge();
        eprintln!("[protogalaxy] step7 K_commit:            {:>8.2?}", t3.elapsed());

        // ── Step 8-9: fold traces at gamma, compute nu(folded) ────────────────
        // E[j] = nu(folded)(ω^j) by the Protogalaxy identity.
        let t4 = Instant::now();
        let folded = fold_traces(&dk_domain, &traces_refs, &gamma);

        let nu_folded = nu_ctx.numerator(&folded);

        if std::env::var("PG_CHECK_STEP6").is_ok() {
            let reference = nu_ctx.numerator_reference(&folded);
            assert_eq!(nu_folded.values.len(), reference.values.len());
            for (i, (fast, refv)) in
                nu_folded.values.iter().zip(reference.values.iter()).enumerate()
            {
                assert!(
                    fast == refv,
                    "step8-9: error_terms differ from reference at row {i}"
                );
            }
            eprintln!("[protogalaxy] step8-9 REFERENCE (check): error_terms identical");
        }

        let error_terms = nu_folded.values;
        eprintln!("[protogalaxy] step8-9 fold+nu_folded:   {:>8.2?}", t4.elapsed());

        // k_at_gamma = E(beta_pg) / Z_dk(gamma).
        // degree(K) may exceed ext_len for large NB_FOLDED, making eval_polynomial
        // on the IFFT'd k_poly aliased.  Since E(beta_pg) = G(gamma) = K(gamma)*Z_dk(gamma)
        // and we know E exactly, we derive K(gamma) from E instead.
        let z_dk_at_gamma = gamma.pow([dk_domain.n]) - F::ONE;
        let e_at_beta_pg: F = error_terms.iter().zip(beta_coeffs.iter()).map(|(e, b)| *e * b).sum();
        let k_at_gamma = e_at_beta_pg * z_dk_at_gamma.invert().expect("Z_dk(gamma) must be nonzero");
        transcript.write(&k_at_gamma)?;

        // ── Step 10: commit to error polynomial ───────────────────────────────
        let t5 = Instant::now();
        let error_lagrange: Polynomial<F, LagrangeCoeff> = Polynomial {
            values: error_terms,
            _marker: PhantomData,
        };
        let error_coeff = domain.lagrange_to_coeff(error_lagrange.clone());
        let error_commitment =
            CS::commit(params, &error_lagrange, PolynomialLabel::Custom("protogalaxy_error".into()));
        transcript.write(&error_commitment)?;
        eprintln!("[protogalaxy] step10 error_commit:      {:>8.2?}", t5.elapsed());

        // ── Step 11: finalise proof with error correction ─────────────────────
        let t6 = Instant::now();
        let folded_trace = into_prover_trace(domain, folded);
        let result = finalise_folded_proof(
            params, &pk, folded_trace, beta_pg, error_coeff, transcript,
        );
        eprintln!("[protogalaxy] step11 finalise_proof:    {:>8.2?}", t6.elapsed());
        result
    }
}



/// Everything `evaluate_numerator` needs besides the trace itself.
///
/// Holds both the optimised and the pre-optimisation ("reference") way of
/// computing the numerator, so `PG_CHECK_STEP6=1` can assert they agree.
struct NuContext<'a, F: WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>> {
    pk: &'a ProvingKey<F, CS>,
    domain: &'a EvaluationDomain<F>,
    l0: &'a Polynomial<F, LagrangeCoeff>,
    l_last: &'a Polynomial<F, LagrangeCoeff>,
    l_active: &'a Polynomial<F, LagrangeCoeff>,
    perm_pk: &'a [Polynomial<F, LagrangeCoeff>],
}

impl<F: WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>> NuContext<'_, F, CS> {
    /// Optimised path: the trace is already in Lagrange form, so the numerator
    /// is evaluated with no basis conversion at all.
    fn numerator(&self, trace: &FoldingProverTrace<F>) -> Polynomial<F, LagrangeCoeff> {
        let lookup_aux: Vec<LookupAux<'_, F, LagrangeCoeff>> = trace
            .lookups
            .iter()
            .map(|l| LookupAux {
                helpers: &l.helper_polys,
                aggregator: &l.aggregator_poly,
                multiplicities: &l.multiplicities,
            })
            .collect();

        self.pk.ev.evaluate_numerator_prepared(
            self.domain,
            &self.pk.vk.cs,
            &trace.advice_polys,
            &trace.instance_polys,
            &self.pk.fixed_values,
            &trace.y,
            trace.beta,
            trace.gamma,
            &trace.theta,
            trace.trash_challenge,
            &lookup_aux,
            &trace.trash_polys,
            &trace.perm_polys,
            self.l0,
            self.l_last,
            self.l_active,
            self.perm_pk,
        )
    }

    /// Reference path, kept for verification only.
    ///
    /// Converts the auxiliary polynomials back to coefficient form and calls the
    /// converting `evaluate_numerator` — i.e. exactly what the prover did before
    /// this optimisation, one FFT per polynomial per evaluation point. Strictly
    /// slower; used only under `PG_CHECK_STEP6`.
    fn numerator_reference(&self, trace: &FoldingProverTrace<F>) -> Polynomial<F, LagrangeCoeff> {
        let to_coeff = |q: &Polynomial<F, LagrangeCoeff>| self.domain.lagrange_to_coeff(q.clone());

        let lookups: Vec<logup::prover::Committed<F>> = trace
            .lookups
            .iter()
            .map(|l| logup::prover::Committed {
                multiplicities: to_coeff(&l.multiplicities),
                helper_polys: l.helper_polys.iter().map(&to_coeff).collect(),
                aggregator_poly: to_coeff(&l.aggregator_poly),
            })
            .collect();
        let trashcans: Vec<trash::prover::Committed<F>> = trace
            .trash_polys
            .iter()
            .map(|q| trash::prover::Committed {
                trash_poly: to_coeff(q),
            })
            .collect();
        let permutations = permutation::prover::Committed {
            sets: trace
                .perm_polys
                .iter()
                .map(|q| permutation::prover::CommittedSet {
                    permutation_product_poly: to_coeff(q),
                })
                .collect(),
        };

        self.pk.ev.evaluate_numerator(
            self.domain,
            &self.pk.vk.cs,
            &trace.advice_polys,
            &trace.instance_polys,
            &self.pk.fixed_values,
            &trace.y,
            trace.beta,
            trace.gamma,
            &trace.theta,
            trace.trash_challenge,
            &lookups,
            &trashcans,
            &permutations,
            self.l0,
            self.l_last,
            self.l_active,
            self.perm_pk,
        )
    }
}

/// Convert a `FoldingProverTrace` back to a `ProverTrace`.
fn into_prover_trace<F: PrimeField + WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    folded: FoldingProverTrace<F>,
) -> ProverTrace<F> {
    let advice_polys: Vec<Polynomial<F, Coeff>> = folded
        .advice_polys
        .iter()
        .map(|p| domain.lagrange_to_coeff(p.clone()))
        .collect();

    let instance_values = folded.instance_polys.clone();
    let instance_polys: Vec<Polynomial<F, Coeff>> = folded
        .instance_polys
        .into_iter()
        .map(|p| domain.lagrange_to_coeff(p))
        .collect();

    // The folding trace is entirely in Lagrange form; the openings and
    // `compute_nu_poly` need coefficient form. This is the only place the
    // conversion is paid, and only for the single folded trace.
    let to_coeff = |p: Polynomial<F, LagrangeCoeff>| domain.lagrange_to_coeff(p);

    let lookups = folded
        .lookups
        .into_iter()
        .map(|l| logup::prover::Committed {
            multiplicities: to_coeff(l.multiplicities),
            helper_polys: l.helper_polys.into_iter().map(to_coeff).collect(),
            aggregator_poly: to_coeff(l.aggregator_poly),
        })
        .collect();

    let trashcans = folded
        .trash_polys
        .into_iter()
        .map(|p| trash::prover::Committed {
            trash_poly: to_coeff(p),
        })
        .collect();

    let permutations = permutation::prover::Committed {
        sets: folded
            .perm_polys
            .into_iter()
            .map(|p| permutation::prover::CommittedSet {
                permutation_product_poly: to_coeff(p),
            })
            .collect(),
    };

    ProverTrace {
        advice_polys,
        instance_polys,
        instance_values,
        lookups,
        trashcans,
        permutations,
        beta: folded.beta,
        gamma: folded.gamma,
        theta: folded.theta,
        trash_challenge: folded.trash_challenge,
        y: folded.y,
    }
}

/// Finalise the protogalaxy proof: same as `finalise_proof` but subtracts the
/// error polynomial from `nu` before dividing, and adds extra PCS queries.
#[allow(clippy::too_many_arguments)]
fn finalise_folded_proof<F, CS, T>(
    params: &CS::Parameters,
    pk: &ProvingKey<F, CS>,
    trace: ProverTrace<F>,
    beta_pg: F,
    error_coeff: Polynomial<F, Coeff>,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: WithSmallOrderMulGroup<3>
        + Sampleable<T::Hash>
        + Hashable<T::Hash>
        + Hash
        + Ord
        + FromUniformBytes<64>,
    CS: PolynomialCommitmentScheme<F>,
    T: Transcript,
    CS::Commitment: Hashable<T::Hash>,
{
    let domain = pk.vk.get_domain();

    // Compute nu(folded_witness) and subtract error correction.
    let nu = compute_nu_poly(pk, &trace);
    let error_ext = domain.coeff_to_extended(error_coeff.clone());
    let nu_corrected = nu - &error_ext;

    // Commit to corrected quotient h = (nu - E) / Z_n.
    let quotient_limbs = compute_h_poly::<F, CS, T>(params, domain, nu_corrected, transcript)?;

    let ProverTrace {
        advice_polys,
        instance_polys,
        lookups,
        trashcans,
        permutations,
        beta,
        gamma,
        theta,
        trash_challenge,
        y,
        ..
    } = trace;

    let x: F = transcript.squeeze_challenge();

    let crate::plonk::prover::Evals {
        fixed_evals,
        instance_evals,
        advice_evals,
        ..
    } = write_evals_to_transcript(pk, 0, &instance_polys, &advice_polys, x, transcript)?;

    // Write the error polynomial evaluation at x.
    let error_at_x = eval_polynomial(&error_coeff.values, x);
    transcript.write(&error_at_x)?;

    let permutations_common = pk.permutation.evaluate(x, transcript)?;
    let permutations_eval = permutations.evaluate(pk, x, transcript)?;
    let lookups_eval: Vec<logup::prover::Evaluated<F>> = lookups
        .into_iter()
        .map(|p| p.evaluate(pk, x, transcript))
        .collect::<Result<Vec<_>, _>>()?;
    let trashcans_eval: Vec<trash::prover::Evaluated<F>> = trashcans
        .into_iter()
        .map(|p| p.evaluate(x, transcript))
        .collect::<Result<Vec<_>, _>>()?;

    let splitting_factor = x.pow_vartime([pk.vk.n() - 1]);
    let xn = splitting_factor * x;

    let expressions = partially_evaluate_identities(
        &pk.vk,
        &fixed_evals,
        &instance_evals,
        &advice_evals,
        &permutations_eval.evaluated,
        lookups_eval.iter().map(|inner| &inner.evaluated),
        trashcans_eval.iter().map(|inner| &inner.evaluated),
        &permutations_common,
        x,
        xn,
        beta,
        gamma,
        &theta,
        trash_challenge,
    );

    let (lin_poly_non_constant_part, _) =
        compute_linearization_poly(expressions, pk, &y, xn, splitting_factor, quotient_limbs);

    let mut queries = compute_queries(
        pk,
        0,
        &instance_polys,
        &advice_polys,
        &permutations_eval,
        &lookups_eval,
        &trashcans_eval,
        x,
        &lin_poly_non_constant_part,
    );

    // Extra queries for the error polynomial.
    queries.push(ProverQuery::new(x, &error_coeff));
    queries.push(ProverQuery::new(beta_pg, &error_coeff));

    CS::multi_open(params, &queries, transcript).map_err(|_| Error::ConstraintSystemFailure)
}
