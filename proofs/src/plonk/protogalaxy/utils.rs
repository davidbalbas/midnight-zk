use std::{
    marker::PhantomData,
    mem::size_of,
    ops::{Add, Mul},
};

use ff::{Field, PrimeField, WithSmallOrderMulGroup};
use midnight_curves::fft::compute_twiddles;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    poly::{EvaluationDomain, LagrangeCoeff, Polynomial},
    utils::arithmetic::eval_polynomial,
};

/// A single logup (LogUp) trace in folding form: polys in Lagrange form.
pub(crate) struct FoldingLogupTrace<F: PrimeField> {
    pub multiplicities: Polynomial<F, LagrangeCoeff>,
    pub helper_polys: Vec<Polynomial<F, LagrangeCoeff>>,
    pub aggregator_poly: Polynomial<F, LagrangeCoeff>,
}

/// A single PLONK prover trace lifted for protogalaxy folding.
///
/// *Every* polynomial is kept in Lagrange (evaluation) form, so
/// `evaluate_numerator_prepared::<LagrangeCoeff>` can be called with no
/// conversion at all. Folding is a linear combination and the change of basis
/// is linear, so the two commute: converting once per input trace (`k` of them)
/// is equivalent to — and far cheaper than — converting once per extended
/// evaluation point (`ext_len` of them, which is what the coefficient-form
/// representation used to cost).
///
/// Coefficient form is still needed for the polynomial openings, but only for
/// the single *folded* trace; see `into_prover_trace`.
pub(crate) struct FoldingProverTrace<F: PrimeField> {
    pub advice_polys: Vec<Polynomial<F, LagrangeCoeff>>,
    pub instance_polys: Vec<Polynomial<F, LagrangeCoeff>>,
    pub lookups: Vec<FoldingLogupTrace<F>>,
    pub trash_polys: Vec<Polynomial<F, LagrangeCoeff>>,
    /// Flattened permutation product polynomials (one per set).
    pub perm_polys: Vec<Polynomial<F, LagrangeCoeff>>,
    pub beta: F,
    pub gamma: F,
    pub theta: Vec<F>,
    pub trash_challenge: F,
    pub y: Vec<F>,
}

impl<F: PrimeField> FoldingProverTrace<F> {
    /// Create a zero-valued trace with the same dimensions as `other`.
    pub fn zero_like(other: &Self) -> Self {
        // All buffers share the domain size. Take it from whichever family is
        // non-empty: a trace need not have advice columns.
        let n = other.poly_values().first().map_or(0, |slot: &&[F]| slot.len());

        let zero = || Polynomial {
            values: vec![F::ZERO; n],
            _marker: PhantomData,
        };

        let lookups = other
            .lookups
            .iter()
            .map(|l| FoldingLogupTrace {
                multiplicities: zero(),
                helper_polys: l.helper_polys.iter().map(|_| zero()).collect(),
                aggregator_poly: zero(),
            })
            .collect();

        FoldingProverTrace {
            advice_polys: (0..other.advice_polys.len()).map(|_| zero()).collect(),
            instance_polys: (0..other.instance_polys.len()).map(|_| zero()).collect(),
            lookups,
            trash_polys: (0..other.trash_polys.len()).map(|_| zero()).collect(),
            perm_polys: (0..other.perm_polys.len()).map(|_| zero()).collect(),
            beta: F::ZERO,
            gamma: F::ZERO,
            theta: vec![F::ZERO; other.theta.len()],
            trash_challenge: F::ZERO,
            y: vec![F::ZERO; other.y.len()],
        }
    }

    /// Every polynomial value buffer of the trace, in a fixed canonical order.
    ///
    /// Traces of the same shape always flatten to the same sequence of buffer
    /// lengths, which is what lets the folding transform be applied
    /// coordinate-wise without knowing anything about the PLONK structure.
    fn poly_values(&self) -> Vec<&[F]> {
        let mut v: Vec<&[F]> = Vec::new();
        v.extend(self.advice_polys.iter().map(|p| p.values.as_slice()));
        v.extend(self.instance_polys.iter().map(|p| p.values.as_slice()));
        for l in self.lookups.iter() {
            v.push(l.multiplicities.values.as_slice());
            v.extend(l.helper_polys.iter().map(|p| p.values.as_slice()));
            v.push(l.aggregator_poly.values.as_slice());
        }
        v.extend(self.trash_polys.iter().map(|p| p.values.as_slice()));
        v.extend(self.perm_polys.iter().map(|p| p.values.as_slice()));
        v
    }

    /// Mutable counterpart of [`Self::poly_values`], in the same order.
    fn poly_values_mut(&mut self) -> Vec<&mut [F]> {
        let mut v: Vec<&mut [F]> = Vec::new();
        v.extend(self.advice_polys.iter_mut().map(|p| p.values.as_mut_slice()));
        v.extend(self.instance_polys.iter_mut().map(|p| p.values.as_mut_slice()));
        for l in self.lookups.iter_mut() {
            v.push(l.multiplicities.values.as_mut_slice());
            v.extend(l.helper_polys.iter_mut().map(|p| p.values.as_mut_slice()));
            v.push(l.aggregator_poly.values.as_mut_slice());
        }
        v.extend(self.trash_polys.iter_mut().map(|p| p.values.as_mut_slice()));
        v.extend(self.perm_polys.iter_mut().map(|p| p.values.as_mut_slice()));
        v
    }

    /// The scalar challenges of the trace, in a fixed canonical order.
    ///
    /// These fold exactly like polynomial coordinates, so the transform treats
    /// them as a handful of extra columns.
    fn challenges(&self) -> Vec<F> {
        let mut v = vec![self.beta, self.gamma];
        v.extend_from_slice(&self.theta);
        v.push(self.trash_challenge);
        v.extend_from_slice(&self.y);
        v
    }

    /// Inverse of [`Self::challenges`].
    fn set_challenges(&mut self, c: &[F]) {
        let n_theta = self.theta.len();
        assert_eq!(c.len(), 3 + n_theta + self.y.len());
        self.beta = c[0];
        self.gamma = c[1];
        self.theta.copy_from_slice(&c[2..2 + n_theta]);
        self.trash_challenge = c[2 + n_theta];
        self.y.copy_from_slice(&c[3 + n_theta..]);
    }
}

impl<F: PrimeField> Add<&FoldingProverTrace<F>> for FoldingProverTrace<F> {
    type Output = Self;

    fn add(mut self, rhs: &FoldingProverTrace<F>) -> Self {
        self.advice_polys
            .par_iter_mut()
            .zip(rhs.advice_polys.par_iter())
            .for_each(|(a, b)| *a = a.clone() + b);
        self.instance_polys
            .par_iter_mut()
            .zip(rhs.instance_polys.par_iter())
            .for_each(|(a, b)| *a = a.clone() + b);
        for (l, r) in self.lookups.iter_mut().zip(rhs.lookups.iter()) {
            l.multiplicities = l.multiplicities.clone() + &r.multiplicities;
            l.helper_polys
                .iter_mut()
                .zip(r.helper_polys.iter())
                .for_each(|(a, b)| *a = a.clone() + b);
            l.aggregator_poly = l.aggregator_poly.clone() + &r.aggregator_poly;
        }
        self.trash_polys
            .par_iter_mut()
            .zip(rhs.trash_polys.par_iter())
            .for_each(|(a, b)| *a = a.clone() + b);
        self.perm_polys
            .par_iter_mut()
            .zip(rhs.perm_polys.par_iter())
            .for_each(|(a, b)| *a = a.clone() + b);
        self.beta += rhs.beta;
        self.gamma += rhs.gamma;
        self.theta
            .iter_mut()
            .zip(rhs.theta.iter())
            .for_each(|(a, b)| *a += b);
        self.trash_challenge += rhs.trash_challenge;
        self.y
            .iter_mut()
            .zip(rhs.y.iter())
            .for_each(|(a, b)| *a += b);
        self
    }
}

impl<F: PrimeField> Mul<F> for FoldingProverTrace<F> {
    type Output = Self;

    fn mul(mut self, scalar: F) -> Self {
        self.advice_polys
            .par_iter_mut()
            .for_each(|p| *p = p.clone() * scalar);
        self.instance_polys
            .par_iter_mut()
            .for_each(|p| *p = p.clone() * scalar);
        for l in self.lookups.iter_mut() {
            l.multiplicities = l.multiplicities.clone() * scalar;
            l.helper_polys
                .iter_mut()
                .for_each(|p| *p = p.clone() * scalar);
            l.aggregator_poly = l.aggregator_poly.clone() * scalar;
        }
        self.trash_polys
            .par_iter_mut()
            .for_each(|p| *p = p.clone() * scalar);
        self.perm_polys
            .par_iter_mut()
            .for_each(|p| *p = p.clone() * scalar);
        self.beta *= scalar;
        self.gamma *= scalar;
        self.theta.iter_mut().for_each(|c| *c *= scalar);
        self.trash_challenge *= scalar;
        self.y.iter_mut().for_each(|c| *c *= scalar);
        self
    }
}

/// Computes all 2^|v| subset-product combinations.
///
/// `pow_vec(v)[i]` = product of `v[j]` for each bit `j` set in `i`.
pub(crate) fn pow_vec<F: Field>(vector: &[F]) -> Vec<F> {
    let mut res = vec![F::ONE];
    for x in vector {
        let extended: Vec<F> = res.iter().map(|v| *v * x).collect();
        res.extend(extended);
    }
    res
}

/// Generic linear combination `∑ scalars[i] * elements[i]` via Horner's method.
///
/// Filters out zero-scalar terms and uses in-place Horner accumulation to
/// avoid cloning the buffer multiple times.
pub(crate) fn linear_combination<F, T>(mut buffer: T, elements: &[&T], scalars: &[F]) -> T
where
    F: Field,
    T: for<'a> Add<&'a T, Output = T> + Mul<F, Output = T>,
{
    assert_eq!(elements.len(), scalars.len());

    let (elements, scalars): (Vec<&T>, Vec<&F>) = elements
        .iter()
        .zip(scalars.iter())
        .filter(|(_, s)| !s.is_zero_vartime())
        .unzip();

    let k = elements.len();
    let mut scalars_owned: Vec<F> = scalars.into_iter().cloned().collect();
    scalars_owned.push(F::ONE);

    let mut c = F::ZERO;
    for i in 0..k {
        buffer = buffer * c;
        buffer = buffer + elements[i];
        c = scalars_owned[i] * scalars_owned[i + 1].invert().unwrap();
    }
    buffer * c
}

/// Evaluates the Lagrange basis polynomials of `domain` at `beta`.
///
/// `result[i] = L_i(beta) = Z_n(beta) * ω^i / (n * (beta - ω^i))`
pub(crate) fn eval_lagrange_on_beta<F: PrimeField + WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    beta: &F,
) -> Vec<F> {
    let n = domain.n as usize;
    let omega = domain.get_omega();
    let n_fe = F::from(n as u64);
    let fixed = (beta.pow([n as u64]) - F::ONE) * n_fe.invert().unwrap();

    let mut omegas = Vec::with_capacity(n);
    omegas.push(F::ONE);
    for _ in 1..n {
        let last = *omegas.last().unwrap();
        omegas.push(last * omega);
    }

    omegas
        .into_par_iter()
        .map(|wi| fixed * wi * (*beta - wi).invert().unwrap())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Column-wise folding transform
//
// Batching computes, for every point of the coset D = ζ·⟨ω_e⟩, the trace
// W(x) = ∑_i L_i(x)·wᵢ interpolating the k input traces over the dk-domain
// H = ⟨ω⟩. Writing the inputs as a k × M matrix A (row i = trace i), the
// output is the matrix product
//
//     B[t][m] = ∑_i L_i(ζ ω_e^t) · A[i][m]
//
// so B[·][m] depends only on A[·][m]: the computation is M independent
// applications of the *same* linear map
//
//     T = (evaluate on D) ∘ (interpolate over H) : F^k → F^E.
//
// `batch_traces` applies T to the k basis vectors eᵢ and then re-expands by
// linearity, which flattens a structured map into a dense E × k matrix —
// costing E·k·M multiply-adds, i.e. Θ(k²·d·M), and re-reading every input
// trace E times.
//
// Applying T directly to the M data columns is quasilinear. Since ω = ω_e^s
// with s = E/k, writing t = r + j·s uniquely (r < s, j < k) gives
//
//     ζ ω_e^t = (ζ ω_e^r) · ω^j,
//
// so D splits into s disjoint cosets α_r·H with α_r = ζ ω_e^r, and evaluating
// a degree-<k polynomial on α_r·H is a coefficient twist plus a size-k DFT:
//
//     W_m(α_r ω^j) = ∑_i (c_i α_r^i) ω^{ij}.
//
// Per column: one size-k inverse DFT and s size-k forward DFTs, i.e.
// O(E·log k) instead of O(E·k). Total Θ(k·d·M·log k).
// ─────────────────────────────────────────────────────────────────────────────

/// Target working-set size for one coordinate block, in bytes.
///
/// A block of `B` coordinates touches `E·B` output scalars; keeping that
/// resident while the block is processed turns the scatter into sequential
/// writes per output row.
const BLOCK_TARGET_BYTES: usize = 256 * 1024;

/// Serial radix-2 DIT FFT of size `2^log_n`, using precomputed twiddles.
///
/// Deliberately serial: this runs inside an already-parallel loop over
/// coordinate blocks, and the transforms are tiny (size `k`). Mirrors
/// `best_fft_with_twiddles` butterfly-for-butterfly so the convention matches
/// `EvaluationDomain` exactly.
fn fft_serial<F: Field>(a: &mut [F], twiddles: &[F], log_n: u32) {
    if log_n == 0 {
        return;
    }
    let n = a.len();
    debug_assert_eq!(n, 1 << log_n);

    for i in 0..n {
        let ri = i.reverse_bits() >> (usize::BITS - log_n);
        if i < ri {
            a.swap(ri, i);
        }
    }

    let mut chunk = 2usize;
    let mut twiddle_chunk = n / 2;
    for _ in 0..log_n {
        for coeffs in a.chunks_mut(chunk) {
            let (left, right) = coeffs.split_at_mut(chunk / 2);

            // Case when the twiddle factor is one.
            let (x0, left) = left.split_at_mut(1);
            let (y0, right) = right.split_at_mut(1);
            let t = y0[0];
            y0[0] = x0[0];
            x0[0] += t;
            y0[0] -= t;

            for (i, (x, y)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
                let mut t = *y;
                t *= twiddles[(i + 1) * twiddle_chunk];
                *y = *x;
                *x += t;
                *y -= t;
            }
        }
        chunk *= 2;
        twiddle_chunk /= 2;
    }
}

/// Precomputed data for the column transform `T : F^k → F^E` of a dk-domain.
struct ColumnTransform<F> {
    log_k: u32,
    k: usize,
    ext_len: usize,
    /// Number of `H`-cosets covering the evaluation coset, `s = E / k`.
    s: usize,
    omega_tw: Vec<F>,
    omega_inv_tw: Vec<F>,
    k_inv: F,
    /// `twist[r][i] = (ζ ω_e^r)^i`
    twist: Vec<Vec<F>>,
}

impl<F: PrimeField + WithSmallOrderMulGroup<3>> ColumnTransform<F> {
    fn new(dk_domain: &EvaluationDomain<F>) -> Self {
        let log_k = dk_domain.k();
        let k = 1usize << log_k;
        let ext_len = dk_domain.extended_len();
        assert_eq!(ext_len % k, 0);
        let s = ext_len / k;

        let (omega_tw, omega_inv_tw) = if log_k == 0 {
            (Vec::new(), Vec::new())
        } else {
            (
                compute_twiddles(&dk_domain.get_omega(), log_k),
                compute_twiddles(&dk_domain.get_omega_inv(), log_k),
            )
        };

        // α_r = ζ·ω_e^r. The coset shift is ζ = F::ZETA because that is what
        // `EvaluationDomain::distribute_powers_zeta` applies (it exploits
        // ζ³ = 1 to twist coefficient j by ζ^{j mod 3} = ζ^j).
        let omega_e = dk_domain.get_extended_omega();
        let mut alpha = F::ZETA;
        let mut twist = Vec::with_capacity(s);
        for _ in 0..s {
            let mut pows = Vec::with_capacity(k);
            let mut p = F::ONE;
            for _ in 0..k {
                pows.push(p);
                p *= alpha;
            }
            twist.push(pows);
            alpha *= omega_e;
        }

        ColumnTransform {
            log_k,
            k,
            ext_len,
            s,
            omega_tw,
            omega_inv_tw,
            k_inv: F::from(k as u64).invert().unwrap(),
            twist,
        }
    }

    /// Interpolates `column` (the `k` input values of one coordinate) over `H`,
    /// leaving its coefficients in `column`.
    #[inline]
    fn interpolate(&self, column: &mut [F]) {
        fft_serial(column, &self.omega_inv_tw, self.log_k);
        for x in column.iter_mut() {
            *x *= self.k_inv;
        }
    }

    /// Evaluates the interpolant whose coefficients are in `coeffs` on the
    /// whole coset, writing `out[t] = W(ζ ω_e^t)`.
    ///
    /// `buf` is scratch of length `k`.
    #[inline]
    fn evaluate_all(&self, coeffs: &[F], buf: &mut [F], out: &mut [F]) {
        for r in 0..self.s {
            let tw = &self.twist[r];
            for i in 0..self.k {
                buf[i] = coeffs[i] * tw[i];
            }
            fft_serial(buf, &self.omega_tw, self.log_k);
            for (j, v) in buf.iter().enumerate() {
                out[r + j * self.s] = *v;
            }
        }
    }

    /// As [`Self::evaluate_all`], but scattering straight into the output rows
    /// at coordinate `m` — avoids a per-coordinate staging buffer.
    #[inline]
    fn evaluate_all_into_rows(&self, coeffs: &[F], buf: &mut [F], rows: &mut [&mut [F]], m: usize) {
        for r in 0..self.s {
            let tw = &self.twist[r];
            for i in 0..self.k {
                buf[i] = coeffs[i] * tw[i];
            }
            fft_serial(buf, &self.omega_tw, self.log_k);
            for (j, v) in buf.iter().enumerate() {
                rows[r + j * self.s][m] = *v;
            }
        }
    }
}

/// Splits `rows` (the `E` output buffers of one polynomial slot) into
/// per-coordinate-block views, so that blocks can be processed in parallel
/// while each thread holds an exclusive `&mut` into every row.
fn split_into_blocks<'a, F>(
    rows: Vec<&'a mut [F]>,
    len: usize,
    block: usize,
) -> Vec<(usize, Vec<&'a mut [F]>)> {
    let mut blocks = Vec::with_capacity(len.div_ceil(block));
    let mut rest = rows;
    let mut off = 0;
    while off < len {
        let take = block.min(len - off);
        let mut here = Vec::with_capacity(rest.len());
        let mut next = Vec::with_capacity(rest.len());
        for r in rest {
            let (a, b) = r.split_at_mut(take);
            here.push(a);
            next.push(b);
        }
        blocks.push((off, here));
        rest = next;
        off += take;
    }
    blocks
}

/// Number of coordinates per block, given `ext_len` output rows.
fn block_size<F>(ext_len: usize) -> usize {
    (BLOCK_TARGET_BYTES / (ext_len * size_of::<F>())).max(1)
}

/// Re-groups the polynomial buffers of the output traces by slot:
/// `result[p][t]` is the buffer of slot `p` in output trace `t`.
fn regroup_by_slot<F: PrimeField>(
    out_traces: &mut [FoldingProverTrace<F>],
    n_slots: usize,
) -> Vec<Vec<&mut [F]>> {
    let mut per_slot: Vec<Vec<&mut [F]>> =
        (0..n_slots).map(|_| Vec::with_capacity(out_traces.len())).collect();
    for trace in out_traces.iter_mut() {
        let slots = trace.poly_values_mut();
        assert_eq!(slots.len(), n_slots);
        for (p, slice) in slots.into_iter().enumerate() {
            per_slot[p].push(slice);
        }
    }
    per_slot
}

/// Folds the scalar challenges of the traces through the column transform.
///
/// There are only a handful of these, so they get the straightforward
/// staged-buffer treatment.
fn fold_challenges<F: PrimeField + WithSmallOrderMulGroup<3>>(
    ct: &ColumnTransform<F>,
    traces: &[&FoldingProverTrace<F>],
    out_traces: &mut [FoldingProverTrace<F>],
) {
    let per_trace: Vec<Vec<F>> = traces.iter().map(|t| t.challenges()).collect();
    let n_ch = per_trace[0].len();

    let mut column = vec![F::ZERO; ct.k];
    let mut buf = vec![F::ZERO; ct.k];
    let mut out = vec![F::ZERO; ct.ext_len];
    let mut folded: Vec<Vec<F>> = (0..ct.ext_len).map(|_| vec![F::ZERO; n_ch]).collect();

    for c in 0..n_ch {
        for (i, x) in column.iter_mut().enumerate() {
            *x = per_trace[i][c];
        }
        ct.interpolate(&mut column);
        ct.evaluate_all(&column, &mut buf, &mut out);
        for (t, v) in out.iter().enumerate() {
            folded[t][c] = *v;
        }
    }

    for (trace, ch) in out_traces.iter_mut().zip(folded.iter()) {
        trace.set_challenges(ch);
    }
}

/// Interpolates `k` traces over the dk-domain, returning one `FoldingProverTrace`
/// per extended dk-domain (coset) point.
///
/// Same contract as [`batch_traces`], but transposes the loop nest: each input
/// scalar is read once instead of `ext_len` times, and the per-coordinate
/// combination is done with FFTs rather than densely — `Θ(k·d·M·log k)`
/// instead of `Θ(k²·d·M)`.
pub(crate) fn batch_traces_fft<F: PrimeField + WithSmallOrderMulGroup<3>>(
    dk_domain: &EvaluationDomain<F>,
    traces: &[&FoldingProverTrace<F>],
) -> Vec<FoldingProverTrace<F>> {
    let ct = ColumnTransform::new(dk_domain);
    assert_eq!(
        traces.len(),
        ct.k,
        "batch_traces_fft expects exactly 2^dk_domain.k() traces"
    );

    let inputs: Vec<Vec<&[F]>> = traces.iter().map(|t| t.poly_values()).collect();
    let n_slots = inputs[0].len();
    let block = block_size::<F>(ct.ext_len);

    let mut out_traces: Vec<FoldingProverTrace<F>> =
        (0..ct.ext_len).map(|_| FoldingProverTrace::zero_like(traces[0])).collect();

    for (p, rows) in regroup_by_slot(&mut out_traces, n_slots).into_iter().enumerate() {
        let slot_inputs: Vec<&[F]> = inputs.iter().map(|t| t[p]).collect();
        let len = slot_inputs[0].len();
        assert!(
            slot_inputs.iter().all(|s| s.len() == len),
            "input traces disagree on the length of polynomial slot {p}"
        );
        assert!(
            rows.iter().all(|r| r.len() == len),
            "output buffer length mismatch on polynomial slot {p}"
        );

        split_into_blocks(rows, len, block).into_par_iter().for_each(|(off, mut rows)| {
            let mut column = vec![F::ZERO; ct.k];
            let mut buf = vec![F::ZERO; ct.k];
            let n_local = rows[0].len();
            for m in 0..n_local {
                for (i, c) in column.iter_mut().enumerate() {
                    *c = slot_inputs[i][off + m];
                }
                ct.interpolate(&mut column);
                ct.evaluate_all_into_rows(&column, &mut buf, &mut rows, m);
            }
        });
    }

    fold_challenges(&ct, traces, &mut out_traces);
    out_traces
}

/// Interpolates `k` traces over the dk-domain, returning one `FoldingProverTrace`
/// per extended dk-domain point.
///
/// At dk-extended point `t`, the returned trace is `∑_i L_i(x_t) * traces[i]`.
///
/// Superseded by [`batch_traces_fft`], which computes the same thing in
/// `Θ(k·d·M·log k)` instead of `Θ(k²·d·M)`. Kept as the reference statement of
/// what the transform is, and as an equivalence oracle.
#[allow(dead_code)]
pub(crate) fn batch_traces<F: PrimeField + WithSmallOrderMulGroup<3>>(
    dk_domain: &EvaluationDomain<F>,
    traces: &[&FoldingProverTrace<F>],
) -> Vec<FoldingProverTrace<F>> {
    let lagrange_polys: Vec<_> = (0..traces.len())
        .map(|i| {
            let mut l = dk_domain.empty_lagrange();
            l[i] = F::ONE;
            l
        })
        .map(|p| dk_domain.lagrange_to_coeff(p))
        .map(|p| dk_domain.coeff_to_extended(p))
        .collect();

    let ext_len = lagrange_polys[0].values.len();

    (0..ext_len)
        .map(|t| {
            let buf = FoldingProverTrace::zero_like(traces[0]);
            let coeffs: Vec<F> = lagrange_polys.iter().map(|l| l.values[t]).collect();
            linear_combination(buf, traces, &coeffs)
        })
        .collect()
}

/// Folds `traces` into a single trace at `gamma` via Lagrange interpolation.
pub(crate) fn fold_traces<F: PrimeField + WithSmallOrderMulGroup<3>>(
    dk_domain: &EvaluationDomain<F>,
    traces: &[&FoldingProverTrace<F>],
    gamma: &F,
) -> FoldingProverTrace<F> {
    let lagrange_polys: Vec<_> = (0..traces.len())
        .map(|i| {
            let mut l = dk_domain.empty_lagrange();
            l[i] = F::ONE;
            l
        })
        .map(|p| dk_domain.lagrange_to_coeff(p))
        .collect();

    let lagrange_at_gamma: Vec<F> = lagrange_polys
        .iter()
        .map(|p| eval_polynomial(&p.values, *gamma))
        .collect();

    let buf = FoldingProverTrace::zero_like(traces[0]);
    linear_combination(buf, traces, &lagrange_at_gamma)
}

#[cfg(test)]
mod tests {
    use midnight_curves::Fq;

    use super::*;

    /// Dimensions of a synthetic [`FoldingProverTrace`].
    #[derive(Clone, Copy)]
    struct Shape {
        n: usize,
        advice: usize,
        instance: usize,
        lookups: usize,
        helpers_per_lookup: usize,
        trash: usize,
        perm_sets: usize,
        n_theta: usize,
        n_y: usize,
    }

    /// A trace of the given shape filled with pseudorandom values.
    ///
    /// Values come from an in-field linear recurrence rather than `Fq::random`
    /// so the test stays cheap. They are dense (neither zero nor one), so none
    /// of the zero/one fast paths in the polynomial operators are hit, and the
    /// zero-scalar filter in [`linear_combination`] never fires.
    fn pseudorandom(shape: &Shape, seed: u64) -> FoldingProverTrace<Fq> {
        let mut state = Fq::from(seed).square() + Fq::from(0x9e37_79b9_7f4a_7c15);
        let mult = Fq::from(0x5851_f42d_4c95_7f2d);
        let incr = Fq::from(0x1405_7b7e_f767_814f);
        let mut vals = move |len: usize| {
            (0..len)
                .map(|_| {
                    state = state * mult + incr;
                    state
                })
                .collect::<Vec<Fq>>()
        };
        macro_rules! poly {
            () => {
                Polynomial {
                    values: vals(shape.n),
                    _marker: PhantomData,
                }
            };
        }

        FoldingProverTrace {
            advice_polys: (0..shape.advice).map(|_| poly!()).collect(),
            instance_polys: (0..shape.instance).map(|_| poly!()).collect(),
            lookups: (0..shape.lookups)
                .map(|_| FoldingLogupTrace {
                    multiplicities: poly!(),
                    helper_polys: (0..shape.helpers_per_lookup).map(|_| poly!()).collect(),
                    aggregator_poly: poly!(),
                })
                .collect(),
            trash_polys: (0..shape.trash).map(|_| poly!()).collect(),
            perm_polys: (0..shape.perm_sets).map(|_| poly!()).collect(),
            beta: vals(1)[0],
            gamma: vals(1)[0],
            theta: vals(shape.n_theta),
            trash_challenge: vals(1)[0],
            y: vals(shape.n_y),
        }
    }

    /// Assert two traces are equal coordinate by coordinate.
    fn assert_traces_eq(a: &FoldingProverTrace<Fq>, b: &FoldingProverTrace<Fq>, ctx: &str) {
        let (pa, pb) = (a.poly_values(), b.poly_values());
        assert_eq!(pa.len(), pb.len(), "{ctx}: number of polynomials differs");
        for (p, (sa, sb)) in pa.iter().zip(pb.iter()).enumerate() {
            assert_eq!(
                sa.len(),
                sb.len(),
                "{ctx}: length of polynomial {p} differs"
            );
            for (m, (x, y)) in sa.iter().zip(sb.iter()).enumerate() {
                assert!(x == y, "{ctx}: polynomial {p} differs at coordinate {m}");
            }
        }
        let (ca, cb) = (a.challenges(), b.challenges());
        assert_eq!(ca.len(), cb.len(), "{ctx}: number of challenges differs");
        for (c, (x, y)) in ca.iter().zip(cb.iter()).enumerate() {
            assert!(x == y, "{ctx}: challenge {c} differs");
        }
    }

    /// [`batch_traces_fft`] must agree with the dense [`batch_traces`] exactly.
    ///
    /// Both compute the same linear map, so this is a strict equality check, not
    /// an approximate one. The degrees are chosen to sweep the number of
    /// `H`-cosets `s = ext_len / k` over 1, 2, 4 and 8, since the coset
    /// decomposition and its stride-`s` scatter are what the fast path gets
    /// wrong if it is wrong at all.
    #[test]
    fn batch_traces_fft_matches_batch_traces() {
        let shapes = [
            // A shape in the spirit of a real PLONK trace.
            Shape {
                n: 32,
                advice: 4,
                instance: 2,
                lookups: 2,
                helpers_per_lookup: 3,
                trash: 1,
                perm_sets: 3,
                n_theta: 4,
                n_y: 24,
            },
            // No lookups, no trashcans, and empty challenge vectors — exercises
            // the empty-slice paths through `challenges`/`set_challenges`.
            Shape {
                n: 16,
                advice: 3,
                instance: 1,
                lookups: 0,
                helpers_per_lookup: 0,
                trash: 0,
                perm_sets: 1,
                n_theta: 0,
                n_y: 0,
            },
        ];

        for (si, shape) in shapes.iter().enumerate() {
            for degree in [2u32, 3, 5, 9] {
                for log_k in 1u32..=4 {
                    let k = 1usize << log_k;
                    let dk_domain = EvaluationDomain::<Fq>::new(degree, log_k);
                    let ext_len = dk_domain.extended_len();

                    let traces: Vec<FoldingProverTrace<Fq>> =
                        (0..k).map(|i| pseudorandom(shape, 1 + i as u64)).collect();
                    let refs: Vec<&FoldingProverTrace<Fq>> = traces.iter().collect();

                    let dense = batch_traces(&dk_domain, &refs);
                    let fast = batch_traces_fft(&dk_domain, &refs);

                    assert_eq!(dense.len(), ext_len, "dense returned the wrong count");
                    assert_eq!(fast.len(), ext_len, "fft returned the wrong count");

                    for (t, (d, f)) in dense.iter().zip(fast.iter()).enumerate() {
                        let ctx = format!(
                            "shape {si}, degree {degree}, k {k}, s {}, point {t}",
                            ext_len / k
                        );
                        assert_traces_eq(d, f, &ctx);
                    }
                }
            }
        }
    }
}
