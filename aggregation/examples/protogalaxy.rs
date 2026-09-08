//! Protogalaxy folding benchmark.
//!
//! Folds `NB_FOLDED` SHA-256 preimage proofs into a single accumulator and
//! verifies the fold. The per-step timings of the folding prover are printed by
//! `ProtogalaxyProver::fold` itself (on stderr).
//!
//! DO NOT add this example to the CI as it is slow.
//!
//! ```text
//! cargo run --release --example protogalaxy                  # widths 2..64
//! cargo run --release --example protogalaxy -- 16            # a single width
//! PG_REPS=5 cargo run --release --example protogalaxy -- 16  # 5 folds, for averaging
//! PG_PLONK=1 cargo run --release --example protogalaxy -- 8  # + PLONK baseline
//! PG_PLONK_ONLY=128 cargo run --release --example protogalaxy # 128 proofs, no fold
//! ```
//!
//! ## Circuit size
//!
//! Step 5 of the folding prover materialises `ext_len` copies of the whole
//! prover trace, where `ext_len ≈ NB_FOLDED · (degree − 1)`. Memory therefore
//! grows linearly in the folding width, and at width 32 it is roughly 16× what
//! a single standard proof needs. `N_SHA` in `circuits/sha_multi.rs` controls
//! the circuit size and hence how large a width fits in RAM.
//!
//! ## SRS
//!
//! Uses a locally generated `ParamsKZG::unsafe_setup` rather than the Filecoin
//! SRS, so the benchmark needs no multi-hundred-MB download. This is *not* a
//! trusted setup and must never be used for anything but benchmarking — timings
//! are unaffected, since only the SRS size matters for performance.

#[path = "circuits/sha_multi.rs"]
mod sha_multi;

use std::{env, time::Instant};

use midnight_circuits::hash::poseidon::PoseidonState;
use midnight_curves::Bls12;
use midnight_proofs::{
    circuit::Value,
    plonk::protogalaxy::{ProtogalaxyProver, ProtogalaxyVerifier},
    poly::{commitment::Guard, kzg::params::ParamsKZG},
    transcript::{CircuitTranscript, Transcript},
};
use midnight_zk_stdlib::{
    cost_model, optimal_k, prove, setup_pk, setup_vk, verify, MidnightCircuit, MidnightPK,
    MidnightVK, Relation,
};
use rand::rngs::OsRng;
use sha_multi::{ShaMultiCircuit as ShaPreimageCircuit, N_SHA};

type F = midnight_curves::Fq;

/// Folding widths run when no width is given on the command line.
const DEFAULT_WIDTHS: [usize; 6] = [2, 4, 8, 16, 32, 64];

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Proves `n` independent instances of the same circuit with the standard PLONK
/// prover — no folding — and reports the total and the per-proof cost.
///
/// The loop is deliberately sequential: `prove` is already internally parallel
/// across all cores, exactly as `fold` is, so `total / n` is the per-proof
/// latency and is directly comparable to a fold of width `n`. Running the
/// outer loop with rayon instead would nest parallelism and measure throughput
/// under contention, which is a different quantity.
fn plonk_bench(
    srs: &ParamsKZG<Bls12>,
    vk: &MidnightVK,
    pk: &MidnightPK<ShaPreimageCircuit>,
    relation: &ShaPreimageCircuit,
    n: usize,
) {
    eprintln!("\n══════════ standard PLONK: {n} independent proofs ══════════");

    // Witness sampling happens outside the timed region; `prove` synthesises
    // the circuit itself, so the timed span is nothing but prove calls.
    let pairs: Vec<_> = (0..n).map(|_| sha_multi::random_instance()).collect();

    let mut proofs = Vec::with_capacity(n);
    let mut times = Vec::with_capacity(n);
    let t_all = Instant::now();
    for (instance, witness) in &pairs {
        let t = Instant::now();
        let proof = prove::<ShaPreimageCircuit, PoseidonState<F>>(
            srs, pk, relation, instance, *witness, OsRng,
        )
        .expect("PLONK prove failed");
        times.push(t.elapsed());
        proofs.push(proof);
    }
    let total = t_all.elapsed();

    times.sort();
    eprintln!("[PLONK {n}× prove] TOTAL             {total:>8.2?}");
    eprintln!(
        "[PLONK] per proof (total / {n})        {:>8.2?}",
        total / n as u32
    );
    eprintln!(
        "[PLONK] min {:.2?}   median {:.2?}   max {:.2?}",
        times[0],
        times[n / 2],
        times[n - 1]
    );
    eprintln!("[PLONK] proof size                   {} B", proofs[0].len());

    // Verify everything, so the timings stand for genuinely valid proofs.
    let params_verifier = srs.verifier_params();
    let t = Instant::now();
    for ((instance, _), proof) in pairs.iter().zip(proofs.iter()) {
        verify::<ShaPreimageCircuit, PoseidonState<F>>(&params_verifier, vk, instance, None, proof)
            .expect("PLONK verify failed");
    }
    let vt = t.elapsed();
    eprintln!(
        "[PLONK {n}× verify] TOTAL            {vt:>8.2?}   per proof {:>8.2?}",
        vt / n as u32
    );
}

/// Folds `NB` instances and reports the timings.
///
/// The per-step breakdown comes from `ProtogalaxyProver::fold`, which prints it
/// to stderr; this function adds the totals.
fn fold_bench<const NB: usize>(
    srs: &ParamsKZG<Bls12>,
    vk: &MidnightVK,
    pk: &MidnightPK<ShaPreimageCircuit>,
    relation: &ShaPreimageCircuit,
    ext_len: usize,
    reps: usize,
) {
    eprintln!("\n══════════ NB_FOLDED = {NB}  (ext_len = {ext_len}) ══════════");
    for rep in 0..reps.max(1) {
        eprintln!("---------- run {} of {} ----------", rep + 1, reps.max(1));
        fold_once::<NB>(srs, vk, pk, relation);
    }
}

/// One fold: fresh witnesses, fold, verify.
fn fold_once<const NB: usize>(
    srs: &ParamsKZG<Bls12>,
    vk: &MidnightVK,
    pk: &MidnightPK<ShaPreimageCircuit>,
    relation: &ShaPreimageCircuit,
) {
    let pairs: [(_, _); NB] = core::array::from_fn(|_| sha_multi::random_instance());

    let t = Instant::now();
    let mut circuits = Vec::with_capacity(NB);
    let mut instances: Vec<Vec<Vec<F>>> = Vec::with_capacity(NB);
    for (instance, witness) in &pairs {
        circuits.push(MidnightCircuit::new(
            relation,
            Value::known(*instance),
            Value::known(*witness),
            None,
        ));
        instances.push(vec![
            vec![],
            ShaPreimageCircuit::format_instance(instance).expect("format_instance failed"),
        ]);
    }
    eprintln!("[witness gen]                        {:>8.2?}", t.elapsed());

    let instances_slices: Vec<Vec<&[F]>> = instances
        .iter()
        .map(|cols| cols.iter().map(|c| c.as_slice()).collect())
        .collect();
    let instances_ref: Vec<&[&[F]]> = instances_slices.iter().map(|c| c.as_slice()).collect();

    // Optional standard-PLONK baseline: NB independent proofs.
    if env::var("PG_PLONK").is_ok() {
        let t = Instant::now();
        let mut total = 0;
        let mut proofs = Vec::with_capacity(NB);
        for (instance, witness) in &pairs {
            let proof = prove::<ShaPreimageCircuit, PoseidonState<F>>(
                srs, pk, relation, instance, *witness, OsRng,
            )
            .expect("PLONK prove failed");
            total += proof.len();
            proofs.push(proof);
        }
        eprintln!(
            "[PLONK {NB}× prove]                    {:>8.2?}   proof size: {total} B",
            t.elapsed()
        );

        let t = Instant::now();
        let params_verifier = srs.verifier_params();
        for ((instance, _), proof) in pairs.iter().zip(proofs.iter()) {
            verify::<ShaPreimageCircuit, PoseidonState<F>>(
                &params_verifier,
                vk,
                instance,
                None,
                proof,
            )
            .expect("PLONK verify failed");
        }
        eprintln!("[PLONK {NB}× verify]                   {:>8.2?}", t.elapsed());
    }

    // ── Fold ──────────────────────────────────────────────────────────────────
    let t = Instant::now();
    let mut prover_transcript = CircuitTranscript::<PoseidonState<F>>::init();
    ProtogalaxyProver::<_, _, NB>::fold(
        srs,
        pk.pk().clone(),
        circuits,
        &instances_ref,
        OsRng,
        &mut prover_transcript,
    )
    .expect("Protogalaxy folding prover failed");
    let fold_time = t.elapsed();

    let proof_bytes = prover_transcript.finalize();
    eprintln!(
        "[protogalaxy] FOLD TOTAL             {fold_time:>8.2?}   proof size: {} B",
        proof_bytes.len()
    );

    // ── Verify the fold ───────────────────────────────────────────────────────
    let t = Instant::now();
    let mut verifier_transcript =
        CircuitTranscript::<PoseidonState<F>>::init_from_bytes(&proof_bytes);
    let guard = ProtogalaxyVerifier::<_, _, NB>::fold(vk.vk(), &instances_ref, &mut verifier_transcript)
        .expect("Protogalaxy folding verifier failed");
    let fold_verify_time = t.elapsed();

    let params_verifier = srs.verifier_params();
    let t = Instant::now();
    guard
        .verify(&params_verifier)
        .expect("Protogalaxy accumulator verification failed");
    let kzg_verify_time = t.elapsed();

    eprintln!(
        "[protogalaxy] verify                 {fold_verify_time:>8.2?}   KZG pairing: {kzg_verify_time:.2?}"
    );
    eprintln!("Protogalaxy folding succeeded for {NB} instances.");
}

fn main() {
    let relation = ShaPreimageCircuit;

    let widths: Vec<usize> = match env::args().nth(1) {
        Some(a) => vec![a.parse().expect("width must be a power-of-two integer")],
        None => DEFAULT_WIDTHS.to_vec(),
    };
    // Number of folds per width; the per-step lines repeat once per run so an
    // external harness can average them.
    let reps = env_usize("PG_REPS", 1);

    // ── Setup ─────────────────────────────────────────────────────────────────
    let t = Instant::now();
    let k = optimal_k(&relation);
    let model = cost_model(&relation, Some(k));
    let srs = ParamsKZG::<Bls12>::unsafe_setup(k, OsRng);
    let vk = setup_vk(&srs, &relation);
    let pk = setup_pk(&relation, &vk);
    println!("Setup completed in {:.2?}", t.elapsed());

    let degree = model.max_deg;
    println!(
        "Circuit: {N_SHA}× SHA-256 per instance, K={k} ({} rows), cs degree {degree}\n\
         advice cols {}, fixed cols {}, lookups {}, trashcans {}, permutations {}",
        1usize << k,
        model.advice_columns,
        model.fixed_columns,
        model.lookups,
        model.trashcans,
        model.permutations,
    );

    // Standard-PLONK baseline: N independent proofs, no folding at all.
    let plonk_only = env_usize("PG_PLONK_ONLY", 0);
    if plonk_only > 0 {
        plonk_bench(&srs, &vk, &pk, &relation, plonk_only);
        return;
    }

    // ext_len of the dk-domain: next power of two >= NB * (degree - 1).
    let ext_len_of = |nb: usize| (nb * (degree - 1)).next_power_of_two().max(nb);

    for w in widths {
        let ext_len = ext_len_of(w);
        match w {
            2 => fold_bench::<2>(&srs, &vk, &pk, &relation, ext_len, reps),
            4 => fold_bench::<4>(&srs, &vk, &pk, &relation, ext_len, reps),
            8 => fold_bench::<8>(&srs, &vk, &pk, &relation, ext_len, reps),
            16 => fold_bench::<16>(&srs, &vk, &pk, &relation, ext_len, reps),
            32 => fold_bench::<32>(&srs, &vk, &pk, &relation, ext_len, reps),
            64 => fold_bench::<64>(&srs, &vk, &pk, &relation, ext_len, reps),
            other => panic!("width {other} not instantiated; add it to the match in main()"),
        }
    }
}
