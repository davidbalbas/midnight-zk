//! Accordion (Protogalaxy k-folding) vs k-Plonk benchmark.
//!
//! For a given circuit and width `k`, measures one of two schemes:
//!
//! - `accordion`: one Protogalaxy fold of `k` fresh instances, then its
//!   verification (fold verifier + KZG check of the returned guard). The fold
//!   is one-shot: it ends with a corrected PLONK proof of the folded instance,
//!   so verification covers all `k` instances.
//! - `plonk`: `k` independent PLONK proofs of `k` fresh instances, then `k`
//!   independent verifications.
//!
//! Each repetition prints one `BENCH {json}` line on stdout. This example is
//! meant to be driven by `aggregation/scripts/bench_accordion.py`, which runs
//! one process per (circuit, scheme, k), averages the repetitions and renders
//! the comparison tables. It can also be run by hand:
//!
//! ```text
//! cargo run --release -p midnight-aggregation --example accordion_bench -- \
//!     <sha|ecdsa> <accordion|plonk> <k> [reps]
//! ```
//!
//! DO NOT add this example to the CI as it is slow.
//!
//! ## What is measured
//!
//! - prover: wall-clock time of `fold` (resp. the `k` sequential `prove`
//!   calls). Circuit synthesis is included; sampling the instance–witness pairs
//!   is not.
//! - prover CPU: user + system CPU time of the whole process over the same
//!   span, from which the driver derives the share of cores in use.
//! - peak RSS: on Linux, the peak resident set size over the prover span (the
//!   kernel's high-water mark is reset right before proving, so setup
//!   transients do not count, while the SRS and keys held in memory do). On
//!   other platforms this falls back to the peak over the whole process.
//! - verifier: wall-clock time of the verification described above.
//! - proof size: bytes of the folding proof (resp. of the `k` PLONK proofs).
//!
//! Both schemes use the Poseidon transcript, i.e. the one an in-circuit
//! verifier would need.
//!
//! ## SRS
//!
//! Uses a locally generated `ParamsKZG::unsafe_setup`. This is *not* a trusted
//! setup and must never be used for anything but benchmarking; timings are
//! unaffected, since only the SRS size matters for performance.

#[path = "circuits/ecdsa_threshold.rs"]
mod ecdsa_threshold;
#[allow(dead_code)] // `NB_PUBLIC_INPUTS` is unused here.
#[path = "circuits/sha_multi.rs"]
mod sha_multi;

use std::{
    env,
    fmt::Debug,
    process,
    time::{Duration, Instant},
};

use ecdsa_threshold::EcdsaThresholdCircuit;
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
use sha_multi::{ShaMultiCircuit, N_SHA};

type F = midnight_curves::Fq;
type H = PoseidonState<F>;

/// Folding widths with an instantiated `ProtogalaxyProver`.
const ACCORDION_WIDTHS: [usize; 6] = [2, 4, 8, 16, 32, 64];

/// Process resource usage, read from the OS.
mod usage {
    use std::time::Duration;

    /// User + system CPU time consumed so far by the whole process (all
    /// threads).
    #[cfg(unix)]
    pub fn cpu_time() -> Duration {
        let ru = rusage();
        let tv = |t: libc::timeval| Duration::new(t.tv_sec as u64, t.tv_usec as u32 * 1000);
        tv(ru.ru_utime) + tv(ru.ru_stime)
    }

    #[cfg(not(unix))]
    pub fn cpu_time() -> Duration {
        Duration::ZERO
    }

    /// Resets the kernel's peak-RSS counter (`VmHWM`) to the current RSS.
    /// Returns `false` where that is unsupported (anything but Linux).
    pub fn reset_peak_rss() -> bool {
        std::fs::write("/proc/self/clear_refs", "5").is_ok()
    }

    /// Peak RSS in bytes: `VmHWM` (since the last reset) on Linux, otherwise
    /// the process-lifetime `ru_maxrss`.
    pub fn peak_rss() -> u64 {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            let hwm_kb = status
                .lines()
                .find_map(|l| l.strip_prefix("VmHWM:"))
                .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok());
            if let Some(kb) = hwm_kb {
                return kb * 1024;
            }
        }
        max_rss_fallback()
    }

    #[cfg(unix)]
    fn max_rss_fallback() -> u64 {
        let max_rss = rusage().ru_maxrss as u64;
        // macOS reports bytes, the other Unixes KiB.
        if cfg!(target_os = "macos") {
            max_rss
        } else {
            max_rss * 1024
        }
    }

    #[cfg(not(unix))]
    fn max_rss_fallback() -> u64 {
        0
    }

    #[cfg(unix)]
    fn rusage() -> libc::rusage {
        // SAFETY: `getrusage` only writes into the provided struct, for which
        // all-zeroes is a valid bit pattern.
        unsafe {
            let mut ru: libc::rusage = std::mem::zeroed();
            libc::getrusage(libc::RUSAGE_SELF, &mut ru);
            ru
        }
    }
}

/// Measurements of one repetition.
struct Measurement {
    prover: Duration,
    prover_cpu: Duration,
    peak_rss: u64,
    verifier: Duration,
    proof_bytes: usize,
}

/// Runs `f` as the prover span: returns its output, wall time, CPU time and
/// peak RSS.
fn measure_prover<T>(f: impl FnOnce() -> T) -> (T, Duration, Duration, u64) {
    usage::reset_peak_rss();
    let cpu = usage::cpu_time();
    let t = Instant::now();
    let out = f();
    let wall = t.elapsed();
    let cpu = usage::cpu_time().saturating_sub(cpu);
    (out, wall, cpu, usage::peak_rss())
}

/// Everything set up once per process and shared by all repetitions.
struct Ctx<'a, R: Relation> {
    relation: &'a R,
    sample: fn() -> (R::Instance, R::Witness),
    k_log: u32,
    srs: ParamsKZG<Bls12>,
    vk: MidnightVK,
    pk: MidnightPK<R>,
}

/// One Protogalaxy fold of `NB` fresh instances, and its verification.
fn accordion_once<R: Relation, const NB: usize>(ctx: &Ctx<R>) -> Measurement
where
    R::Error: Debug,
{
    let pairs: Vec<_> = (0..NB).map(|_| (ctx.sample)()).collect();

    let circuits: Vec<_> = pairs
        .iter()
        .map(|(instance, witness)| {
            MidnightCircuit::new(
                ctx.relation,
                Value::known(instance.clone()),
                Value::known(witness.clone()),
                Some(ctx.k_log),
            )
        })
        .collect();
    let instances: Vec<Vec<Vec<F>>> = pairs
        .iter()
        .map(|(instance, _)| {
            vec![
                vec![],
                R::format_instance(instance).expect("format_instance failed"),
            ]
        })
        .collect();
    let instances_slices: Vec<Vec<&[F]>> = instances
        .iter()
        .map(|cols| cols.iter().map(|c| c.as_slice()).collect())
        .collect();
    let instances_ref: Vec<&[&[F]]> = instances_slices.iter().map(|c| c.as_slice()).collect();

    // `fold` takes the proving key by value.
    let pk = ctx.pk.pk().clone();
    let params_verifier = ctx.srs.verifier_params();

    let (proof, prover, prover_cpu, peak_rss) = measure_prover(|| {
        let mut transcript = CircuitTranscript::<H>::init();
        ProtogalaxyProver::<_, _, NB>::fold(
            &ctx.srs,
            pk,
            circuits,
            &instances_ref,
            OsRng,
            &mut transcript,
        )
        .expect("Protogalaxy folding prover failed");
        transcript.finalize()
    });

    let t = Instant::now();
    let mut transcript = CircuitTranscript::<H>::init_from_bytes(&proof);
    let guard = ProtogalaxyVerifier::<_, _, NB>::fold(ctx.vk.vk(), &instances_ref, &mut transcript)
        .expect("Protogalaxy folding verifier failed");
    guard
        .verify(&params_verifier)
        .expect("Protogalaxy accumulator verification failed");
    let verifier = t.elapsed();

    Measurement {
        prover,
        prover_cpu,
        peak_rss,
        verifier,
        proof_bytes: proof.len(),
    }
}

/// `k` independent PLONK proofs of `k` fresh instances, and their
/// verification.
///
/// The loop is deliberately sequential: `prove` is already internally parallel
/// across all cores, exactly as `fold` is.
fn plonk_once<R: Relation>(ctx: &Ctx<R>, k: usize) -> Measurement
where
    R::Error: Debug,
{
    let (instances, witnesses): (Vec<_>, Vec<_>) = (0..k).map(|_| (ctx.sample)()).unzip();
    let params_verifier = ctx.srs.verifier_params();

    let (proofs, prover, prover_cpu, peak_rss) = measure_prover(|| {
        instances
            .iter()
            .zip(witnesses)
            .map(|(instance, witness)| {
                prove::<R, H>(&ctx.srs, &ctx.pk, ctx.relation, instance, witness, OsRng)
                    .expect("PLONK prove failed")
            })
            .collect::<Vec<_>>()
    });

    let t = Instant::now();
    for (instance, proof) in instances.iter().zip(proofs.iter()) {
        verify::<R, H>(&params_verifier, &ctx.vk, instance, None, proof)
            .expect("PLONK verify failed");
    }
    let verifier = t.elapsed();

    Measurement {
        prover,
        prover_cpu,
        peak_rss,
        verifier,
        proof_bytes: proofs.iter().map(Vec::len).sum(),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Scheme {
    Accordion,
    Plonk,
}

impl Scheme {
    fn name(self) -> &'static str {
        match self {
            Scheme::Accordion => "accordion",
            Scheme::Plonk => "plonk",
        }
    }
}

fn measure_once<R: Relation>(ctx: &Ctx<R>, scheme: Scheme, k: usize) -> Measurement
where
    R::Error: Debug,
{
    match (scheme, k) {
        (Scheme::Plonk, _) => plonk_once(ctx, k),
        (Scheme::Accordion, 2) => accordion_once::<R, 2>(ctx),
        (Scheme::Accordion, 4) => accordion_once::<R, 4>(ctx),
        (Scheme::Accordion, 8) => accordion_once::<R, 8>(ctx),
        (Scheme::Accordion, 16) => accordion_once::<R, 16>(ctx),
        (Scheme::Accordion, 32) => accordion_once::<R, 32>(ctx),
        (Scheme::Accordion, 64) => accordion_once::<R, 64>(ctx),
        (Scheme::Accordion, _) => unreachable!("width validated in main"),
    }
}

/// Number of threads the provers may use: `RAYON_NUM_THREADS` if set, else
/// the available parallelism.
fn threads() -> usize {
    env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

fn run<R: Relation>(
    relation: &R,
    sample: fn() -> (R::Instance, R::Witness),
    circuit: &str,
    description: &str,
    scheme: Scheme,
    k: usize,
    reps: usize,
) where
    R::Error: Debug,
{
    let t = Instant::now();
    let k_log = optimal_k(relation);
    let model = cost_model(relation, Some(k_log));
    let srs = ParamsKZG::<Bls12>::unsafe_setup(k_log, OsRng);
    let vk = setup_vk(&srs, relation);
    let pk = setup_pk(relation, &vk);
    let setup = t.elapsed();
    let rss_scope = if usage::reset_peak_rss() {
        "prover"
    } else {
        "process"
    };

    println!(
        "BENCH {{\"type\":\"setup\",\"circuit\":\"{circuit}\",\"description\":\"{description}\",\
         \"scheme\":\"{}\",\"k\":{k},\"reps\":{reps},\"k_log\":{k_log},\"rows\":{},\
         \"table_rows\":{},\"degree\":{},\"advice_columns\":{},\"fixed_columns\":{},\
         \"lookups\":{},\"permutations\":{},\"nb_public_inputs\":{},\"threads\":{},\
         \"peak_rss_scope\":\"{}\",\"setup_s\":{:.6}}}",
        scheme.name(),
        model.rows,
        model.table_rows,
        model.max_deg,
        model.advice_columns,
        model.fixed_columns,
        model.lookups,
        model.permutations,
        R::format_instance(&sample().0).expect("format_instance failed").len(),
        threads(),
        rss_scope,
        setup.as_secs_f64(),
    );
    eprintln!(
        "[{circuit}/{}/k={k}] setup {setup:.2?} (K={k_log}, {} rows)",
        scheme.name(),
        model.rows
    );

    let ctx = Ctx {
        relation,
        sample,
        k_log,
        srs,
        vk,
        pk,
    };

    for rep in 0..reps {
        let m = measure_once(&ctx, scheme, k);
        println!(
            "BENCH {{\"type\":\"rep\",\"rep\":{rep},\"prover_s\":{:.6},\"prover_cpu_s\":{:.6},\
             \"peak_rss_bytes\":{},\"verifier_s\":{:.6},\"proof_bytes\":{}}}",
            m.prover.as_secs_f64(),
            m.prover_cpu.as_secs_f64(),
            m.peak_rss,
            m.verifier.as_secs_f64(),
            m.proof_bytes,
        );
        eprintln!(
            "[{circuit}/{}/k={k}] rep {}/{reps}: prover {:.2?}, verifier {:.2?}, \
             proof {} B, peak RSS {} MiB",
            scheme.name(),
            rep + 1,
            m.prover,
            m.verifier,
            m.proof_bytes,
            m.peak_rss >> 20,
        );
    }
}

fn usage_and_exit() -> ! {
    eprintln!(
        "usage: accordion_bench <sha|ecdsa> <accordion|plonk> <k> [reps]\n\
         \n  accordion widths: {ACCORDION_WIDTHS:?}; plonk accepts any k >= 1"
    );
    process::exit(2);
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 3 || args.len() > 4 {
        usage_and_exit();
    }
    let scheme = match args[1].as_str() {
        "accordion" => Scheme::Accordion,
        "plonk" => Scheme::Plonk,
        _ => usage_and_exit(),
    };
    let k: usize = args[2].parse().unwrap_or_else(|_| usage_and_exit());
    let reps: usize = match args.get(3) {
        Some(r) => r.parse().unwrap_or_else(|_| usage_and_exit()),
        None => 1,
    };
    if k == 0 || reps == 0 || (scheme == Scheme::Accordion && !ACCORDION_WIDTHS.contains(&k)) {
        usage_and_exit();
    }

    match args[0].as_str() {
        "sha" => run(
            &ShaMultiCircuit,
            sha_multi::random_instance,
            "sha",
            &format!("{N_SHA}x SHA-256 of a 192-bit preimage"),
            scheme,
            k,
            reps,
        ),
        "ecdsa" => run(
            &EcdsaThresholdCircuit,
            ecdsa_threshold::random_instance,
            "ecdsa",
            &format!(
                "{}-out-of-{} ECDSA (secp256k1) threshold signature",
                ecdsa_threshold::T,
                ecdsa_threshold::N
            ),
            scheme,
            k,
            reps,
        ),
        _ => usage_and_exit(),
    }
}
