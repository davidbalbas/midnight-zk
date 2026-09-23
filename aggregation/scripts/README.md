# Accordion vs k-Plonk benchmark

`bench_accordion.py` compares two ways of proving `k` instances of the same
circuit:

- **Accordion**: a single Protogalaxy fold of the `k` instances (one-shot: the
  fold ends with a corrected PLONK proof of the folded instance).
- **k-Plonk**: `k` independent PLONK proofs, generated one after another.

It runs both schemes for `k = 2, 4, 8, 16, 32, 64` on two circuits:

| Name | Circuit | Size |
|---|---|---|
| `sha` | 4 × SHA-256 of a 192-bit preimage ([`sha_multi.rs`](../examples/circuits/sha_multi.rs)) | K = 13 |
| `ecdsa` | 15-out-of-16 ECDSA (secp256k1) threshold signature ([`ecdsa_threshold.rs`](../examples/circuits/ecdsa_threshold.rs)) | K = 17 |

Each cell is averaged over 5 repetitions. For each circuit you get one table
that puts the two schemes side by side for every `k`, reporting prover time,
verifier time, proof size, max memory and the share of CPU cores in use.

## Requirements

- The Rust toolchain pinned by the repository (`rust-toolchain.toml`; `rustup`
  installs it automatically).
- Python ≥ 3.8. Only the standard library is used.
- Linux is recommended. On macOS the benchmark runs, but *max memory* is the
  whole-process peak (setup included) rather than the proving-phase peak, and
  there is no memory guard (see below).

## Quick start

From the repository root:

```sh
# Everything: both circuits, both schemes, k = 2..64, 5 repetitions.
python3 aggregation/scripts/bench_accordion.py

# A few-minute smoke test.
python3 aggregation/scripts/bench_accordion.py --widths 2,4 --reps 1

# Only the SHA-256 circuit (about 8 minutes).
python3 aggregation/scripts/bench_accordion.py --circuits sha
```

The script builds the `accordion_bench` example in release mode, then runs it
once per (circuit, k, scheme). Progress is printed per repetition. At the end
the tables are printed and saved to
`target/accordion-bench/<timestamp>/`:

| File | Contents |
|---|---|
| `tables.md` | One Markdown table per circuit, with machine and circuit details |
| `summary.csv` | The same aggregates, one row per (circuit, scheme, k) |
| `results.json` | Every raw repetition, plus setup and machine metadata |
| `logs/` | Full stdout/stderr of each run, including the fold's per-step timings |

The output files are rewritten after every run, so an interrupted benchmark
(Ctrl-C) still leaves the results obtained so far.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--circuits sha,ecdsa` | both | Circuits to run |
| `--schemes accordion,plonk` | both | Schemes to run |
| `--widths 2,4,8` | `2,4,8,16,32,64` | Values of `k`. Accordion supports only these six; k-Plonk accepts any `k ≥ 1` |
| `--reps N` | `5` | Repetitions per cell |
| `--out DIR` | `target/accordion-bench/<timestamp>` | Output directory |
| `--mem-limit GIB` | 90% of available memory | Kill a run whose RSS exceeds this and report it as `OOM`; `0` disables. Linux only |
| `--timeout MIN` | none | Kill a single (circuit, scheme, k) run after this many minutes |
| `--threads N` | all cores | Prover threads (sets `RAYON_NUM_THREADS`) |
| `--cooldown SEC` | `0` | Pause between runs, so a laptop can cool down |
| `--no-build` | | Reuse the existing release binary |
| `--render DIR` | | Only regenerate `tables.md` and `summary.csv` from `DIR/results.json` |

## Runtime and memory

Accordion's memory grows linearly in `k`, because the folding prover
materialises about `4k` copies of the circuit trace. k-Plonk's memory does not
depend on `k`. Reference figures per repetition (means where several were
run), measured on an Intel Core Ultra 7 258V (8 cores, 30 GiB RAM, `balanced`
power profile). Values marked `~` are extrapolated.

| Circuit | Scheme | | k=2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SHA-256 | Accordion | prover | 0.6 s | 1.2 s | 2.3 s | 4.6 s | 8.2 s | 16 s |
| | | memory | 0.23 GiB | 0.28 GiB | 0.37 GiB | 0.56 GiB | 0.94 GiB | 1.7 GiB |
| SHA-256 | k-Plonk | prover | 0.6 s | 1.8 s | 3.4 s | 6.9 s | 13 s | 26 s |
| | | memory | 0.12 GiB | 0.12 GiB | 0.12 GiB | 0.12 GiB | 0.12 GiB | 0.12 GiB |
| ECDSA | Accordion | prover | 17 s | 25 s | 47 s | 95 s | ~190 s | ~380 s |
| | | memory | 3.6 GiB | 4.4 GiB | 6.1 GiB | 9.4 GiB | ~17 GiB | ~31 GiB |
| ECDSA | k-Plonk | prover | 15 s | 28 s | ~57 s | ~115 s | ~230 s | ~460 s |
| | | memory | 1.8 GiB | 1.8 GiB | 1.8 GiB | 1.8 GiB | 1.8 GiB | 1.8 GiB |

A full default run (5 repetitions, plus a few seconds of setup per cell)
therefore takes about **8 minutes for SHA-256** and about **2.5 hours for
ECDSA**. ECDSA with Accordion needs about 17 GiB of free memory at `k = 32`
and about 31 GiB at `k = 64`.

When a run exceeds the memory limit, the script kills it before the machine
starts swapping and reports `OOM (> X GiB)` in the table. Larger `k` for the
same circuit and scheme are then skipped. k-Plonk keeps running at every `k`.
The default limit is 90% of the memory that is *available when the script
starts*, so close memory-hungry applications first, or pass `--mem-limit`
explicitly.

## Getting stable numbers

- Plug laptops into AC power and select the performance power profile, e.g.
  `powerprofilesctl set performance` on Linux. The script prints a warning when
  the power profile is not `performance`, and records it in `tables.md`. Under `balanced`/`powersave`, the same run can vary by 30% or
  more.
- Close other CPU-heavy programs; both provers use every core.
- On laptops, short runs (small `k`, SHA-256) fit within the turbo-boost
  budget while long ones hit sustained power limits. Per-proof k-Plonk times
  can therefore look better at `k = 2` than at larger `k` on the same machine.
- For each `k`, Accordion and k-Plonk run back to back, so slow thermal drift
  affects both similarly. `--cooldown 30` reduces it further.
- Check the `±` columns: a large standard deviation means the machine was not
  quiet.

## What exactly is measured

Every (circuit, scheme, k) runs in its own process. The process does the setup
once (SRS, verifying and proving keys), then runs the repetitions. Each
repetition samples `k` fresh instance–witness pairs outside the timed region.

- **Prover**: wall-clock time of `ProtogalaxyProver::fold` (Accordion), or of
  `k` sequential `prove` calls (k-Plonk). The provers are internally parallel,
  so the calls are not run concurrently. Circuit synthesis (witness
  generation) is included.
- **Verifier**: `ProtogalaxyVerifier::fold` plus the KZG check of the guard it
  returns (Accordion), or `k` sequential `verify` calls (k-Plonk). PLONK proofs
  are not batch-verified.
- **Proof size**: the Accordion folding proof, or the total size of the `k`
  PLONK proofs.
- **Max memory**: peak resident set size during the prover span, taken as the
  maximum over repetitions. On Linux the kernel's high-water mark is reset
  just before proving, so setup transients are excluded. Memory that stays
  resident, such as the SRS and keys, is included. The Accordion prover takes
  the proving key by value, so its figure includes one extra copy of the
  proving key: about 60 MiB for SHA-256 and 0.9 GiB for ECDSA. k-Plonk does
  not pay for that copy.
- **Cores used**: prover CPU time (user + system, all threads) divided by
  prover wall time × available threads.

Both schemes use the Poseidon transcript (the one an in-circuit verifier
needs). The SRS comes from `ParamsKZG::unsafe_setup`: it is **not** a trusted
setup and must only be used for benchmarking. Timings are unaffected, since
only the SRS size matters.

## Running a single configuration by hand

```sh
cargo run --release -p midnight-aggregation --example accordion_bench -- ecdsa accordion 8 5
```

The arguments are `<sha|ecdsa> <accordion|plonk> <k> [reps]`. The example
prints one `BENCH {json}` line for the setup and one per repetition on stdout.
Human-readable progress, including the fold's per-step timings, goes to
stderr.
