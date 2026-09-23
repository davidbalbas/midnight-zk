# Accordion vs k-Plonk benchmark

Note: The benchmark pipeline has been co-authored with Claude Opus 5.

`bench_accordion.py` compares two ways of proving `k` instances of the same
circuit:

- **Accordion**: a single Accordion fold of the `k` instances (one-shot: the
  fold ends with a corrected PLONK proof of the folded instance).
- **k-Plonk**: `k` independent PLONK proofs, generated one after another.

Two circuits are available:

| Name | Circuit |
|---|---|
| `sha` | SHA-256 of a 192-bit preimage ([`sha_preimage.rs`](../examples/circuits/sha_preimage.rs)) |
| `ecdsa` | 15-out-of-16 ECDSA (secp256k1) threshold signature ([`ecdsa_threshold.rs`](../examples/circuits/ecdsa_threshold.rs)) |

For each circuit, the script prints one table that puts the two schemes side by
side for every `k`. The table reports prover time, verifier time, proof size,
max memory and the share of CPU cores in use, averaged over several
repetitions.

## Requirements

- The Rust toolchain pinned by the repository (`rust-toolchain.toml`; `rustup`
  installs it automatically).
- Python 3.8 or later. Only the standard library is used.
- Linux is recommended. On macOS the benchmark runs, but *max memory* is the
  whole-process peak (setup included) rather than the proving-phase peak, and
  there is no memory guard.

## Running

From the repository root:

```sh
# Everything: both circuits, both schemes, k = 2, 4, 8, 16, 32, 64, 5 repetitions.
python3 aggregation/scripts/bench_accordion.py

# Quick smoke test.
python3 aggregation/scripts/bench_accordion.py --widths 2,4 --reps 1

# Only the SHA-256 circuit.
python3 aggregation/scripts/bench_accordion.py --circuits sha

# Only k-Plonk, at widths Accordion does not support.
python3 aggregation/scripts/bench_accordion.py --schemes plonk --widths 3,5,100
```

The script builds the `accordion_bench` example in release mode, then runs one
process per circuit. Each process does the setup (SRS and keys) once, then runs
every `k` in increasing order, with Accordion and k-Plonk back to back. Progress
is printed after every repetition.

The ECDSA circuit is much larger than SHA-256, so a full ECDSA run takes hours
and Accordion needs a lot of memory at large `k`. Start with `--circuits sha`,
or with a subset of `--widths`, to get a feel for your machine.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--circuits sha,ecdsa` | both | Circuits to run |
| `--schemes accordion,plonk` | both | Schemes to run |
| `--widths 2,4,8` | `2,4,8,16,32,64` | Values of `k`. Accordion supports only these six; k-Plonk accepts any `k ≥ 1` |
| `--reps N` | `5` | Repetitions per table cell |
| `--out DIR` | `target/accordion-bench/<timestamp>` | Output directory |
| `--mem-limit GIB` | 90% of available memory | Memory guard: kill a run whose RSS exceeds this; `0` disables it. Linux only |
| `--timeout MIN` | none | Stop a single (scheme, k) run after this many minutes |
| `--threads N` | all cores | Prover threads (sets `RAYON_NUM_THREADS`) |
| `--cooldown SEC` | `0` | Pause between runs, so a laptop can cool down |
| `--no-build` | | Reuse the existing release binary instead of rebuilding |
| `--render DIR` | | Only regenerate `tables.md` and `summary.csv` from `DIR/results.json` |

## Output

At the end, the tables are printed and saved in the output directory:

| File | Contents |
|---|---|
| `tables.md` | One Markdown table per circuit, with machine, toolchain and circuit details |
| `summary.csv` | The same aggregates, one row per (circuit, scheme, k) |
| `results.json` | Every raw repetition, plus setup and machine metadata |
| `logs/<circuit>.log` | Full output of each circuit's process, including the fold's per-step timings |

The files are rewritten after every repetition. An interrupted run (Ctrl-C)
still leaves everything measured so far, and `--render DIR` regenerates the
tables from it.

## Memory guard and failures

Accordion's memory grows linearly with `k`, because the folding prover
materialises a number of copies of the circuit trace proportional to `k`.
k-Plonk's memory does not depend on `k`.

When a run's memory exceeds the limit, the script kills it before the machine
starts swapping and reports `OOM (> X GiB)` in the table. It then skips larger
`k` for the same circuit and scheme, and continues with the remaining runs in a
fresh process, which repeats the setup. Timeouts and crashes are handled the
same way, without the skipping.

The default limit is 90% of the memory that is *available when the script
starts*. Close memory-hungry applications first, or pass `--mem-limit`
explicitly.

## Getting stable numbers

- Plug laptops into AC power and select the performance power profile, e.g.
  `powerprofilesctl set performance` on Linux. The script prints a warning when
  the power profile is not `performance`, and records it in `tables.md`.
- Close other CPU-heavy programs; both provers use every core.
- On laptops, short runs fit within the turbo-boost budget while long ones hit
  sustained power limits. This can make small `k` look relatively faster.
  `--cooldown` reduces the effect.
- Check the `±` columns: a large standard deviation means the machine was not
  quiet.

## What exactly is measured

The setup (SRS, verifying and proving keys) is not part of any measurement.
Each repetition samples `k` fresh instance–witness pairs outside the timed
region.

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
  maximum over repetitions. On Linux, just before proving, freed heap memory is
  handed back to the OS (`malloc_trim`) and the kernel's high-water mark is
  reset. Neither setup transients nor earlier runs in the same process count.
  Memory that stays resident, such as the SRS and keys, is included. The
  Accordion prover takes the proving key by value, so its figure includes one
  extra copy of the proving key, which k-Plonk does not pay for.
- **Cores used**: prover CPU time (user + system, all threads) divided by
  prover wall time × available threads.

Both schemes use the Poseidon transcript (the one an in-circuit verifier
needs). The SRS comes from `ParamsKZG::unsafe_setup`: it is **not** a trusted
setup and must only be used for benchmarking. Timings are unaffected, since
only the SRS size matters.

## Running a single configuration by hand

```sh
cargo run --release -p midnight-aggregation --example accordion_bench -- ecdsa 5 accordion:8 plonk:8
```

The arguments are `<sha|ecdsa> <reps> <accordion|plonk>:<k>...`. The example
does the setup once, then runs the listed configurations in order. On stdout
it prints one `BENCH {json}` line for the setup, one as each configuration
starts, and one per repetition. Human-readable progress, including the fold's
per-step timings, goes to stderr.
