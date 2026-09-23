#!/usr/bin/env python3
"""Accordion (Protogalaxy k-folding) vs k-Plonk benchmark driver.

Builds and runs the `accordion_bench` example of `midnight-aggregation` once
per (circuit, scheme, k), averages the repetitions and renders one comparison
table per circuit. See `README.md` next to this file for usage.

Only the Python standard library is needed.
"""

import argparse
import datetime
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = "accordion_bench"
PACKAGE = "midnight-aggregation"

CIRCUITS = {"sha": "SHA-256", "ecdsa": "ECDSA"}
SCHEMES = {"accordion": "Accordion", "plonk": "k-Plonk"}
# Widths with an instantiated `ProtogalaxyProver` in the example.
ACCORDION_WIDTHS = [2, 4, 8, 16, 32, 64]

GIB = 1 << 30
MIB = 1 << 20


# ── Helpers ──────────────────────────────────────────────────────────────────


def log(msg=""):
    print(msg, flush=True)


def csv_list(kind):
    def parse(s):
        items = [x.strip() for x in s.split(",") if x.strip()]
        if kind is int:
            return [int(x) for x in items]
        return items

    return parse


def run_text(cmd, **kw):
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=True, **kw
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_first(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def meminfo_bytes(key):
    """A /proc/meminfo entry in bytes (Linux only), or None."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def rss_bytes(pid):
    """Current RSS of a process (Linux only), or None."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def machine_info():
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    if cpu is None and sys.platform == "darwin":
        cpu = run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
    ram = meminfo_bytes("MemTotal")
    if ram is None and sys.platform == "darwin":
        ram = int(run_text(["sysctl", "-n", "hw.memsize"]) or 0) or None
    return {
        "cpu": cpu or platform.processor() or "unknown",
        "logical_cpus": os.cpu_count(),
        "ram_bytes": ram,
        "os": f"{platform.system()} {platform.release()}",
        "cpu_governor": read_first(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ),
        "platform_profile": read_first("/sys/firmware/acpi/platform_profile"),
        "rustc": run_text(["rustc", "-V"], cwd=REPO),
        "git_commit": run_text(["git", "rev-parse", "--short", "HEAD"], cwd=REPO),
        "git_branch": run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO),
        "git_dirty": bool(run_text(["git", "status", "--porcelain"], cwd=REPO)),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
    }


def build():
    """Builds the example in release mode and returns the binary's path."""
    log(f"Building `{EXAMPLE}` (release)...")
    cmd = [
        "cargo", "build", "--release", "-p", PACKAGE,
        "--example", EXAMPLE, "--message-format=json",
    ]
    out = subprocess.run(cmd, cwd=REPO, stdout=subprocess.PIPE, text=True)
    if out.returncode != 0:
        sys.exit("error: cargo build failed")
    for line in out.stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            msg.get("reason") == "compiler-artifact"
            and msg.get("target", {}).get("name") == EXAMPLE
            and msg.get("executable")
        ):
            return msg["executable"]
    sys.exit("error: could not locate the built example binary")


# ── Running one configuration ────────────────────────────────────────────────


def run_config(binary, circuit, scheme, k, reps, env, log_path, mem_limit, timeout):
    """Runs one (circuit, scheme, k) process and returns its result record."""
    record = {
        "circuit": circuit,
        "scheme": scheme,
        "k": k,
        "status": "ok",
        "setup": None,
        "reps": [],
        "log": str(log_path),
    }
    cmd = [binary, circuit, scheme, str(k), str(reps)]
    t0 = time.monotonic()
    with open(log_path, "a") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=log_file, text=True, env=env
        )

        def read_stdout():
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                if not line.startswith("BENCH "):
                    continue
                data = json.loads(line[len("BENCH "):])
                if data.pop("type") == "setup":
                    record["setup"] = data
                    log(f"    setup {data['setup_s']:.1f} s")
                else:
                    record["reps"].append(data)
                    log(
                        f"    rep {data['rep'] + 1}/{reps}: "
                        f"prover {data['prover_s']:.2f} s, "
                        f"verifier {data['verifier_s'] * 1e3:.1f} ms, "
                        f"peak {data['peak_rss_bytes'] / MIB:,.0f} MiB"
                    )

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()

        peak_seen = 0
        try:
            while proc.poll() is None:
                rss = rss_bytes(proc.pid)
                if rss is not None:
                    peak_seen = max(peak_seen, rss)
                    if mem_limit and rss > mem_limit:
                        proc.kill()
                        record["status"] = "oom"
                        record["killed_by"] = "guard"
                        break
                if timeout and time.monotonic() - t0 > timeout:
                    proc.kill()
                    record["status"] = "timeout"
                    break
                time.sleep(0.2)
        finally:
            # Never leave a benchmark running behind an interrupted driver.
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            reader.join()

    record["wall_s"] = time.monotonic() - t0
    record["peak_rss_seen_bytes"] = peak_seen
    record["returncode"] = proc.returncode
    if record["status"] == "ok" and proc.returncode != 0:
        # SIGKILL we did not send is almost always the kernel's OOM killer.
        if proc.returncode == -signal.SIGKILL:
            record["status"] = "oom"
            record["killed_by"] = "kernel"
        else:
            record["status"] = "error"
    if record["status"] == "ok" and len(record["reps"]) != reps:
        record["status"] = "error"
    return record


# ── Aggregation and rendering ────────────────────────────────────────────────


def mean_sd(values):
    """Mean and sample standard deviation (None for a single value)."""
    m = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    return m, sd


def summarize(record):
    """Per-configuration aggregates (means over repetitions)."""
    reps = record["reps"]
    if record["status"] != "ok" or not reps:
        return None
    threads = record["setup"]["threads"]
    prover = [r["prover_s"] for r in reps]
    verifier = [r["verifier_s"] * 1e3 for r in reps]
    cores = [r["prover_cpu_s"] / r["prover_s"] / threads * 100 for r in reps]
    return {
        "prover_s": mean_sd(prover),
        "verifier_ms": mean_sd(verifier),
        "proof_bytes": statistics.fmean(r["proof_bytes"] for r in reps),
        "peak_rss_mib": max(r["peak_rss_bytes"] for r in reps) / MIB,
        "cores_pct": statistics.fmean(cores),
        "prover_cpu_s": statistics.fmean(r["prover_cpu_s"] for r in reps),
        "threads": threads,
        "reps": len(reps),
    }


def fmt_msd(msd, digits):
    m, sd = msd
    if sd is None:
        return f"{m:,.{digits}f}"
    return f"{m:,.{digits}f} ± {sd:,.{digits}f}"


def failure_text(record, mem_limit):
    status = record["status"]
    if status == "oom":
        limit = record.get("mem_limit_bytes") or mem_limit
        if record.get("killed_by") == "guard" and limit:
            return f"OOM (> {limit / GIB:.1f} GiB)"
        return "OOM (killed by the kernel)"
    if status == "timeout":
        return "timeout"
    if status == "skipped":
        return record.get("reason", "skipped")
    return "failed (see log)"


def circuit_header(setup):
    return (
        f"{setup['description']}. K = {setup['k_log']} "
        f"({1 << setup['k_log']:,} rows, {setup['rows']:,} used by gates, "
        f"{setup['table_rows']:,} by tables), degree {setup['degree']}, "
        f"{setup['advice_columns']} advice / {setup['fixed_columns']} fixed columns, "
        f"{setup['lookups']} lookups, {setup['permutations']} permutation columns, "
        f"{setup['nb_public_inputs']} public inputs."
    )


def render_markdown(results):
    meta = results["meta"]
    m = meta["machine"]
    lines = ["# Accordion vs k-Plonk", ""]
    ram = f"{m['ram_bytes'] / GIB:.0f} GiB RAM" if m.get("ram_bytes") else "RAM unknown"
    power = ", ".join(
        f"{name} `{m[key]}`"
        for key, name in (("platform_profile", "platform profile"), ("cpu_governor", "governor"))
        if m.get(key)
    )
    commit = f"`{m['git_commit']}`{' (dirty)' if m['git_dirty'] else ''}"
    lines += [
        f"- Date: {meta['started']}",
        f"- Commit: {commit} on `{m['git_branch']}`",
        f"- Machine: {m['cpu']}, {m['logical_cpus']} logical CPUs, {ram}, {m['os']}"
        + (f"; {power}" if power else ""),
        f"- Toolchain: {m['rustc']}, release profile",
        f"- Repetitions: {meta['reps']} per cell (mean ± sample standard deviation)",
        "- Proof system: PLONK over BLS12-381 with KZG (locally generated, *insecure* "
        "`unsafe_setup` SRS), Poseidon transcript for both schemes",
        "",
        "Columns:",
        "",
        "- **Prover**: wall-clock time to fold k instances (Accordion) or to produce "
        "k independent PLONK proofs, one after another (k-Plonk). Includes circuit "
        "synthesis; excludes key generation and instance sampling.",
        "- **Verifier**: fold verifier + KZG check of the accumulator (Accordion), or "
        "k independent PLONK verifications (k-Plonk).",
        "- **Proof size**: the folding proof, or all k PLONK proofs together.",
        "- **Max memory**: peak resident set size of the prover process during proving "
        + (
            "(includes the SRS and keys held in memory; setup transients excluded)."
            if all(
                (r.get("setup") or {}).get("peak_rss_scope", "prover") == "prover"
                for r in results["runs"]
            )
            else "(on this platform: over the whole process, setup included)."
        ),
        "- **Cores used**: prover CPU time / (prover wall time × available threads).",
        "",
    ]

    for circuit in meta["circuits"]:
        runs = [r for r in results["runs"] if r["circuit"] == circuit]
        if not runs:
            continue
        setup = next((r["setup"] for r in runs if r.get("setup")), None)
        threads = setup["threads"] if setup else None
        lines += [f"## {CIRCUITS[circuit]}", ""]
        if setup:
            lines += [circuit_header(setup), ""]
        lines += [
            "| k | Scheme | Prover (s) | Verifier (ms) | Proof size (KiB) | "
            f"Max memory (MiB) | Cores used{f' (of {threads})' if threads else ''} |",
            "|---:|:---|---:|---:|---:|---:|---:|",
        ]
        for k in meta["widths"]:
            for scheme in meta["schemes"]:
                rec = next(
                    (r for r in runs if r["k"] == k and r["scheme"] == scheme), None
                )
                if rec is None:
                    continue
                s = summarize(rec)
                if s is None:
                    cells = [failure_text(rec, meta.get("mem_limit_bytes"))] + ["—"] * 4
                else:
                    cells = [
                        fmt_msd(s["prover_s"], 2),
                        fmt_msd(s["verifier_ms"], 1),
                        f"{s['proof_bytes'] / 1024:,.1f}",
                        f"{s['peak_rss_mib']:,.0f}",
                        f"{s['cores_pct']:.0f}%",
                    ]
                lines.append(f"| {k} | {SCHEMES[scheme]} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def fmt_sd(sd, digits):
    return "" if sd is None else f"{sd:.{digits}f}"


def render_csv(results):
    header = [
        "circuit", "scheme", "k", "status", "reps",
        "prover_s_mean", "prover_s_sd", "verifier_ms_mean", "verifier_ms_sd",
        "proof_bytes", "peak_rss_mib", "cores_used_pct", "prover_cpu_s_mean",
        "threads", "k_log",
    ]
    rows = [",".join(header)]
    for rec in results["runs"]:
        s = summarize(rec)
        k_log = (rec.get("setup") or {}).get("k_log", "")
        if s is None:
            rows.append(
                f"{rec['circuit']},{rec['scheme']},{rec['k']},{rec['status']},"
                f"{len(rec['reps'])}" + "," * 9 + f",{k_log}"
            )
            continue
        rows.append(
            ",".join(
                str(x)
                for x in (
                    rec["circuit"], rec["scheme"], rec["k"], rec["status"], s["reps"],
                    f"{s['prover_s'][0]:.6f}", fmt_sd(s["prover_s"][1], 6),
                    f"{s['verifier_ms'][0]:.4f}", fmt_sd(s["verifier_ms"][1], 4),
                    f"{s['proof_bytes']:.0f}", f"{s['peak_rss_mib']:.1f}",
                    f"{s['cores_pct']:.2f}", f"{s['prover_cpu_s']:.6f}",
                    s["threads"], k_log,
                )
            )
        )
    return "\n".join(rows) + "\n"


def write_outputs(out_dir, results):
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / "summary.csv").write_text(render_csv(results))
    md = render_markdown(results)
    (out_dir / "tables.md").write_text(md + "\n")
    return md


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark Accordion (Protogalaxy k-folding) against k independent "
        "PLONK proofs, and render one comparison table per circuit.",
    )
    p.add_argument("--circuits", type=csv_list(str), default=list(CIRCUITS),
                   help="comma-separated subset of: " + ", ".join(CIRCUITS) + " (default: all)")
    p.add_argument("--schemes", type=csv_list(str), default=list(SCHEMES),
                   help="comma-separated subset of: " + ", ".join(SCHEMES) + " (default: all)")
    p.add_argument("--widths", type=csv_list(int), default=ACCORDION_WIDTHS,
                   help="comma-separated values of k (default and Accordion-supported: "
                   + ",".join(map(str, ACCORDION_WIDTHS)) + ")")
    p.add_argument("--reps", type=int, default=5,
                   help="repetitions per (circuit, scheme, k), averaged (default: 5)")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default: target/accordion-bench/<timestamp>)")
    p.add_argument("--mem-limit", type=float, default=None, metavar="GIB",
                   help="kill a run whose RSS exceeds this many GiB and report it as OOM "
                   "(default: 90%% of the memory available at start; 0 disables; Linux only)")
    p.add_argument("--timeout", type=float, default=None, metavar="MIN",
                   help="kill a (circuit, scheme, k) run after this many minutes")
    p.add_argument("--threads", type=int, default=None,
                   help="number of prover threads (sets RAYON_NUM_THREADS)")
    p.add_argument("--cooldown", type=float, default=0, metavar="SEC",
                   help="pause between runs, to let a laptop cool down (default: 0)")
    p.add_argument("--no-build", action="store_true",
                   help="do not rebuild; use the existing release binary")
    p.add_argument("--render", type=Path, default=None, metavar="DIR",
                   help="only re-render tables.md and summary.csv from DIR/results.json")
    args = p.parse_args()

    for c in args.circuits:
        if c not in CIRCUITS:
            p.error(f"unknown circuit {c!r}")
    for s in args.schemes:
        if s not in SCHEMES:
            p.error(f"unknown scheme {s!r}")
    if not args.widths or any(k < 1 for k in args.widths):
        p.error("widths must be positive integers")
    if "accordion" in args.schemes:
        bad = [k for k in args.widths if k not in ACCORDION_WIDTHS]
        if bad:
            p.error(f"Accordion does not support k in {bad}; use a subset of "
                    f"{ACCORDION_WIDTHS} or --schemes plonk")
    if args.reps < 1:
        p.error("--reps must be at least 1")
    return args


def main():
    args = parse_args()

    if args.render:
        results = json.loads((args.render / "results.json").read_text())
        log(write_outputs(args.render, results))
        return

    machine = machine_info()
    # With intel_pstate the governor reads "powersave" even under the
    # performance profile, so it is only consulted when there is no profile.
    power_mode = machine["platform_profile"] or machine["cpu_governor"]
    if power_mode not in (None, "performance"):
        log(
            f"warning: power profile is {machine['platform_profile']!r} "
            f"(governor {machine['cpu_governor']!r}); timings will be noisier than "
            "under 'performance'. See README.md."
        )

    if args.mem_limit is None:
        avail = meminfo_bytes("MemAvailable")
        mem_limit = int(avail * 0.9) if avail else None
    else:
        mem_limit = int(args.mem_limit * GIB) or None

    if args.no_build:
        target_dir = Path(os.environ.get("CARGO_TARGET_DIR", REPO / "target"))
        binary = str(target_dir / "release" / "examples" / EXAMPLE)
        if not Path(binary).exists():
            sys.exit(f"error: {binary} not found; run without --no-build")
    else:
        binary = build()

    started = datetime.datetime.now()
    out_dir = args.out or REPO / "target" / "accordion-bench" / started.strftime(
        "%Y%m%d-%H%M%S"
    )
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # Runs the slow reference path of step 6 alongside the fast one.
    env.pop("PG_CHECK_STEP6", None)
    if args.threads:
        env["RAYON_NUM_THREADS"] = str(args.threads)
        machine["rayon_num_threads"] = str(args.threads)

    results = {
        "meta": {
            "started": started.strftime("%Y-%m-%d %H:%M"),
            "circuits": args.circuits,
            "schemes": args.schemes,
            "widths": args.widths,
            "reps": args.reps,
            "mem_limit_bytes": mem_limit,
            "machine": machine,
        },
        "runs": [],
    }

    log(f"Output directory: {out_dir}")
    log(
        "Memory guard: "
        + (f"{mem_limit / GIB:.1f} GiB" if mem_limit else "disabled")
    )

    # After an OOM, larger widths of the same (circuit, scheme) are skipped.
    oom_at = {}
    first = True
    try:
        for circuit in args.circuits:
            for k in args.widths:
                for scheme in args.schemes:
                    label = f"{CIRCUITS[circuit]} / {SCHEMES[scheme]} / k={k}"
                    if (circuit, scheme) in oom_at:
                        log(f"==> {label}: skipped")
                        results["runs"].append({
                            "circuit": circuit, "scheme": scheme, "k": k,
                            "status": "skipped", "setup": None, "reps": [],
                            "reason": f"skipped (OOM at k={oom_at[(circuit, scheme)]})",
                        })
                        write_outputs(out_dir, results)
                        continue
                    if not first and args.cooldown:
                        time.sleep(args.cooldown)
                    first = False
                    log(f"==> {label}")
                    rec = run_config(
                        binary, circuit, scheme, k, args.reps, env,
                        out_dir / "logs" / f"{circuit}-{scheme}-k{k}.log",
                        mem_limit,
                        args.timeout * 60 if args.timeout else None,
                    )
                    rec["mem_limit_bytes"] = mem_limit
                    results["runs"].append(rec)
                    if rec["status"] != "ok":
                        log(f"    {rec['status'].upper()} after {rec['wall_s']:.0f} s "
                            f"(log: {rec['log']})")
                    if rec["status"] == "oom":
                        oom_at[(circuit, scheme)] = k
                    write_outputs(out_dir, results)
    except KeyboardInterrupt:
        log("\nInterrupted; writing partial results.")

    log("")
    log(write_outputs(out_dir, results))
    log(f"Results written to {out_dir}/{{tables.md,summary.csv,results.json,logs/}}")


if __name__ == "__main__":
    main()
