"""Host-side bridge to the out-of-process engines (Chrono, AGX).

Both engines live in interpreters the ostrich venv cannot import (pychrono in the
micromamba env at ~/.local/opt/chrono-env, AGX python inside the `agx-env`
distrobox container). The contract is JSON files: the host writes a batch of jobs,
invokes the engine runner once per batch, and reads back a batch of results.
Batching matters for AGX, which re-validates its license online on every launch.

Job:    {id, dt, duration_s, settle_time_s, control:{t,lrr}, scene:[...],
         ground:{mu}, params:{engine-specific}, log_wheel_omega}
Result: {id, dt, pose:[[x,y,z,qx,qy,qz,qw]...], wheel_omega, wall_clock_s,
         n_steps, stable, diverged_at_s, params_effective}

Files are exchanged under engines/tmp/, which is inside the user's home and
therefore visible at the same absolute path inside the distrobox container.
"""

import json
import pathlib
import subprocess

import numpy as np

ENGINES_DIR = pathlib.Path(__file__).parent
TMP_DIR = ENGINES_DIR / "tmp"

CHRONO_PYTHON = pathlib.Path.home() / ".local" / "opt" / "chrono-env" / "bin" / "python"
# Source-built Chrono with the vehicle module (SCM terrain); the conda env lacks it.
CHRONO_ENV_SH = (pathlib.Path.home() / "projects" / "helhest_stack-tier1" / "scripts"
                 / "chrono_env.sh")
AGX_CONTAINER = "agx-env"
AGX_SETUP = "/opt/Algoryx/AGX-2.42.1.0/setup_env.bash"


def make_job(job_id: str, commands: np.ndarray, dt: float, params: dict | None = None,
             scene: list | None = None, ground_mu: float = 0.8,
             settle_time_s: float = 1.0) -> dict:
    """A job from an already-prepared command array [T,3] (see common.prepare_commands)."""
    T = commands.shape[0]
    return {
        "id": job_id,
        "dt": dt,
        "duration_s": T * dt,
        "settle_time_s": settle_time_s,
        "control": {"t": (np.arange(T) * dt).tolist(), "lrr": np.asarray(commands).tolist()},
        "scene": scene or [],
        "ground": {"mu": ground_mu},
        "params": params or {},
        "log_wheel_omega": False,
    }


def _run_batch(jobs: list[dict], argv: list[str], tag: str, timeout: float) -> list[dict]:
    TMP_DIR.mkdir(exist_ok=True)
    jobs_file = TMP_DIR / f"jobs_{tag}.json"
    out_file = TMP_DIR / f"out_{tag}.json"
    out_file.unlink(missing_ok=True)
    with open(jobs_file, "w") as f:
        json.dump({"jobs": jobs}, f)
    proc = subprocess.run(argv + ["--jobs", str(jobs_file), "--out", str(out_file)],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not out_file.exists():
        raise RuntimeError(
            f"{tag} runner failed (rc={proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout[-3000:]}\n--- stderr ---\n{proc.stderr[-3000:]}")
    with open(out_file) as f:
        return json.load(f)["results"]


def run_chrono(jobs: list[dict], timeout: float = 3600.0, procs: int = 1) -> list[dict]:
    """Chrono replays are single-threaded and independent, so a big batch can be
    split across `procs` concurrent runner processes. Result order matches jobs."""
    runner = ENGINES_DIR / "chrono_replay_junior.py"
    argv = [str(CHRONO_PYTHON), str(runner)]
    if procs <= 1 or len(jobs) <= 1:
        return _run_batch(jobs, argv, "chrono", timeout)
    from concurrent.futures import ThreadPoolExecutor
    procs = min(procs, len(jobs))
    chunks = [jobs[i::procs] for i in range(procs)]
    with ThreadPoolExecutor(max_workers=procs) as ex:
        futs = [ex.submit(_run_batch, chunk, argv, f"chrono_{i}", timeout)
                for i, chunk in enumerate(chunks)]
        by_id = {r["id"]: r for f in futs for r in f.result()}
    return [by_id[j["id"]] for j in jobs]


def run_chrono_scm(jobs: list[dict], timeout: float = 14400.0, procs: int = 1) -> list[dict]:
    """SCM-terrain jobs need pychrono.vehicle, which only the source build has;
    chrono_env.sh sets PYTHONPATH/LD_LIBRARY_PATH and execs its python."""
    runner = ENGINES_DIR / "chrono_replay_junior.py"
    argv = [str(CHRONO_ENV_SH), str(runner)]
    if procs <= 1 or len(jobs) <= 1:
        return _run_batch(jobs, argv, "chrono_scm", timeout)
    from concurrent.futures import ThreadPoolExecutor
    procs = min(procs, len(jobs))
    chunks = [jobs[i::procs] for i in range(procs)]
    with ThreadPoolExecutor(max_workers=procs) as ex:
        futs = [ex.submit(_run_batch, chunk, argv, f"chrono_scm_{i}", timeout)
                for i, chunk in enumerate(chunks)]
        by_id = {r["id"]: r for f in futs for r in f.result()}
    return [by_id[j["id"]] for j in jobs]


def run_agx(jobs: list[dict], timeout: float = 3600.0) -> list[dict]:
    runner = ENGINES_DIR / "agx_replay_junior.py"
    # Home is bind-mounted into the container at the same path, so the runner,
    # jobs and output files resolve identically on both sides.
    # /usr/bin/python3 pins the container's own interpreter (3.14, matching the
    # AGX bindings); a bare `python3` can resolve to the host's via the merged PATH.
    inner = (f"source {AGX_SETUP} && export AGX_LICENSE_FILE=$HOME/agx-license.lic && "
             f'exec /usr/bin/python3 "$@"')
    argv = ["distrobox", "enter", AGX_CONTAINER, "--", "bash", "-lc", inner, "_",
            str(runner)]
    return _run_batch(jobs, argv, "agx", timeout)
