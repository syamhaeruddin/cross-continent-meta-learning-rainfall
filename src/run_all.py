# Runs all gate trainings in parallel from the command line (existing output files are skipped), then the evaluation and the report.
#   python src/run_all.py gates     -> RG x2, JG-E, JG-S, LRO x2, scaling curve (4 k x 5 draws; k=28 has one draw)
#   python src/run_all.py all       -> gates + evaluate + report
# The number of parallel processes is set with the environment variable WORKERS (default 6).
import os, sys, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, SRC, REGION_LIST

WORKERS = int(os.environ.get("WORKERS", "6")); PY = sys.executable
G = os.path.join(SRC, "gates.py")
jobs = []
for name in ["JG-E", "JG-S"]: jobs.append((f"pred_jg_{name}.npz", [PY, G, "joint", name]))
for r in REGION_LIST: jobs.append((f"pred_rg_{r}.npz", [PY, G, "region", r])); jobs.append((f"pred_lro_{r}.npz", [PY, G, "lro", r]))
for k in [4, 8, 14, 28]:
    for d in range(1 if k == 28 else 5): jobs.append((f"pred_scale_k{k}_d{d}.npz", [PY, G, "scale", str(k), str(d)]))


def run(jobs):
    jobs = [(f, c) for f, c in jobs if not os.path.exists(os.path.join(OUT, f))]
    print(len(jobs), "jobs,", WORKERS, "workers", flush=True); running = []; t0 = time.time()
    while jobs or running:
        while jobs and len(running) < WORKERS:
            f, c = jobs.pop(0); log = open(os.path.join(OUT, f.replace(".npz", ".log")), "w")
            running.append((f, subprocess.Popen(c, stdout=log, stderr=subprocess.STDOUT), log))
        for f, pr, log in list(running):
            if pr.poll() is not None:
                log.close(); running.remove((f, pr, log)); print("finished", f, "rc", pr.returncode, round((time.time() - t0) / 60, 1), "min", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    run(jobs)
    if what == "all":
        subprocess.run([PY, os.path.join(SRC, "evaluate.py")], check=True)
        subprocess.run([PY, os.path.join(SRC, "report.py")], check=True)
