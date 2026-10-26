"""Modal runner for scripts/jspace.py: fit the J-lens and R-lens on RLVR-hotpot, then read out the RLVR baseline and the
best exact adapter on the hotpot test rollouts. Outputs land in rlcr-exact-data:/jspace.

    modal run scripts/modal_jspace.py::fit          # both lenses, in parallel (one H100 each)
    modal run scripts/modal_jspace.py::readout      # both models, in parallel
    modal volume get rlcr-exact-data /jspace .
"""
import modal

VOL, HF, GPUQ = "/vol", "/hf", "/data/gpuq-runs/rrenaud"
OUT = f"{VOL}/jspace"
TEST = f"{GPUQ}/rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137/rollouts_hotpot_fresh_test.jsonl"
# Best exact run by test greedy Brier (0.187): rlcr11, 1000 steps on the 4k-question train cache.
ADAPTER = f"{GPUQ}/rlcr-hotpot-rlvr-frozen/run-20261019T233718Z-f6f79693af/checkpoints/rlcr11_exact_lr0.00037_W0.02_s0_F0.053"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.8.0", "transformers", "peft", "scikit-learn", "scipy", "numpy", "hf_transfer", "datasets",
                 "jlens @ git+https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e")
    .env({"HF_HOME": HF, "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(".", "/root/rlcr-exact", ignore=[".git", "data", "**/__pycache__", "rlcr-*"])
)
data = modal.Volume.from_name("rlcr-exact-data")
hf_cache = modal.Volume.from_name("rlcr-hf-cache", create_if_missing=True)
app = modal.App("rlcr-jspace", image=image)


@app.function(gpu="H100", volumes={VOL: data, HF: hf_cache}, timeout=4 * 3600)
def run(args: list[str]):
    import os, subprocess, sys
    repo = "/root/rlcr-exact"
    if not os.path.exists(f"{repo}/rlcr-reward-conf-weighted-tt-scaling"):
        os.symlink(f"{VOL}/code", f"{repo}/rlcr-reward-conf-weighted-tt-scaling")
    if not os.path.exists(GPUQ):
        os.makedirs(os.path.dirname(GPUQ), exist_ok=True)
        os.symlink(f"{VOL}/gpuq-runs", GPUQ)
    os.makedirs(OUT, exist_ok=True)
    code = subprocess.run([sys.executable, "scripts/jspace.py", *args], cwd=repo).returncode
    hf_cache.commit(); data.commit()
    if code:
        raise RuntimeError(f"jspace.py {args[0]} exited {code}")


@app.function(gpu="L40S", volumes={VOL: data, HF: hf_cache}, timeout=3 * 3600)
def run_l4(args: list[str]):
    run.local(args)


@app.local_entrypoint()
def fit(n_prompts: int = 48):
    list(run.map([["fit", "--kind", k, "--out", f"{OUT}/lens_{k}.pt", "--n-prompts", str(n_prompts)] for k in ("j", "r")]))


@app.local_entrypoint()
def readout(limit: int = 0):
    base = ["readout", "--lens", f"j={OUT}/lens_j.pt", "--lens", f"r={OUT}/lens_r.pt", "--test", TEST, "--out", OUT]
    lim = ["--limit", str(limit)] if limit else []
    list(run.map([base + lim, base + lim + ["--adapter", ADAPTER]]))


@app.local_entrypoint()
def readout_steps(run: str, steps: str = "50,100,150,200,250,300,350,400,450,500", limit: int = 0):
    """Logit + J + R-lens readout at the confidence site for each checkpoint of a RunPod run (500 = its final adapter):
    top-20 tokens at layers 18-26, contrast probes at every layer."""
    import subprocess
    ck = subprocess.run(["modal", "volume", "ls", "rlcr-exact-data", f"/runpod-runs/{run}/checkpoints"],
                        capture_output=True, text=True, check=True).stdout.split()[0].strip()
    final = steps.split(",")[-1]
    ads = ["--adapter", "base=-"] + [a for s in steps.split(",") for a in
           ("--adapter", f"s{s}={VOL}/{ck}" + ("" if s == final else f"/step{s}"))]
    args = ["readout-steps", "--lens", f"j={OUT}/lens_j.pt", "--lens", f"r={OUT}/lens_r.pt", "--test", TEST, "--out", f"{OUT}/steps-{run}", *ads]
    run_l4.remote(args + (["--limit", str(limit)] if limit else []))
