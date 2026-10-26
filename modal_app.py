"""Modal runner for train_frozen.py (Stage A, cached rollouts) on a rented GPU.

The data volume `rlcr-exact-data` mirrors the gpuq run directories under /gpuq-runs (copied from the 4090 box) and
holds the conf-vote repo's logprob_spans.py under /code. Hub checkpoints are cached in `rlcr-hf-cache`. Each run
writes its log and adapter to /modal-runs/<name> on the data volume.

    modal run modal_app.py::smoke                        # 10 steps of --objective rps at the rlcr11 leader config
    modal run modal_app.py::train --name NAME --args "--objective rps --lr 3.7e-4 ..."   # rlcr11 data preset added
    modal volume get rlcr-exact-data /modal-runs/NAME .  # fetch a run
"""
import shlex

import modal

VOL, HF = "/vol", "/hf"
# The gpuq paths, as the box has them: the container links /data/gpuq-runs/rrenaud to the volume's copy, so paths
# recorded in caches (ref_topk's meta.json checks its rollouts' absolute path) and in the notes work unchanged.
GPUQ = "/data/gpuq-runs/rrenaud"
RLVR_CACHE = f"{GPUQ}/rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137"
RLVR11_REF = f"{GPUQ}/rlcr-hotpot-rlvr-frozen/run-20261018T050522Z-eeac77918e/ref_topk_train"
# TASK=hotpot-rlvr-11 in run_frozen.sh
HOTPOT_RLVR_11 = (f"--scorer rlcr11 --model mehuldamani/hotpot-v2-correctness-7b "
                  f"--train {RLVR_CACHE}/rollouts_hotpot_fresh_train.jsonl --dev {RLVR_CACHE}/rollouts_hotpot_fresh_dev.jsonl "
                  f"--ref-topk {RLVR11_REF}")
LEADER = "--lr 3.7e-4 --answer-kl 0.02 --support-floor 0.053"  # Stage A rlcr11 leader (NOTES, 2026-10-18)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "transformers", "peft", "scikit-learn", "scipy", "numpy", "hf_transfer")
    .env({"HF_HOME": HF, "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(".", "/root/rlcr-exact", ignore=[".git", "data", "**/__pycache__", "rlcr-*"])
)
data = modal.Volume.from_name("rlcr-exact-data")
hf_cache = modal.Volume.from_name("rlcr-hf-cache", create_if_missing=True)
app = modal.App("rlcr-exact", image=image)


@app.function(gpu="L40S", volumes={VOL: data, HF: hf_cache}, timeout=6 * 3600)
def run(name: str, args: str, script: str = "scripts/train_frozen.py"):
    import os, subprocess, sys
    repo = "/root/rlcr-exact"
    if not os.path.exists(f"{repo}/rlcr-reward-conf-weighted-tt-scaling"):  # train_frozen imports logprob_spans from here
        os.symlink(f"{VOL}/code", f"{repo}/rlcr-reward-conf-weighted-tt-scaling")
    if not os.path.exists(GPUQ):
        os.makedirs(os.path.dirname(GPUQ), exist_ok=True)
        os.symlink(f"{VOL}/gpuq-runs", GPUQ)
    out = f"{VOL}/modal-runs/{name}"
    os.makedirs(out, exist_ok=True)
    subprocess.run([sys.executable, "-c", "import torch, transformers, peft; print('versions', torch.__version__, "
                    "transformers.__version__, peft.__version__, torch.cuda.get_device_name())"], check=True)
    cmd = [sys.executable, script, *shlex.split(args), "--out", out]
    print(" ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=repo).returncode
    hf_cache.commit(); data.commit()
    if code:
        raise RuntimeError(f"{script} exited {code}")


@app.local_entrypoint()
def smoke(objective: str = "rps", name: str = "", extra: str = ""):
    """10 steps on the rlcr11 data, a 32-row dev eval at steps 0 and 10, no test. extra: more train_frozen.py args."""
    run.remote(name or f"smoke-{objective}", f"{HOTPOT_RLVR_11} {LEADER} --objective {objective} --seed 0 --steps 10 "
               f"--eval-every 10 --dev-limit 32 --drift-rows 8 {extra}")


@app.local_entrypoint()
def probe_tags(questions: int = 500):
    """Step-0 readout of candidate pre-answer tag names (scripts/tag_probe.py) on train questions."""
    run.remote("tag-probe", f"{RLVR_CACHE}/rollouts_hotpot_fresh_train.jsonl --questions {questions}", "scripts/tag_probe.py")


@app.local_entrypoint()
def train(name: str, args: str):
    run.remote(name, f"{HOTPOT_RLVR_11} {args}")


vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "transformers==4.56.2", "hf_transfer")
    .env({"HF_HOME": HF, "HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir(".", "/root/rlcr-exact", ignore=[".git", "data", "**/__pycache__", "rlcr-*"])
)


@app.function(image=vllm_image, gpu="L40S", volumes={VOL: data, HF: hf_cache}, timeout=3 * 3600)
def tts_gen_remote(out: str, k: int = 8, limit: int = 0, temperature: float = 0.7, splits: str = "dev,test"):
    import subprocess, sys
    ins = [f"{VOL}/gpuq-runs/{RLVR_CACHE.removeprefix(GPUQ + '/')}/rollouts_hotpot_fresh_{s}.jsonl" for s in splits.split(",")]
    cmd = [sys.executable, "scripts/tts_gen.py", *ins, "--out", f"{VOL}/tts/{out}", "--k", str(k), "--temperature", str(temperature)] + (["--limit", str(limit)] if limit else [])
    import os; os.makedirs(f"{VOL}/tts", exist_ok=True)
    code = subprocess.run(cmd, cwd="/root/rlcr-exact").returncode
    hf_cache.commit(); data.commit()
    if code:
        raise RuntimeError(f"tts_gen exited {code}")


@app.local_entrypoint()
def tts_gen(out: str = "rollouts_k8.jsonl", k: int = 8, limit: int = 0, temperature: float = 0.7, splits: str = "dev,test"):
    """k rollouts per dev and test question (RLVR-hotpot, T 0.7) to /tts/OUT on the data volume."""
    tts_gen_remote.remote(out, k, limit, temperature, splits)


@app.local_entrypoint()
def tts_score(rollouts: str = "rollouts_k8.jsonl", name: str = "tts-score-k8", runs: str = "", prior: str = ""):
    """Score /tts/ROLLOUTS with the base model and each RunPod run's adapter (comma-separated run names); writes
    /modal-runs/NAME/scores.jsonl."""
    adapters = " ".join(f"{r}={VOL}/runpod-runs/{r}" for r in runs.split(","))
    run.remote(name, f"{VOL}/tts/{rollouts} {adapters}" + (f" --prior {prior}" if prior else ""), "scripts/tts_score.py")
