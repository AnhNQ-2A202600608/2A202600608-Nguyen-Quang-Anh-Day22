# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB5 — Merge + Deploy + GGUF  (OPTIONAL / BONUS)
#
# > **Optional (bonus).** Core lab = NB1--NB4. GGUF export builds llama.cpp at
# > runtime and is the most fragile step --- skip on free Colab T4 if short on time.
#
# **Stack:** Unsloth `merge_and_unload` + `save_pretrained_gguf(quantization='Q4_K_M')`
# + llama-cpp-python smoke test.
# Maps to deck §7.1 lab brief: "merge adapter, quantize GGUF, serve với vLLM".
#
# > **Mục tiêu:** export the SFT+DPO adapter as a deployable GGUF Q4_K_M file
# > (~1.5 GB on 3B / ~4 GB on 7B), then smoke-test it through llama-cpp-python.
# > Final cell shows the optional vLLM serving command (BigGPU only).

# %% [markdown]
# ## 0. Setup

# %%
import os
import json
from pathlib import Path

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
BASE_MODEL = (
    "unsloth/Qwen2.5-3B-bnb-4bit" if COMPUTE_TIER == "T4"
    else "unsloth/Qwen2.5-7B-bnb-4bit"
)
MAX_LEN = 512 if COMPUTE_TIER == "T4" else 1024

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DPO_PATH = REPO_ROOT / "adapters" / "dpo"
MERGED_PATH = REPO_ROOT / "adapters" / "merged-fp16"
GGUF_DIR = REPO_ROOT / "gguf"
MERGED_PATH.mkdir(parents=True, exist_ok=True)
GGUF_DIR.mkdir(parents=True, exist_ok=True)

assert DPO_PATH.exists(), "NB3 must run first"

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"DPO adapter:     {DPO_PATH}")
print(f"merged output:   {MERGED_PATH}")
print(f"GGUF output:     {GGUF_DIR}")

# %%
import torch

assert torch.cuda.is_available()

# %% [markdown]
# ## 1. Load DPO model + merge adapter

# %%
from unsloth import FastLanguageModel
from peft import PeftModel
import json
import gc

# STEP 1: Load SFT model directly using Unsloth's native loader (avows PeftModel merging issues)
SFT_PATH = REPO_ROOT / "adapters" / "sft-mini"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(SFT_PATH),
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=True,
)

# Save SFT-merged model
model.save_pretrained_merged(
    str(MERGED_PATH),
    tokenizer,
    save_method="merged_16bit",
)
print(f"Saved merged SFT FP16 to {MERGED_PATH}")

# Free memory before loading again
del model
gc.collect()
torch.cuda.empty_cache()

# Patch config.json of SFT-merged model to remove quantization_config
config_path = MERGED_PATH / "config.json"
if config_path.exists():
    with open(config_path, "r") as f:
        config = json.load(f)
    if "quantization_config" in config:
        del config["quantization_config"]
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("Removed quantization_config from SFT-merged config.json")

# STEP 2: Temporarily patch DPO adapter config to use the SFT-merged model as its base
dpo_config_path = DPO_PATH / "adapter_config.json"
orig_base = None
if dpo_config_path.exists():
    with open(dpo_config_path, "r") as f:
        dpo_config = json.load(f)
    orig_base = dpo_config.get("base_model_name_or_path")
    dpo_config["base_model_name_or_path"] = str(MERGED_PATH)
    with open(dpo_config_path, "w") as f:
        json.dump(dpo_config, f, indent=2)
    print(f"Temporarily changed DPO base_model_name_or_path to {MERGED_PATH}")

# Load SFT-merged model + DPO adapter natively
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(DPO_PATH),
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=False,  # Load the FP16 merged SFT model
)

# Restore original base model in config
if orig_base and dpo_config_path.exists():
    with open(dpo_config_path, "r") as f:
        dpo_config = json.load(f)
    dpo_config["base_model_name_or_path"] = orig_base
    with open(dpo_config_path, "w") as f:
        json.dump(dpo_config, f, indent=2)
    print("Restored original base_model_name_or_path in DPO adapter config")

# Save final SFT+DPO merged model
model.save_pretrained_merged(
    str(MERGED_PATH),
    tokenizer,
    save_method="merged_16bit",
)
print(f"Saved final SFT+DPO merged FP16 to {MERGED_PATH}")

# Free memory
del model
gc.collect()
torch.cuda.empty_cache()

# Patch final config.json to remove quantization_config again (in case save_pretrained_merged re-introduced it)
if config_path.exists():
    with open(config_path, "r") as f:
        config = json.load(f)
    if "quantization_config" in config:
        del config["quantization_config"]
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("Removed quantization_config from final merged config.json")

# STEP 3: Reload the final merged model for GGUF quantization
from unsloth import FastLanguageModel as FLM

model, tokenizer = FLM.from_pretrained(
    model_name=str(MERGED_PATH),
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=False,
)

# %%
# Save GGUF in 1 quantization tier (Q4_K_M). Add more tiers below if you want the
# +3 "GGUF release published" rigor add-on.
model.save_pretrained_gguf(
    str(GGUF_DIR),
    tokenizer,
    quantization_method="q4_k_m",
)
print(f"Saved GGUF Q4_K_M to {GGUF_DIR}")

# %% [markdown]
# ### 3a. Optional — additional quantization tiers (for the +3 rigor add-on)

# %%
# Uncomment if you want Q5_K_M + Q8_0 too (~2× total disk space).
# Each adds ~30s for an extra GGUF file.
#
# model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q5_k_m")
# model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q8_0")

# %%
import os

print("GGUF files:")
for p in sorted(GGUF_DIR.iterdir()):
    if p.suffix == ".gguf":
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:50s}  {size_mb:>8.1f} MB")

del model
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# ## 4. Smoke test with llama-cpp-python

# %%
from llama_cpp import Llama

# Find the Q4_K_M GGUF
gguf_files = list(GGUF_DIR.glob("*Q4_K_M*.gguf")) + list(GGUF_DIR.glob("*q4_k_m*.gguf"))
assert gguf_files, "No Q4_K_M GGUF found — step 3 may have failed"
gguf_path = gguf_files[0]
print(f"Loading: {gguf_path.name}")

# n_gpu_layers=-1 offloads all layers to GPU if compiled with CUDA/Metal/Vulkan
llm = Llama(
    model_path=str(gguf_path),
    n_ctx=MAX_LEN,
    n_gpu_layers=-1,           # all layers on GPU; falls back to CPU if no GPU compile
    verbose=False,
)
print("Loaded.")

# %% [markdown]
# ### 4a. Smoke prompt + response (deliverable: `06-gguf-smoke.png`)

# %%
SMOKE_PROMPT = "Giải thích ngắn gọn (3 câu) cách thuật toán Bubble sort hoạt động."

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": SMOKE_PROMPT}],
    max_tokens=200,
    temperature=0.0,
)

print(f"PROMPT:\n  {SMOKE_PROMPT}\n")
print(f"RESPONSE (Q4_K_M GGUF, llama-cpp-python):\n  {response['choices'][0]['message']['content']}")
print(f"\nTokens used: {response['usage']}")

# %% [markdown]
# ## 5. Optional — vLLM serving (BigGPU only)
#
# vLLM provides production-grade OpenAI-compatible serving. **Requires CUDA GPU
# with ≥ 16 GB VRAM** and `vllm` installed (see `requirements-biggpu.txt`).
# On T4 tier this cell will OOM. Skip on T4.
#
# Run in a SEPARATE terminal (NOT in the notebook — vLLM blocks until killed):
#
# ```bash
# pip install vllm                         # once
# vllm serve adapters/merged-fp16 \
#   --port 8000 \
#   --max-model-len 1024 \
#   --gpu-memory-utilization 0.9
# ```
#
# Then test:
#
# ```bash
# curl http://localhost:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{"model": "merged-fp16", "messages": [{"role": "user", "content": "Hello"}]}'
# ```
#
# **Why not in the notebook?** vLLM's process model doesn't play nicely with
# Jupyter — it expects to own the GPU + a long-running HTTP server. Run it as
# a sidecar process. The deck mentions vLLM as the deploy target; for actual
# production you'd containerize this command. For the lab, llama-cpp-python in
# step 4 is the graded artifact.

# %% [markdown]
# ## 6. Save deployment metadata

# %%
deploy_meta = {
    "compute_tier": COMPUTE_TIER,
    "base_model": BASE_MODEL,
    "merged_path": str(MERGED_PATH),
    "gguf_path": str(gguf_path),
    "gguf_size_mb": round(gguf_path.stat().st_size / 1e6, 1),
    "quantization": "q4_k_m",
    "smoke_prompt": SMOKE_PROMPT,
    "smoke_response": response["choices"][0]["message"]["content"],
}
(REPO_ROOT / "data" / "eval" / "deploy_meta.json").parent.mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "data" / "eval" / "deploy_meta.json").write_text(
    json.dumps(deploy_meta, ensure_ascii=False, indent=2)
)
print("Saved data/eval/deploy_meta.json")

# %% [markdown]
# ## 7. Submission checklist
#
# Bạn vừa hoàn thành core lab. Trước khi submit:
#
# 1. **Run** `make verify` — gatekeeper sẽ list missing artifacts.
# 2. **Take screenshots** vào `submission/screenshots/` (xem `submission/screenshots/README.md`).
# 3. **Fill** `submission/REFLECTION.md` — đặc biệt là § 3 (reward curves analysis,
#    cross-reference deck §3.4) và § 6 (single change that mattered most).
# 4. **(Optional)** Pick a rigor add-on từ rubric.md (β-sweep, HF push, GGUF
#    release, W&B link, cross-judge).
# 5. **(Optional)** Pick a `BONUS-CHALLENGE.md` provocation cho creative bonus.
#
# Push public repo + paste URL vào VinUni LMS Day-22 box.
#
# Câu hỏi cuối để brainstorm trước khi đóng laptop:
#
# > **The deck says:** "DPO + 30 min A100 + 2k UltraFeedback → 3.2 → 4.1 helpfulness."
# > **You measured:** _<your win-rate from NB4>_.
# > **Why might they differ?** Dataset (English vs VN), base model (Qwen2.5-3B vs
# > deck's unspecified base), judge bias, sample size (8 prompts vs deck's full eval).
# > Đó chính là § 6 trong REFLECTION — what 1 change would close the gap.
