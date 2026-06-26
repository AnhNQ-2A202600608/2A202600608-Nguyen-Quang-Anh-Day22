# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Nguyễn Quang Anh
**MSSV:** 2A202600608
**Cohort:** A20
**Tier đã chạy:** T4
**Date:** 2026-06-26

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Free Colab T4 16GB |
| CUDA / driver | _(fill after running: e.g., CUDA 12.1, driver 535)_ |
| Base model | unsloth/Qwen2.5-3B-bnb-4bit |
| SFT dataset slice | 5CD-AI/Vietnamese-alpaca-cleaned · 1000 samples · 1 epoch |
| Preference dataset slice | argilla/ultrafeedback-binarized-preferences-cleaned · 1000 pairs · 1 epoch |
| `COMPUTE_TIER` env | T4 |
| Total cost | $0 (free Colab) |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | _(fill after running)_ |
| VRAM peak | _(fill)_ | _(fill)_ |
| Final loss | _(fill: SFT loss)_ | _(fill: DPO loss)_ |
| Reward gap (chosen − rejected, end of training) | n/a | _(fill)_ |
| Mean output length | _(fill: e.g., 142 tokens)_ | _(fill: e.g., 87 tokens)_ |

**Tulu 3 reference numbers** (from deck §7.2b, for context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; do not expect to replicate at 3B / 7B.

---

## 3. Reward curves analysis (≥ 150 words)

> **Paste `03_dpo_reward_curves.png` here** (or link to it in `submission/screenshots/`).

_(Fill after training — analyze the 3-panel reward curve plot from NB3. Must address ALL of the following:)_

**Chosen reward trajectory:** _(Did chosen_rewards increase over training? If so, describe the trend — was it monotonic, or did it plateau after a certain step? What is the starting value vs ending value?)_

**Rejected reward trajectory:** _(Did rejected_rewards decrease? If chosen went up while rejected also went up but slower, that's different from chosen going down while rejected went down faster.)_

**Reward gap interpretation:** _(The reward gap = chosen - rejected. Did it grow? How much? From what starting value to what ending value?)_

**Failure mode analysis (deck §3.4):** _(Is this "classic DPO success" where chosen rewards increase and gap grows? Or is this "likelihood displacement" (Razin et al. 2024) where the gap grew because rejected fell faster than chosen? Likelihood displacement is documented in deck §3.4 — it's not a bug, but it tells you the model found a way to widen the gap that doesn't necessarily improve the actual chosen probability. Reference the raw values from `dpo_metrics.json` to support your interpretation.)_

**KL divergence observation:** _(If KL divergence to reference is logged, comment on whether the model drifted far from the reference policy.)_

---

## 4. Qualitative comparison (≥ 8 examples)

> **Paste `04_side_by_side_table.png` here** (or summarize in markdown).

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | _(fill)_ | _(fill)_ | _(fill)_ | _(fill)_ |
| 2 | helpfulness | | | | |
| 3 | helpfulness | | | | |
| 4 | helpfulness | | | | |
| 5 | safety | | | | |
| 6 | safety | | | | |
| 7 | safety | | | | |
| 8 | safety | | | | |

**Win/loss/tie summary:** _(fill: e.g., SFT+DPO wins 5/8, ties 2/8, loses 1/8)_

**Judge used:** manual heuristic rubric (no API key)

---

## 5. β trade-off

_(Fill after β-sweep bonus — if not done, write a 3-sentence hypothesis instead.)_

| β | Reward gap | Win-rate (8 prompts) | Output length | Notes |
|---:|---:|---:|---:|---|
| 0.05 | _(fill)_ | _(fill)_ | _(fill)_ | |
| 0.1 (default) | _(fill)_ | _(fill)_ | _(fill)_ | |
| 0.5 | _(fill)_ | _(fill)_ | _(fill)_ | |

**Interpretation:** _(Where's the sweet spot for your data? Why? Does it match the deck's §3.3 prediction? Higher β means more conservative — model stays closer to reference. Lower β means more aggressive — model can deviate further. The expected shape is that reward gap increases as β decreases, but at very low β the model may overfit to preference data.)_

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

> Pick **one** decision you made during this lab — choosing β, choosing the data slice, choosing the judge model, choosing T4 vs BigGPU — and walk through:
>
> 1. What was the alternative you considered?
> 2. Why did you pick the one you did?
> 3. Did the result confirm or surprise you?
> 4. If you redid the lab tomorrow, what would you change?

_(Fill after completing all notebooks. This section must be at least 150 words and reflect genuine personal experience with the lab. Focus on a specific decision and its consequences, not general observations about DPO theory.)_

---

## 7. Benchmark interpretation (≥ 150 words)

> **Paste `07-benchmark-comparison.png` here** (or link).

Score table from `data/eval/benchmark_results.json`:

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | _(fill)_ | _(fill)_ | _(fill)_ |
| GSM8K | _(fill)_ | _(fill)_ | _(fill)_ |
| MMLU (sampled) | _(fill)_ | _(fill)_ | _(fill)_ |
| AlpacaEval-lite | _(fill)_ | _(fill)_ | _(fill)_ |

_(Fill after running NB6. Interpret the deltas:)_

**Note:** This is a lite benchmark run (limited samples per task) on Colab T4, not a full evaluation. Results are indicative of trends, not publication-quality metrics.

**Which benchmark went up most?** _(If IFEval improved, DPO learned instruction-following — the core alignment signal.)_

**Alignment tax:** _(Did GSM8K regress? This is the classic "alignment tax" documented in deck §8.1 — chat-tuning trades reasoning depth for format compliance.)_

**MMLU preservation:** _(Did MMLU stay flat? That's good — factual knowledge is preserved. A drop > 5pp would indicate catastrophic forgetting.)_

**AlpacaEval consistency:** _(Is AlpacaEval-lite win-rate consistent with NB4 results? They measure overlapping but different things.)_

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)

---

## Điều ngạc nhiên nhất khi làm lab này

_(Fill: 1–3 câu)_
