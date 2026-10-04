"""
Thesis Experiment: Multi-Prompt Linguistic Evaluation of LLM Fact-Checking
Runs Gemma 2 9B-IT via vLLM on Kaggle 2x T4 GPUs.

Experiments:
  - Experiment 1: Zero-shot evaluation with 3 prompt styles (neutral, instructed, CoT)
  - Experiment 2: Slang ablation study (replace Indian slang with English equivalents)
"""
import json
import time
import re
import argparse
import pandas as pd
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Slang replacement dictionary for Ablation Study
# ---------------------------------------------------------------------------
SLANG_MAP = {
    "ghotala": "scam",
    "tamasha": "spectacle",
    "jumla": "false promise",
    "dhokha": "betrayal",
    "babu": "official",
}

def de_slang(text: str) -> str:
    """Replace Indian political slang with neutral English equivalents."""
    for slang, replacement in SLANG_MAP.items():
        text = re.sub(rf'\b{slang}\b', replacement, text, flags=re.IGNORECASE)
    return text

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
PROMPTS = {
    "neutral": (
        'Article: {article}\n'
        'Post: {post}\n'
        'Does this post contain misinformation based strictly on the facts '
        'in the article? Answer only "Yes" or "No".'
    ),
    "instructed": (
        'You are an expert fact-checker specializing in Indian political news. '
        'Your task is to determine if a social media post contains misinformation.\n'
        'Article: {article}\n'
        'Post: {post}\n'
        'Does this post contain misinformation? Answer only "Yes" or "No".'
    ),
    "cot": (
        'You are a meticulous fact-checker.\n'
        'Article: {article}\n'
        'Post: {post}\n'
        'Step 1: List the key facts from the Article.\n'
        'Step 2: Compare each fact to the claims in the Post.\n'
        'Step 3: State your final verdict starting with exactly '
        '"Verdict: Yes" if the post contains misinformation, '
        'or "Verdict: No" if it does not.'
    ),
}

# Max tokens per prompt type (CoT needs more room for reasoning)
MAX_TOKENS = {
    "neutral": 16,
    "instructed": 16,
    "cot": 512,
}

# ---------------------------------------------------------------------------
# Prediction parsing
# ---------------------------------------------------------------------------
def parse_prediction(text: str, prompt_type: str) -> str:
    """Extract a misinfo / not_misinfo label from the raw LLM output."""
    text = text.strip().lower()

    if not text:
        return "not_misinfo"  # empty response → treat as refusal / no detection

    if prompt_type == "cot":
        # Look for "Verdict: Yes" or "Verdict: No" anywhere in the response
        match = re.search(r'verdict\s*:\s*(yes|no)', text)
        if match:
            return "misinfo" if match.group(1) == "yes" else "not_misinfo"
        # Fallback: check the very last word
        last_word = text.split()[-1].strip(".,!?")
        return "misinfo" if last_word == "yes" else "not_misinfo"
    else:
        # For neutral / instructed: model should reply with just "Yes" or "No"
        first_word = text.split()[0].strip(".,!?")
        return "misinfo" if first_word == "yes" else "not_misinfo"

# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment(llm: LLM, df: pd.DataFrame, prompt_type: str,
                   use_ablated_slang: bool = False) -> list[str]:
    """Build prompts, run batched inference, return list of predictions."""

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS[prompt_type],
    )

    # Build all prompts
    prompts = []
    for _, row in df.iterrows():
        post = de_slang(row["post"]) if use_ablated_slang else row["post"]
        content = PROMPTS[prompt_type].format(article=row["article"], post=post)
        # Gemma 2 instruction-tuned chat format
        gemma_prompt = (
            f"<start_of_turn>user\n{content}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        prompts.append(gemma_prompt)

    print(f"  Sending {len(prompts)} prompts (max_tokens={MAX_TOKENS[prompt_type]})...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} prompts/sec)")

    predictions = [
        parse_prediction(out.outputs[0].text, prompt_type)
        for out in outputs
    ]
    return predictions

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run thesis experiments with vLLM on Kaggle T4 x2"
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Path to result.jsonl")
    parser.add_argument("--model", type=str, default="google/gemma-2-9b-it",
                        help="HuggingFace model ID")
    parser.add_argument("--output", type=str, default="experiment_results.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    # ---- Load dataset ----
    print(f"Loading dataset from {args.data} ...")
    data = []
    with open(args.data, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    df = pd.DataFrame(data)
    print(f"  Loaded {len(df)} rows  |  Labels: {df['label'].value_counts().to_dict()}")

    # ---- Initialise vLLM ----
    #
    # Key decisions for Kaggle 2x T4 (15 GB VRAM each):
    #   - tensor_parallel_size=2   → split model across both GPUs
    #   - dtype="bfloat16"         → Gemma 2 REJECTS float16; PyTorch ≥2.0
    #                                supports bfloat16 on T4 via software emulation
    #   - gpu_memory_utilization=0.85 → leave headroom to prevent OOM
    #   - enforce_eager=True       → avoids CUDA-graph issues with Gemma 2
    #   - max_model_len=4096       → sufficient for our prompts
    #
    print(f"\nLoading {args.model} with vLLM (tensor_parallel=2, bfloat16) ...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=2,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        dtype="half",
        trust_remote_code=True,
        enforce_eager=True,
    )
    print("Model loaded successfully!\n")

    results_df = df.copy()

    # ---- Experiment 1: Prompt sensitivity (3 styles) ----
    for p_type in ["neutral", "instructed", "cot"]:
        print(f"=== Experiment: Prompt style '{p_type}' ===")
        preds = run_experiment(llm, df, p_type, use_ablated_slang=False)
        results_df[f"pred_{p_type}"] = preds

    # ---- Experiment 2: Slang ablation (neutral prompt, slang replaced) ----
    print("=== Experiment: Slang Ablation (neutral prompt, de-slanged) ===")
    preds = run_experiment(llm, df, "neutral", use_ablated_slang=True)
    results_df["pred_ablation"] = preds

    # ---- Save ----
    results_df.to_csv(args.output, index=False)
    print(f"\nAll experiments complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()
