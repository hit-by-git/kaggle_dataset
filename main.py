import json
import pandas as pd
from vllm import LLM, SamplingParams
import re
import argparse
import os

# Slang replacement dictionary for Ablation Study (Experiment 3)
SLANG_MAP = {
    "ghotala": "scam",
    "tamasha": "spectacle",
    "jumla": "false promise",
    "dhokha": "betrayal",
    "babu": "official"
}

def de_slang(text):
    for slang, replacement in SLANG_MAP.items():
        # Case insensitive replacement
        text = re.sub(rf'\b{slang}\b', replacement, text, flags=re.IGNORECASE)
    return text

PROMPTS = {
    "neutral": """Article: {article}
Post: {post}
Does this post contain misinformation based strictly on the facts in the article? Answer only "Yes" or "No".""",

    "instructed": """You are an expert fact-checker specializing in Indian political news. Your task is to determine if a social media post contains misinformation.
Article: {article}
Post: {post}
Does this post contain misinformation? Answer only "Yes" or "No".""",

    "cot": """You are a meticulous fact-checker. 
Article: {article}
Post: {post}
First, list the key facts from the Article. Then compare them to the Post. Finally, provide your verdict on whether the post contains misinformation starting with the exact phrase "Conclusion: Yes" or "Conclusion: No"."""
}

def run_experiment(llm, df, prompt_type, use_ablated_slang=False):
    # Temperature 0 for deterministic outputs
    sampling_params = SamplingParams(temperature=0.0, max_tokens=150)
    
    prompts = []
    for _, row in df.iterrows():
        post = row['post']
        if use_ablated_slang:
            post = de_slang(post)
            
        content = PROMPTS[prompt_type].format(article=row['article'], post=post)
        # Using Gemma 2 instruction format
        gemma_prompt = f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"
        prompts.append(gemma_prompt)
        
    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    
    predictions = []
    for output in outputs:
        text = output.outputs[0].text.strip().lower()
        if prompt_type == "cot":
            if "conclusion: yes" in text or "\nyes" in text[-10:]:
                predictions.append("misinfo")
            else:
                predictions.append("not_misinfo")
        else:
            if text.startswith("yes") or "yes" in text[:10]:
                predictions.append("misinfo")
            else:
                predictions.append("not_misinfo")
                
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to result.jsonl")
    parser.add_argument("--model", type=str, default="google/gemma-2-9b-it")
    args = parser.parse_args()

    print("Loading dataset...")
    data = []
    with open(args.data, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    
    # Initialize vLLM 
    # tensor_parallel_size=2 is critical for Kaggle's dual T4 GPUs
    # dtype="half" because T4 doesn't support bfloat16 natively
    print(f"Loading {args.model} with vLLM on 2 GPUs...")
    llm = LLM(
        model=args.model, 
        tensor_parallel_size=2, 
        max_model_len=4096, 
        dtype="half",
        trust_remote_code=True,
        enforce_eager=True # Recommended for Gemma 2 on some vLLM versions to avoid CUDA graph issues
    )
    
    results_df = df.copy()
    
    # Experiment 2: Prompt Sensitivity (running Neutral, Instructed, and CoT)
    for p_type in ["neutral", "instructed", "cot"]:
        print(f"\n--- Running Experiment: Prompt style '{p_type}' ---")
        preds = run_experiment(llm, df, p_type, use_ablated_slang=False)
        results_df[f'pred_{p_type}'] = preds
        
    # Experiment 3: Slang Ablation (running Neutral prompt on de-slanged Hinglish)
    print("\n--- Running Experiment: Slang Ablation (Neutral prompt) ---")
    preds = run_experiment(llm, df, "neutral", use_ablated_slang=True)
    results_df['pred_ablation'] = preds
    
    results_df.to_csv("experiment_results.csv", index=False)
    print("\nAll experiments complete! Results saved to experiment_results.csv")
    
if __name__ == "__main__":
    main()
