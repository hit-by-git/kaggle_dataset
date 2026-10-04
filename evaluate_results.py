import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse

def print_metrics(y_true, y_pred, title=""):
    print(f"\n--- {title} ---")
    acc = accuracy_score(y_true, y_pred)
    # Using 'misinfo' as the positive class
    pos_label = 'misinfo'
    prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    cm = confusion_matrix(y_true, y_pred, labels=['not_misinfo', 'misinfo'])
    print("Confusion Matrix (TN, FP | FN, TP):")
    print(cm)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="experiment_results.csv", help="Path to experiment_results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    
    print("====== THESIS EVALUATION REPORT ======")
    
    # 1. Base Accuracy by Language (Neutral Prompt)
    print("\n\n1. LANGUAGE GAP ANALYSIS (Neutral Prompt)")
    for lang in df['language'].unique():
        subset = df[df['language'] == lang]
        print_metrics(subset['label'], subset['pred_neutral'], title=f"Language: {lang}")
        
    # 2. Prompt Sensitivity Analysis (English vs Hinglish)
    print("\n\n2. PROMPT SENSITIVITY ANALYSIS (Hinglish Subset)")
    hinglish_df = df[df['language'] == 'Hinglish (Roman script)']
    for p_type in ['neutral', 'instructed', 'cot']:
        print_metrics(hinglish_df['label'], hinglish_df[f'pred_{p_type}'], title=f"Prompt: {p_type}")
        
    # 3. Slang Ablation Study
    print("\n\n3. SLANG ABLATION STUDY (Hinglish Subset)")
    print_metrics(hinglish_df['label'], hinglish_df['pred_neutral'], title="Original Hinglish (with slang)")
    print_metrics(hinglish_df['label'], hinglish_df['pred_ablation'], title="De-slanged Hinglish (slang replaced)")

if __name__ == "__main__":
    main()
