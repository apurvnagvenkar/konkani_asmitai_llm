import pandas as pd
import evaluate
import argparse
import json

# -------------------------
# Parse command-line arguments
# -------------------------
parser = argparse.ArgumentParser(description="Evaluate model outputs using BLEU and ROUGE")
parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file with outputs")
parser.add_argument("--output_file", type=str, required=True, help="Path to save evaluation results (txt or json)")
args = parser.parse_args()

# -------------------------
# Load CSV
# -------------------------
df = pd.read_csv(args.csv_file)

# Extract ground truth and predictions
references = df["output"].astype(str).tolist()
predictions = df["model_output"].astype(str).tolist()

# -------------------------
# Load metrics
# -------------------------
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

bleu_score = bleu.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=predictions, references=references)

# -------------------------
# Combine results
# -------------------------
results = {
    "BLEU": bleu_score["bleu"],
    "ROUGE-1": rouge_score["rouge1"],
    "ROUGE-2": rouge_score["rouge2"],
    "ROUGE-L": rouge_score["rougeL"],
}

# Print
print("\n--- Evaluation Results ---")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# Save
if args.output_file.endswith(".json"):
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
else:
    with open(args.output_file, "w") as f:
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")

print(f"\nEvaluation saved to {args.output_file}")
