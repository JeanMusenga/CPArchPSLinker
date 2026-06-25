import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from openai import OpenAI

# =========================================================
# Configuration
# =========================================================
DATA_PATH = "XXXX"  # Specify dataset PATH
OUTPUT_PATH = "xxxx"  # Specify output file name

MODEL_NAME = "gpt-4o-mini"   # or "gpt-4o"
N_SPLITS = 10
REQUEST_DELAY = 0.2

client = OpenAI()  # requires OPENAI_API_KEY


# =========================================================
# Load Dataset
# =========================================================
df = pd.read_excel(DATA_PATH).reset_index(drop=True)

X_problem = df["problem_text"].values
X_solution = df["solution_text"].values
y_true = df["label"].values


def build_prompt(problem_text, solution_text):

    return f"""
Given an architectural problem and a candidate post (e.g., an architectural solution), determine whether the post is relevant to the given problem.

Input:
Architectural Problem: {problem_text}
Candidate Post: {solution_text}

Output Format:
Return only a single value: 1 (relevant) or 0 (not relevant). Do not output any additional text, explanation, or formatting.
""".strip()


# =========================================================
# GPT-4o Inference Function
# =========================================================
def gpt4o_relevance(problem_text, solution_text):

    prompt = build_prompt(problem_text, solution_text)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
            max_tokens=5
            top_p=1
        )

        output = response.choices[0].message.content.strip()

        # ---------------- Robust binary parsing ----------------
        if output == "1":
            return 1
        if output == "0":
            return 0

        # fallback parsing (robust to minor deviations)
        if "1" in output and "0" not in output:
            return 1
        if "0" in output and "1" not in output:
            return 0

        for ch in reversed(output):
            if ch in ["0", "1"]:
                return int(ch)

        return 0

    except Exception as e:
        print("GPT-4o API error:", e)
        return 0


# =========================================================
# 10-Fold Cross Validation
# =========================================================
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

fold_metrics = []
all_predictions = []

for fold, (train_idx, test_idx) in enumerate(kf.split(df)):

    print(f"\nRunning Fold {fold+1}/{N_SPLITS}")

    y_fold_true = []
    y_fold_pred = []

    for idx in test_idx:

        problem_text = X_problem[idx]
        solution_text = X_solution[idx]
        true_label = y_true[idx]

        pred_label = gpt4o_relevance(problem_text, solution_text)

        y_fold_true.append(true_label)
        y_fold_pred.append(pred_label)

        all_predictions.append({
            "fold": fold + 1,
            "problem_text": problem_text,
            "solution_text": solution_text,
            "gold_label": true_label,
            "pred_label": pred_label
        })

        time.sleep(REQUEST_DELAY)

    # ---------------- Fold-level metrics ----------------
    fold_metrics.append({
        "fold": fold + 1,
        "precision": precision_score(y_fold_true, y_fold_pred, zero_division=0),
        "recall": recall_score(y_fold_true, y_fold_pred, zero_division=0),
        "f1": f1_score(y_fold_true, y_fold_pred, zero_division=0),
        "accuracy": accuracy_score(y_fold_true, y_fold_pred)
    })


# =========================================================
# Save Results 
# =========================================================
pred_df = pd.DataFrame(all_predictions)
metrics_df = pd.DataFrame(fold_metrics)

pred_df.to_excel("gpt4o_relevance_predictions.xlsx", index=False)
metrics_df.to_excel(OUTPUT_PATH, index=False)

print("\n===== GPT-4o Relevance Identification Completed =====")
print(metrics_df.mean(numeric_only=True))


====Results=====
Precision: 0.9130 
Recall: 0.8400
F1-score: 0.8950
