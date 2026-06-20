import pandas as pd
import subprocess
import json
import time
import tempfile
import os
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ------------------ Settings ------------------
PATH = "XXXX"  # Specify dataset PATH
OLLAMA_MODEL = "llama3.3"
N_SPLITS = 10
OUTPUT_PATH = "xxxx"  # specify output file name

# ------------------ Load Data ------------------
df = pd.read_excel(PATH).reset_index(drop=True)

X_problem = df["problem_text"].values
X_solution = df["solution_text"].values
y_true = df["label"].values

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

all_results = []

# ------------------ LLM Function ------------------
def judge_relevance(problem_text, solution_text):

    prompt = f"""
Given an architectural problem extracted from a GitHub issue/commit and a candidate Stack Overflow post, determine whether the post provides a relevant architectural solution to the given problem.

Input:
Architectural Problem: {problem_text}
Candidate Stack Overflow Post: {solution_text}

Output Format:
Return only a single value: 1 (relevant) or 0 (not relevant). Do not output any additional text, explanation, or formatting.
""".strip()

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt", encoding="utf-8") as tmpfile:
        tmpfile.write(prompt)
        tmpfile_path = tmpfile.name

    try:
        result = subprocess.run(
            ["ollama", "query", OLLAMA_MODEL, "--prompt-file", tmpfile_path, "--json"],
            capture_output=True,
            text=True,
            check=True
        )

        response_json = json.loads(result.stdout)
        response_text = response_json.get("response", "").strip().lower()

        # -------- ROBUST PARSING --------
        if "1" in response_text and "0" not in response_text:
            return 1
        if "0" in response_text and "1" not in response_text:
            return 0

        # fallback: last digit heuristic
        for ch in reversed(response_text):
            if ch in ["0", "1"]:
                return int(ch)

        return 0

    except Exception as e:
        print("Error:", e)
        return 0

    finally:
        os.remove(tmpfile_path)


# ------------------ 10-Fold CV ------------------
fold_metrics = []

for fold, (train_idx, test_idx) in enumerate(kf.split(df)):

    print(f"Running fold {fold+1}/{N_SPLITS}")

    y_fold_true = []
    y_fold_pred = []

    for idx in test_idx:

        problem_text = X_problem[idx]
        solution_text = X_solution[idx]
        true_label = y_true[idx]

        pred_label = judge_relevance(problem_text, solution_text)

        y_fold_true.append(true_label)
        y_fold_pred.append(pred_label)

        all_results.append({
            "fold": fold + 1,
            "problem_text": problem_text,
            "solution_text": solution_text,
            "gold_label": true_label,
            "pred_label": pred_label
        })

        time.sleep(0.2)

    # ---- metrics per fold ----
    fold_metrics.append({
        "fold": fold + 1,
        "precision": precision_score(y_fold_true, y_fold_pred, zero_division=0),
        "recall": recall_score(y_fold_true, y_fold_pred, zero_division=0),
        "f1": f1_score(y_fold_true, y_fold_pred, zero_division=0),
        "accuracy": accuracy_score(y_fold_true, y_fold_pred)
    })


# ------------------ Save Outputs ------------------
pd.DataFrame(all_results).to_excel("llama_cv_predictions.xlsx", index=False)
pd.DataFrame(fold_metrics).to_excel(OUTPUT_PATH, index=False)

print("10-fold CV completed. Results saved.")