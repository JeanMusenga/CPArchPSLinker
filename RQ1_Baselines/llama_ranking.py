import pandas as pd
import subprocess
import json
import time
import tempfile
import os
import numpy as np

# ------------------ Settings ------------------
INPUT_PATH = "xxx"        #load inference_dataset
GROUND_TRUTH_PATH = "xxx"     #ground_truth_data

RANKED_OUTPUT_PATH = "xxx"    #Llama_ranked_output
METRICS_OUTPUT_PATH = "xxx"   #Llma_evaluation_results

MAX_CANDIDATES = 15
TOP_K = 5
OLLAMA_MODEL = "llama3.3"

HIT_K_VALUES = [3, 5, 10, 15]


# ------------------ Load Dataset ------------------
df = pd.read_excel(INPUT_PATH)
results = []


# ------------------ LLM Ranking Function ------------------
def rank_candidates_ollama(problem_text, candidate_texts, candidate_ids, top_k=5):

    candidate_list_text = "\n".join(
        [
            f"{i+1}-{candidate_ids[i]}: {candidate_texts[i]}"
            for i in range(len(candidate_texts))
        ]
    )

    prompt = f"""
Given the following architectural problem, rank the candidate posts (e.g., architectural solutions) according to their relevance to the problem.

Input:
Architectural Problem: {problem_text}

Architectural Solutions:
{candidate_list_text}

Output Format:
Return ONLY a Top-{top_k} ranked list of post identifiers ordered from most relevant to least relevant, formatted as:
[1-21174209, 2-66989236, 3-58820361, ...]

Do not output any additional text, explanation, or formatting.
"""

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write(prompt)
            tmp_path = f.name

        result = subprocess.run(
            ["ollama", "query", OLLAMA_MODEL, "--prompt-file", tmp_path, "--json"],
            capture_output=True,
            text=True,
            check=True
        )

        response_json = json.loads(result.stdout)
        response_text = response_json.get("response", "")

        ranked_ids = []
        cleaned = response_text.replace("[", "").replace("]", "")

        for item in cleaned.split(","):
            item = item.strip()
            if "-" in item:
                ranked_ids.append(item)

        if not ranked_ids:
            return candidate_ids[:top_k]

        return ranked_ids[:top_k]

    except Exception as e:
        print("Error calling Ollama:", e)
        return candidate_ids[:top_k]

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ------------------ Ranking ------------------
for commit_id, group in df.groupby("commit_id"):

    problem_text = group.problem_text.iloc[0]

    shuffled_group = group.sample(frac=1, random_state=42)

    candidate_texts = shuffled_group.solution_text.tolist()[:MAX_CANDIDATES]
    candidate_ids = shuffled_group.solution_index.tolist()[:MAX_CANDIDATES]

    ranked_ids = rank_candidates_ollama(
        problem_text,
        candidate_texts,
        candidate_ids,
        top_k=TOP_K
    )

    id_to_text = dict(zip(
        [f"{i+1}-{candidate_ids[i]}" for i in range(len(candidate_ids))],
        candidate_texts
    ))

    id_to_original = dict(zip(
        [f"{i+1}-{candidate_ids[i]}" for i in range(len(candidate_ids))],
        candidate_ids
    ))

    for rank_position, rid in enumerate(ranked_ids, start=1):
        if rid in id_to_text:
            results.append({
                "commit_id": commit_id,
                "problem_text": problem_text,
                "solution_index": id_to_original[rid],
                "solution_text": id_to_text[rid],
                "rank": rank_position
            })

    time.sleep(0.5)


# ------------------ Save Ranked Output ------------------
ranked_df = pd.DataFrame(results)
ranked_df.to_excel(RANKED_OUTPUT_PATH, index=False)
print("Ranking completed. Saved to:", RANKED_OUTPUT_PATH)


# ------------------ Stage 2: Evaluation ------------------
ground_truth_df = pd.read_excel(GROUND_TRUTH_PATH)

merged_df = ranked_df.merge(
    ground_truth_df[['commit_id', 'solution_index', 'label']],
    on=['commit_id', 'solution_index'],
    how='left'
)

metrics_list = []

for commit_id, group in merged_df.groupby("commit_id"):

    group = group.sort_values("rank")
    labels = group.label.fillna(0).tolist()

    metrics = {}

    # Hit@K
    for k in HIT_K_VALUES:
        metrics[f"Hit@{k}"] = int(any(labels[:k]))

    # MRR
    try:
        first_rel = next(i + 1 for i, l in enumerate(labels) if l == 1)
        metrics["MRR"] = 1.0 / first_rel
    except StopIteration:
        metrics["MRR"] = 0.0

    metrics["commit_id"] = commit_id
    metrics_list.append(metrics)

metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_excel(METRICS_OUTPUT_PATH, index=False)

print("Evaluation metrics saved to:", METRICS_OUTPUT_PATH)

overall_metrics = metrics_df[["Hit@3", "Hit@5", "Hit@10", "Hit@15", "MRR"]].mean()

print("\nOverall Metrics:")
print(overall_metrics)

====LlamaRanking Results=====
HR@3: 0.7571
HR@5: 0.7962
HR@10: 0.8249
HR@15: 0.8845
MRR: 0.6319
