import pandas as pd
import time
import json
import openai
import math

# ------------------ OpenAI API Key ------------------
openai.api_key = "YOUR_OPENAI_API_KEY_HERE"

# ------------------ Settings ------------------
INFERENCE_PATH = "Inference_Experiment_dataset.xlsx"
GROUND_TRUTH_PATH = "ground_truth_Evaluation.xlsx"
RANKED_OUTPUT_PATH = "gpt4o_ranked_output_shuffled.xlsx"
METRICS_OUTPUT_PATH = "gpt4o_evaluation_results_shuffled.xlsx"

MAX_CANDIDATES = 15
MODEL = "gpt-4o"
BATCH_SIZE = 5
MAX_RETRIES = 5
RETRY_WAIT = 10
TOP_K = 5
HIT_K_VALUES = [3, 5, 10, 15]

# ------------------ Load Data ------------------
df = pd.read_excel(INFERENCE_PATH)
commit_groups = list(df.groupby("commit_id"))
results = []

# ------------------ GPT-4o Ranking Function ------------------
def rank_candidates_gpt4o_batch(commit_batch):
    batch_rankings = []

    for commit_id, group in commit_batch:
        problem_text = group.problem_text.iloc[0]

        # Shuffle candidates (bias removal)
        shuffled_group = group.sample(frac=1, random_state=42)

        candidate_texts = shuffled_group.solution_text.tolist()[:MAX_CANDIDATES]
        candidate_ids = shuffled_group.solution_index.tolist()[:MAX_CANDIDATES]

        # Format candidates as paper requires: 1-21174209
        candidate_list_text = "\n".join(
            [f"{i+1}-{candidate_ids[i]}: {candidate_texts[i]}"
             for i in range(len(candidate_texts))]
        )

        prompt = f"""
You are an expert software architect.

Given the following architectural problem, rank the candidate architectural solutions extracted from Stack Overflow according to their relevance to the problem.

Input:
Architectural Problem: {problem_text}

Architectural Solutions:
{candidate_list_text}

Output Format:
Return ONLY a Top-{TOP_K} ranked list of architectural solution identifiers ordered from most relevant to least relevant, formatted as:
[1-21174209, 2-66989236, 3-58820361, ...]

Do not output any additional text, explanation, or formatting.
"""

        # ------------------ GPT-4o Call ------------------
        for attempt in range(MAX_RETRIES):
            try:
                response = openai.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=128
                )

                text = response.choices[0].message.content.strip()

                # ------------------ Parse structured output ------------------
                ranked_ids = []
                cleaned = text.replace("[", "").replace("]", "")

                for item in cleaned.split(","):
                    item = item.strip()
                    if "-" in item:
                        ranked_ids.append(item)

                if len(ranked_ids) == 0:
                    ranked_ids = [
                        f"{i+1}-{candidate_ids[i]}"
                        for i in range(min(len(candidate_ids), TOP_K))
                    ]

                batch_rankings.append(
                    (commit_id, candidate_texts, candidate_ids, ranked_ids[:TOP_K])
                )
                break

            except Exception as e:
                print(f"Attempt {attempt+1} failed for {commit_id}: {e}")
                time.sleep(RETRY_WAIT)

        else:
            print(f"All retries failed for {commit_id}. Using fallback order.")
            ranked_ids = [
                f"{i+1}-{candidate_ids[i]}"
                for i in range(min(len(candidate_ids), TOP_K))
            ]
            batch_rankings.append(
                (commit_id, candidate_texts, candidate_ids, ranked_ids)
            )

    return batch_rankings


# ------------------ Run Batches ------------------
num_batches = math.ceil(len(commit_groups) / BATCH_SIZE)

for b in range(num_batches):
    batch = commit_groups[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
    batch_rankings = rank_candidates_gpt4o_batch(batch)

    for commit_id, candidate_texts, candidate_ids, ranked_ids in batch_rankings:

        id_to_text = {
            f"{i+1}-{candidate_ids[i]}": candidate_texts[i]
            for i in range(len(candidate_texts))
        }

        id_to_original = {
            f"{i+1}-{candidate_ids[i]}": candidate_ids[i]
            for i in range(len(candidate_texts))
        }

        for rank_position, rid in enumerate(ranked_ids, start=1):
            if rid in id_to_text:
                results.append({
                    "commit_id": commit_id,
                    "rank": rank_position,
                    "solution_index": id_to_original[rid],
                    "solution_text": id_to_text[rid]
                })

    time.sleep(0.5)

# ------------------ Save Ranked Output ------------------
ranked_df = pd.DataFrame(results)
ranked_df.to_excel(RANKED_OUTPUT_PATH, index=False)
print("GPT-4o ranking saved to:", RANKED_OUTPUT_PATH)


# ------------------ Evaluation ------------------
ground_truth_df = pd.read_excel(GROUND_TRUTH_PATH)

merged_df = ranked_df.merge(
    ground_truth_df[['commit_id', 'solution_index', 'label']],
    on=['commit_id', 'solution_index'],
    how='left'
)

metrics_list = []

for commit_id, group in merged_df.groupby("commit_id"):
    group = group.sort_values("rank")
    labels = group.label.tolist()

    metrics = {}

    for k in HIT_K_VALUES:
        metrics[f"Hit@{k}"] = int(any(labels[:k]))

    try:
        first_rel = next(i + 1 for i, l in enumerate(labels) if l == 1)
        metrics["MRR"] = 1.0 / first_rel
    except StopIteration:
        metrics["MRR"] = 0.0

    metrics["commit_id"] = commit_id
    metrics_list.append(metrics)

metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_excel(METRICS_OUTPUT_PATH, index=False)

overall_metrics = metrics_df[["Hit@3", "Hit@5", "Hit@10", "Hit@15", "MRR"]].mean()

print("Evaluation metrics saved to:", METRICS_OUTPUT_PATH)
print("Overall Metrics:")
print(overall_metrics)

====GPTRanking Results=====
HR@3: 0.7841
HR@5: 0.8271
HR@10: 0.9249
HR@15: 0.9845
MRR: 0.7545