import pandas as pd 
import subprocess 
import json 
import time 
import tempfile 
import os 

# ------------------ Settings ------------------
PATH = "xxx"      #load inference_dataset
MAX_CANDIDATES = 15
TOP_K = 5
OLLAMA_MODEL = "llama3.3"
OUTPUT_PATH = "xxx"    #llama_ranked_output

# ------------------ Load Dataset ------------------
df = pd.read_excel(PATH)
results = []

# ------------------ Ranking Function ------------------
def rank_candidates_ollama(problem_text, candidate_texts, candidate_ids, top_k=5):

    # Format candidates as required by paper
    # Example: 1-21174209: text
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
Return ONLY a Top-{top_k} ranked list of architectural solution identifiers ordered from most relevant to least relevant, formatted as:
[1-21174209, 2-66989236, 3-58820361, ...]

Do not output any additional text, explanation, or formatting.
"""

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
        response_text = response_json.get("response", "")

        # ------------------ Parse structured IDs ------------------
        # Expected: [1-21174209, 2-66989236, ...]
        ranked_ids = []

        cleaned = response_text.replace("[", "").replace("]", "")
        for item in cleaned.split(","):
            item = item.strip()
            if "-" in item:
                ranked_ids.append(item)

        if len(ranked_ids) == 0:
            return candidate_ids[:top_k]

        return ranked_ids[:top_k]

    except Exception as e:
        print("Error calling Ollama:", e)
        return candidate_ids[:top_k]

    finally:
        os.remove(tmpfile_path)

# ------------------ Ranking ------------------
for commit_id, group in df.groupby("commit_id"):
    problem_text = group.problem_text.iloc[0]

    # Shuffle to remove positional bias
    shuffled_group = group.sample(frac=1, random_state=42)

    candidate_texts = shuffled_group.solution_text.tolist()[:MAX_CANDIDATES]
    candidate_ids = shuffled_group.solution_index.tolist()[:MAX_CANDIDATES]

    ranked_ids = rank_candidates_ollama(
        problem_text,
        candidate_texts,
        candidate_ids,
        top_k=TOP_K
    )

    # Map returned IDs back to original data
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
                "p_text": problem_text,
                "solution_index": id_to_original[rid],
                "solution_text": id_to_text[rid],
                "rank": rank_position
            })

    time.sleep(0.5)

# ------------------ Save ------------------
pd.DataFrame(results).to_excel(OUTPUT_PATH, index=False)
print("LLaMA ranking completed. Saved to:", OUTPUT_PATH)

====LlamaRanking Results=====
HR@3: 0.7771
HR@5: 0.7962
HR@10: 0.8249
HR@15: 0.8845
MRR: 0.7319
