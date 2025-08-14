import requests
import time
import csv
from datetime import datetime
from promptlist import evaluation_prompts

API_URL = "http://localhost:3002/generate_instructions/"
OUTPUT_FILE = "evaluation_results.csv"

headers = [
    "timestamp", "domain", "title", "prompt",
    "gnn_instructions", "mcts_instructions", "rmodel_instructions",
    "gnn_score", "mcts_score", "rmodel_score"
]

# Dummy graph (replace if you want to simulate real graphs)
empty_graph = {"nodes": [], "links": []}

from app import calculate_instruction_metrics  # if externalized
from sentence_transformers import SentenceTransformer

classifier_model = SentenceTransformer("all-MiniLM-L6-v2")

results = []

for idx, item in enumerate(evaluation_prompts):
    print(f"\nPrompt {idx+1}/{len(evaluation_prompts)} - {item['title']}")

    payload = {
        "prompt": item["prompt"],
        "user_nodes": [],
        "gnn_subgraph": empty_graph,
        "mcts_subgraph": empty_graph
    }

    try:
        res = requests.post(API_URL, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()

        gnn = data.get("gnn_instructions", "").strip()
        mcts = data.get("mcts_instructions", "").strip()
        rmodel = data.get("rmodel_instructions", "").strip()

        # Metrics from your internal evaluator
        metrics = calculate_instruction_metrics(item["prompt"], [], gnn, mcts, rmodel)

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "domain": item["domain"],
            "title": item["title"],
            "prompt": item["prompt"],
            "gnn_instructions": gnn,
            "mcts_instructions": mcts,
            "rmodel_instructions": rmodel,
            "gnn_score": metrics["overall_effectiveness"]["gnn"],
            "mcts_score": metrics["overall_effectiveness"]["mcts"],
            "rmodel_score": metrics["overall_effectiveness"]["rmodel"]
        }

        results.append(row)

    except Exception as e:
        print(f"Error on prompt {idx+1}: {e}")
        continue

# Save to CSV
with open(OUTPUT_FILE, "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Finished. Results saved to: {OUTPUT_FILE}")
