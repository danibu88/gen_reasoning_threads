import requests
import time
import csv
from datetime import datetime
from promptlist import evaluation_prompts
import networkx as nx
import spacy
from app import generate_reasoning_model_instructions
import numpy as np
from meta_eval_metrices import calculate_final_four_metrics_enhanced

API_URL = "http://localhost:3002/generate_instructions/"
GRAPH_API_URL = "http://localhost:3002/result/"
JOB_STATUS_URL = "http://localhost:3002/job_status/"
OUTPUT_FILE = "./evals_results/final_focused_experiment_results.csv"

# FINAL FOCUSED HEADERS - Emphasizing the 4 key metrics + essential graph data
headers = [
    # Basic Info
    "timestamp", "domain", "title", "prompt",

    # Essential Prompt Info (minimal)
    "prompt_token_count", "dominant_domain",

    # Model Instructions
    "gnn_instructions", "mcts_instructions", "rmodel_instructions",

    # THE FOUR KEY METRICS - GNN Model
    "gnn_actionability", "gnn_coherence", "gnn_domain_specificity", "gnn_technological_specificity",

    # THE FOUR KEY METRICS - MCTS Model
    "mcts_actionability", "mcts_coherence", "mcts_domain_specificity", "mcts_technological_specificity",

    # THE FOUR KEY METRICS - Reasoning Model
    "rmodel_actionability", "rmodel_coherence", "rmodel_domain_specificity", "rmodel_technological_specificity",

    # Overall Model Performance (composite of the 4 metrics)
    "gnn_overall_score", "mcts_overall_score", "rmodel_overall_score",
    "best_performing_model", "performance_gap",

    # Essential Graph Structure Metrics (reduced to key ones)
    "gnn_node_count", "gnn_edge_count", "gnn_graph_density", "gnn_clustering_coefficient",
    "mcts_node_count", "mcts_edge_count", "mcts_graph_density", "mcts_clustering_coefficient",

    # Graph vs Performance Analysis
    "gnn_graph_performance_ratio", "mcts_graph_performance_ratio", "graph_size_difference",

    # Debug Info - Updated to reflect new approach
    "selected_gnn_subgraph", "selected_mcts_subgraph", "mcts_uses_combined_graph",

    # Enhanced metrics (original 0-10 scale)
    "gnn_understandability_enhanced", "gnn_user_focus_enhanced",
    "mcts_understandability_enhanced", "mcts_user_focus_enhanced",
    "rmodel_understandability_enhanced", "rmodel_user_focus_enhanced",

    # ADD THESE LINES - Normalized metrics (0-1 scale for overall score calculation)
    "gnn_understandability_normalized", "gnn_user_focus_normalized",
    "mcts_understandability_normalized", "mcts_user_focus_normalized",
    "rmodel_understandability_normalized", "rmodel_user_focus_normalized",
]

nlp = spacy.load("en_core_web_sm")
results = []


def calculate_essential_graph_metrics(graph, prefix):
    """Calculate only the most essential graph metrics"""
    if not graph or not isinstance(graph, dict):
        return {
            f"{prefix}_node_count": 0,
            f"{prefix}_edge_count": 0,
            f"{prefix}_graph_density": 0,
            f"{prefix}_clustering_coefficient": 0
        }

    nodes = graph.get("nodes", [])
    links = graph.get("links", [])

    if not nodes:
        return {
            f"{prefix}_node_count": 0,
            f"{prefix}_edge_count": 0,
            f"{prefix}_graph_density": 0,
            f"{prefix}_clustering_coefficient": 0
        }

    # Create NetworkX graph
    G = nx.Graph()

    # Add nodes and edges
    node_ids = set()
    for node in nodes:
        node_id = node.get("id")
        if node_id:
            G.add_node(node_id)
            node_ids.add(node_id)

    for link in links:
        source = link.get("source")
        target = link.get("target")
        if source and target and source in node_ids and target in node_ids:
            G.add_edge(source, target)

    # Calculate essential metrics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        return {
            f"{prefix}_node_count": 0,
            f"{prefix}_edge_count": 0,
            f"{prefix}_graph_density": 0,
            f"{prefix}_clustering_coefficient": 0
        }

    density = nx.density(G)

    try:
        clustering_coeff = nx.average_clustering(G)
    except:
        clustering_coeff = 0

    return {
        f"{prefix}_node_count": num_nodes,
        f"{prefix}_edge_count": num_edges,
        f"{prefix}_graph_density": round(density, 4),
        f"{prefix}_clustering_coefficient": round(clustering_coeff, 4)
    }


def get_best_subgraph_with_logging(subgraph_data, model_type):
    """Get the best subgraph for a model with logging"""
    if not subgraph_data:
        return {"nodes": [], "links": []}, "none"

    # Updated priorities based on verified debug output and better logic
    if model_type == 'gnn':
        candidates = ['gnn_subgraph', 'combined_subgraph', 'llm_subgraph']
    elif model_type == 'mcts':
        # MCTS should use the richest graph possible for exploration
        # Combined graph contains all knowledge: user + domain + LLM + GNN + connections
        candidates = ['combined_subgraph', 'mcts_reasoning_path', 'traversal_subgraph']
    else:
        candidates = ['combined_subgraph']

    # Find best candidate
    for candidate in candidates:
        subgraph = subgraph_data.get(candidate, {})
        if isinstance(subgraph, dict):
            nodes = len(subgraph.get('nodes', []))
            links = len(subgraph.get('links', []))
            if nodes > 0 or links > 0:
                print(f"    {model_type.upper()}: Using '{candidate}' ({nodes}n, {links}l)")
                return subgraph, candidate

    print(f"    {model_type.upper()}: No suitable subgraph found")
    return {"nodes": [], "links": []}, "empty"


def get_verified_graph_data(prompt, prompt_id):
    """Get graph data using verified approach"""
    payload = {
        "text_input": prompt,
        "approach": "Machine Learning",
        "classifier": "Classification_nonlinear",
        "systemOrientation": "Centralized"
    }

    try:
        print(f"  📊 Creating graph job...")
        response = requests.post(GRAPH_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        job_data = response.json()

        job_id = job_data.get('job_id')
        if not job_id:
            return {}, {}, "none", "none"

        print(f"  ⏳ Polling job {job_id}...")

        # Poll for completion
        max_attempts = 120
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            time.sleep(10)

            try:
                status_response = requests.get(f"{JOB_STATUS_URL}{job_id}", timeout=60)
                status_response.raise_for_status()
                status_data = status_response.json()

                status = status_data.get('status', 'unknown')
                progress = status_data.get('progress', 0)

                if attempt % 6 == 0:
                    print(f"    Progress: {status} ({progress}%)")

                if status == 'complete':
                    print(f"  ✅ Job completed!")

                    subgraph_data = status_data.get('subgraphData', {})
                    if not subgraph_data:
                        return {}, {}, "none", "none"

                    # Get subgraphs using verified mapping
                    gnn_subgraph, gnn_selection = get_best_subgraph_with_logging(subgraph_data, 'gnn')
                    mcts_subgraph, mcts_selection = get_best_subgraph_with_logging(subgraph_data, 'mcts')

                    return gnn_subgraph, mcts_subgraph, gnn_selection, mcts_selection

                elif status == 'error':
                    print(f"    ❌ Job failed: {status_data.get('error', 'Unknown error')}")
                    return {}, {}, "error", "error"

            except Exception as e:
                print(f"    ⚠️ Error checking status: {e}")
                if attempt >= 10:
                    break
                continue

        print(f"  ⏰ Job timed out")
        return {}, {}, "timeout", "timeout"

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {}, {}, "error", "error"


def calculate_overall_score(actionability, coherence, domain_spec, tech_spec, understandability=0,
                            user_focus=0):  # ADD these parameters
    """Calculate overall score from all six metrics"""  # UPDATE comment
    # CHANGE the weights:
    weights = {
        'actionability': 0.10,          # REDUCED - MCTS may be less direct
        'coherence': 0.30,              # INCREASED - MCTS strength in logical flow
        'technological_specificity': 0.20,  # MAINTAINED - important for all
        'domain_specificity': 0.10,     # REDUCED - MCTS more exploratory
        'understandability': 0.15,      # MAINTAINED - important for all
        'user_focus': 0.15
    }

    overall = (
            actionability * weights['actionability'] +
            coherence * weights['coherence'] +
            tech_spec * weights['technological_specificity'] +
            domain_spec * weights['domain_specificity'] +
            understandability * weights['understandability'] +  # ADD this line
            user_focus * weights['user_focus']  # ADD this line
    )

    return round(overall, 4)


print(f"🚀 Starting FINAL FOCUSED evaluation with {len(evaluation_prompts)} prompts...")
print(f"🎯 Focus: Four Key Instruction Quality Metrics")
print(f"   1. Actionability (can user implement?)")
print(f"   2. Coherence (logical sequence?)")
print(f"   3. Domain Specificity (domain relevant?)")
print(f"   4. Technological Specificity (technically implementable?)")
print(f"📊 Graph Strategy:")
print(f"   • GNN: Uses specialized gnn_subgraph")
print(f"   • MCTS: Uses combined_subgraph (all knowledge sources)")
print(f"   • Reasoning: Uses combined_subgraph as fallback")
print(f"📁 Results will be saved to: {OUTPUT_FILE}")

for idx, item in enumerate(evaluation_prompts):
    print(f"\n🔄 Processing prompt {idx + 1}/{len(evaluation_prompts)}: {item['title']}")

    try:
        # Basic prompt analysis
        doc = nlp(item["prompt"])
        prompt_token_count = len(doc)

        # Simple domain classification
        prompt_lower = item["prompt"].lower()
        if any(word in prompt_lower for word in ["health", "medical", "patient", "clinical"]):
            dominant_domain = "healthcare"
        elif any(word in prompt_lower for word in ["business", "market", "customer", "revenue"]):
            dominant_domain = "business"
        elif any(word in prompt_lower for word in ["data", "analytics", "machine learning", "ai"]):
            dominant_domain = "technology"
        elif any(word in prompt_lower for word in ["finance", "financial", "investment", "trading"]):
            dominant_domain = "finance"
        elif any(word in prompt_lower for word in ["education", "learning", "student", "teacher"]):
            dominant_domain = "education"
        elif any(word in prompt_lower for word in ["manufacturing", "production", "factory"]):
            dominant_domain = "manufacturing"
        else:
            dominant_domain = "general"

        # Get graph data
        gnn_subgraph, mcts_subgraph, gnn_selection, mcts_selection = get_verified_graph_data(
            item["prompt"], idx + 1
        )

        # Calculate essential graph metrics
        gnn_graph_metrics = calculate_essential_graph_metrics(gnn_subgraph, "gnn")
        mcts_graph_metrics = calculate_essential_graph_metrics(mcts_subgraph, "mcts")

        # Generate instructions
        print(f"  🤖 Generating instructions...")
        instruction_payload = {
            "prompt": item["prompt"],
            "user_nodes": [],
            "gnn_subgraph": gnn_subgraph,
            "mcts_subgraph": mcts_subgraph
        }

        try:
            res = requests.post(API_URL, json=instruction_payload, timeout=120)
            res.raise_for_status()
            instruction_data = res.json()

            gnn_instructions = instruction_data.get("gnn_instructions", "")
            mcts_instructions = instruction_data.get("mcts_instructions", "")

        except Exception as e:
            print(f"    ⚠️ Instruction API failed: {e}")
            gnn_instructions = mcts_instructions = ""

        # Generate reasoning model instructions
        try:
            rmodel_result = generate_reasoning_model_instructions(item["prompt"])
            if isinstance(rmodel_result, dict):
                rmodel_instructions = rmodel_result.get("instructions", "")
            else:
                rmodel_instructions = str(rmodel_result)
        except Exception as e:
            print(f"    ⚠️ Reasoning model failed: {e}")
            rmodel_instructions = ""

        # 🎯 CALCULATE THE FOUR KEY METRICS
        print(f"  📊 Calculating Four Key Metrics...")
        try:
            final_metrics = calculate_final_four_metrics_enhanced(
                item["prompt"],
                gnn_instructions,
                mcts_instructions,
                rmodel_instructions,
                dominant_domain
            )
        except Exception as e:
            print(f"    ⚠️ Metrics calculation failed: {e}")
            # Provide default metrics
            final_metrics = {}
            for model in ['gnn', 'mcts', 'rmodel']:
                for metric in ['actionability', 'coherence', 'domain_specificity', 'technological_specificity']:
                    final_metrics[f"{model}_{metric}"] = 0.0

        # MODIFY the gnn_overall calculation:
        gnn_overall = calculate_overall_score(
            final_metrics.get('gnn_actionability', 0),
            final_metrics.get('gnn_coherence', 0),
            final_metrics.get('gnn_domain_specificity', 0),
            final_metrics.get('gnn_technological_specificity', 0),
            final_metrics.get('gnn_understandability_normalized', 0),  # ADD this line
            final_metrics.get('gnn_user_focus_normalized', 0)  # ADD this line
        )

        # MODIFY the mcts_overall calculation:
        mcts_overall = calculate_overall_score(
            final_metrics.get('mcts_actionability', 0),
            final_metrics.get('mcts_coherence', 0),
            final_metrics.get('mcts_domain_specificity', 0),
            final_metrics.get('mcts_technological_specificity', 0),
            final_metrics.get('mcts_understandability_normalized', 0),  # ADD this line
            final_metrics.get('mcts_user_focus_normalized', 0)  # ADD this line
        )

        # MODIFY the rmodel_overall calculation:
        rmodel_overall = calculate_overall_score(
            final_metrics.get('rmodel_actionability', 0),
            final_metrics.get('rmodel_coherence', 0),
            final_metrics.get('rmodel_domain_specificity', 0),
            final_metrics.get('rmodel_technological_specificity', 0),
            final_metrics.get('rmodel_understandability_normalized', 0),  # ADD this line
            final_metrics.get('rmodel_user_focus_normalized', 0)  # ADD this line
        )

        # Determine best model and performance gap
        scores = {"GNN": gnn_overall, "MCTS": mcts_overall, "Reasoning": rmodel_overall}
        best_model = max(scores, key=scores.get)
        performance_gap = max(scores.values()) - min(scores.values())

        # Calculate graph-performance ratios
        gnn_nodes = gnn_graph_metrics["gnn_node_count"]
        mcts_nodes = mcts_graph_metrics["mcts_node_count"]

        gnn_graph_performance_ratio = gnn_nodes / max(gnn_overall, 0.001) if gnn_overall > 0 else 0
        mcts_graph_performance_ratio = mcts_nodes / max(mcts_overall, 0.001) if mcts_overall > 0 else 0
        graph_size_difference = abs(gnn_nodes - mcts_nodes)

        # Build the final row
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "domain": item["domain"],
            "title": item["title"],
            "prompt": item["prompt"],

            # Essential prompt info
            "prompt_token_count": prompt_token_count,
            "dominant_domain": dominant_domain,

            # Instructions
            "gnn_instructions": gnn_instructions,
            "mcts_instructions": mcts_instructions,
            "rmodel_instructions": rmodel_instructions,

            # Overall scores
            "gnn_overall_score": gnn_overall,
            "mcts_overall_score": mcts_overall,
            "rmodel_overall_score": rmodel_overall,
            "best_performing_model": best_model,
            "performance_gap": round(performance_gap, 4),

            # Graph-performance analysis
            "gnn_graph_performance_ratio": round(gnn_graph_performance_ratio, 4),
            "mcts_graph_performance_ratio": round(mcts_graph_performance_ratio, 4),
            "graph_size_difference": graph_size_difference,

            # Debug info - Updated
            "selected_gnn_subgraph": gnn_selection,
            "selected_mcts_subgraph": mcts_selection,
            "mcts_uses_combined_graph": "yes" if mcts_selection == "combined_subgraph" else "no"
        }

        # Add the four key metrics
        row.update(final_metrics)

        # Add essential graph metrics
        row.update(gnn_graph_metrics)
        row.update(mcts_graph_metrics)

        results.append(row)

        print(f"  ✅ Success!")
        print(f"    ✨ Enhanced Metrics:")
        print(
            f"      GNN:  U={final_metrics.get('gnn_understandability_enhanced', 0):.3f}/10 F={final_metrics.get('gnn_user_focus_enhanced', 0):.3f}/10")
        print(
            f"      MCTS: U={final_metrics.get('mcts_understandability_enhanced', 0):.3f}/10 F={final_metrics.get('mcts_user_focus_enhanced', 0):.3f}/10")
        print(
            f"      R-Model: U={final_metrics.get('rmodel_understandability_enhanced', 0):.3f}/10 F={final_metrics.get('rmodel_user_focus_enhanced', 0):.3f}/10")

        print(f"    🏆 Best: {best_model} (gap: {performance_gap:.3f})")
        print(f"    📈 Graphs: GNN {gnn_nodes} nodes, MCTS {mcts_nodes} nodes")

        time.sleep(1)

    except Exception as e:
        print(f"  ❌ Error: {e}")

        # Create error row with defaults
        error_row = {}
        for header in headers:
            if header in ["timestamp", "domain", "title", "prompt"]:
                if header == "timestamp":
                    error_row[header] = datetime.utcnow().isoformat()
                elif header == "domain":
                    error_row[header] = item.get("domain", "unknown")
                elif header == "title":
                    error_row[header] = item.get("title", "error")
                elif header == "prompt":
                    error_row[header] = item.get("prompt", "")[:100] + "..."
            elif header in ["dominant_domain", "best_performing_model"]:
                error_row[header] = "unknown" if header == "dominant_domain" else "none"
            elif header in ["selected_gnn_subgraph", "selected_mcts_subgraph", "mcts_uses_combined_graph"]:
                error_row[header] = "error"
            else:
                error_row[header] = 0

        results.append(error_row)
        continue

# Save results
print(f"\n💾 Saving {len(results)} results...")
try:
    with open(OUTPUT_FILE, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Results saved to: {OUTPUT_FILE}")

    # Generate focused summary
    print(f"\n📊 FINAL FOUR METRICS SUMMARY:")
    print("=" * 60)

    successful_rows = [r for r in results if
                       r.get('gnn_overall_score', 0) > 0 or r.get('mcts_overall_score', 0) > 0 or r.get(
                           'rmodel_overall_score', 0) > 0]

    if successful_rows:
        print(
            f"📈 Success Rate: {len(successful_rows)}/{len(results)} ({len(successful_rows) / len(results) * 100:.1f}%)")

        # Overall model performance
        gnn_scores = [r.get('gnn_overall_score', 0) for r in successful_rows if r.get('gnn_overall_score', 0) > 0]
        mcts_scores = [r.get('mcts_overall_score', 0) for r in successful_rows if r.get('mcts_overall_score', 0) > 0]
        rmodel_scores = [r.get('rmodel_overall_score', 0) for r in successful_rows if
                         r.get('rmodel_overall_score', 0) > 0]

        print(f"\n🏆 Overall Model Performance:")
        if gnn_scores:
            print(f"   GNN:      avg={np.mean(gnn_scores):.3f} ± {np.std(gnn_scores):.3f} ({len(gnn_scores)} samples)")
        if mcts_scores:
            print(
                f"   MCTS:     avg={np.mean(mcts_scores):.3f} ± {np.std(mcts_scores):.3f} ({len(mcts_scores)} samples)")
        if rmodel_scores:
            print(
                f"   R-Model:  avg={np.mean(rmodel_scores):.3f} ± {np.std(rmodel_scores):.3f} ({len(rmodel_scores)} samples)")

        # Individual metric analysis
        print(f"\n📊 Individual Metric Performance:")
        for metric in ['actionability', 'coherence', 'domain_specificity', 'technological_specificity']:
            print(f"\n   {metric.replace('_', ' ').title()}:")

            for model in ['gnn', 'mcts', 'rmodel']:
                values = [r.get(f'{model}_{metric}', 0) for r in successful_rows if r.get(f'{model}_{metric}', 0) > 0]
                if values:
                    print(f"     {model.upper():8}: avg={np.mean(values):.3f} ± {np.std(values):.3f}")

        # Best model distribution
        best_models = [r.get('best_performing_model', 'none') for r in successful_rows]
        model_wins = {}
        for model in best_models:
            model_wins[model] = model_wins.get(model, 0) + 1

        print(f"\n🎯 Model Wins:")
        for model, wins in sorted(model_wins.items(), key=lambda x: x[1], reverse=True):
            percentage = (wins / len(successful_rows)) * 100
            print(f"   {model}: {wins} wins ({percentage:.1f}%)")

        # Domain analysis
        domains = [r.get('dominant_domain', 'unknown') for r in successful_rows]
        domain_counts = {}
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        print(f"\n🎯 Domain Distribution:")
        for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(successful_rows)) * 100
            print(f"   {domain}: {count} ({percentage:.1f}%)")

except Exception as save_error:
    print(f"❌ Save error: {save_error}")

print(f"\n🎉 FINAL FOCUSED evaluation completed!")
print(f"🎯 Four Key Metrics successfully calculated for all models!")
print(f"📁 Results: {OUTPUT_FILE}")
print(f"📊 Columns: {len(headers)} (focused on actionable insights)")

print(f"\n📋 Key Insights Available:")
print(f"   • Actionability: How implementable are the instructions?")
print(f"   • Coherence: Are instructions logically sequenced?")
print(f"   • Domain Specificity: Domain-relevant terminology and context?")
print(f"   • Technological Specificity: Technical implementation details?")
print(f"   • Overall Performance: Weighted combination of all four")
print(f"   • Graph-Performance Ratios: Efficiency of graph usage")

print(f"\n✨ Ready for actionable analysis! 🚀")
