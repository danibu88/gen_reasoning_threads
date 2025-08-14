import requests
import json
import time
from datetime import datetime


def debug_single_prompt_full_pipeline(prompt):
    """
    Debug a single prompt through the entire pipeline to understand where graph data is lost
    """
    print("🔍 COMPREHENSIVE GRAPH DATA DEBUG")
    print("=" * 80)
    print(f"Prompt: {prompt[:100]}...")
    print("=" * 80)

    GRAPH_API_URL = "http://localhost:3002/result/"
    JOB_STATUS_URL = "http://localhost:3002/job_status/"
    INSTRUCTION_API_URL = "http://localhost:3002/generate_instructions/"

    # Step 1: Create job and get raw response
    payload = {
        "text_input": prompt,
        "approach": "Machine Learning",
        "classifier": "Classification_nonlinear",
        "systemOrientation": "Centralized"
    }

    try:
        print("\n1️⃣ CREATING JOB...")
        print(f"   Payload: {json.dumps(payload, indent=2)}")

        response = requests.post(GRAPH_API_URL, json=payload, timeout=120)
        print(f"   Response Status: {response.status_code}")
        print(f"   Response Headers: {dict(response.headers)}")

        response.raise_for_status()
        job_data = response.json()
        print(f"   Job Response: {json.dumps(job_data, indent=2)}")

        job_id = job_data.get('job_id')
        if not job_id:
            print("   ❌ No job ID returned!")
            return None

        print(f"   ✅ Job created: {job_id}")

        # Step 2: Poll and capture ALL status updates
        print(f"\n2️⃣ POLLING JOB STATUS...")

        status_history = []
        max_attempts = 60
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            time.sleep(5)  # Check every 5 seconds

            try:
                status_response = requests.get(f"{JOB_STATUS_URL}{job_id}", timeout=30)
                status_response.raise_for_status()
                status_data = status_response.json()

                status = status_data.get('status', 'unknown')
                progress = status_data.get('progress', 0)

                # Store each status update
                status_update = {
                    'attempt': attempt,
                    'timestamp': datetime.now().isoformat(),
                    'status': status,
                    'progress': progress,
                    'has_subgraphData': 'subgraphData' in status_data,
                    'response_keys': list(status_data.keys())
                }
                status_history.append(status_update)

                print(f"   Attempt {attempt:2d}: {status} ({progress}%) - Keys: {status_update['response_keys']}")

                if status == 'complete':
                    print(f"\n3️⃣ JOB COMPLETED! ANALYZING RESPONSE...")

                    # Save complete final response
                    with open('./evals_results/debug_final_response.json', 'w') as f:
                        json.dump(status_data, f, indent=2)
                    print(f"   💾 Complete response saved to: ./evals_results/debug_final_response.json")

                    # Analyze subgraphData in detail
                    subgraph_data = status_data.get('subgraphData', {})
                    if not subgraph_data:
                        print("   ❌ NO SUBGRAPH DATA FOUND!")
                        print(f"   Available keys: {list(status_data.keys())}")

                        # Check if data is in different location
                        for key, value in status_data.items():
                            if isinstance(value, dict) and ('nodes' in str(value) or 'links' in str(value)):
                                print(f"   🔍 Found potential graph data in key '{key}': {type(value)}")
                        return None

                    print(f"\n4️⃣ SUBGRAPH DATA ANALYSIS:")
                    print(f"   Type: {type(subgraph_data)}")
                    print(f"   Keys: {list(subgraph_data.keys()) if isinstance(subgraph_data, dict) else 'Not a dict'}")

                    # Analyze each subgraph
                    graph_summary = {}
                    for key, value in subgraph_data.items():
                        if isinstance(value, dict):
                            nodes = value.get('nodes', [])
                            links = value.get('links', [])

                            graph_summary[key] = {
                                'nodes': len(nodes),
                                'links': len(links),
                                'has_data': len(nodes) > 0 or len(links) > 0,
                                'sample_node': nodes[0] if nodes else None,
                                'sample_link': links[0] if links else None
                            }

                            print(f"\n   🔸 {key}:")
                            print(f"      Nodes: {len(nodes)}")
                            print(f"      Links: {len(links)}")
                            if nodes:
                                print(f"      Sample node: {json.dumps(nodes[0], indent=8)}")
                            if links:
                                print(f"      Sample link: {json.dumps(links[0], indent=8)}")

                    # Save graph summary
                    with open('./evals_results/debug_graph_summary.json', 'w') as f:
                        json.dump(graph_summary, f, indent=2)

                    # Step 3: Test instruction generation with actual graph data
                    print(f"\n5️⃣ TESTING INSTRUCTION GENERATION...")

                    # Extract best graphs for each model
                    best_graphs = find_best_graphs_for_models(subgraph_data)
                    print(f"   Best graphs selected: {json.dumps(best_graphs, indent=2)}")

                    # Test instruction generation
                    instruction_payload = {
                        "prompt": prompt,
                        "user_nodes": [],
                        "gnn_subgraph": subgraph_data.get(best_graphs['gnn'], {"nodes": [], "links": []}),
                        "mcts_subgraph": subgraph_data.get(best_graphs['mcts'], {"nodes": [], "links": []})
                    }

                    print(f"   Instruction payload graph sizes:")
                    print(f"      GNN: {len(instruction_payload['gnn_subgraph'].get('nodes', []))} nodes")
                    print(f"      MCTS: {len(instruction_payload['mcts_subgraph'].get('nodes', []))} nodes")

                    try:
                        inst_response = requests.post(INSTRUCTION_API_URL, json=instruction_payload, timeout=60)
                        inst_response.raise_for_status()
                        instruction_data = inst_response.json()

                        print(f"   ✅ Instructions generated successfully!")
                        print(f"      GNN instructions: {len(instruction_data.get('gnn_instructions', ''))} chars")
                        print(f"      MCTS instructions: {len(instruction_data.get('mcts_instructions', ''))} chars")

                        # Save instruction response
                        with open('./evals_results/debug_instruction_response.json', 'w') as f:
                            json.dump(instruction_data, f, indent=2)

                    except Exception as inst_error:
                        print(f"   ❌ Instruction generation failed: {inst_error}")

                    return {
                        'subgraph_data': subgraph_data,
                        'graph_summary': graph_summary,
                        'best_graphs': best_graphs,
                        'status_history': status_history
                    }

                elif status == 'error':
                    print(f"   ❌ Job failed: {status_data.get('error', 'Unknown error')}")
                    return None

            except Exception as e:
                print(f"   ⚠️ Error checking status: {e}")

        print(f"   ⏰ Job timed out after {max_attempts} attempts")

        # Save status history for analysis
        with open('./evals_results/debug_status_history.json', 'w') as f:
            json.dump(status_history, f, indent=2)

        return None

    except Exception as e:
        print(f"❌ Error in debug pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_best_graphs_for_models(subgraph_data):
    """
    Find the best subgraph for each model based on actual data content
    """
    if not subgraph_data:
        return {'gnn': None, 'mcts': None, 'reasoning': None}

    # Score each subgraph based on content
    graph_scores = {}

    for key, value in subgraph_data.items():
        if isinstance(value, dict):
            nodes = value.get('nodes', [])
            links = value.get('links', [])

            node_count = len(nodes)
            link_count = len(links)

            # Calculate basic score
            content_score = node_count + link_count

            # Bonus for specific graph types
            type_bonus = 0
            if 'gnn' in key.lower():
                type_bonus = 100
            elif 'mcts' in key.lower() or 'reasoning' in key.lower():
                type_bonus = 90
            elif 'combined' in key.lower():
                type_bonus = 80
            elif 'llm' in key.lower():
                type_bonus = 70
            elif 'domain' in key.lower():
                type_bonus = 60

            total_score = content_score + type_bonus
            graph_scores[key] = {
                'score': total_score,
                'nodes': node_count,
                'links': link_count,
                'content_score': content_score,
                'type_bonus': type_bonus
            }

    # Sort by score
    sorted_graphs = sorted(graph_scores.items(), key=lambda x: x[1]['score'], reverse=True)

    # Assign best graphs to models
    best_graphs = {'gnn': None, 'mcts': None, 'reasoning': None}

    # Try to assign specific graphs first
    for key, score_data in sorted_graphs:
        if 'gnn' in key.lower() and not best_graphs['gnn']:
            best_graphs['gnn'] = key
        elif ('mcts' in key.lower() or 'reasoning' in key.lower()) and not best_graphs['mcts']:
            best_graphs['mcts'] = key

    # Fill remaining with best available
    for key, score_data in sorted_graphs:
        if score_data['content_score'] > 0:  # Has actual content
            if not best_graphs['gnn']:
                best_graphs['gnn'] = key
            elif not best_graphs['mcts']:
                best_graphs['mcts'] = key
            elif not best_graphs['reasoning']:
                best_graphs['reasoning'] = key

    # Use best graph for all if no specific assignments
    if sorted_graphs and sorted_graphs[0][1]['content_score'] > 0:
        best_key = sorted_graphs[0][0]
        for model in ['gnn', 'mcts', 'reasoning']:
            if not best_graphs[model]:
                best_graphs[model] = best_key

    return best_graphs


def test_direct_backend_call():
    """
    Test calling the backend directly to see what's returned
    """
    print("\n🧪 TESTING DIRECT BACKEND CALL")
    print("=" * 50)

    # Import the function directly
    try:
        from subgraph_enricher import subgraph_enricher

        # Test with simple data
        concepts = ["healthcare", "AI", "system", "patient", "data"]
        targets = ["management", "analysis", "recommendations", "monitoring", "alerts"]
        relations = ["uses", "analyzes", "provides", "enables", "sends"]

        print(f"Calling subgraph_enricher directly...")
        print(f"Concepts: {concepts}")
        print(f"Targets: {targets}")
        print(f"Relations: {relations}")

        result = subgraph_enricher(concepts, targets, relations, debug_mode=True)

        print(f"\n📊 Direct call results:")
        for key, value in result.items():
            if isinstance(value, dict) and ('nodes' in value or 'links' in value):
                nodes = len(value.get('nodes', []))
                links = len(value.get('links', []))
                print(f"   {key}: {nodes} nodes, {links} links")
            elif isinstance(value, list):
                print(f"   {key}: {len(value)} items")
            else:
                print(f"   {key}: {type(value)}")

        # Save result
        with open('./evals_results/debug_direct_call_result.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)

        return result

    except ImportError as e:
        print(f"   ❌ Cannot import subgraph_enricher: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Error in direct call: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test with the healthcare prompt from your results
    test_prompt = "At SimpleHealth, we've identified a groundbreaking opportunity to revolutionize elderly care with our AI-powered MedBox concept. This device reminds seniors to take their medications and uses AI to analyze health data and make informed suggestions about treatment plans."

    print("🚀 Starting comprehensive graph data debugging...")

    # Test 1: Full pipeline debug
    pipeline_result = debug_single_prompt_full_pipeline(test_prompt)

    # Test 2: Direct backend call
    direct_result = test_direct_backend_call()

    # Summary
    print(f"\n📋 DEBUG SUMMARY")
    print("=" * 30)
    print(f"Pipeline result: {'✅ Success' if pipeline_result else '❌ Failed'}")
    print(f"Direct call result: {'✅ Success' if direct_result else '❌ Failed'}")

    if pipeline_result:
        graph_summary = pipeline_result.get('graph_summary', {})
        non_empty_graphs = [k for k, v in graph_summary.items() if v.get('has_data', False)]
        print(f"Non-empty graphs found: {len(non_empty_graphs)}")
        print(f"Graph names with data: {non_empty_graphs}")

    print(f"\n📁 Debug files saved:")
    print(f"   • ./evals_results/debug_final_response.json")
    print(f"   • ./evals_results/debug_graph_summary.json")
    print(f"   • ./evals_results/debug_instruction_response.json")
    print(f"   • ./evals_results/debug_status_history.json")
    print(f"   • ./evals_results/debug_direct_call_result.json")

    print(f"\n✅ Debug complete! Check the files above for detailed analysis.")