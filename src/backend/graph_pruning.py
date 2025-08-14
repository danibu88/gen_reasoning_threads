import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple, Dict

# Load model
BERTmodel = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')
ALLOWED_EDGE_LABELS = {"implements", "enables", "solves", "is_related_to"}


def clean_string(s):
    if not isinstance(s, str):
        s = str(s)
    return s.replace('\xa0', ' ').replace('\n', ' ').replace('\t', ' ').strip()


def dfs_traverse(graph, starting_ids: set, max_depth: int = 3) -> set:
    adjacency = {}
    for link in graph['links']:
        adjacency.setdefault(link['source'], []).append(link['target'])
        adjacency.setdefault(link['target'], []).append(link['source'])

    visited = set()
    for start in starting_ids:
        stack = [(start, 0)]
        while stack:
            node, depth = stack.pop()
            if node in visited or depth > max_depth:
                continue
            visited.add(node)
            for neighbor in adjacency.get(node, []):
                stack.append((neighbor, depth + 1))
    return visited


def prune_graph_by_similarity(
        graph: Dict,
        user_triples: List[Tuple[str, str, str]],
        similarity_threshold: float = 0.65,
        similarity_method: str = "cosine",
        max_hops: int = 2
) -> Dict:
    # Check if valid inputs are available
    if not user_triples:
        print(f"[Pruning] No user triples provided for pruning")
        return {"nodes": [], "links": []}  # Return empty result

    if not graph.get('nodes') or not graph.get('links'):
        print(
            f"[Pruning] Graph doesn't have nodes or links: nodes={len(graph.get('nodes', []))}, links={len(graph.get('links', []))}")
        return {"nodes": [], "links": []}  # Return empty result

    # Print some of the user triples for debugging
    print(f"[Pruning] User triples sample: {user_triples[:2]}")

    # Create triple embeddings from user input
    triples = [f"{clean_string(s)} {clean_string(p)} {clean_string(o)}" for s, p, o in user_triples]

    try:
        # Directly look for exact matches first
        user_entities = set()
        for s, p, o in user_triples:
            user_entities.add(clean_string(s))
            user_entities.add(clean_string(o))

        exact_matches = []
        for i, node in enumerate(graph['nodes']):
            if clean_string(node['label']) in user_entities:
                exact_matches.append(i)

        print(f"[Pruning] Found {len(exact_matches)} exact matches for user entities")

        # Compute embeddings
        user_embeds = BERTmodel.encode(triples, convert_to_tensor=True, show_progress_bar=False)

        # Create node embeddings
        node_texts = [clean_string(node['label']) for node in graph['nodes']]
        node_ids = [node['id'] for node in graph['nodes']]
        node_embeds = BERTmodel.encode(node_texts, convert_to_tensor=True, show_progress_bar=False)

        # Compute similarity scores
        if similarity_method == "dot":
            scores = util.dot_score(user_embeds, node_embeds).cpu().numpy()
        else:
            scores = util.cos_sim(user_embeds, node_embeds).cpu().numpy()

        # Get max similarity for each node
        max_scores = scores.max(axis=0)

        # Try multiple thresholds to ensure we get some results
        thresholds = [similarity_threshold, 0.6, 0.55, 0.5, 0.45, 0.4]
        above_thresh = np.array([], dtype=int)  # Initialize as empty integer array

        for threshold in thresholds:
            # Find nodes above threshold or top-K
            above_thresh = np.where(max_scores >= threshold)[0]

            if len(above_thresh) > 5:
                print(f"[Pruning] Found {len(above_thresh)} nodes above threshold {threshold}")
                break
            else:
                print(f"[Pruning] Only {len(above_thresh)} nodes above threshold {threshold}, trying lower threshold")

        # Convert indices to integers
        above_thresh = [int(idx) for idx in above_thresh]

        # Add exact matches to the relevant indices
        relevant_indices = list(set(above_thresh + exact_matches))

        # Also add top-K nodes regardless of threshold
        top_k_indices = np.argsort(-max_scores)[:30]
        top_k_indices = [int(idx) for idx in top_k_indices]  # Convert to integers

        relevant_indices = list(set(relevant_indices + top_k_indices))

        # If still no matches, use top 50 as fallback
        if len(relevant_indices) < 10:
            print(f"[Pruning] Less than 10 relevant nodes found, using top 50 as fallback")
            top_50_indices = np.argsort(-max_scores)[:50]
            relevant_indices = [int(idx) for idx in top_50_indices]  # Convert to integers

        # Get corresponding node IDs
        try:
            relevant_ids = set([node_ids[i] for i in relevant_indices])
        except Exception as e:
            print(f"[Pruning] Error getting node IDs: {e}, using fallback")
            # Fallback if indices are invalid
            relevant_ids = set([node_ids[min(i, len(node_ids) - 1)] for i in range(min(30, len(node_ids)))])

        # Log the highest scoring nodes for debugging
        top_5_indices = [int(idx) for idx in np.argsort(-max_scores)[:5]]  # Convert to integers
        print(f"[Pruning] Top 5 similar nodes:")
        for idx in top_5_indices:
            if 0 <= idx < len(node_texts):  # Check index bounds
                print(f"  - {node_texts[idx]} (score: {max_scores[idx]:.4f})")
            else:
                print(f"  - Index {idx} out of range for node_texts length {len(node_texts)}")

        # Traversal to find connected nodes
        print(f"[Pruning] Before Traversal: {len(relevant_ids)}")
        expanded_ids = dfs_traverse(graph, relevant_ids, max_depth=max_hops)
        print(f"[Pruning] After Traversal: {len(expanded_ids)}")

        # Debug link structure
        if graph['links']:
            link_sample = graph['links'][0]
            print(f"[Pruning] Link sample: {link_sample}")
            print(
                f"[Pruning] Link source type: {type(link_sample['source'])}, target type: {type(link_sample['target'])}")

            # Convert IDs to correct type for comparison
            if isinstance(link_sample['source'], str) and not all(isinstance(id, str) for id in expanded_ids):
                expanded_ids_str = {str(id) for id in expanded_ids}
                print(
                    f"[Pruning] Converting IDs to strings. Before: {len(expanded_ids)}, After: {len(expanded_ids_str)}")
                expanded_ids = expanded_ids_str

        # Filter nodes and links
        pruned_nodes = [node for node in graph['nodes'] if node['id'] in expanded_ids]

        # Handle both string and non-string IDs in links
        pruned_links = []
        for link in graph['links']:
            source = link['source']
            target = link['target']

            # Convert to string for comparison if needed
            source_str = str(source) if not isinstance(source, str) else source
            target_str = str(target) if not isinstance(target, str) else target

            if source_str in expanded_ids and target_str in expanded_ids:
                pruned_links.append(link)

        print(f"[Pruning] Kept {len(pruned_nodes)} nodes and {len(pruned_links)} links after pruning + traversal")

        # Final check - if we have very few or no links, select some nodes and their connections
        if len(pruned_links) < 3:
            print(f"[Pruning] Very few links found ({len(pruned_links)}), using alternative selection")

            # Take top 20 most similar nodes
            top_indices = [int(idx) for idx in np.argsort(-max_scores)[:20]]  # Convert to integers

            # Get IDs for these nodes
            top_node_ids = [node_ids[i] for i in top_indices if 0 <= i < len(node_ids)]

            # For each of these nodes, find all links where they appear
            fallback_links = []
            for link in graph['links']:
                source = link['source']
                target = link['target']

                # Convert to string for comparison if needed
                source_str = str(source) if not isinstance(source, str) else source
                target_str = str(target) if not isinstance(target, str) else target

                if source_str in top_node_ids or target_str in top_node_ids:
                    fallback_links.append(link)

            # Get all node IDs involved in these links
            fallback_node_ids = set()
            for link in fallback_links:
                fallback_node_ids.add(str(link['source']))
                fallback_node_ids.add(str(link['target']))

            # Get the corresponding nodes
            fallback_nodes = [node for node in graph['nodes'] if str(node['id']) in fallback_node_ids]

            print(f"[Pruning] Fallback selected {len(fallback_nodes)} nodes and {len(fallback_links)} links")
            return {"nodes": fallback_nodes, "links": fallback_links}

        return {"nodes": pruned_nodes, "links": pruned_links}

    except Exception as e:
        print(f"[Pruning] Error during graph pruning: {str(e)}")
        import traceback
        traceback.print_exc()

        # Provide a simple fallback that should never fail
        try:
            # Take random sample of 20 nodes and their connections
            sample_size = min(20, len(graph.get('nodes', [])))
            sample_nodes = graph.get('nodes', [])[:sample_size]
            sample_node_ids = {str(node['id']) for node in sample_nodes}

            sample_links = []
            for link in graph.get('links', []):
                source = str(link['source']) if not isinstance(link['source'], str) else link['source']
                target = str(link['target']) if not isinstance(link['target'], str) else link['target']

                if source in sample_node_ids and target in sample_node_ids:
                    sample_links.append(link)

            return {"nodes": sample_nodes, "links": sample_links}
        except:
            # Ultimate fallback
            return {"nodes": [], "links": []}
