from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, util
from neo4j_subgraph_agent import fetch_full_graph_from_neo4j
from llm_kg_enhancer import enhance_subgraph_with_llm
from graph_pruning import prune_graph_by_similarity
import logging
import pandas as pd
from collections import defaultdict
import uuid
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from knowledge_graph_gnn import KnowledgeGraphProcessor
import hashlib
from graph_mcts_agent import improved_gnn_mcts_agent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BERTmodel = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')

category_prototypes = {
    "Business": ["customer", "market", "revenue", "cost", "value proposition"],
    "System": ["architecture", "module", "process", "workflow", "interface"],
    "Data": ["database", "dataset", "schema", "data pipeline", "ETL"],
    "Technology": ["API", "TensorFlow", "cloud", "Docker", "GPU"]
}

category_embeddings = {
    category: BERTmodel.encode(examples, convert_to_tensor=True, show_progress_bar=False).mean(dim=0)
    for category, examples in category_prototypes.items()
}


def get_consistent_id_for_entity(entity_label, prefix="gnn"):
    """
    Generate a consistent ID for an entity based on its label.
    This ensures the same entity always gets the same ID.

    Args:
        entity_label: The label of the entity
        prefix: The prefix to use for the ID

    Returns:
        A consistent ID string for this entity
    """
    # Create a deterministic hash from the entity label
    hash_obj = hashlib.md5(entity_label.encode())
    hash_hex = hash_obj.hexdigest()

    # Create a consistent ID using the hash
    return f"{prefix}_{hash_hex[:8]}"

def classify_term(term: str) -> str:
    # Handle None or empty terms
    if term is None or term == "":
        return "other"  # Default category

    try:
        term_embedding = BERTmodel.encode(term, convert_to_tensor=True, show_progress_bar=False)
        similarities = {
            category: float(util.cos_sim(term_embedding, prototype_embedding))
            for category, prototype_embedding in category_embeddings.items()
        }
        return max(similarities, key=similarities.get)
    except Exception as e:
        print(f"Error in classify_term for term '{term}': {e}")
        return "other"  # Default category on error


def debug_user_graph_state(user_records, stage_name, debug_mode=True):
    """Debug function to track user graph changes at each stage"""
    if not debug_mode:
        return

    print(f"\n{'=' * 60}")
    print(f"USER GRAPH DEBUG - {stage_name}")
    print(f"{'=' * 60}")

    # Basic stats
    print(f"Number of user records: {len(user_records)}")

    # Extract all entities
    all_entities = set()
    origins = set()

    for record in user_records:
        if record:
            all_entities.add(record.get("subject_label", ""))
            all_entities.add(record.get("object_label", ""))
            origins.add(record.get("origin", "unknown"))

    # Remove empty strings
    all_entities.discard("")

    print(f"Unique entities: {len(all_entities)}")
    print(f"Origins found: {list(origins)}")

    # Show entities (first 10)
    entities_list = sorted(list(all_entities))
    print(f"First 10 entities: {entities_list[:10]}")

    # Check for suspicious entities (solution nodes, etc.)
    suspicious = []
    for entity in all_entities:
        if entity and isinstance(entity, str):
            entity_lower = entity.lower()
            if any(keyword in entity_lower for keyword in
                   ['solution', 'implement', 'system', 'technology', 'api', 'framework']):
                suspicious.append(entity)

    if suspicious:
        print(f"⚠️  SUSPICIOUS ENTITIES (might be from enrichment): {suspicious[:5]}")

    # Show sample records
    print(f"Sample records (first 3):")
    for i, record in enumerate(user_records[:3]):
        print(
            f"  {i + 1}: {record.get('subject_label', '')} --{record.get('predicate', '')}-> {record.get('object_label', '')} [origin: {record.get('origin', '')}]")

    print(f"{'=' * 60}\n")

def connect_user_domain_triples(user_records, domain_records, similarity_threshold=0.7):
    """
    Create connections between user and domain concepts based on semantic similarity.
    """
    # Use an efficient model for similarity calculations
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Extract unique entities from user triples
    user_entities = set()
    for record in user_records:
        user_entities.add(record["subject_label"])
        user_entities.add(record["object_label"])

    # Extract unique entities from domain triples
    domain_entities = set()
    for record in domain_records:
        domain_entities.add(record["subject_label"])
        domain_entities.add(record["object_label"])

    # Calculate embeddings in batches for efficiency
    user_texts = list(user_entities)
    domain_texts = list(domain_entities)

    if not user_texts or not domain_texts:
        return []

    # Encode all entities at once (more efficient)
    user_embeddings = model.encode(user_texts, convert_to_tensor=True, show_progress_bar=False)
    domain_embeddings = model.encode(domain_texts, convert_to_tensor=True, show_progress_bar=False)

    # Calculate cosine similarity between all pairs
    similarity_matrix = util.cos_sim(user_embeddings, domain_embeddings)

    # Extract connections above threshold
    connections = []
    for i, user_entity in enumerate(user_texts):
        for j, domain_entity in enumerate(domain_texts):
            similarity = similarity_matrix[i, j].item()

            if similarity >= similarity_threshold:
                connection = {
                    "subject_id": f"conn_{uuid.uuid4().hex[:8]}",
                    "subject_label": user_entity,
                    "predicate": "ns0__Is_similar_to",
                    "object_id": f"conn_{uuid.uuid4().hex[:8]}",
                    "object_label": domain_entity,
                    "origin": "connection",
                    "similarity": similarity
                }
                connections.append(connection)

    # Sort by similarity and take top matches
    connections.sort(key=lambda x: x["similarity"], reverse=True)
    top_connections = connections[:min(15, len(connections))]

    return top_connections


def apply_gnn_to_subgraph(all_records, seed_entities, debug_mode=False):
    """
    Apply Graph Neural Network to enhance the subgraph.
    Uses a lightweight approach without full DGL dependency.

    Args:
        all_records: List of all triple records
        seed_entities: List of important user entities to focus on
        debug_mode: Whether to print debug information

    Returns:
        List of new triple records generated by GNN analysis
    """
    try:
        # Create a networkx graph from records
        G = nx.DiGraph()

        # Add all nodes from records
        entities = set()
        for record in all_records:
            entities.add(record["subject_label"])
            entities.add(record["object_label"])

        # Map entities to indices for embedding
        entity_to_idx = {entity: i for i, entity in enumerate(entities)}
        idx_to_entity = {i: entity for entity, i in entity_to_idx.items()}

        # Add nodes to graph
        for entity in entities:
            G.add_node(entity)

        # Add edges to graph
        for record in all_records:
            G.add_edge(
                record["subject_label"],
                record["object_label"],
                label=record["predicate"]
            )

        if debug_mode:
            print(f"Created graph with {len(G.nodes)} nodes and {len(G.edges)} edges")

        # Create simple node feature vectors (using node degree as a feature)
        node_features = np.zeros((len(entities), 2))
        for i, entity in enumerate(entities):
            in_degree = G.in_degree(entity)
            out_degree = G.out_degree(entity)
            node_features[i] = [in_degree, out_degree]

        # Simple 1-hop message passing to get node embeddings
        embeddings = np.zeros((len(entities), 10))

        # Initialize with random values
        np.random.seed(42)
        embeddings = np.random.rand(len(entities), 10)

        # Update with 1-hop neighborhood info
        for _ in range(3):  # 3 rounds of message passing
            new_embeddings = np.copy(embeddings)
            for entity in entities:
                neighbors = list(G.neighbors(entity))
                if neighbors:
                    neighbor_indices = [entity_to_idx[n] for n in neighbors]
                    neighbor_embeds = embeddings[neighbor_indices]
                    new_embeddings[entity_to_idx[entity]] += np.mean(neighbor_embeds, axis=0)

            # Normalize
            norm = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
            embeddings = new_embeddings / (norm + 1e-10)  # Avoid division by zero

        # Find semantic community based on seed entities
        if seed_entities:
            seed_indices = [entity_to_idx[e] for e in seed_entities if e in entity_to_idx]
            if seed_indices:
                # Get seed entity embeddings
                seed_embeddings = embeddings[seed_indices]
                seed_mean = np.mean(seed_embeddings, axis=0)

                # Calculate similarity to seed entities
                similarities = cosine_similarity(embeddings, seed_mean.reshape(1, -1)).flatten()

                # Get top similar entities
                top_indices = np.argsort(similarities)[-20:]  # Top 20 entities
                community = [idx_to_entity[i] for i in top_indices]

                # Generate new triples from community
                new_records = []
                for i in range(len(community)):
                    for j in range(i + 1, len(community)):
                        if G.has_edge(community[i], community[j]):
                            # Find existing predicate
                            predicates = [data['label'] for _, _, data in G.edges(data=True)
                                          if (_, _) == (community[i], community[j])]
                            predicate = predicates[0] if predicates else "ns0__Is_related_to"
                        else:
                            predicate = "ns0__Is_related_to"

                        # Add in both directions
                        new_records.append({
                            "subject_id": get_consistent_id_for_entity(community[i], "gnn"),
                            "subject_label": community[i],
                            "predicate": predicate,
                            "object_id": get_consistent_id_for_entity(community[j], "gnn"),
                            "object_label": community[j],
                            "origin": "gnn"
                        })

                if debug_mode:
                    print(f"GNN generated {len(new_records)} new connections in community")

                return new_records

        return []
    except ImportError:
        if debug_mode:
            print("NetworkX not available, skipping GNN analysis")
        return []
    except Exception as e:
        if debug_mode:
            print(f"Error in GNN analysis: {e}")
            import traceback
            traceback.print_exc()
        return []


def subgraph_enricher(concepts: List[str], targets: List[str], relations: List[str], debug_mode=True):
    """
    Enrich a subgraph with knowledge graph data, LLM-generated triples, and GNN-based connections.

    Args:
        concepts: List of concepts from user input
        targets: List of targets from user input
        relations: List of relations from user input
        debug_mode: Whether to print debug information

    Returns:
        Dictionary with subgraph data
    """
    if debug_mode:
        logger.info(
            f"Starting subgraph enrichment with {len(concepts)} concepts, {len(targets)} targets, {len(relations)} relations")

    # Step 1: Create user triples from input
    user_triples = list(zip(concepts, relations, targets))
    user_records = []
    for s, p, o in user_triples:
        user_records.append({
            "subject_id": f"usr_{uuid.uuid4().hex[:8]}",
            "subject_label": s,
            "predicate": p,
            "object_id": f"usr_{uuid.uuid4().hex[:8]}",
            "object_label": o,
            "origin": "user"
        })

    debug_user_graph_state(user_records, "STEP 1 - INITIAL USER RECORDS", debug_mode)

    if debug_mode:
        logger.info(f"Created {len(user_records)} user records")

    # Step 2: Fetch domain knowledge from Neo4j
    try:
        knowledge_graph = fetch_full_graph_from_neo4j()

        if debug_mode:
            node_count = len(knowledge_graph.get('nodes', []))
            link_count = len(knowledge_graph.get('links', []))
            logger.info(f"Fetched knowledge graph with {node_count} nodes and {link_count} links")

            # Extract domain records from knowledge graph
        full_domain_records = []
        for link in knowledge_graph.get('links', []):
            source_node = next((n for n in knowledge_graph.get('nodes', []) if n['id'] == link['source']), {})
            target_node = next((n for n in knowledge_graph.get('nodes', []) if n['id'] == link['target']), {})

            full_domain_records.append({
                "subject_id": source_node.get('id', ''),
                "subject_label": source_node.get('label', ''),
                "predicate": link.get('label', ''),
                "object_id": target_node.get('id', ''),
                "object_label": target_node.get('label', ''),
                "origin": "domain"
            })

        if debug_mode:
            logger.info(f"Extracted {len(full_domain_records)} full domain records")
    except Exception as e:
        if debug_mode:
            logger.error(f"Error fetching knowledge graph: {e}")
            import traceback
            traceback.print_exc()
        full_domain_records = []
        knowledge_graph = {'nodes': [], 'links': []}

    # Step 3: Create domain knowledge threads for pruning
    try:
        # Convert to the format expected by pruning function
        graph_dict = knowledge_graph  # Already in the right format with 'nodes' and 'links'

        # Make sure all user entities are in the list of things to prune
        user_entities = set(
            [rec["subject_label"] for rec in user_records] + [rec["object_label"] for rec in user_records])

        # Create triples for pruning
        user_triples = [(rec["subject_label"], rec["predicate"], rec["object_label"]) for rec in user_records]

        # Prune the graph to get a relevant subgraph
        pruned_graph = prune_graph_by_similarity(
            graph_dict,
            user_triples=user_triples,
            similarity_threshold=0.85,
            similarity_method="cosine",
            max_hops=1
        )

        # Convert pruned graph to domain records
        domain_records = []
        for link in pruned_graph.get('links', []):
            source_id = link.get('source')
            target_id = link.get('target')

            # Find source and target nodes
            source_node = next((n for n in pruned_graph.get('nodes', []) if str(n.get('id')) == str(source_id)), None)
            target_node = next((n for n in pruned_graph.get('nodes', []) if str(n.get('id')) == str(target_id)), None)

            if source_node and target_node:
                domain_records.append({
                    "subject_id": source_node.get('id', ''),
                    "subject_label": source_node.get('label', ''),
                    "predicate": link.get('label', ''),
                    "object_id": target_node.get('id', ''),
                    "object_label": target_node.get('label', ''),
                    "origin": "domain"
                })

        if debug_mode:
            logger.info(f"Pruned graph has {len(domain_records)} triples")

    except Exception as e:
        if debug_mode:
            logger.error(f"Error pruning graph: {e}")
            import traceback
            traceback.print_exc()
        # Use a heavily limited subset as fallback
        if len(full_domain_records) > 30:
            domain_records = full_domain_records[:30]  # Just take the first 30 as fallback
        else:
            domain_records = full_domain_records

    # Step 4: Build node dictionary and label-to-ids mapping
    node_dict = {}
    label_to_ids = defaultdict(list)

    # Add domain nodes to dictionary
    for node in knowledge_graph.get('nodes', []):
        node_id = node.get('id')
        label = node.get('label')
        if node_id and label:
            node_dict[node_id] = label
            label_to_ids[label].append(node_id)

    # Add user nodes to dictionary
    for record in user_records:
        node_dict[record["subject_id"]] = record["subject_label"]
        node_dict[record["object_id"]] = record["object_label"]
        label_to_ids[record["subject_label"]].append(record["subject_id"])
        label_to_ids[record["object_label"]].append(record["object_id"])

    # Step 5: Find semantic connections between user and domain concepts
    user_entities = list(
        set([rec["subject_label"] for rec in user_records] + [rec["object_label"] for rec in user_records]))
    connection_records = connect_user_domain_triples(user_records, domain_records, similarity_threshold=0.55)

    if debug_mode:
        logger.info(f"Created {len(connection_records)} connection records")

    # Step 6: Enhance with LLM-generated triples (Evaluation)
    llm_records = enhance_subgraph_with_llm(
        pruned_graph,
        max_triples=15,
        debug_mode=debug_mode,
        node_dict=node_dict,
        label_to_ids=label_to_ids,
        user_entities=user_entities
    )

    if debug_mode:
        logger.info(f"Generated {len(llm_records)} LLM records")

    # Step 7: Combine all records
    all_records = user_records + domain_records + list(llm_records) + connection_records

    # Step 8: Apply GNN for intelligent subgraph generation and traversal
    # Only use GNN if we have enough data
    gnn_records = []
    if len(all_records) > 10:
        try:
            # Replace your current entity_types creation with this more robust version:
            entity_types = {}

            # Map entities to types based on domain knowledge
            for node in knowledge_graph.get('nodes', []):
                node_label = node.get('label', '')

                # Skip if label is None or empty
                if not node_label:
                    continue

                # Get domain_type either from node or classify it
                domain_type = None
                if node.get('domain_type'):
                    domain_type = node.get('domain_type').lower()
                else:
                    # Use classify_term to get domain_type
                    classified = classify_term(node_label)
                    if classified:
                        domain_type = classified.lower()
                    # Fallback to keyword-based classification
                    elif 'business' in node_label.lower():
                        domain_type = 'business'
                    elif 'system' in node_label.lower():
                        domain_type = 'system'
                    elif 'data' in node_label.lower():
                        domain_type = 'data'
                    elif 'technology' in node_label.lower() or 'tech' in node_label.lower():
                        domain_type = 'technology'

                # Only add to entity_types if we determined a domain_type
                if domain_type in ('business', 'system', 'data', 'technology'):
                    entity_types[node_label] = domain_type

            # Also classify user entities to ensure they have domain_types
            for entity in user_entities:
                if entity not in entity_types:
                    classified = classify_term(entity)
                    if classified:
                        entity_types[entity] = classified.lower()

            if debug_mode:
                logger.info(f"Created entity type mappings for {len(entity_types)} entities")
                if entity_types:
                    logger.info(f"Entity types sample: {list(entity_types.items())[:5]}")
                else:
                    logger.warning("No entity types were created!")

            # Initialize GNN processor
            gnn_processor = KnowledgeGraphProcessor()

            # Process with GNN
            gnn_result = gnn_processor.process(
                all_records,
                user_entities,
                entity_types=entity_types,
                max_hops=2,
                debug_mode=debug_mode
            )

            # Extract GNN-generated connections
            gnn_records = gnn_result.get('connections', [])

            if debug_mode:
                logger.info(f"Generated {len(gnn_records)} GNN-enhanced connections")

            # Add GNN records to the mix
            all_records.extend(gnn_records)
        except Exception as e:
            if debug_mode:
                logger.error(f"Error applying GNN: {e}")
                import traceback
                traceback.print_exc()

    # Step 9: Generate nodes and links from all records
    nodes = {}
    links = []

    for record in all_records:
        # Skip invalid records
        if not record or "subject_id" not in record or "object_id" not in record:
            continue

        subject_id = record.get("subject_id", "")
        object_id = record.get("object_id", "")
        subject_label = record.get("subject_label", "")
        object_label = record.get("object_label", "")
        origin = record.get("origin", "unknown")

        if not subject_id or not object_id:
            continue

        # Check if the subject or object labels look like predicates
        is_subject_predicate = isinstance(subject_label, str) and (
                subject_label.startswith("ns0__") or
                subject_label.startswith("rdfs__") or
                subject_label.startswith("owl__")
        )

        is_object_predicate = isinstance(object_label, str) and (
                object_label.startswith("ns0__") or
                object_label.startswith("rdfs__") or
                object_label.startswith("owl__")
        )

        # Add nodes if not already present and ensure labels are not None
        if subject_id not in nodes and not is_subject_predicate:
            # Ensure subject_label is a string and not empty
            if not subject_label or not isinstance(subject_label, str):
                subject_label = f"Entity_{subject_id[-8:]}"

            domain_type = classify_term(subject_label) if subject_label else "other"
            nodes[subject_id] = {
                "id": subject_id,
                "label": subject_label,
                "group": origin,
                "origin_label": origin,
                "domain_type": domain_type
            }

        if object_id not in nodes and not is_object_predicate:
            # Ensure object_label is a string and not empty
            if not object_label or not isinstance(object_label, str):
                object_label = f"Entity_{object_id[-8:]}"

            domain_type = classify_term(object_label) if object_label else "other"
            nodes[object_id] = {
                "id": object_id,
                "label": object_label,
                "group": origin,
                "origin_label": origin,
                "domain_type": domain_type
            }

        # Add link only if both nodes are valid
        if subject_id in nodes and object_id in nodes:
            links.append({
                "source": subject_id,
                "target": object_id,
                "label": record.get("predicate", ""),
                "origin": origin
            })

    # Step 10: Create subgraphs by origin

    def filter_by_origin(origin):
        """Filter nodes and links by origin"""
        # More flexible filtering for GNN nodes
        filtered_nodes = [n for n in nodes.values() if
                          n.get("group") == origin and
                          n.get("origin_label") == origin]

        # Get IDs of all filtered nodes
        node_ids = {n["id"] for n in filtered_nodes}

        # Filter links to only include those between filtered nodes
        filtered_links = [l for l in links if
                          l.get("origin") == origin and
                          l.get("source") in node_ids and
                          l.get("target") in node_ids]

        return filtered_nodes, filtered_links

    user_nodes, user_links = filter_by_origin("user")
    domain_nodes, domain_links = filter_by_origin("domain")
    llm_nodes, llm_links = filter_by_origin("llm")
    connection_nodes, connection_links = filter_by_origin("connection")
    gnn_nodes, gnn_links = filter_by_origin("gnn")

    # Create a combined GNN and connection subgraph for traversal visualization
    traversal_nodes = connection_nodes + gnn_nodes
    traversal_links = connection_links + gnn_links

    # Clean up the traversal subgraph - remove any nodes with predicate-like labels
    traversal_nodes = [
        n for n in traversal_nodes if
        isinstance(n.get("label"), str) and
        not (n["label"].startswith("ns0__") or
             n["label"].startswith("rdfs__") or
             n["label"].startswith("owl__"))
    ]

    # Make sure traversal links only reference valid nodes
    traversal_node_ids = {n["id"] for n in traversal_nodes}
    traversal_links = [
        l for l in traversal_links if
        l["source"] in traversal_node_ids and
        l["target"] in traversal_node_ids
    ]
    if debug_mode:
        logger.info(f"Final graph stats:")
        logger.info(f"- User subgraph: {len(user_nodes)} nodes, {len(user_links)} links")
        logger.info(f"- Domain subgraph: {len(domain_nodes)} nodes, {len(domain_links)} links")
        logger.info(f"- LLM subgraph: {len(llm_nodes)} nodes, {len(llm_links)} links")
        logger.info(f"- Connection subgraph: {len(connection_nodes)} nodes, {len(connection_links)} links")
        logger.info(f"- GNN subgraph: {len(gnn_nodes)} nodes, {len(gnn_links)} links")
        logger.info(f"- Traversal subgraph: {len(traversal_nodes)} nodes, {len(traversal_links)} links")
        logger.info(f"- Combined: {len(nodes)} nodes, {len(links)} links")

    debug_user_graph_state(user_nodes, "STEP Last - AFTER DOMAIN KNOWLEDGE", debug_mode)

    # Step 11: Calculate similarity metrics
    user_concepts = {r["subject_label"] for r in user_records} | {r["object_label"] for r in user_records}
    domain_concepts = {r["subject_label"] for r in domain_records} | {r["object_label"] for r in domain_records}
    matches = list(user_concepts & domain_concepts)
    similarity = len(matches) / max(len(user_concepts | domain_concepts), 1)

    # Step12: Apply MCTS for optimal reasoning path extraction
    if len(all_records) > 10:
        try:
            # Extract entity embeddings from GNN result
            entity_embeddings = gnn_result.get('embeddings', {})

            # Run MCTS to extract optimal reasoning path
            mcts_result = improved_gnn_mcts_agent(
                user_records=user_records,
                domain_records=domain_records,
                llm_records=llm_records,
                gnn_records=gnn_records,
                entity_embeddings=entity_embeddings,
                user_entities=user_entities,
                max_iterations=2000,
                debug_mode=debug_mode
            )

            if debug_mode:
                mcts_nodes = mcts_result.get('nodes', [])
                mcts_links = mcts_result.get('links', [])
                logger.info(f"[MCTS]: Generated MCTS reasoning path with {len(mcts_nodes)} nodes and {len(mcts_links)} "
                            f"links")
        except Exception as e:
            if debug_mode:
                logger.error(f"Error applying MCTS: {e}")
                import traceback
                traceback.print_exc()
            mcts_result = {"nodes": [], "links": []}
    else:
        mcts_result = {"nodes": [], "links": []}


    # Step 13: Return the final result
    return {
        "user_subgraph": {"nodes": user_nodes, "links": user_links},
        "domain_subgraph": {"nodes": domain_nodes, "links": domain_links},
        "llm_subgraph": {"nodes": llm_nodes, "links": llm_links},
        "connection_subgraph": {"nodes": connection_nodes, "links": connection_links},
        "gnn_subgraph": {"nodes": gnn_nodes, "links": gnn_links},
        "traversal_subgraph": {"nodes": traversal_nodes, "links": traversal_links},
        "combined_subgraph": {"nodes": list(nodes.values()), "links": links},
        "matches": matches,
        "similarity": similarity,
        "llm_records": list(llm_records),
        "domain_records": domain_records,
        "user_records": user_records,
        "connection_records": connection_records,
        "gnn_records": gnn_records,
        "mcts_reasoning_path": mcts_result
    }
