"""
Knowledge Graph GNN Module
--------------------------
A graph neural network implementation for knowledge graph enrichment, 
designed to find semantic connections between entities across different
abstraction levels (Business → System → Data → Technology).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GATConv
import uuid
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import os
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("knowledge_graph_gnn")

# Check for CUDA and set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")


class KnowledgeGraphGNN(nn.Module):
    """
    GNN model for knowledge graph embedding with multi-hop message passing.
    Uses GraphConv layers for initial representation, followed by GAT layers
    for attention-based message passing.
    """

    def __init__(self, in_feats, hidden_size=128, n_layers=3, dropout=0.2):
        super(KnowledgeGraphGNN, self).__init__()
        self.layers = nn.ModuleList()

        # Initial convolutional layer with allow_zero_in_degree=True
        self.layers.append(GraphConv(in_feats, hidden_size, activation=F.relu, allow_zero_in_degree=True))

        # Middle layers with GAT for attention
        for _ in range(n_layers - 2):
            self.layers.append(GATConv(hidden_size, hidden_size, num_heads=4,
                                       feat_drop=dropout, attn_drop=dropout, allow_zero_in_degree=True))

        # Final convolutional layer with allow_zero_in_degree=True
        self.layers.append(GraphConv(hidden_size, hidden_size, allow_zero_in_degree=True))

        # Projection layer for final embedding
        self.projection = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features

        # Apply layers with residual connections
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GATConv):
                # For GAT layers, handle the multi-head output
                h_new = layer(g, h)
                h_new = h_new.mean(dim=1)  # Average across attention heads
            else:
                h_new = layer(g, h)

            # Apply residual connection except for first layer
            if i > 0:
                h = h_new + h
            else:
                h = h_new

            h = self.dropout(h)

        # Final projection
        h = self.projection(h)
        return h

    def get_embeddings(self, g, features):
        """Get normalized embeddings for all nodes"""
        with torch.no_grad():
            h = self.forward(g, features)
            # L2 normalize embeddings
            h_norm = F.normalize(h, p=2, dim=1)
            return h_norm


class KnowledgeGraphProcessor:
    """
    Processes knowledge graph triples, builds a GNN, and generates enriched connections.
    """

    def __init__(self, model_path=None, embedding_dim=384, hidden_dim=128, use_pretrained=True):
        """
        Initialize the knowledge graph processor.

        Args:
            model_path: Path to saved model weights, if available
            embedding_dim: Dimension of initial node embeddings
            hidden_dim: Hidden dimension in GNN
            use_pretrained: Whether to use pretrained sentence embeddings
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.model_path = model_path
        self.use_pretrained = use_pretrained

        # Load sentence transformer for initial node features
        self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # Initialize GNN model (will be created when needed)
        self.model = None

        # Track entity information
        self.entity_to_idx = {}
        self.idx_to_entity = {}
        self.relation_types = set()

    # Fix for prepare_graph_data method to handle None values in entity list

    def prepare_graph_data(self, triple_records):
        """
        Convert triple records to a DGL graph with features.

        Args:
            triple_records: List of knowledge graph triples

        Returns:
            g: DGL graph
            features: Node features tensor
            entity_list: List of all entities
        """
        # Extract all unique entities and relations
        entities = set()
        for record in triple_records:
            subject_label = record.get("subject_label", "")
            object_label = record.get("object_label", "")

            # Only add non-None, non-empty entities
            if subject_label is not None and subject_label != "":
                entities.add(subject_label)
            if object_label is not None and object_label != "":
                entities.add(object_label)

            # Add predicate to relation types if it exists
            predicate = record.get("predicate", "")
            if predicate:
                self.relation_types.add(predicate)

        # Convert set to list and filter out any remaining None values as a safety measure
        entity_list = [e for e in entities if e is not None]

        if not entity_list:
            logger.warning("No valid entities found in triple records")
            return None, None, []

        # Create entity to index mapping
        self.entity_to_idx = {e: i for i, e in enumerate(entity_list)}
        self.idx_to_entity = {i: e for i, e in enumerate(entity_list)}

        # Create node features using sentence transformer
        logger.info(f"Generating embeddings for {len(entity_list)} entities")

        try:
            # Add debug logging to check for issues
            if len(entity_list) < 10:
                logger.info(f"Entity list sample: {entity_list}")
            else:
                logger.info(f"Entity list sample (first 10): {entity_list[:10]}")

            # Ensure all entities are strings
            entity_list_str = [str(e) for e in entity_list]
            node_features = self.sentence_model.encode(entity_list_str, convert_to_tensor=True, show_progress_bar=False)

            # Convert to torch tensor
            node_features = node_features.to(DEVICE)
        except Exception as e:
            logger.error(f"Error encoding entities: {e}")
            # If there's an issue with encoding, try to identify problematic entities
            for i, entity in enumerate(entity_list):
                try:
                    # Try to encode each entity individually to find the problematic one
                    self.sentence_model.encode([str(entity)], convert_to_tensor=True, show_progress_bar=False)
                except Exception as inner_e:
                    logger.error(f"Problem with entity {i}: '{entity}' - {type(entity)} - Error: {inner_e}")

            # Return empty data to avoid crashing
            return None, None, []

        # Create graph edges
        src_nodes = []
        dst_nodes = []

        # Track successfully added edges for debugging
        added_edges = 0
        skipped_edges = 0

        for record in triple_records:
            subject_label = record.get("subject_label", "")
            object_label = record.get("object_label", "")

            if (subject_label in self.entity_to_idx and
                    object_label in self.entity_to_idx):
                src_nodes.append(self.entity_to_idx[subject_label])
                dst_nodes.append(self.entity_to_idx[object_label])
                added_edges += 1
            else:
                skipped_edges += 1

        logger.info(f"Added {added_edges} edges to graph, skipped {skipped_edges} edges due to missing entities")

        if len(src_nodes) == 0:
            logger.warning("No edges could be added to the graph. Creating dummy edges to enable GNN processing.")
            # Create dummy self-loops if no edges could be added
            for i in range(len(entity_list)):
                src_nodes.append(i)
                dst_nodes.append(i)

        try:
            # Create DGL graph
            g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)), num_nodes=len(entity_list))

            # Add self-loops to handle zero in-degree nodes
            g = dgl.add_self_loop(g)
            logger.info(f"Added self-loops, now graph has {g.number_of_edges()} edges")

            g = g.to(DEVICE)
            g.ndata['feat'] = node_features

            return g, node_features, entity_list
        except Exception as e:
            logger.error(f"Error creating graph: {e}")
            return None, None, []

    def get_consistent_id(self, entity_label, entity_type="entity"):
        """
        Generate a consistent ID for an entity based on its label.
        This ensures the same entity always gets the same ID.

        Args:
            entity_label: The label of the entity
            entity_type: The type of entity (helps create more descriptive IDs)

        Returns:
            A consistent ID string for this entity
        """
        # Create a deterministic hash from the entity label
        import hashlib
        # Use MD5 for a relatively short, fixed-length hash
        hash_obj = hashlib.md5(entity_label.encode())
        hash_hex = hash_obj.hexdigest()

        # Create a consistent ID using the hash
        return f"gnn_{hash_hex[:8]}"

    def initialize_model(self, input_dim):
        """Initialize the GNN model"""
        self.model = KnowledgeGraphGNN(input_dim, self.hidden_dim).to(DEVICE)

        # Load pretrained weights if available
        if self.model_path and os.path.exists(self.model_path):
            try:
                logger.info(f"Loading model from {self.model_path}")
                self.model.load_state_dict(torch.load(self.model_path))
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

    def get_embeddings(self, g, features):
        """Get node embeddings from the GNN"""
        if self.model is None:
            self.initialize_model(features.shape[1])

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.get_embeddings(g, features)
            return embeddings

    def find_semantic_connections(self, embeddings, entity_list, user_entities,
                                  similarity_threshold=0.6, max_connections=20, max_hops=2):
        """
        Find semantically connected entities based on GNN embeddings with deeper traversal.
        Fixed to handle None values for object_label.
        """
        # Get indices of user entities that exist in the graph
        user_indices = [self.entity_to_idx[e] for e in user_entities if e in self.entity_to_idx]

        if not user_indices:
            logger.warning("No user entities found in the graph. Using backup method to find user indices.")

            # Try to find closest matches using string similarity
            for user_entity in user_entities:
                best_match = None
                best_score = 0

                for graph_entity in entity_list:
                    # Simple string similarity - count common words
                    user_words = set(user_entity.lower().split())
                    graph_words = set(graph_entity.lower().split())
                    common_words = len(user_words.intersection(graph_words))

                    if common_words > 0 and common_words > best_score:
                        best_score = common_words
                        best_match = graph_entity

                if best_match:
                    idx = self.entity_to_idx[best_match]
                    if idx not in user_indices:
                        user_indices.append(idx)
                        logger.info(f"Added approximate match: {user_entity} → {best_match}")

        # If still no matches, use top connected entities as seeds
        if not user_indices and len(entity_list) > 0:
            logger.warning("Using fallback - selecting random nodes as seeds")
            # Just use the first few entities in the graph
            user_indices = list(range(min(3, len(entity_list))))

        # If we have no valid indices or entity list is empty, return empty list
        if not user_indices or not entity_list:
            return []

        # Get embeddings for user entities
        user_embeddings = embeddings[user_indices]

        # Compute similarity between user entities and all others
        all_similarities = torch.mm(user_embeddings, embeddings.t())

        # Find connections for each user entity
        connections = []
        visited_pairs = set()  # To avoid duplicate connections

        # First-hop connections (direct connections)
        for i, user_idx in enumerate(user_indices):
            user_entity = entity_list[user_idx]
            similarities = all_similarities[i]

            # Get top similar entities for first hop
            values, indices = torch.topk(similarities, min(50, len(similarities)))

            for j, idx in enumerate(indices.tolist()):
                if idx == user_idx:
                    continue  # Skip self

                similar_entity = entity_list[idx]
                similarity = values[j].item()

                pair_key = f"{user_entity}|{similar_entity}"
                if pair_key in visited_pairs:
                    continue

                visited_pairs.add(pair_key)

                if similarity >= similarity_threshold:
                    # Determine relation type based on similarity
                    if similarity > 0.8:
                        predicate = "ns0__Is_strongly_related_to"
                    else:
                        predicate = "ns0__Is_semantically_related_to"

                    connection = {
                        "subject_id": self.get_consistent_id(user_entity, "subject"),
                        "subject_label": user_entity,
                        "predicate": predicate,
                        "object_id": self.get_consistent_id(similar_entity, "object"),
                        "object_label": similar_entity,
                        "origin": "gnn",
                        "similarity": similarity
                    }
                    connections.append(connection)

        # If max_hops >= 2, add multi-hop connections
        if max_hops >= 2 and connections:
            # Extract first-hop target entities
            first_hop_targets = [conn["object_label"] for conn in connections]

            # Find indices in entity list
            first_hop_indices = [self.entity_to_idx[e] for e in first_hop_targets if e in self.entity_to_idx]

            if first_hop_indices:
                # Get embeddings for first-hop targets
                first_hop_embeddings = embeddings[first_hop_indices]

                # Compute similarities with all entities
                hop2_similarities = torch.mm(first_hop_embeddings, embeddings.t())

                # For each first-hop target, find its most similar entities
                for i, hop1_idx in enumerate(first_hop_indices):
                    if i >= len(first_hop_targets):
                        continue  # Safety check

                    hop1_entity = first_hop_targets[i]
                    similarities = hop2_similarities[i]

                    # Get top similar entities
                    values, indices = torch.topk(similarities, min(10, len(similarities)))

                    for j, idx in enumerate(indices.tolist()):
                        # Skip self and entities already connected to user
                        if idx == hop1_idx or idx in user_indices:
                            continue

                        hop2_entity = entity_list[idx]
                        similarity = values[j].item()

                        # Skip already visited pairs
                        pair_key = f"{hop1_entity}|{hop2_entity}"
                        if pair_key in visited_pairs:
                            continue

                        visited_pairs.add(pair_key)

                        # Use an even lower threshold for second-hop connections
                        if similarity >= (similarity_threshold - 0.2):
                            predicate = "ns0__Is_indirectly_related_to"

                            connection = {
                                "subject_id": self.get_consistent_id(user_entity, "subject"),
                                "subject_label": user_entity,
                                "predicate": predicate,
                                "object_id": self.get_consistent_id(similar_entity, "object"),
                                "object_label": similar_entity,
                                "origin": "gnn",
                                "similarity": similarity
                            }
                            connections.append(connection)

        # After finding connections, filter to prioritize business/system/data/technology nodes
        tech_connections = []
        other_connections = []

        for connection in connections:
            # Check if the object entity contains tech-related terms
            # FIX: Handle None values for object_label
            object_label = connection.get("object_label", "")
            if object_label is None:
                object_label = ""

            object_label_lower = object_label.lower()

            if any(term in object_label_lower for term in
                   ["tech", "technology", "api", "system", "data", "algorithm",
                    "cloud", "device", "model", "ai", "learning", "neural",
                    "business", "service", "workflow"]):
                tech_connections.append(connection)
            else:
                other_connections.append(connection)

        # Prioritize technology connections, but keep some others
        prioritized_connections = tech_connections + other_connections
        max_to_return = min(max_connections, len(prioritized_connections))

        # If we have no connections, create some basic ones
        if not prioritized_connections and len(entity_list) > 3:
            logger.warning("No connections found, creating basic connections")
            # Take a few entities and connect them
            for i in range(min(3, len(user_indices))):
                user_idx = user_indices[i]
                user_entity = entity_list[user_idx]

                # Connect to a few other entities
                for j in range(3):
                    target_idx = (user_idx + j + 1) % len(entity_list)
                    target_entity = entity_list[target_idx]

                    connection = {
                        "subject_id": self.get_consistent_id(user_entity, "subject"),
                        "subject_label": user_entity,
                        "predicate": "ns0__Is_related_to",
                        "object_id": self.get_consistent_id(target_entity, "object"),
                        "object_label": target_entity,
                        "origin": "gnn",
                        "similarity": 0.7
                    }
                    prioritized_connections.append(connection)

            max_to_return = len(prioritized_connections)

        return prioritized_connections[:max_to_return]

    def find_cross_level_connections(self, embeddings, entity_list, entity_types, user_entities,
                                     similarity_threshold=0.5):
        """
        Find connections across different levels of abstraction
        (Business → System → Data → Technology)

        Args:
            embeddings: Node embeddings from GNN
            entity_list: List of all entities
            entity_types: Dictionary mapping entities to their types
                        (business, system, data, technology)
            user_entities: List of user-specified entities to focus on
            similarity_threshold: Threshold for including a connection

        Returns:
            List of new triple records for cross-level connections
        """
        if not entity_types:
            return []

        # Group entities by type
        entities_by_type = defaultdict(list)
        for i, entity in enumerate(entity_list):
            entity_type = entity_types.get(entity, 'unknown')
            entities_by_type[entity_type].append((i, entity))

        # Define hierarchy levels for traversal
        hierarchy = ['business', 'system', 'data', 'technology']

        # Find connections that span across the hierarchy
        connections = []

        user_indices = [self.entity_to_idx[e] for e in user_entities if e in self.entity_to_idx]

        # Add this fallback if user_indices is empty:
        if not user_indices and entity_list:
            logger.warning("No user entities found in the graph. Using backup method to find user indices.")

            # Try to find closest matches using string similarity
            for user_entity in user_entities:
                best_match = None
                best_score = 0

                for graph_entity in entity_list:
                    # Simple string similarity - count common words
                    user_words = set(user_entity.lower().split())
                    graph_words = set(graph_entity.lower().split())
                    common_words = len(user_words.intersection(graph_words))

                    if common_words > 0 and common_words > best_score:
                        best_score = common_words
                        best_match = graph_entity

                if best_match:
                    idx = self.entity_to_idx[best_match]
                    if idx not in user_indices:
                        user_indices.append(idx)
                        logger.info(f"Added approximate match: {user_entity} → {best_match}")

            # If still no matches, use top connected entities as seeds
            if not user_indices:
                logger.warning("Using fallback - selecting most connected nodes as seeds")
                # Calculate node degrees
                degrees = []
                for i, entity in enumerate(entity_list):
                    in_degree = self.sentence_model.encode(entity, convert_to_tensor=True,
                                                           show_progress_bar=False).norm().item()
                    degrees.append((i, in_degree))

                # Take top 3 by degree
                degrees.sort(key=lambda x: x[1], reverse=True)
                user_indices = [idx for idx, _ in degrees[:3]]

        # Get embeddings for user entities
        user_embeddings = embeddings[user_indices]

        # For each user entity, find connections to each level
        for level_idx, level in enumerate(hierarchy):
            # Skip if no entities at this level
            if level not in entities_by_type:
                continue

            # Get entities at this level
            level_entities = entities_by_type[level]
            level_indices = [idx for idx, _ in level_entities]

            # Skip if no entities at this level
            if not level_indices:
                continue

            # Get embeddings for entities at this level
            level_embeddings = embeddings[level_indices]

            # Compute similarity between user entities and level entities
            similarities = torch.mm(user_embeddings, level_embeddings.t())

            # For each user entity, find the most similar entities at this level
            for i, user_idx in enumerate(user_indices):
                user_entity = entity_list[user_idx]

                # Get top 2 similar entities at this level
                values, relative_indices = torch.topk(similarities[i], min(2, len(similarities[i])))

                for j, rel_idx in enumerate(relative_indices.tolist()):
                    level_entity = level_entities[rel_idx][1]
                    similarity = values[j].item()

                    if similarity >= similarity_threshold:
                        # Create a connection that explicitly shows the level
                        predicate = f"ns0__Is_implemented_by_{level}"

                        connection = {
                            "subject_id": self.get_consistent_id(user_entity, "subject"),
                            "subject_label": user_entity,
                            "predicate": predicate,
                            "object_id": self.get_consistent_id(level_entity, "object"),  # FIXED: use level_entity
                            "object_label": level_entity,  # FIXED: use level_entity
                            "origin": "gnn",
                            "similarity": similarity
                        }
                        connections.append(connection)

        return connections

    def process(self, triple_records, user_entities, entity_types=None, max_hops=2, debug_mode=False):
        """
        Process knowledge graph triples, find semantic connections,
        and generate enriched subgraph.

        Args:
            triple_records: List of knowledge graph triples
            user_entities: List of user-specified entities to focus on
            entity_types: Optional dict mapping entities to types
            max_hops: Maximum number of hops for traversal
            debug_mode: Whether to print debug information

        Returns:
            Dict containing embeddings and new connections
        """
        if debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        if len(triple_records) < 2:
            logger.warning("Not enough triples to process")
            return {"embeddings": {}, "connections": []}

        logger.info(f"Processing {len(triple_records)} triples with GNN")

        # Make sure user_entities have valid values
        valid_user_entities = [e for e in user_entities if e and isinstance(e, str) and len(e) > 0]
        if not valid_user_entities:
            logger.warning("No valid user entities provided, using the first 2 entities from triple records")
            try:
                # Extract 2 random entities from triple records as fallback
                entity_set = set()
                for rec in triple_records[:20]:  # Just check first 20 records
                    entity_set.add(rec.get("subject_label", ""))
                    entity_set.add(rec.get("object_label", ""))
                valid_user_entities = list(entity_set)[:2]  # Take first 2
            except:
                logger.warning("Could not extract entities from triple records")
                valid_user_entities = ["Entity1", "Entity2"]  # Default fallback

        try:
            # Prepare graph data
            g, features, entity_list = self.prepare_graph_data(triple_records)
            if not entity_list:
                logger.error("No entities found in triple_records")
                return {"embeddings": {}, "connections": []}

            logger.info(f"Created graph with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")

            # Get node embeddings from GNN
            embeddings = self.get_embeddings(g, features)

            # Find semantic connections
            semantic_connections = self.find_semantic_connections(
                embeddings, entity_list, valid_user_entities,
                similarity_threshold=0.5,
                max_hops=max_hops
            )

            logger.info(f"Found {len(semantic_connections)} semantic connections")

            # Find cross-level connections if entity types are available
            cross_level_connections = []
            if entity_types:
                cross_level_connections = self.find_cross_level_connections(
                    embeddings, entity_list, entity_types, valid_user_entities
                )
                logger.info(f"Found {len(cross_level_connections)} cross-level connections")

            # Combine connections
            all_connections = semantic_connections + cross_level_connections

            # Convert embeddings to dictionary
            embedding_dict = {entity_list[i]: emb.cpu().numpy().tolist()
                              for i, emb in enumerate(embeddings)}

            return {
                "embeddings": embedding_dict,
                "connections": all_connections
            }

        except Exception as e:
            logger.error(f"Error processing graph with GNN: {e}")
            import traceback
            traceback.print_exc()
            return {"embeddings": {}, "connections": []}
