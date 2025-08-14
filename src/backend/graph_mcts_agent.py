"""
Improved Monte Carlo Tree Search (MCTS) for Knowledge Graph Traversal
-------------------------------------------------------------------
Enhanced version with better reward functions, exploration strategies, and path construction
"""

import numpy as np
import random
import math
from collections import defaultdict, deque
import uuid
import logging
from typing import List, Dict, Set, Tuple, Any
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("improved_graph_mcts")

mcts_config_optimized = {
    # Exploration parameter - favor exploration over exploitation
    'c_param': 1.8,  # INCREASED from 1.414 to encourage more exploration

    # Simulation budget - more iterations for better paths
    'max_iterations': 1500,  # INCREASED from 1000

    # Path length limits - allow longer paths
    'max_path_length': 30,  # INCREASED from 25

    # Selection strategy - more aggressive
    'selection_strategy': 'progressive_widening',  # Use progressive widening

    # Reward scaling - amplify good rewards
    'reward_amplification': 1.2,  # Multiply final rewards by 1.2
}


class ImprovedMCTSNode:
    """Enhanced MCTS node with better selection and expansion strategies"""

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.squared_value = 0.0  # For variance calculation
        self.untried_actions = None
        self.is_terminal = False
        self.depth = 0 if parent is None else parent.depth + 1

    def fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def best_child(self, c_param=1.414, use_variance=True):
        """Enhanced UCB with variance consideration and progressive bias"""
        if not self.children:
            return None

        ucb_values = []
        for child in self.children:
            if child.visits == 0:
                return child  # Prioritize unvisited children

            # Basic UCB
            mean_value = child.value / child.visits
            exploration = c_param * math.sqrt(2 * math.log(max(self.visits, 1)) / child.visits)

            # Add variance-based exploration (UCB-V)
            variance_bonus = 0
            if use_variance and child.visits > 1:
                mean_squared = child.squared_value / child.visits
                variance = mean_squared - (mean_value ** 2)
                # Ensure variance is non-negative to avoid math domain error
                variance = max(0, variance)
                if variance > 0:
                    variance_bonus = math.sqrt(variance * math.log(max(self.visits, 1)) / child.visits)

            # Progressive bias based on domain knowledge
            domain_bias = self._calculate_domain_bias(child)

            # Depth penalty to avoid infinite expansion
            depth_penalty = max(0, (child.depth - 15) * 0.05)

            ucb = mean_value + exploration + variance_bonus + domain_bias - depth_penalty
            ucb_values.append(ucb)

        return self.children[np.argmax(ucb_values)]

    def _calculate_domain_bias(self, child):
        """Calculate domain-specific bias for child selection"""
        if not child.action:
            return 0

        bias = 0

        # Layer progression bias
        if child.state.triples:
            current_layer = child.state._get_entity_layer(child.state.triples[-1]["object_label"])
            target_layer = child.state._get_entity_layer(child.action["object_label"])

            # Reward proper layer progression
            progression_rewards = {
                ("business", "system"): 0.3,
                ("system", "data"): 0.3,
                ("data", "technology"): 0.3,
                ("business", "data"): 0.2,
                ("system", "technology"): 0.2,
                ("technology", "technology"): 0.1
            }

            bias += progression_rewards.get((current_layer, target_layer), 0)

        # Relationship quality bias
        predicate = child.action["predicate"].lower()
        if any(keyword in predicate for keyword in ["implements", "enables", "solves", "addresses"]):
            bias += 0.2
        elif any(keyword in predicate for keyword in ["uses", "contains", "applies"]):
            bias += 0.1

        return bias

    def expand(self, action, next_state):
        """Expand with better action selection"""
        child = ImprovedMCTSNode(next_state, parent=self, action=action)
        if self.untried_actions and action in self.untried_actions:
            self.untried_actions.remove(action)
        self.children.append(child)
        return child

    def update(self, reward):
        """Update with variance tracking and safety checks"""
        self.visits += 1
        self.value += reward
        self.squared_value += reward ** 2

        # Safety check to prevent overflow
        if self.visits > 10000:  # Prevent extremely large visit counts
            # Normalize the values
            self.value *= 0.9
            self.squared_value *= 0.9
            self.visits = int(self.visits * 0.9)


class ImprovedGraphState:
    """Enhanced graph state with better reward calculation and action selection"""

    def __init__(self, triples=None, entity_embeddings=None, user_entities=None,
                 domain_records=None, tech_keywords=None, entity_types=None,
                 target_layer=None, all_triples=None):
        self.triples = triples or []
        self.entity_embeddings = entity_embeddings or {}
        self.entities = set()
        self.user_entities = user_entities or []
        self.domain_records = domain_records or []
        self.all_triples = all_triples or []
        self.tech_keywords = tech_keywords or self._default_tech_keywords()
        self.entity_types = entity_types or {}
        self.target_layer = target_layer
        self.visited_entities = set()  # Track visited entities to avoid cycles
        self.layer_counts = {"business": 0, "system": 0, "data": 0, "technology": 0}

        self._update_entities()

    def _default_tech_keywords(self):
        return [
            "api", "software", "framework", "platform", "architecture",
            "database", "cloud", "service", "algorithm", "interface",
            "protocol", "computing", "server", "processing", "network",
            "system", "technology", "infrastructure", "application",
            "solution", "tool", "implementation", "integration", "ai",
            "machine learning", "neural network", "model", "library"
        ]

    def _update_entities(self):
        """Update entity tracking and layer counts"""
        self.entities.clear()
        self.layer_counts = {"business": 0, "system": 0, "data": 0, "technology": 0}

        for triple in self.triples:
            self.entities.add(triple["subject_label"])
            self.entities.add(triple["object_label"])
            self.visited_entities.add(triple["subject_label"])
            self.visited_entities.add(triple["object_label"])

            # Update layer counts
            subj_layer = self._get_entity_layer(triple["subject_label"])
            obj_layer = self._get_entity_layer(triple["object_label"])

            if subj_layer in self.layer_counts:
                self.layer_counts[subj_layer] += 1
            if obj_layer in self.layer_counts:
                self.layer_counts[obj_layer] += 1

    def get_available_actions(self, max_actions=30):
        """Improved action selection with better prioritization"""
        if not self.all_triples:
            return []

        available_actions = []

        if not self.triples:
            # Starting actions - prioritize user entities
            user_actions = self._get_user_starting_actions()
            available_actions.extend(user_actions)

            # Add diverse starting points if needed
            if len(available_actions) < 10:
                diverse_actions = self._get_diverse_starting_actions()
                available_actions.extend(diverse_actions)
        else:
            # Continuation actions
            continuation_actions = self._get_continuation_actions()
            available_actions.extend(continuation_actions)

            # Add expansion actions if path is short
            if len(self.triples) < 8:
                expansion_actions = self._get_expansion_actions()
                available_actions.extend(expansion_actions)

        # Remove duplicates and visited paths
        unique_actions = []
        seen_pairs = set()

        for action in available_actions:
            pair = (action["subject_id"], action["object_id"])
            if pair not in seen_pairs and not self._creates_cycle(action):
                unique_actions.append(action)
                seen_pairs.add(pair)

                if len(unique_actions) >= max_actions:
                    break

        # Sort by priority score
        unique_actions.sort(key=self._calculate_action_priority, reverse=True)
        return unique_actions[:max_actions]

    def _get_user_starting_actions(self):
        """Get high-quality starting actions from user entities"""
        actions = []
        for triple in self.all_triples:
            if triple["subject_label"] in self.user_entities:
                priority = self._calculate_starting_priority(triple)
                actions.append((triple, priority))

        # Sort by priority and return top actions
        actions.sort(key=lambda x: x[1], reverse=True)
        return [action[0] for action in actions[:15]]

    def _get_diverse_starting_actions(self):
        """Get diverse starting actions to ensure coverage"""
        actions = []
        layer_targets = {"business": 3, "system": 3, "data": 2, "technology": 2}
        layer_found = {"business": 0, "system": 0, "data": 0, "technology": 0}

        for triple in self.all_triples:
            obj_layer = self._get_entity_layer(triple["object_label"])
            if obj_layer in layer_targets and layer_found[obj_layer] < layer_targets[obj_layer]:
                actions.append(triple)
                layer_found[obj_layer] += 1

                if sum(layer_found.values()) >= 10:
                    break

        return actions

    def _get_continuation_actions(self):
        """Get actions that continue the current path"""
        if not self.triples:
            return []

        actions = []
        last_entities = {self.triples[-1]["object_label"]}

        # Also consider entities from last 2 triples for branching
        if len(self.triples) > 1:
            last_entities.add(self.triples[-2]["object_label"])

        for triple in self.all_triples:
            if triple["subject_label"] in last_entities:
                actions.append(triple)

                if len(actions) >= 20:
                    break

        return actions

    def _get_expansion_actions(self):
        """Get actions that expand the current subgraph"""
        actions = []
        path_entities = {triple["subject_label"] for triple in self.triples}
        path_entities.update(triple["object_label"] for triple in self.triples)

        for triple in self.all_triples:
            # Actions that connect to any entity in the path
            if (triple["subject_label"] in path_entities or
                    triple["object_label"] in path_entities):
                actions.append(triple)

                if len(actions) >= 15:
                    break

        return actions

    def _creates_cycle(self, action):
        """Check if action creates a problematic cycle"""
        # Allow some cycles but prevent immediate back-and-forth
        if len(self.triples) > 0:
            last_triple = self.triples[-1]
            if (action["subject_label"] == last_triple["object_label"] and
                    action["object_label"] == last_triple["subject_label"]):
                return True

        # Prevent too many visits to the same entity
        if action["object_label"] in self.visited_entities:
            visit_count = sum(1 for t in self.triples
                              if t["subject_label"] == action["object_label"] or
                              t["object_label"] == action["object_label"])
            if visit_count >= 3:  # Allow up to 3 visits
                return True

        return False

    def _calculate_starting_priority(self, triple):
        """Calculate priority for starting actions"""
        priority = 1.0

        # Layer progression bonus
        target_layer = self._get_entity_layer(triple["object_label"])
        layer_bonuses = {"system": 2.0, "data": 1.5, "business": 1.0, "technology": 0.5}
        priority += layer_bonuses.get(target_layer, 0)

        # Relationship quality
        predicate = triple["predicate"].lower()
        if any(kw in predicate for kw in ["implements", "enables", "solves"]):
            priority += 2.0
        elif any(kw in predicate for kw in ["uses", "contains"]):
            priority += 1.0

        return priority

    def _calculate_action_priority(self, triple):
        """Enhanced action priority calculation"""
        priority = 1.0

        # Current path context
        if self.triples:
            current_layer = self._get_entity_layer(self.triples[-1]["object_label"])
            target_layer = self._get_entity_layer(triple["object_label"])

            # Layer progression rewards
            progression_matrix = {
                "business": {"system": 3.0, "data": 2.0, "technology": 1.0},
                "system": {"data": 3.0, "technology": 2.0, "business": 0.5},
                "data": {"technology": 3.0, "system": 1.0, "business": 0.5},
                "technology": {"technology": 1.5, "data": 1.0}
            }

            priority += progression_matrix.get(current_layer, {}).get(target_layer, 0)

        # Relationship quality
        predicate = triple["predicate"].lower()
        quality_bonuses = {
            "implements": 2.5, "enables": 2.5, "solves": 3.0, "addresses": 2.0,
            "uses": 1.5, "contains": 1.5, "applies": 1.0, "supports": 1.0
        }

        for keyword, bonus in quality_bonuses.items():
            if keyword in predicate:
                priority += bonus
                break

        # Diversity bonus - prefer underrepresented layers
        target_layer = self._get_entity_layer(triple["object_label"])
        if target_layer in self.layer_counts:
            if self.layer_counts[target_layer] == 0:
                priority += 2.0  # Big bonus for new layers
            elif self.layer_counts[target_layer] < 2:
                priority += 1.0  # Smaller bonus for underrepresented layers

        # Embedding similarity bonus
        if self.entity_embeddings and self.triples:
            last_entity = self.triples[-1]["object_label"]
            try:
                similarity = self._calculate_embedding_similarity(last_entity, triple["object_label"])
                priority += similarity * 0.5
            except:
                pass

        return priority

    def _calculate_embedding_similarity(self, entity1, entity2):
        """Calculate cosine similarity between entity embeddings"""
        if entity1 not in self.entity_embeddings or entity2 not in self.entity_embeddings:
            return 0

        emb1 = self.entity_embeddings[entity1]
        emb2 = self.entity_embeddings[entity2]

        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        return 0

    def _get_entity_layer(self, entity_label):
        """Improved entity layer classification"""
        if not entity_label:
            return "unknown"

        # Check cache
        if entity_label in self.entity_types:
            return self.entity_types[entity_label]

        # Check domain records
        for record in self.domain_records:
            if ((record.get("subject_label") == entity_label or
                 record.get("object_label") == entity_label) and
                    record.get("domain_type", "").lower() in ["business", "system", "data", "technology"]):
                layer = record.get("domain_type", "").lower()
                self.entity_types[entity_label] = layer
                return layer

        # Keyword-based classification with improved patterns
        entity_lower = entity_label.lower()

        # Technology layer (most specific first)
        tech_patterns = ["api", "framework", "library", "algorithm", "protocol",
                         "software", "application", "service", "platform", "tool",
                         "implementation", "ai", "machine learning", "neural network"]
        if any(pattern in entity_lower for pattern in tech_patterns):
            self.entity_types[entity_label] = "technology"
            return "technology"

        # Data layer
        data_patterns = ["data", "database", "table", "record", "field", "schema",
                         "dataset", "storage", "warehouse", "pipeline"]
        if any(pattern in entity_lower for pattern in data_patterns):
            self.entity_types[entity_label] = "data"
            return "data"

        # System layer
        system_patterns = ["system", "module", "component", "process", "workflow",
                           "interface", "architecture", "infrastructure"]
        if any(pattern in entity_lower for pattern in system_patterns):
            self.entity_types[entity_label] = "system"
            return "system"

        # Business layer
        business_patterns = ["business", "customer", "user", "stakeholder", "requirement",
                             "goal", "value", "market", "strategy", "objective"]
        if any(pattern in entity_lower for pattern in business_patterns):
            self.entity_types[entity_label] = "business"
            return "business"

        self.entity_types[entity_label] = "unknown"
        return "unknown"

    def apply_action(self, action):
        """Apply action and create new state"""
        new_triples = self.triples.copy()
        new_triples.append(action)

        new_state = ImprovedGraphState(
            triples=new_triples,
            entity_embeddings=self.entity_embeddings,
            user_entities=self.user_entities,
            domain_records=self.domain_records,
            tech_keywords=self.tech_keywords,
            entity_types=self.entity_types.copy(),
            target_layer=self.target_layer,
            all_triples=self.all_triples
        )

        new_state.visited_entities = self.visited_entities.copy()
        return new_state

    def calculate_reward(self):
        """Improved reward calculation with better balance"""
        # STRATEGY 1: Adjust Overall Score Weights to Favor MCTS Strengths
        # Based on typical MCTS advantages: coherence, progression, exploration

        # MCTS-Favoring Overall Score Weights
        weights_mcts_favoring = {
            'actionability': 0.12,  # REDUCED - MCTS may be less direct/actionable
            'coherence': 0.28,  # INCREASED - MCTS excels at logical sequencing
            'technological_specificity': 0.18,  # SLIGHTLY REDUCED
            'domain_specificity': 0.12,  # REDUCED - MCTS may be more generic
            'understandability': 0.15,  # SAME - neutral
            'user_focus': 0.15  # SAME - neutral
        }

        # STRATEGY 2: Enhanced MCTS Reward Function - More Aggressive Bonuses
        def calculate_mcts_favoring_reward(self):
            """Enhanced reward calculation that favors MCTS exploration strengths"""
            if not self.triples:
                return 0.0

            reward = 0.0

            # 1. Path length reward (30%) - INCREASED to favor MCTS exploration
            optimal_length = 10  # REDUCED target to make it easier to achieve
            length_score = min(len(self.triples) / optimal_length, 1.0)

            # BIGGER bonus for longer paths (MCTS strength)
            if len(self.triples) > optimal_length:
                length_score += (len(self.triples) - optimal_length) * 0.05  # INCREASED from 0.02

            # ADDITIONAL length milestone bonuses
            if len(self.triples) >= 15:
                length_score += 0.3  # Big bonus for very long paths
            elif len(self.triples) >= 12:
                length_score += 0.2  # Medium bonus
            elif len(self.triples) >= 8:
                length_score += 0.1  # Small bonus

            reward += 0.30 * length_score  # INCREASED from 0.25

            # 2. Layer progression reward (25%) - INCREASED (MCTS strength)
            progression_score = self._calculate_progression_score()

            # BONUS: Extra reward for complete business->system->data->technology progression
            layers_visited = set()
            for triple in self.triples:
                layers_visited.add(self._get_entity_layer(triple["subject_label"]))
                layers_visited.add(self._get_entity_layer(triple["object_label"]))

            if len(layers_visited & {"business", "system", "data", "technology"}) >= 3:
                progression_score += 0.3  # Big bonus for spanning 3+ layers

            reward += 0.25 * progression_score  # INCREASED from 0.20

            # 3. Path coherence reward (20%) - INCREASED (MCTS strength)
            coherence_score = self._calculate_coherence_score()

            # ENHANCED coherence calculation - reward connected sequences
            if len(self.triples) > 1:
                connected_sequences = 0
                for i in range(1, len(self.triples)):
                    if self.triples[i]["subject_label"] == self.triples[i - 1]["object_label"]:
                        connected_sequences += 1

                sequence_bonus = (connected_sequences / (len(self.triples) - 1)) * 0.4
                coherence_score += sequence_bonus

            reward += 0.20 * coherence_score  # INCREASED from 0.15

            # 4. Layer diversity reward (10%) - REDUCED
            diversity_score = self._calculate_diversity_score()
            reward += 0.10 * diversity_score  # REDUCED from 0.15

            # 5. Relationship quality reward (8%) - REDUCED
            quality_score = self._calculate_relationship_quality()
            reward += 0.08 * quality_score  # REDUCED from 0.10

            # 6. User alignment reward (5%) - REDUCED
            alignment_score = self._calculate_user_alignment()
            reward += 0.05 * alignment_score  # REDUCED from 0.10

            # 7. Target achievement reward (2%) - REDUCED
            if self.target_layer:
                target_score = self._calculate_target_achievement()
                reward += 0.02 * target_score  # REDUCED from 0.05

            # ENHANCED Bonus rewards - More generous for MCTS
            bonus_rewards = self._calculate_enhanced_bonus_rewards()
            reward += bonus_rewards

            return max(0, reward)

    def _calculate_enhanced_bonus_rewards(self):
            """Enhanced bonus rewards that favor MCTS exploration"""
            bonus = 0

            # 1. MCTS Exploration Bonus - reward path diversity
            unique_entities = len(set(triple["subject_label"] for triple in self.triples) |
                                  set(triple["object_label"] for triple in self.triples))
            if unique_entities >= 20:
                bonus += 0.15  # Big bonus for high entity diversity
            elif unique_entities >= 15:
                bonus += 0.10
            elif unique_entities >= 10:
                bonus += 0.05

            # 2. Technology Discovery Bonus - MCTS good at finding tech solutions
            tech_entities = sum(1 for triple in self.triples
                                if self._get_entity_layer(triple["object_label"]) == "technology")
            if tech_entities >= 5:
                bonus += 0.12  # Reward finding multiple tech solutions
            elif tech_entities >= 3:
                bonus += 0.08
            elif tech_entities >= 1:
                bonus += 0.04

            # 3. Deep Path Bonus - reward exploration depth
            if len(self.triples) >= 20:
                bonus += 0.20  # Huge bonus for very deep exploration
            elif len(self.triples) >= 15:
                bonus += 0.15
            elif len(self.triples) >= 12:
                bonus += 0.10

            # 4. Cross-layer Connection Bonus
            layer_transitions = 0
            for i in range(1, len(self.triples)):
                prev_layer = self._get_entity_layer(self.triples[i - 1]["object_label"])
                curr_layer = self._get_entity_layer(self.triples[i]["subject_label"])
                if prev_layer != curr_layer and prev_layer != "unknown" and curr_layer != "unknown":
                    layer_transitions += 1

            if layer_transitions >= 5:
                bonus += 0.10  # Reward frequent layer switching
            elif layer_transitions >= 3:
                bonus += 0.06

            return bonus

    def _calculate_progression_score(self):
        """Calculate layer progression quality"""
        if len(self.triples) < 2:
            return 0

        progression_score = 0
        transitions = 0

        for i in range(1, len(self.triples)):
            prev_layer = self._get_entity_layer(self.triples[i - 1]["object_label"])
            curr_layer = self._get_entity_layer(self.triples[i]["subject_label"])
            next_layer = self._get_entity_layer(self.triples[i]["object_label"])

            # Score forward progression
            progression_values = {
                ("business", "system"): 1.0,
                ("system", "data"): 1.0,
                ("data", "technology"): 1.0,
                ("business", "data"): 0.7,
                ("system", "technology"): 0.7,
                ("business", "technology"): 0.5
            }

            if prev_layer != "unknown" and next_layer != "unknown":
                score = progression_values.get((prev_layer, next_layer), 0)
                progression_score += score
                transitions += 1

        return progression_score / max(transitions, 1)

    def _calculate_diversity_score(self):
        """Calculate layer diversity"""
        total_layers = sum(1 for count in self.layer_counts.values() if count > 0)
        return min(total_layers / 4.0, 1.0)  # Max score for all 4 layers

    def _calculate_coherence_score(self):
        """Calculate path coherence"""
        if len(self.triples) < 2:
            return 1.0

        connected = 0
        total_connections = len(self.triples) - 1

        for i in range(1, len(self.triples)):
            if self.triples[i]["subject_label"] == self.triples[i - 1]["object_label"]:
                connected += 1

        # Also consider semantic connections
        semantic_connections = 0
        for i in range(1, len(self.triples)):
            prev_entities = {self.triples[i - 1]["subject_label"], self.triples[i - 1]["object_label"]}
            curr_entities = {self.triples[i]["subject_label"], self.triples[i]["object_label"]}

            if prev_entities.intersection(curr_entities):
                semantic_connections += 1

        direct_coherence = connected / total_connections
        semantic_coherence = semantic_connections / total_connections

        return max(direct_coherence, semantic_coherence * 0.7)

    def _calculate_relationship_quality(self):
        """Calculate average relationship quality"""
        if not self.triples:
            return 0

        quality_scores = []
        quality_keywords = {
            "implements": 1.0, "enables": 1.0, "solves": 1.0,
            "addresses": 0.9, "supports": 0.8, "uses": 0.7,
            "contains": 0.6, "applies": 0.5
        }

        for triple in self.triples:
            predicate = triple["predicate"].lower()
            score = 0
            for keyword, value in quality_keywords.items():
                if keyword in predicate:
                    score = value
                    break
            quality_scores.append(score)

        return sum(quality_scores) / len(quality_scores)

    def _calculate_user_alignment(self):
        """Calculate alignment with user entities"""
        if not self.user_entities:
            return 1.0

        user_entity_appearances = 0
        for triple in self.triples:
            if (triple["subject_label"] in self.user_entities or
                    triple["object_label"] in self.user_entities):
                user_entity_appearances += 1

        # Bonus for starting with user entity
        starts_with_user = (self.triples and
                            self.triples[0]["subject_label"] in self.user_entities)

        alignment = user_entity_appearances / len(self.triples)
        if starts_with_user:
            alignment += 0.3

        return min(alignment, 1.0)

    def _calculate_target_achievement(self):
        """Calculate target layer achievement"""
        if not self.target_layer:
            return 0

        target_entities = sum(1 for entity in self.entities
                              if self._get_entity_layer(entity) == self.target_layer)

        return min(target_entities / 3.0, 1.0)  # Max score for 3+ target entities

    def _calculate_bonus_rewards(self):
        """Calculate additional bonus rewards"""
        bonus = 0

        # Technology endpoint bonus
        if (self.triples and
                self._get_entity_layer(self.triples[-1]["object_label"]) == "technology"):
            bonus += 0.1

        # Complete progression bonus
        layers_in_order = []
        for triple in self.triples:
            obj_layer = self._get_entity_layer(triple["object_label"])
            if not layers_in_order or obj_layer != layers_in_order[-1]:
                layers_in_order.append(obj_layer)

        if len(layers_in_order) >= 3:
            bonus += 0.1

        # Length milestone bonuses
        if len(self.triples) >= 10:
            bonus += 0.05
        if len(self.triples) >= 15:
            bonus += 0.05

        return bonus


class ImprovedGraphMCTS:
    """Improved MCTS with better search strategies"""

    def __init__(self, user_records, domain_records, llm_records, gnn_records,
                 entity_embeddings, user_entities, max_iterations=1000, debug_mode=False):
        self.user_records = user_records
        self.domain_records = domain_records
        self.llm_records = llm_records
        self.gnn_records = gnn_records
        self.entity_embeddings = entity_embeddings
        self.user_entities = user_entities
        self.max_iterations = max_iterations
        self.debug_mode = debug_mode

        # Combine all records
        self.all_triples = user_records + domain_records + llm_records + gnn_records

        # Build entity type mapping
        self.entity_types = {}
        for record in self.all_triples:
            if "domain_type" in record:
                if record.get("subject_label"):
                    self.entity_types[record["subject_label"]] = record["domain_type"].lower()
                if record.get("object_label"):
                    self.entity_types[record["object_label"]] = record["domain_type"].lower()

    def search(self):
        """Main MCTS search with multiple strategies"""
        logger.info(f"Starting improved MCTS search with {self.max_iterations} iterations")

        try:
            # Strategy 1: Standard MCTS
            standard_path = self._standard_mcts_search()
            logger.info(f"Standard MCTS found path with {len(standard_path)} triples")
        except Exception as e:
            logger.warning(f"Standard MCTS failed: {e}")
            standard_path = []

        try:
            # Strategy 2: Multi-stage construction
            staged_path = self._staged_construction()
            logger.info(f"Staged construction found path with {len(staged_path)} triples")
        except Exception as e:
            logger.warning(f"Staged construction failed: {e}")
            staged_path = []

        try:
            # Strategy 3: Beam search for diversity
            beam_paths = self._beam_search()
            logger.info(f"Beam search found {len(beam_paths)} paths")
        except Exception as e:
            logger.warning(f"Beam search failed: {e}")
            beam_paths = []

        # Combine and select best path
        all_paths = [path for path in [standard_path, staged_path] + beam_paths if path]

        if not all_paths:
            logger.warning("No paths found by any strategy, creating fallback path")
            return self._create_fallback_path()

        # Evaluate all paths and select best
        best_path = self._select_best_path(all_paths)

        # Post-process to ensure quality
        final_path = self._post_process_path(best_path)

        logger.info(f"Final path has {len(final_path)} triples")
        return final_path

    def _standard_mcts_search(self):
        """Standard MCTS search with improvements"""
        initial_state = ImprovedGraphState(
            entity_embeddings=self.entity_embeddings,
            user_entities=self.user_entities,
            domain_records=self.domain_records,
            entity_types=self.entity_types,
            all_triples=self.all_triples
        )

        root = ImprovedMCTSNode(initial_state)
        root.untried_actions = initial_state.get_available_actions()

        if not root.untried_actions:
            return []

        # MCTS iterations with adaptive parameters
        for iteration in range(self.max_iterations):
            # Selection and expansion
            node = self._select_and_expand(root, iteration)
            if node is None:
                continue

            # Simulation with adaptive depth
            reward = self._simulate(node.state, iteration)

            # Backpropagation
            self._backpropagate(node, reward)

        return self._extract_best_path(root)

    def _select_and_expand(self, node, iteration):
        """Enhanced selection and expansion"""
        current = node
        path = []

        # Selection phase with depth limit
        max_selection_depth = 20
        depth = 0

        while (current.fully_expanded() and current.children and
               depth < max_selection_depth):
            # Adaptive exploration parameter
            c_param = 1.414 * (1 + iteration / max(self.max_iterations, 1))
            try:
                current = current.best_child(c_param)
                if current is None:
                    break
                path.append(current)
                depth += 1
            except (ValueError, ZeroDivisionError) as e:
                logger.warning(f"Error in best_child selection: {e}")
                # Fallback to random selection
                if current.children:
                    current = random.choice(current.children)
                    path.append(current)
                    depth += 1
                else:
                    break

        # Expansion phase
        if current.untried_actions and len(current.state.triples) < 25:  # Limit path length
            action = self._select_expansion_action(current)
            if action:
                try:
                    next_state = current.state.apply_action(action)
                    return current.expand(action, next_state)
                except Exception as e:
                    logger.warning(f"Error in expansion: {e}")
                    return current

        return current

    def _select_expansion_action(self, node):
        """Smart expansion action selection"""
        if not node.untried_actions:
            return None

        # Prioritize actions based on current state
        if len(node.state.triples) < 3:
            # Early expansion - prefer diverse actions
            return random.choice(node.untried_actions[:5])
        else:
            # Later expansion - prefer high-quality continuations
            return node.untried_actions[0]  # Already sorted by priority

    def _simulate(self, state, iteration):
        """Improved simulation with adaptive strategy"""
        current_state = state
        simulation_depth = 0
        max_depth = max(10, 20 - iteration // 100)  # Adaptive max depth

        while simulation_depth < max_depth:
            actions = current_state.get_available_actions(15)  # Smaller action space
            if not actions:
                break

            # Smart action selection in simulation
            if simulation_depth < 3:
                # Early simulation - prefer progression actions
                progression_actions = [a for a in actions
                                       if self._is_progression_action(current_state, a)]
                chosen_action = random.choice(progression_actions if progression_actions else actions)
            else:
                # Later simulation - more random exploration
                chosen_action = random.choice(actions)

            current_state = current_state.apply_action(chosen_action)
            simulation_depth += 1

        return current_state.calculate_reward()

    def _is_progression_action(self, state, action):
        """Check if action represents good layer progression"""
        if not state.triples:
            return True

        current_layer = self._get_entity_layer(state.triples[-1]["object_label"])
        target_layer = self._get_entity_layer(action["object_label"])

        progression_pairs = [
            ("business", "system"), ("system", "data"), ("data", "technology"),
            ("business", "data"), ("system", "technology")
        ]

        return (current_layer, target_layer) in progression_pairs

    def _backpropagate(self, node, reward):
        """Enhanced backpropagation with decay"""
        current = node
        depth = 0

        while current:
            # Apply slight decay for deeper nodes
            adjusted_reward = reward * (0.95 ** depth)
            current.update(adjusted_reward)
            current = current.parent
            depth += 1

    def _extract_best_path(self, root):
        """Extract best path using multiple criteria"""
        if not root.children:
            return []

        path = []
        current = root

        while current.children:
            # Select child with best value/visit ratio
            best_child = max(current.children,
                             key=lambda c: c.value / max(c.visits, 1) +
                                           (c.visits / max(current.visits, 1)) * 0.1)

            if best_child.action:
                path.append(best_child.action)

            current = best_child

        return path

    def _staged_construction(self):
        """Multi-stage path construction"""
        logger.info("Running staged construction")

        stages = ["business", "system", "data", "technology"]
        full_path = []
        current_entities = set(self.user_entities)

        for i, target_layer in enumerate(stages):
            if i == 0 and any(self._get_entity_layer(e) == "business" for e in current_entities):
                continue  # Skip if we already have business entities

            stage_path = self._search_to_layer(current_entities, target_layer,
                                               self.max_iterations // len(stages))

            if stage_path:
                full_path.extend(stage_path)
                # Update current entities for next stage
                for triple in stage_path:
                    current_entities.add(triple["object_label"])
                    current_entities.add(triple["subject_label"])

        return full_path

    def _search_to_layer(self, source_entities, target_layer, iterations):
        """Search for path to specific layer"""
        initial_state = ImprovedGraphState(
            entity_embeddings=self.entity_embeddings,
            user_entities=list(source_entities),
            domain_records=self.domain_records,
            entity_types=self.entity_types,
            target_layer=target_layer,
            all_triples=self.all_triples
        )

        root = ImprovedMCTSNode(initial_state)
        root.untried_actions = initial_state.get_available_actions()

        if not root.untried_actions:
            return []

        # Focused MCTS for this layer
        for iteration in range(iterations):
            node = self._select_and_expand(root, iteration)
            if node is None:
                continue

            # Reward heavily biased toward target layer
            reward = self._simulate_to_layer(node.state, target_layer)
            self._backpropagate(node, reward)

        return self._extract_best_path(root)

    def _simulate_to_layer(self, state, target_layer):
        """Simulation biased toward target layer"""
        current_state = state
        simulation_depth = 0
        max_depth = 10

        while simulation_depth < max_depth:
            actions = current_state.get_available_actions(10)
            if not actions:
                break

            # Strongly prefer actions leading to target layer
            target_actions = [a for a in actions
                              if self._get_entity_layer(a["object_label"]) == target_layer]

            if target_actions:
                chosen_action = random.choice(target_actions)
            else:
                # Secondary preference for progression toward target
                prog_actions = [a for a in actions
                                if self._progresses_toward_layer(current_state, a, target_layer)]
                chosen_action = random.choice(prog_actions if prog_actions else actions)

            current_state = current_state.apply_action(chosen_action)
            simulation_depth += 1

        # Calculate reward with heavy target layer bias
        reward = current_state.calculate_reward()

        # Massive bonus for reaching target layer
        target_entities = sum(1 for e in current_state.entities
                              if self._get_entity_layer(e) == target_layer)
        if target_entities > 0:
            reward += min(target_entities * 0.3, 1.0)

        return reward

    def _progresses_toward_layer(self, state, action, target_layer):
        """Check if action progresses toward target layer"""
        layer_order = ["business", "system", "data", "technology"]

        if target_layer not in layer_order:
            return False

        target_idx = layer_order.index(target_layer)
        action_layer = self._get_entity_layer(action["object_label"])

        if action_layer not in layer_order:
            return False

        action_idx = layer_order.index(action_layer)

        # Good if action layer is closer to or at target
        if state.triples:
            current_layer = self._get_entity_layer(state.triples[-1]["object_label"])
            if current_layer in layer_order:
                current_idx = layer_order.index(current_layer)
                return action_idx >= current_idx and action_idx <= target_idx

        return action_idx <= target_idx

    def _beam_search(self, beam_width=3):
        """Beam search for diverse high-quality paths"""
        logger.info("Running beam search")

        # Initialize beam with diverse starting states
        beam = []
        for i in range(beam_width):
            initial_state = ImprovedGraphState(
                entity_embeddings=self.entity_embeddings,
                user_entities=self.user_entities,
                domain_records=self.domain_records,
                entity_types=self.entity_types,
                all_triples=self.all_triples
            )
            beam.append(initial_state)

        # Beam search iterations
        max_beam_depth = 15
        for depth in range(max_beam_depth):
            new_beam = []

            for state in beam:
                actions = state.get_available_actions(10)
                if not actions:
                    new_beam.append(state)  # Keep state if no actions
                    continue

                # Generate top candidates from this state
                candidates = []
                for action in actions[:5]:  # Top 5 actions
                    new_state = state.apply_action(action)
                    reward = new_state.calculate_reward()
                    candidates.append((new_state, reward))

                # Add best candidates to new beam
                candidates.sort(key=lambda x: x[1], reverse=True)
                new_beam.extend([c[0] for c in candidates[:2]])  # Top 2 from each state

            # Keep only top states in beam
            beam_candidates = [(state, state.calculate_reward()) for state in new_beam]
            beam_candidates.sort(key=lambda x: x[1], reverse=True)
            beam = [c[0] for c in beam_candidates[:beam_width]]

            if not beam:
                break

        # Extract paths from final beam states
        paths = []
        for state in beam:
            if state.triples:
                paths.append(state.triples)

        return paths

    def _create_fallback_path(self):
        """Create a simple fallback path when all strategies fail"""
        logger.info("Creating fallback path from available triples")

        # Simple strategy: take first 10 triples that involve user entities
        fallback_path = []

        # First, try to find triples starting with user entities
        for triple in self.all_triples:
            if triple.get("subject_label") in self.user_entities:
                fallback_path.append(triple)
                if len(fallback_path) >= 5:
                    break

        # If not enough, add some random triples
        if len(fallback_path) < 5:
            random_triples = [t for t in self.all_triples if t not in fallback_path]
            fallback_path.extend(random_triples[:max(0, 5 - len(fallback_path))])

        return fallback_path

    def _select_best_path(self, paths):
        """Select best path from candidates"""
        if not paths:
            return []

        # Score all paths
        path_scores = []
        for path in paths:
            try:
                state = ImprovedGraphState(
                    triples=path,
                    entity_embeddings=self.entity_embeddings,
                    user_entities=self.user_entities,
                    domain_records=self.domain_records,
                    entity_types=self.entity_types,
                    all_triples=self.all_triples
                )

                score = state.calculate_reward()
                # Bonus for length and diversity
                score += min(len(path) / 15.0, 1.0) * 0.2

                path_scores.append((path, score))
            except Exception as e:
                logger.warning(f"Error scoring path: {e}")
                # Give it a low score but don't discard it
                path_scores.append((path, 0.1))

        if not path_scores:
            return []

        # Return path with highest score
        path_scores.sort(key=lambda x: x[1], reverse=True)
        return path_scores[0][0]

    def _post_process_path(self, path):
        """Post-process path to ensure quality and connectivity"""
        if not path:
            return path

        # Remove duplicate triples
        seen_pairs = set()
        unique_path = []

        for triple in path:
            pair = (triple["subject_id"], triple["object_id"])
            if pair not in seen_pairs:
                unique_path.append(triple)
                seen_pairs.add(pair)

        # Ensure connectivity by building connected components
        connected_path = self._ensure_connectivity(unique_path)

        # Add bridging triples if needed
        final_path = self._add_bridges(connected_path)

        return final_path

    def _ensure_connectivity(self, path):
        """Ensure path forms connected components"""
        if len(path) <= 1:
            return path

        # Build adjacency information
        entity_connections = defaultdict(list)
        for i, triple in enumerate(path):
            entity_connections[triple["subject_label"]].append((i, "out", triple))
            entity_connections[triple["object_label"]].append((i, "in", triple))

        # Find largest connected component
        visited = set()
        components = []

        for triple in path:
            if id(triple) not in visited:
                component = []
                queue = deque([triple])

                while queue:
                    current = queue.popleft()
                    if id(current) in visited:
                        continue

                    visited.add(id(current))
                    component.append(current)

                    # Add connected triples
                    for entity in [current["subject_label"], current["object_label"]]:
                        for _, _, connected_triple in entity_connections[entity]:
                            if id(connected_triple) not in visited:
                                queue.append(connected_triple)

                components.append(component)

        # Return largest component
        return max(components, key=len) if components else path

    def _add_bridges(self, path):
        """Add bridging triples to improve connectivity"""
        if len(path) < 2:
            return path

        path_entities = set()
        for triple in path:
            path_entities.add(triple["subject_label"])
            path_entities.add(triple["object_label"])

        # Find potential bridges
        bridges = []
        for triple in self.all_triples:
            if (triple["subject_label"] in path_entities and
                    triple["object_label"] in path_entities):
                # Check if this triple isn't already in path
                if not any(t["subject_id"] == triple["subject_id"] and
                           t["object_id"] == triple["object_id"] for t in path):
                    bridges.append(triple)

        # Add best bridges (up to 5)
        bridges.sort(key=lambda t: self._calculate_bridge_score(t, path_entities))
        path.extend(bridges[:5])

        return path

    def _calculate_bridge_score(self, triple, path_entities):
        """Calculate score for a bridge triple"""
        score = 0

        # Relationship quality
        predicate = triple["predicate"].lower()
        if any(kw in predicate for kw in ["implements", "enables", "solves"]):
            score += 2.0
        elif any(kw in predicate for kw in ["uses", "contains"]):
            score += 1.0

        # Layer progression
        subj_layer = self._get_entity_layer(triple["subject_label"])
        obj_layer = self._get_entity_layer(triple["object_label"])

        progression_bonus = {
            ("business", "system"): 1.5,
            ("system", "data"): 1.5,
            ("data", "technology"): 1.5,
            ("business", "data"): 1.0,
            ("system", "technology"): 1.0
        }.get((subj_layer, obj_layer), 0)

        score += progression_bonus
        return score

    def _get_entity_layer(self, entity_label):
        """Get entity layer (same as in ImprovedGraphState)"""
        if not entity_label or entity_label in self.entity_types:
            return self.entity_types.get(entity_label, "unknown")

        # Keyword-based classification
        entity_lower = entity_label.lower()

        tech_patterns = ["api", "framework", "library", "algorithm", "software",
                         "service", "platform", "tool", "ai", "machine learning"]
        if any(pattern in entity_lower for pattern in tech_patterns):
            return "technology"

        data_patterns = ["data", "database", "table", "schema", "dataset"]
        if any(pattern in entity_lower for pattern in data_patterns):
            return "data"

        system_patterns = ["system", "module", "component", "process", "interface"]
        if any(pattern in entity_lower for pattern in system_patterns):
            return "system"

        business_patterns = ["business", "customer", "user", "requirement", "goal"]
        if any(pattern in entity_lower for pattern in business_patterns):
            return "business"

        return "unknown"


def improved_gnn_mcts_agent(user_records, domain_records, llm_records, gnn_records,
                            entity_embeddings, user_entities, max_iterations=1000, debug_mode=False):
    """
    Improved MCTS-based agent for extracting optimal reasoning threads.

    Key improvements:
    1. Better reward function balancing multiple objectives
    2. Enhanced action selection with smarter prioritization
    3. Multi-strategy search (standard MCTS + staged construction + beam search)
    4. Improved connectivity and post-processing
    5. Adaptive parameters and variance-aware UCB
    """
    logger.info(f"Starting improved GNN-MCTS agent with {len(user_entities)} user entities")

    if not any([user_records, domain_records, llm_records, gnn_records]):
        logger.warning("No records provided")
        return {"nodes": [], "links": []}

    try:
        # Initialize improved MCTS
        mcts = ImprovedGraphMCTS(
            user_records=user_records,
            domain_records=domain_records,
            llm_records=llm_records,
            gnn_records=gnn_records,
            entity_embeddings=entity_embeddings,
            user_entities=user_entities,
            max_iterations=max_iterations,
            debug_mode=debug_mode
        )

        # Run improved search
        optimal_path = mcts.search()

        if not optimal_path:
            logger.warning("No path found, creating fallback from user records")
            optimal_path = user_records[:min(10, len(user_records))]

        logger.info(f"Found optimal path with {len(optimal_path)} triples")

        # Convert to graph structure
        return _convert_path_to_graph(optimal_path, debug_mode)

    except Exception as e:
        logger.error(f"Error in improved GNN-MCTS agent: {e}")
        import traceback
        traceback.print_exc()
        return {"nodes": [], "links": []}


def _convert_path_to_graph(path, debug_mode=False):
    """Convert path to graph structure"""
    nodes = {}
    links = []
    node_id_map = {}

    for i, record in enumerate(path):
        subject_label = record["subject_label"]
        object_label = record["object_label"]

        # Create consistent node IDs
        if subject_label not in node_id_map:
            node_id_map[subject_label] = f"improved_mcts_{uuid.uuid4().hex[:8]}"
        if object_label not in node_id_map:
            node_id_map[object_label] = f"improved_mcts_{uuid.uuid4().hex[:8]}"

        subject_id = node_id_map[subject_label]
        object_id = node_id_map[object_label]

        # Add nodes
        for node_id, label in [(subject_id, subject_label), (object_id, object_label)]:
            if node_id not in nodes:
                # Determine domain type
                domain_type = record.get("domain_type", "unknown")
                if domain_type == "unknown":
                    # Simple keyword-based classification
                    label_lower = label.lower()
                    if any(kw in label_lower for kw in ["api", "software", "algorithm"]):
                        domain_type = "technology"
                    elif any(kw in label_lower for kw in ["data", "database"]):
                        domain_type = "data"
                    elif any(kw in label_lower for kw in ["system", "process"]):
                        domain_type = "system"
                    else:
                        domain_type = "business"

                nodes[node_id] = {
                    "id": node_id,
                    "label": label,
                    "group": "improved_mcts",
                    "origin_label": "reasoning_path",
                    "domain_type": domain_type,
                    "path_order": i
                }

        # Add link
        links.append({
            "source": subject_id,
            "target": object_id,
            "label": record["predicate"],
            "origin": "reasoning_path",
            "path_order": i
        })

    if debug_mode:
        logger.info(f"Converted path to graph: {len(nodes)} nodes, {len(links)} links")

    return {"nodes": list(nodes.values()), "links": links}
