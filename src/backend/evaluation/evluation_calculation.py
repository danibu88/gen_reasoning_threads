import math
import numpy as np
import networkx as nx
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import re
from collections import Counter
import statistics
from app import calculate_instruction_metrics


class ComprehensiveMetricsCalculator:
    def __init__(self):
        # Initialize models
        self.nlp = spacy.load("en_core_web_sm")
        self.classifier_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize GPT-2 for perplexity calculation
        try:
            self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            self.gpt2_model.eval()
        except Exception as e:
            print(f"Warning: Could not load GPT-2 for perplexity calculation: {e}")
            self.gpt2_model = None
            self.gpt2_tokenizer = None

        # Domain-specific term sets
        self.domain_terms = {
            "technology": ["api", "database", "framework", "algorithm", "python", "javascript",
                           "system", "interface", "model", "cloud", "server", "client", "docker",
                           "machine learning", "ai", "analytics", "neural", "transform", "optimization",
                           "deployment", "architecture", "microservices", "kubernetes", "devops"],
            "business": ["customer", "market", "revenue", "cost", "value proposition", "strategy",
                         "stakeholder", "roi", "kpi", "metrics", "growth", "scalability", "profit",
                         "investment", "budget", "resource", "productivity", "efficiency"],
            "data_science": ["dataset", "feature", "regression", "classification", "clustering",
                             "preprocessing", "visualization", "statistics", "correlation", "variance",
                             "distribution", "sampling", "validation", "cross-validation", "overfitting"],
            "healthcare": ["patient", "diagnosis", "treatment", "clinical", "medical", "therapeutic",
                           "healthcare", "hospital", "physician", "symptoms", "disease", "medication"],
            "finance": ["portfolio", "investment", "risk", "return", "asset", "equity", "bond",
                        "derivative", "volatility", "liquidity", "capital", "trading", "market"]
        }

        # Create domain embeddings for semantic similarity
        self.domain_embeddings = {}
        for domain, terms in self.domain_terms.items():
            domain_text = " ".join(terms)
            self.domain_embeddings[domain] = self.classifier_model.encode(domain_text, convert_to_tensor=True)

    def calculate_prompt_complexity(self, prompt):
        """Calculate prompt complexity metrics"""
        metrics = {}

        # 1. Token Count
        doc = self.nlp(prompt)
        metrics['token_count'] = len(doc)

        # 2. Syntactic Complexity
        metrics.update(self._calculate_syntactic_complexity(doc))

        # 3. Perplexity
        metrics['perplexity'] = self._calculate_perplexity(prompt)

        return metrics

    def _calculate_syntactic_complexity(self, doc):
        """Calculate syntactic complexity metrics"""
        sentences = list(doc.sents)
        if not sentences:
            return {
                'avg_sentence_length': 0,
                'max_dependency_depth': 0,
                'avg_dependency_depth': 0,
                'clause_density': 0,
                'syntactic_complexity_score': 0
            }

        sentence_lengths = [len(sent) for sent in sentences]
        dependency_depths = []
        total_clauses = 0

        for sent in sentences:
            # Calculate dependency depth for each sentence
            max_depth = 0
            for token in sent:
                depth = self._get_dependency_depth(token)
                max_depth = max(max_depth, depth)
            dependency_depths.append(max_depth)

            # Count clauses (simplified: count conjunctions and relative pronouns)
            clauses = len([token for token in sent if token.dep_ in ['conj', 'relcl', 'advcl', 'ccomp', 'xcomp']])
            total_clauses += clauses

        avg_sentence_length = statistics.mean(sentence_lengths)
        max_dependency_depth = max(dependency_depths) if dependency_depths else 0
        avg_dependency_depth = statistics.mean(dependency_depths) if dependency_depths else 0
        clause_density = total_clauses / len(sentences) if sentences else 0

        # Composite syntactic complexity score
        syntactic_complexity_score = (
                (avg_sentence_length / 20) * 0.3 +  # Normalized by typical sentence length
                (avg_dependency_depth / 5) * 0.4 +  # Normalized by typical depth
                (clause_density / 2) * 0.3  # Normalized by typical clause density
        )

        return {
            'avg_sentence_length': round(avg_sentence_length, 2),
            'max_dependency_depth': max_dependency_depth,
            'avg_dependency_depth': round(avg_dependency_depth, 2),
            'clause_density': round(clause_density, 2),
            'syntactic_complexity_score': round(syntactic_complexity_score, 4)
        }

    def _get_dependency_depth(self, token):
        """Calculate dependency depth for a token"""
        depth = 0
        current = token
        while current.head != current:  # Not root
            depth += 1
            current = current.head
            if depth > 50:  # Prevent infinite loops
                break
        return depth

    def _calculate_perplexity(self, text):
        """Calculate perplexity using GPT-2"""
        if not self.gpt2_model or not self.gpt2_tokenizer:
            return 0.0

        try:
            # Tokenize text
            encodings = self.gpt2_tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
            input_ids = encodings.input_ids

            # Calculate perplexity
            with torch.no_grad():
                outputs = self.gpt2_model(input_ids, labels=input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()

            return round(perplexity, 4)
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return 0.0

    def calculate_domain_specificity(self, prompt):
        """Calculate domain specificity metrics"""
        metrics = {}

        # 1. Domain Term Frequency
        domain_frequencies = {}
        prompt_lower = prompt.lower()

        for domain, terms in self.domain_terms.items():
            count = sum(1 for term in terms if term in prompt_lower)
            domain_frequencies[f'{domain}_term_frequency'] = count

        metrics.update(domain_frequencies)

        # Overall domain term density
        total_terms = sum(domain_frequencies.values())
        word_count = len(prompt.split())
        metrics['overall_domain_density'] = round(total_terms / word_count if word_count > 0 else 0, 4)

        # 2. Semantic Similarity to domain corpora
        prompt_embedding = self.classifier_model.encode(prompt, convert_to_tensor=True)

        domain_similarities = {}
        for domain, domain_embedding in self.domain_embeddings.items():
            similarity = util.cos_sim(prompt_embedding, domain_embedding).item()
            domain_similarities[f'{domain}_semantic_similarity'] = round(similarity, 4)

        metrics.update(domain_similarities)

        # Find dominant domain
        max_similarity = max(domain_similarities.values())
        dominant_domain = \
        [k.replace('_semantic_similarity', '') for k, v in domain_similarities.items() if v == max_similarity][0]
        metrics['dominant_domain'] = dominant_domain
        metrics['max_domain_similarity'] = max_similarity

        return metrics

    def calculate_clarity_coherence(self, prompt):
        """Calculate clarity and coherence metrics"""
        metrics = {}

        # 1. Readability Scores
        try:
            metrics['flesch_reading_ease'] = round(flesch_reading_ease(prompt), 2)
            metrics['flesch_kincaid_grade'] = round(flesch_kincaid_grade(prompt), 2)
        except:
            metrics['flesch_reading_ease'] = 0.0
            metrics['flesch_kincaid_grade'] = 0.0

        # 2. Logical Flow Assessment
        metrics.update(self._assess_logical_flow(prompt))

        return metrics

    def _assess_logical_flow(self, prompt):
        """Assess logical flow of the prompt"""
        doc = self.nlp(prompt)
        sentences = list(doc.sents)

        if len(sentences) < 2:
            return {
                'coherence_score': 1.0,
                'transition_word_density': 0.0,
                'sentence_similarity_variance': 0.0,
                'logical_flow_score': 1.0
            }

        # Count transition words/phrases
        transition_words = ['however', 'therefore', 'furthermore', 'moreover', 'consequently',
                            'additionally', 'meanwhile', 'nevertheless', 'thus', 'hence',
                            'first', 'second', 'finally', 'then', 'next', 'also', 'because']

        transition_count = sum(1 for word in prompt.lower().split() if word in transition_words)
        transition_density = transition_count / len(prompt.split())

        # Calculate sentence-to-sentence similarity
        sentence_embeddings = [self.classifier_model.encode(str(sent), convert_to_tensor=True) for sent in sentences]
        similarities = []

        for i in range(len(sentence_embeddings) - 1):
            sim = util.cos_sim(sentence_embeddings[i], sentence_embeddings[i + 1]).item()
            similarities.append(sim)

        similarity_variance = statistics.variance(similarities) if len(similarities) > 1 else 0.0
        avg_similarity = statistics.mean(similarities) if similarities else 0.0

        # Composite logical flow score
        logical_flow_score = (
                (transition_density * 2) * 0.3 +  # Higher transition density = better flow
                avg_similarity * 0.4 +  # Higher avg similarity = better coherence
                (1 - similarity_variance) * 0.3  # Lower variance = more consistent flow
        )

        return {
            'coherence_score': round(avg_similarity, 4),
            'transition_word_density': round(transition_density, 4),
            'sentence_similarity_variance': round(similarity_variance, 4),
            'logical_flow_score': round(max(0, min(1, logical_flow_score)), 4)
        }

    def calculate_graph_metrics(self, graph_data):
        """Calculate comprehensive graph metrics"""
        if not graph_data or not graph_data.get('nodes'):
            return self._empty_graph_metrics()

        # Create NetworkX graph
        G = nx.Graph()

        # Add nodes
        for node in graph_data.get('nodes', []):
            node_id = node.get('id', '')
            if node_id:
                G.add_node(node_id, **node)

        # Add edges
        for edge in graph_data.get('links', []):
            source = edge.get('source', '')
            target = edge.get('target', '')
            if source and target and source in G.nodes() and target in G.nodes():
                G.add_edge(source, target, **edge)

        if G.number_of_nodes() == 0:
            return self._empty_graph_metrics()

        metrics = {}

        # 1. Structural Metrics
        metrics.update(self._calculate_structural_metrics(G))

        # 2. Centrality Measures
        metrics.update(self._calculate_centrality_measures(G))

        # 3. Consistency and Coherence
        metrics.update(self._calculate_graph_consistency(G))

        return metrics

    def _calculate_structural_metrics(self, G):
        """Calculate structural graph metrics"""
        node_count = G.number_of_nodes()
        edge_count = G.number_of_edges()

        if node_count == 0:
            return {'node_count': 0, 'edge_count': 0, 'average_degree': 0}

        degrees = [d for n, d in G.degree()]
        average_degree = statistics.mean(degrees) if degrees else 0

        return {
            'node_count': node_count,
            'edge_count': edge_count,
            'average_degree': round(average_degree, 2),
            'degree_variance': round(statistics.variance(degrees) if len(degrees) > 1 else 0, 2),
            'max_degree': max(degrees) if degrees else 0,
            'min_degree': min(degrees) if degrees else 0
        }

    def _calculate_centrality_measures(self, G):
        """Calculate centrality measures"""
        if G.number_of_nodes() == 0:
            return {
                'avg_degree_centrality': 0,
                'max_degree_centrality': 0,
                'avg_betweenness_centrality': 0,
                'max_betweenness_centrality': 0,
                'avg_closeness_centrality': 0,
                'max_closeness_centrality': 0
            }

        # Degree centrality
        degree_centrality = nx.degree_centrality(G)
        degree_values = list(degree_centrality.values())

        # Betweenness centrality
        try:
            betweenness_centrality = nx.betweenness_centrality(G)
            betweenness_values = list(betweenness_centrality.values())
        except:
            betweenness_values = [0] * len(degree_values)

        # Closeness centrality (only for connected components)
        try:
            if nx.is_connected(G):
                closeness_centrality = nx.closeness_centrality(G)
                closeness_values = list(closeness_centrality.values())
            else:
                closeness_values = [0] * len(degree_values)
        except:
            closeness_values = [0] * len(degree_values)

        return {
            'avg_degree_centrality': round(statistics.mean(degree_values), 4),
            'max_degree_centrality': round(max(degree_values), 4),
            'avg_betweenness_centrality': round(statistics.mean(betweenness_values), 4),
            'max_betweenness_centrality': round(max(betweenness_values), 4),
            'avg_closeness_centrality': round(statistics.mean(closeness_values), 4),
            'max_closeness_centrality': round(max(closeness_values), 4)
        }

    def _calculate_graph_consistency(self, G):
        """Calculate graph consistency and coherence metrics"""
        if G.number_of_nodes() < 3:
            return {
                'clustering_coefficient': 0,
                'graph_density': 0,
                'path_length_variance': 0,
                'connected_components': 1 if G.number_of_nodes() > 0 else 0,
                'largest_component_ratio': 1.0 if G.number_of_nodes() > 0 else 0.0
            }

        # Clustering coefficient
        try:
            clustering_coeff = nx.average_clustering(G)
        except:
            clustering_coeff = 0

        # Graph density
        density = nx.density(G)

        # Path length analysis
        path_lengths = []
        connected_components = list(nx.connected_components(G))

        for component in connected_components:
            if len(component) > 1:
                subgraph = G.subgraph(component)
                try:
                    lengths = dict(nx.all_pairs_shortest_path_length(subgraph))
                    for source in lengths:
                        for target, length in lengths[source].items():
                            if source != target:
                                path_lengths.append(length)
                except:
                    pass

        path_length_variance = statistics.variance(path_lengths) if len(path_lengths) > 1 else 0

        # Component analysis
        num_components = len(connected_components)
        largest_component_size = max(len(comp) for comp in connected_components) if connected_components else 0
        largest_component_ratio = largest_component_size / G.number_of_nodes() if G.number_of_nodes() > 0 else 0

        return {
            'clustering_coefficient': round(clustering_coeff, 4),
            'graph_density': round(density, 4),
            'path_length_variance': round(path_length_variance, 4),
            'avg_path_length': round(statistics.mean(path_lengths) if path_lengths else 0, 2),
            'connected_components': num_components,
            'largest_component_ratio': round(largest_component_ratio, 4)
        }

    def _empty_graph_metrics(self):
        """Return empty graph metrics when graph is empty or invalid"""
        return {
            'node_count': 0, 'edge_count': 0, 'average_degree': 0, 'degree_variance': 0,
            'max_degree': 0, 'min_degree': 0, 'avg_degree_centrality': 0,
            'max_degree_centrality': 0, 'avg_betweenness_centrality': 0,
            'max_betweenness_centrality': 0, 'avg_closeness_centrality': 0,
            'max_closeness_centrality': 0, 'clustering_coefficient': 0,
            'graph_density': 0, 'path_length_variance': 0, 'avg_path_length': 0,
            'connected_components': 0, 'largest_component_ratio': 0
        }

    def calculate_all_metrics(self, prompt, gnn_subgraph, mcts_subgraph):
        """Calculate all metrics for a given prompt and graphs"""
        all_metrics = {}

        # Prompt metrics
        all_metrics['prompt_complexity'] = self.calculate_prompt_complexity(prompt)
        all_metrics['domain_specificity'] = self.calculate_domain_specificity(prompt)
        all_metrics['clarity_coherence'] = self.calculate_clarity_coherence(prompt)

        # Graph metrics
        all_metrics['gnn_graph_metrics'] = self.calculate_graph_metrics(gnn_subgraph)
        all_metrics['mcts_graph_metrics'] = self.calculate_graph_metrics(mcts_subgraph)

        return all_metrics


# Function to integrate with existing system
def calculate_comprehensive_metrics(prompt, user_nodes, gnn_instructions, mcts_instructions,
                                    rmodel_instructions, gnn_subgraph, mcts_subgraph):
    """
    Enhanced version of the original calculate_instruction_metrics function
    that includes comprehensive prompt and graph metrics
    """
    # Initialize calculator
    calculator = ComprehensiveMetricsCalculator()

    # Calculate comprehensive metrics
    comprehensive_metrics = calculator.calculate_all_metrics(prompt, gnn_subgraph, mcts_subgraph)

    # Calculate original instruction metrics
    original_metrics = calculate_instruction_metrics(prompt, user_nodes, gnn_instructions,
                                                     mcts_instructions, rmodel_instructions)

    # Combine all metrics
    combined_metrics = {
        'instruction_metrics': original_metrics,
        'prompt_metrics': comprehensive_metrics
    }

    return combined_metrics


# Flatten metrics for CSV export
def flatten_metrics_for_csv(metrics):
    """Flatten nested metrics dictionary for CSV export"""
    flattened = {}

    def flatten_dict(d, parent_key=''):
        for k, v in d.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            if isinstance(v, dict):
                flatten_dict(v, new_key)
            else:
                flattened[new_key] = v

    flatten_dict(metrics)
    return flattened