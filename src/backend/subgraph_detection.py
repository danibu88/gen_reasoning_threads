"""
Improved subgraph detection module with better handling of isolated nodes.

This module provides functions for detecting meaningful subgraphs in knowledge graphs
using community detection and other advanced partitioning techniques.
"""

import networkx as nx
import math
from typing import List, Dict, Set, Tuple, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def identify_predicate_nodes(nodes: List[Dict]) -> Set[str]:
    """
    Identify predicate/relationship nodes in the graph that often act as hubs.

    Args:
        nodes: List of node dictionaries from the graph

    Returns:
        Set of node IDs that are predicates
    """
    predicate_prefixes = ('ns0__', 'ns2__', 'rdfs__', 'owl__', 'rdf__')
    return {node['id'] for node in nodes
            if any(node['id'].startswith(prefix) for prefix in predicate_prefixes)}

def calculate_node_degrees(links: List[Dict]) -> Dict[str, int]:
    """
    Calculate the degree of each node in the graph.

    Args:
        links: List of link dictionaries from the graph

    Returns:
        Dictionary mapping node IDs to their degrees
    """
    node_degrees = {}
    for link in links:
        node_degrees[link['source']] = node_degrees.get(link['source'], 0) + 1
        node_degrees[link['target']] = node_degrees.get(link['target'], 0) + 1
    return node_degrees

def identify_isolated_nodes(nodes: List[Dict], links: List[Dict]) -> Set[str]:
    """
    Identify nodes that have no connections.

    Args:
        nodes: List of node dictionaries
        links: List of link dictionaries

    Returns:
        Set of node IDs that are isolated (have no connections)
    """
    all_node_ids = {node['id'] for node in nodes}
    connected_nodes = set()

    for link in links:
        connected_nodes.add(link['source'])
        connected_nodes.add(link['target'])

    return all_node_ids - connected_nodes

def handle_isolated_nodes(graph: Dict[str, List], subgraphs: List[Dict]) -> List[Dict]:
    """
    Distribute isolated nodes to existing subgraphs based on similarity.

    Args:
        graph: Original graph with 'nodes' and 'links'
        subgraphs: List of subgraph dictionaries

    Returns:
        Updated list of subgraphs with isolated nodes assigned
    """
    if not subgraphs:
        return subgraphs

    # Find isolated nodes
    isolated_node_ids = identify_isolated_nodes(graph['nodes'], graph['links'])
    isolated_nodes = [node for node in graph['nodes'] if node['id'] in isolated_node_ids]

    logger.info(f"[Isolated] Found {len(isolated_nodes)} isolated nodes")

    if not isolated_nodes:
        return subgraphs

    # If we have too many isolated nodes, limit to a reasonable number
    max_isolated_nodes = 20
    if len(isolated_nodes) > max_isolated_nodes:
        logger.info(f"[Isolated] Limiting to {max_isolated_nodes} isolated nodes")
        isolated_nodes = isolated_nodes[:max_isolated_nodes]

    # Distribute nodes among subgraphs (round-robin for simplicity)
    for i, node in enumerate(isolated_nodes):
        # Add to subgraph with fewest nodes for balance
        subgraphs.sort(key=lambda x: len(x['nodes']))
        subgraph = subgraphs[0]

        # Add the node
        subgraph['nodes'].append(node)
        subgraph['size'] = len(subgraph['nodes'])

        logger.info(f"[Isolated] Added node {node['id']} to subgraph {subgraphs.index(subgraph) + 1}")

    return subgraphs

def build_weighted_graph(nodes: List[Dict],
                         links: List[Dict],
                         predicate_nodes: Optional[Set[str]] = None,
                         node_degrees: Optional[Dict[str, int]] = None) -> nx.Graph:
    """
    Build a networkx graph with weighted edges to reduce hub node influence.

    Args:
        nodes: List of node dictionaries
        links: List of link dictionaries
        predicate_nodes: Optional pre-identified predicate nodes
        node_degrees: Optional pre-calculated node degrees

    Returns:
        A networkx Graph with nodes and weighted edges
    """
    G = nx.Graph()  # Undirected for community detection

    # Add nodes with attributes
    for node in nodes:
        G.add_node(node['id'], **node)

    # Calculate node degrees if not provided
    if node_degrees is None:
        node_degrees = calculate_node_degrees(links)

    # Get predicate nodes if not provided
    if predicate_nodes is None:
        predicate_nodes = identify_predicate_nodes(nodes)

    # Add edges with weights inversely proportional to node degree
    for link in links:
        source, target = link['source'], link['target']

        # Calculate edge weight - downweight connections through hub nodes
        source_degree = node_degrees.get(source, 1)
        target_degree = node_degrees.get(target, 1)

        # Base weight starts at 1.0
        weight = 1.0

        # Apply degree-based penalty for hub nodes
        if source_degree > 5 or target_degree > 5:
            weight /= math.sqrt(source_degree * target_degree) + 0.1

        # Apply additional penalty for predicate nodes that often act as hubs
        if source in predicate_nodes or target in predicate_nodes:
            weight *= 0.7

        # Add the edge with weight and original attributes
        G.add_edge(source, target, weight=weight, **link)

    return G

def extract_communities(graph: Dict[str, List],
                        min_nodes: int = 7,
                        max_nodes: int = 50,
                        max_communities: int = 3,
                        handle_isolated: bool = True) -> List[Dict]:
    """
    Extract meaningful subgraphs using community detection.

    Args:
        graph: Dictionary with 'nodes' and 'links' keys
        min_nodes: Minimum nodes required in a community
        max_nodes: Maximum nodes allowed in a community
        max_communities: Maximum number of communities to return
        handle_isolated: Whether to distribute isolated nodes to subgraphs

    Returns:
        List of subgraphs, each as a dict with 'nodes' and 'links'
    """
    logger.info(f"[Community] Starting community detection on {len(graph['nodes'])} nodes")

    # Check for isolated nodes
    isolated_node_ids = identify_isolated_nodes(graph['nodes'], graph['links'])
    if isolated_node_ids:
        logger.info(f"[Community] Found {len(isolated_node_ids)} isolated nodes")

    # Calculate node degrees and identify predicates
    node_degrees = calculate_node_degrees(graph['links'])
    predicate_nodes = identify_predicate_nodes(graph['nodes'])
    logger.info(f"[Community] Found {len(predicate_nodes)} predicate nodes")

    # Build weighted graph
    G = build_weighted_graph(
        graph['nodes'],
        graph['links'],
        predicate_nodes,
        node_degrees
    )

    # Try Louvain community detection (the most effective method)
    try:
        import community as community_louvain

        partition = community_louvain.best_partition(G)

        # Organize nodes by community
        communities = {}
        for node_id, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = set()
            communities[community_id].add(node_id)

        logger.info(f"[Community] Found {len(communities)} communities")
        community_sizes = [len(nodes) for nodes in communities.values()]
        logger.info(f"[Community] Community sizes: {sorted(community_sizes, reverse=True)}")

        # Filter communities by size
        valid_communities = {
            comm_id: nodes for comm_id, nodes in communities.items()
            if min_nodes <= len(nodes) <= max_nodes
        }

        logger.info(f"[Community] Found {len(valid_communities)} valid communities with {min_nodes}-{max_nodes} nodes")

        result_subgraphs = _process_valid_communities(graph, G, valid_communities, max_communities, min_nodes, max_nodes)

    except ImportError:
        logger.info("[Community] Louvain method not available, falling back to edge betweenness")
        result_subgraphs = _fallback_to_edge_betweenness(graph, G, min_nodes, max_nodes, max_communities)

    # Handle isolated nodes if requested
    if handle_isolated and result_subgraphs:
        result_subgraphs = handle_isolated_nodes(graph, result_subgraphs)

    return result_subgraphs

def _process_valid_communities(graph: Dict[str, List],
                              G: nx.Graph,
                              valid_communities: Dict[Any, Set[str]],
                              max_communities: int,
                              min_nodes: int,
                              max_nodes: int) -> List[Dict]:
    """
    Process valid communities into subgraphs.

    Args:
        graph: Original graph dictionary with 'nodes' and 'links'
        G: NetworkX graph
        valid_communities: Dictionary of community_id -> set of node_ids
        max_communities: Maximum number of communities to return
        min_nodes: Minimum nodes required in a community
        max_nodes: Maximum nodes allowed in a community

    Returns:
        List of subgraphs as dictionaries
    """
    if not valid_communities:
        return _fallback_to_edge_betweenness(graph, G, min_nodes, max_nodes, max_communities)

    # Create result subgraphs
    result_subgraphs = []
    for comm_id, node_ids in list(valid_communities.items())[:max_communities]:
        subgraph_nodes = [node for node in graph['nodes'] if node['id'] in node_ids]
        subgraph_links = [
            link for link in graph['links']
            if link['source'] in node_ids and link['target'] in node_ids
        ]

        # Verify subgraph is connected
        sg = nx.Graph()
        sg.add_nodes_from(node_ids)
        sg.add_edges_from([(link['source'], link['target']) for link in subgraph_links])

        if nx.is_connected(sg):
            result_subgraphs.append({
                "nodes": subgraph_nodes,
                "links": subgraph_links,
                "community_id": comm_id,
                "size": len(subgraph_nodes)
            })
        else:
            # If not connected, use only the largest component
            largest_comp = max(nx.connected_components(sg), key=len)
            filtered_nodes = [node for node in subgraph_nodes if node['id'] in largest_comp]
            filtered_links = [
                link for link in subgraph_links
                if link['source'] in largest_comp and link['target'] in largest_comp
            ]

            result_subgraphs.append({
                "nodes": filtered_nodes,
                "links": filtered_links,
                "community_id": comm_id,
                "size": len(filtered_nodes)
            })

    logger.info(f"[Community] Returning {len(result_subgraphs)} subgraphs")
    for i, sg in enumerate(result_subgraphs):
        logger.info(f"[Community] Subgraph {i+1}: {len(sg['nodes'])} nodes, {len(sg['links'])} links")

    return result_subgraphs

def _fallback_to_edge_betweenness(graph: Dict[str, List],
                                 G: nx.Graph,
                                 min_nodes: int,
                                 max_nodes: int,
                                 max_communities: int) -> List[Dict]:
    """
    Fallback to edge betweenness when Louvain method fails.

    Args:
        graph: Original graph dictionary with 'nodes' and 'links'
        G: NetworkX graph
        min_nodes: Minimum nodes required in a community
        max_nodes: Maximum nodes allowed in a community
        max_communities: Maximum number of communities to return

    Returns:
        List of subgraphs as dictionaries
    """
    logger.info("[Community] Trying edge betweenness clustering")

    # Identify and remove bridges between communities
    try:
        betweenness = nx.edge_betweenness_centrality(G, weight='weight')
        edges_to_remove = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:int(G.number_of_edges() * 0.1)]

        G_cut = G.copy()
        for edge, _ in edges_to_remove:
            G_cut.remove_edge(*edge)

        # Get new components after edge removal
        new_components = list(nx.connected_components(G_cut))
        new_components.sort(key=len, reverse=True)

        logger.info(f"[Community] Edge betweenness found {len(new_components)} components")
        logger.info(f"[Community] Component sizes: {sorted([len(comp) for comp in new_components], reverse=True)}")

        valid_communities = {
            i: comp for i, comp in enumerate(new_components)
            if min_nodes <= len(comp) <= max_nodes
        }

        if valid_communities:
            return _process_valid_communities(
                graph, G, valid_communities, max_communities, min_nodes, max_nodes
            )

        # If no valid communities, try spectral clustering
        return _fallback_to_spectral_clustering(
            graph, G, min_nodes, max_nodes, max_communities, new_components
        )

    except Exception as e:
        logger.warning(f"[Community] Edge betweenness failed: {str(e)}")
        return _fallback_to_spectral_clustering(
            graph, G, min_nodes, max_nodes, max_communities
        )

def _fallback_to_spectral_clustering(graph: Dict[str, List],
                                    G: nx.Graph,
                                    min_nodes: int,
                                    max_nodes: int,
                                    max_communities: int,
                                    components: List[Set[str]] = None) -> List[Dict]:
    """
    Fallback to spectral clustering when other methods fail.

    Args:
        graph: Original graph dictionary with 'nodes' and 'links'
        G: NetworkX graph
        min_nodes: Minimum nodes required in a community
        max_nodes: Maximum nodes allowed in a community
        max_communities: Maximum number of communities to return
        components: Optional list of components to process

    Returns:
        List of subgraphs as dictionaries
    """
    # Try spectral clustering
    try:
        from sklearn.cluster import SpectralClustering
        import numpy as np

        logger.info("[Community] Trying spectral clustering")

        # Create adjacency matrix
        nodes_list = list(G.nodes())
        node_indices = {node: i for i, node in enumerate(nodes_list)}

        # Get adjacency matrix with weights
        adjacency = nx.to_numpy_array(G, weight='weight')

        # Apply spectral clustering
        n_clusters = min(10, len(nodes_list) // 10 + 1)  # Reasonable number of clusters
        clustering = SpectralClustering(n_clusters=n_clusters,
                                       affinity='precomputed',
                                       assign_labels='discretize',
                                       random_state=42).fit(adjacency)

        # Organize nodes by cluster
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = set()
            clusters[label].add(nodes_list[i])

        logger.info(f"[Community] Spectral clustering found {len(clusters)} clusters")
        logger.info(f"[Community] Cluster sizes: {sorted([len(nodes) for nodes in clusters.values()], reverse=True)}")

        # Filter clusters by size
        valid_clusters = {
            cluster_id: nodes for cluster_id, nodes in clusters.items()
            if min_nodes <= len(nodes) <= max_nodes
        }

        if valid_clusters:
            return _process_valid_communities(
                graph, G, valid_clusters, max_communities, min_nodes, max_nodes
            )

        # Last fallback - use connected components with size filtering
        return _final_fallback(graph, G, min_nodes, max_nodes, max_communities, components)

    except Exception as e:
        logger.warning(f"[Community] Spectral clustering failed: {str(e)}")
        return _final_fallback(graph, G, min_nodes, max_nodes, max_communities, components)


def _final_fallback(graph: Dict[str, List],
                    G: nx.Graph,
                    min_nodes: int,
                    max_nodes: int,
                    max_communities: int,
                    components: List[Set[str]] = None) -> List[Dict]:
    """
    Final fallback using basic size capping of components.

    Args:
        graph: Original graph dictionary with 'nodes' and 'links'
        G: NetworkX graph
        min_nodes: Minimum nodes required in a community
        max_nodes: Maximum nodes allowed in a community
        max_communities: Maximum number of communities to return
        components: Optional list of components to process

    Returns:
        List of subgraphs as dictionaries
    """
    logger.info("[Community] Using basic connected components with size filtering")

    if not components:
        components = list(nx.connected_components(G))

    components.sort(key=len, reverse=True)

    result_subgraphs = []
    for i, comp in enumerate(components[:max_communities]):
        if len(comp) >= min_nodes:
            nodes_subset = comp
            if len(comp) > max_nodes:
                # Take central nodes
                subgraph = G.subgraph(comp)
                try:
                    centrality = nx.eigenvector_centrality_numpy(subgraph, weight='weight')
                except:
                    centrality = nx.degree_centrality(subgraph)  # Fallback

                central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
                nodes_subset = {node for node, _ in central_nodes}

            subgraph_nodes = [node for node in graph['nodes'] if node['id'] in nodes_subset]
            subgraph_links = [
                link for link in graph['links']
                if link['source'] in nodes_subset and link['target'] in nodes_subset
            ]

            result_subgraphs.append({
                "nodes": subgraph_nodes,
                "links": subgraph_links,
                "community_id": i,
                "size": len(subgraph_nodes)
            })

    # If we still have nothing, just take a slice of the largest component
    if not result_subgraphs and components:
        largest = list(components[0])
        nodes_subset = largest[:max_nodes] if len(largest) > max_nodes else largest

        subgraph_nodes = [node for node in graph['nodes'] if node['id'] in nodes_subset]
        subgraph_links = [
            link for link in graph['links']
            if link['source'] in nodes_subset and link['target'] in nodes_subset
        ]

        result_subgraphs.append({
            "nodes": subgraph_nodes,
            "links": subgraph_links,
            "community_id": 0,
            "size": len(subgraph_nodes)
        })

    logger.info(f"[Community] Final fallback returning {len(result_subgraphs)} subgraphs")
    for i, sg in enumerate(result_subgraphs):
        logger.info(f"[Community] Subgraph {i + 1}: {len(sg['nodes'])} nodes, {len(sg['links'])} links")

    return result_subgraphs


def analyze_graph_connectivity(graph: Dict[str, List]) -> Dict:
    """
    Analyze the connectivity of the graph for debugging purposes.

    Args:
        graph: Dictionary with 'nodes' and 'links'

    Returns:
        Dictionary with analysis results
    """
    node_ids = {node['id'] for node in graph['nodes']}

    # Calculate node degrees
    node_degrees = calculate_node_degrees(graph['links'])

    # Find isolated nodes
    connected_nodes = set(node_degrees.keys())
    isolated_nodes = node_ids - connected_nodes

    # Find predicate nodes
    predicate_nodes = identify_predicate_nodes(graph['nodes'])

    # Find high-degree nodes (potential hubs)
    hub_threshold = 5
    hub_nodes = {node: degree for node, degree in node_degrees.items() if degree >= hub_threshold}

    # Calculate average degree
    avg_degree = sum(node_degrees.values()) / len(node_degrees) if node_degrees else 0

    # Build a networkx graph for component analysis
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    G.add_edges_from([(link['source'], link['target']) for link in graph['links']])

    # Analyze connected components
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)

    return {
        "total_nodes": len(node_ids),
        "total_links": len(graph['links']),
        "isolated_nodes": len(isolated_nodes),
        "isolated_node_ids": list(isolated_nodes),
        "predicate_nodes": len(predicate_nodes),
        "predicate_node_ids": list(predicate_nodes),
        "hub_nodes": len(hub_nodes),
        "hub_node_ids": list(hub_nodes.keys()),
        "average_degree": avg_degree,
        "max_degree": max(node_degrees.values()) if node_degrees else 0,
        "connected_components": len(components),
        "component_sizes": [len(comp) for comp in components],
    }