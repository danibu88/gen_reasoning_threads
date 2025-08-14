import React, {useState, useEffect, useMemo, useCallback} from 'react';
import {useLocation} from 'react-router-dom';
import {
    Container,
    Grid,
    Typography,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Button,
    ButtonGroup,
    Paper,
    CircularProgress, List, ListItem, ListItemIcon, ListItemText, Divider, Box, Chip
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import Graph from "./Graph";
import APIService from './APIService';
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord';
import {useStyles} from './styles';
import {styled} from '@mui/material/styles';
import * as d3 from "d3";
import InstructionComparison from './InstructionComparison';

const CombinedResults = () => {
        const classes = useStyles();
        const {state} = useLocation();
        const [topPapers, setTopPapers] = useState([]);
        const [restPapers, setRestPapers] = useState([]);
        const [allPapers, setAllPapers] = useState([]);
        const [graphDescription, setGraphDescription] = useState("");
        const [error, setError] = useState(null);
        const [papersSummary, setPapersSummary] = useState("");
        const [isLoading, setIsLoading] = useState(true);
        const [isSummaryLoading, setIsSummaryLoading] = useState(false);
        const [isSummaryGenerated, setIsSummaryGenerated] = useState(false);
        const [highlightedPrompt, setHighlightedPrompt] = useState('');
        let [searchData, setSearchData] = useState(state?.search || {});
        const HighlightedText = styled('span')(({theme}) => ({
            backgroundColor: theme.palette.warning.light,
            padding: '0 4px',
            borderRadius: '2px',
        }));
        const nodeColors = {
            'data': '#377eb8',
            'system': '#4daf4a',
            'business': '#984ea3',
            'technology': '#ff7f00',
            'other': '#999999'
        };

        const [useConsolidated, setUseConsolidated] = useState(true);
        const [activeSubgraph, setActiveSubgraph] = useState("combined");
        const [highlightTraversal, setHighlightTraversal] = useState(false);
        const [useGraphForInstructions, setUseGraphForInstructions] = useState('mcts'); // 'mcts' or 'combined'
        const consolidateGraph = (nodes, links, maxNodes = 40) => {
            if (!nodes || !Array.isArray(nodes) || nodes.length === 0 ||
                !links || !Array.isArray(links) || links.length === 0) {
                console.warn("No valid nodes or links to consolidate");
                return {nodes: [], links: []};
            }

            console.log("Consolidating graph with", nodes.length, "nodes and", links.length, "links");

            try {
                // Step 1: Create a map of node labels to aggregated nodes
                const nodeMap = {};

                // First collect all nodes by label
                nodes.forEach(node => {
                    const label = node.label;
                    if (!label) return; // Skip nodes without labels

                    if (!nodeMap[label]) {
                        nodeMap[label] = {
                            ...node,
                            id: `consolidated_${label.replace(/\s+/g, '_')}`,
                            originalIds: [node.id],
                            count: 1,
                            origins: [node.origin_label || node.group || "other"],
                            // Preserve domain_type for coloring
                            domain_type: node.domain_type || "other"
                        };
                    } else {
                        nodeMap[label].originalIds.push(node.id);
                        nodeMap[label].count += 1;

                        // Track all origins
                        const origin = node.origin_label || node.group || "other";
                        if (origin && !nodeMap[label].origins.includes(origin)) {
                            nodeMap[label].origins.push(origin);
                        }

                        // Use the most specific group (user > gnn > llm > connection > domain)
                        const groupPriority = {
                            "user": 5,
                            "gnn": 4,
                            "llm": 3,
                            "connection": 2,
                            "domain": 1,
                            "other": 0
                        };

                        const currentGroup = nodeMap[label].group || "other";
                        const newGroup = node.group || "other";

                        if (groupPriority[newGroup] > groupPriority[currentGroup]) {
                            nodeMap[label].group = newGroup;
                            nodeMap[label].origin_label = newGroup;
                        }

                        // Prefer more specific domain_type if available
                        if ((node.domain_type === "business" || node.domain_type === "system" ||
                                node.domain_type === "data" || node.domain_type === "technology") &&
                            nodeMap[label].domain_type === "other") {
                            nodeMap[label].domain_type = node.domain_type;
                        }
                    }
                });

                // Create a map from original ID to consolidated ID
                const idMap = {};
                nodes.forEach(node => {
                    if (node.label) {
                        idMap[node.id] = `consolidated_${node.label.replace(/\s+/g, '_')}`;
                    }
                });

                // Step 2: Consolidate links
                const linkMap = {};

                links.forEach(link => {
                    // Skip links where source or target is missing from idMap
                    if (!idMap[link.source] || !idMap[link.target]) {
                        return;
                    }

                    const source = idMap[link.source];
                    const target = idMap[link.target];

                    // Skip self-links created by consolidation
                    if (source === target) return;

                    const key = `${source}-${target}-${link.label || 'unknown'}`;

                    if (!linkMap[key]) {
                        linkMap[key] = {
                            ...link,
                            source,
                            target,
                            count: 1,
                            weight: 1,
                            origins: [link.origin || "other"]
                        };
                    } else {
                        linkMap[key].count += 1;
                        linkMap[key].weight = Math.min(5, linkMap[key].count);

                        if (link.origin && !linkMap[key].origins.includes(link.origin)) {
                            linkMap[key].origins.push(link.origin);
                        }
                    }
                });

                // Step 3: Calculate node importance and filter if too many nodes
                let consolidatedNodes = Object.values(nodeMap);

                // If we have too many nodes, prioritize the most connected ones
                if (consolidatedNodes.length > maxNodes) {
                    // Calculate node importance based on connection count
                    const nodeConnections = {};
                    Object.values(linkMap).forEach(link => {
                        nodeConnections[link.source] = (nodeConnections[link.source] || 0) + 1;
                        nodeConnections[link.target] = (nodeConnections[link.target] || 0) + 1;
                    });

                    // Add connection count as importance to nodes
                    consolidatedNodes = consolidatedNodes.map(node => ({
                        ...node,
                        importance: nodeConnections[node.id] || 0
                    }));

                    // Sort by importance (connection count) and take top maxNodes
                    consolidatedNodes.sort((a, b) => b.importance - a.importance);
                    consolidatedNodes = consolidatedNodes.slice(0, maxNodes);

                    // Get the IDs of filtered nodes
                    const filteredNodeIds = new Set(consolidatedNodes.map(node => node.id));

                    // Filter links to only include connections between remaining nodes
                    const consolidatedLinks = Object.values(linkMap).filter(link =>
                        filteredNodeIds.has(link.source) && filteredNodeIds.has(link.target)
                    );

                    return {
                        nodes: consolidatedNodes,
                        links: consolidatedLinks
                    };
                }

                // No need to filter, return all consolidated nodes and links
                const consolidatedLinks = Object.values(linkMap);

                console.log("Consolidated to", consolidatedNodes.length, "nodes and", consolidatedLinks.length, "links");

                return {
                    nodes: consolidatedNodes,
                    links: consolidatedLinks
                };
            } catch (error) {
                console.error("Error in consolidateGraph:", error);
                return {nodes, links};  // Return original on error
            }
        };

        //mcts functions
        // Calculate path coherence score
        const calculateCoherence = useCallback((gnnGraph, mctsPath) => {
            if (!gnnGraph || !mctsPath || !gnnGraph.links || !mctsPath.links ||
                gnnGraph.links.length === 0 || mctsPath.links.length === 0) {
                return "N/A";
            }

            try {
                // Count connected links in MCTS path (links that form a chain)
                const mctsLinks = mctsPath.links;
                const connectedCount = mctsLinks.reduce((count, link, index) => {
                    if (index === 0) return count;

                    const prevLink = mctsLinks[index - 1];
                    // Check if current link connects to previous link
                    if (link.source === prevLink.target ||
                        link.target === prevLink.source ||
                        link.source === prevLink.source ||
                        link.target === prevLink.target) {
                        return count + 1;
                    }
                    return count;
                }, 0);

                // Calculate coherence as percentage of connected links
                const coherenceScore = mctsLinks.length > 1
                    ? (connectedCount / (mctsLinks.length - 1)) * 100
                    : 100;

                return `${Math.round(coherenceScore)}%`;
            } catch (error) {
                console.error("Error calculating coherence:", error);
                return "Error";
            }
        }, []);

        // Calculate coverage of user concepts
        const calculateCoverage = useCallback((userRecords, mctsPath) => {
            if (!userRecords || !mctsPath || !mctsPath.nodes ||
                userRecords.length === 0 || mctsPath.nodes.length === 0) {
                return "N/A";
            }

            try {
                // Extract user entities
                const userEntities = new Set();
                userRecords.forEach(record => {
                    userEntities.add(record.subject_label);
                    userEntities.add(record.object_label);
                });

                // Check how many user entities are in MCTS path
                const mctsEntities = new Set(mctsPath.nodes.map(node => node.label));

                let matchCount = 0;
                userEntities.forEach(entity => {
                    if (mctsEntities.has(entity)) {
                        matchCount++;
                    }
                });

                // Calculate coverage percentage
                const coverageScore = (matchCount / userEntities.size) * 100;
                return `${Math.round(coverageScore)}%`;
            } catch (error) {
                console.error("Error calculating coverage:", error);
                return "Error";
            }
        }, []);

        // Calculate simplification score
        const calculateSimplification = useCallback((gnnGraph, mctsPath) => {
            if (!gnnGraph || !mctsPath || !gnnGraph.nodes || !mctsPath.nodes ||
                gnnGraph.nodes.length === 0) {
                return "N/A";
            }

            try {
                // Calculate reduction in nodes and links while preserving important information
                const gnnNodeCount = gnnGraph.nodes.length;
                const mctsNodeCount = mctsPath.nodes.length;

                // If MCTS has no nodes, return 0
                if (mctsNodeCount === 0) return "0%";

                // Calculate the reduction percentage
                const reductionPercentage = ((gnnNodeCount - mctsNodeCount) / gnnNodeCount) * 100;

                // A good simplification reduces size but not too much
                // Range from 0 to 100, with optimal at 40-70% reduction
                let simplificationScore;

                if (reductionPercentage < 0) {
                    // MCTS is larger than GNN (not simplified)
                    simplificationScore = "0%";
                } else if (reductionPercentage > 95) {
                    // Too much reduction (lost too much information)
                    simplificationScore = "Low";
                } else {
                    // Normal range
                    simplificationScore = `${Math.round(reductionPercentage)}%`;
                }

                return simplificationScore;
            } catch (error) {
                console.error("Error calculating simplification:", error);
                return "Error";
            }
        }, []);

        const calculateOverallQuality = useCallback((gnnGraph, mctsPath, userRecords) => {
            if (!gnnGraph || !mctsPath || !userRecords) {
                return "N/A";
            }

            try {
                // Get the numeric values from the other metrics
                const coherenceStr = calculateCoherence(gnnGraph, mctsPath);
                const coverageStr = calculateCoverage(userRecords, mctsPath);
                const simplificationStr = calculateSimplification(gnnGraph, mctsPath);

                // Convert to numbers
                const coherence = parseInt(coherenceStr, 10) || 0;
                const coverage = parseInt(coverageStr, 10) || 0;
                const simplification = parseInt(simplificationStr, 10) || 0;

                // Weight the metrics (can be adjusted based on importance)
                const weightedScore = (
                    coherence * 0.4 +
                    coverage * 0.4 +
                    (simplification > 0 && simplification < 95 ? simplification * 0.2 : 0)
                );

                // Convert to a 0-10 scale
                const finalScore = Math.min(10, Math.max(0, Math.round(weightedScore / 10)));

                return finalScore;
            } catch (error) {
                console.error("Error calculating overall quality:", error);
                return "N/A";
            }
        }, [calculateCoherence, calculateCoverage, calculateSimplification]);

        const graphData = useMemo(() => {
            // Skip if no state or subgraphData
            if (!state || !state.subgraphData) {
                console.warn("No subgraphData available");
                return {nodes: [], links: []};
            }

            // Get subgraph based on activeSubgraph
            const data = state.subgraphData;
            let subgraphKey = `${activeSubgraph}_subgraph`;

            // Special handling for MCTS path
            if (activeSubgraph === "mcts") {
                return data.mcts_reasoning_path || {nodes: [], links: []};
            }
            // Special handling for traversal which is a combination of GNN and connection
            if (activeSubgraph === "traversal") {
                return data.traversal_subgraph || {nodes: [], links: []};
            }

            // Special handling for GNN to make sure it's found
            if (activeSubgraph === "gnn") {
                // If gnn_subgraph is empty, try to use gnn_records to build it
                const gnnSubgraph = data.gnn_subgraph;
                if ((!gnnSubgraph || !gnnSubgraph.nodes || gnnSubgraph.nodes.length === 0) && data.gnn_records) {
                    console.log("GNN subgraph was empty, rebuilding from gnn_records");
                    // Build nodes and links from gnn_records
                    const nodes = [];
                    const links = [];
                    const nodeMap = new Map();

                    data.gnn_records.forEach(record => {
                        if (!nodeMap.has(record.subject_label)) {
                            nodeMap.set(record.subject_label, {
                                id: record.subject_id,
                                label: record.subject_label,
                                group: "gnn",
                                origin_label: "gnn",
                                domain_type: record.domain_type || "other"
                            });
                        }

                        if (!nodeMap.has(record.object_label)) {
                            nodeMap.set(record.object_label, {
                                id: record.object_id,
                                label: record.object_label,
                                group: "gnn",
                                origin_label: "gnn",
                                domain_type: record.domain_type || "other"
                            });
                        }

                        links.push({
                            source: record.subject_id,
                            target: record.object_id,
                            label: record.predicate,
                            origin: "gnn"
                        });
                    });

                    nodeMap.forEach(node => nodes.push(node));
                    return {nodes, links};
                }
                return gnnSubgraph || {nodes: [], links: []};
            }

            // Get the selected subgraph data, with fallbacks for each property
            const selectedSubgraph = data[subgraphKey] || data.combined_subgraph || {nodes: [], links: []};

            // Ensure nodes and links are arrays
            const nodes = Array.isArray(selectedSubgraph.nodes) ? selectedSubgraph.nodes : [];
            const links = Array.isArray(selectedSubgraph.links) ? selectedSubgraph.links : [];

            console.log(`Selected ${activeSubgraph} subgraph:`, {
                nodeCount: nodes.length,
                linkCount: links.length
            });

            // Apply consolidation if enabled
            if (useConsolidated && nodes.length > 0) {
                try {
                    return consolidateGraph(nodes, links);
                } catch (error) {
                    console.error("Error in consolidateGraph:", error);
                    return {nodes, links};
                }
            } else {
                return {nodes, links};
            }
        }, [state, activeSubgraph, useConsolidated]);

        // Update the animateReasoningPath function in CombinedResults.jsx
        const animateReasoningPath = () => {
            // No need to check isReasoningPath since this button only appears when MCTS is active
            if (!graphData || !graphData.nodes || graphData.nodes.length < 2) {
                return;
            }

            // Find the SVG element
            const svg = d3.select('.graph-container svg');
            if (!svg.node()) {
                console.error("SVG element not found");
                return;
            }

            // Sort nodes by path_order
            const orderedNodes = [...graphData.nodes]
                .filter(n => n.path_order !== undefined)
                .sort((a, b) => a.path_order - b.path_order);

            if (orderedNodes.length < 2) {
                console.warn("Not enough ordered nodes for animation");
                return;
            }

            // Reset all nodes and links
            svg.selectAll('circle').attr('opacity', 0.3);
            svg.selectAll('line').attr('opacity', 0.1);
            svg.selectAll('text').attr('opacity', 0.3);

            // Animate through the path
            let currentIndex = 0;

            const interval = setInterval(() => {
                // Highlight current node and all previous nodes
                for (let i = 0; i <= currentIndex; i++) {
                    if (i < orderedNodes.length) {
                        const currentNode = orderedNodes[i];

                        // Find and highlight the node - with safe checks
                        svg.selectAll('circle')
                            .filter(d => d && d.id === currentNode.id)
                            .attr('opacity', 1)
                            .attr('r', d => {
                                // Safely check origin_label
                                if (d && typeof d.origin_label === 'string' && d.origin_label === 'user') {
                                    return 30;
                                }
                                return 25;
                            });

                        // Highlight node text
                        svg.selectAll('text')
                            .filter(d => d && d.id === currentNode.id)
                            .attr('opacity', 1)
                            .attr('font-weight', 'bold');

                        // Highlight link if we're past the first node
                        if (i > 0) {
                            const prevNode = orderedNodes[i - 1];
                            svg.selectAll('line')
                                .filter(d => {
                                    if (!d || !d.source || !d.target) return false;

                                    // Handle both string IDs and object references
                                    const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
                                    const targetId = typeof d.target === 'object' ? d.target.id : d.target;

                                    return (sourceId === prevNode.id && targetId === currentNode.id) ||
                                        (targetId === prevNode.id && sourceId === currentNode.id);
                                })
                                .attr('opacity', 1)
                                .attr('stroke-width', 3);
                        }
                    }
                }

                currentIndex++;
                if (currentIndex >= orderedNodes.length) {
                    clearInterval(interval);

                    // Reset after a delay
                    setTimeout(() => {
                        // Reset with safer checks
                        svg.selectAll('circle')
                            .attr('opacity', 1)
                            .attr('r', d => {
                                if (d && typeof d.origin_label === 'string' && d.origin_label === 'user') {
                                    return 25;
                                }
                                return 20;
                            });

                        svg.selectAll('line')
                            .attr('opacity', 0.6)
                            .attr('stroke-width', 1.5);

                        svg.selectAll('text')
                            .attr('opacity', 1)
                            .attr('font-weight', 'normal');
                    }, 2000);
                }
            }, 1000); // Animate at 1-second intervals
        };

        useEffect(() => {
            console.log("CombinedResults mounted with state:", {
                hasState: !!state,
                stateProperties: state ? Object.keys(state) : [],
                hasPrompt: !!state?.prompt,
                hasSearch: !!state?.search,
                hasSubgraphData: !!state?.subgraphData,
                subgraphDataProperties: state?.subgraphData ? Object.keys(state.subgraphData) : [],
                hasCombinedSubgraph: !!state?.subgraphData?.combined_subgraph,
                combinedSubgraphNodes: state?.subgraphData?.combined_subgraph?.nodes?.length || 0,
                combinedSubgraphLinks: state?.subgraphData?.combined_subgraph?.links?.length || 0
            });
        }, [state]);

        function getNodeGroup(nodeName) {
            if (!nodeName || typeof nodeName !== 'string') {
                return 'other'; // Default group for undefined or non-string nodes
            }
            const lowerName = nodeName.toLowerCase();
            if (lowerName.includes('data')) return 'data';
            if (lowerName.includes('system')) return 'system';
            if (lowerName.includes('business')) return 'business';
            if (lowerName.includes('technology')) return 'technology';
            return 'other';
        }

        const generateGraphDescription = useCallback((nodes, links) => {
            if (nodes.length === 0 || links.length === 0) {
                return "No graph data available.";
            }

            const nodeCount = new Set(nodes.map(node => node.id)).size;
            const uniqueLinks = new Set(links.map(link => `${link.source}-${link.target}-${link.label}`));
            const linkCount = uniqueLinks.size;
            const nodeGroups = [...new Set(nodes.map(node => node.group))];
            const relationshipTypes = [...new Set(links.map(link => link.label))];

            return `This graph represents a solution with ${nodeCount} unique components across ${nodeGroups.length} categories: ${nodeGroups.join(", ")}. 
             These components are connected through ${linkCount} unique relationships, including ${relationshipTypes.slice(0, 5).join(", ")}${relationshipTypes.length > 5 ? ", and others" : ""}. 
             This structure suggests a ${nodeCount > 10 ? "complex" : "simple"} system with ${linkCount > nodeCount ? "high" : "moderate"} interconnectivity. 
             Key components include ${[...new Set(nodes.slice(0, 3).map(node => node.id))].join(", ")}, which play central roles in the system.`;
        }, []);


        const handleGenerateSummary = useCallback(async () => {
            if (isSummaryGenerated) return;
            setIsSummaryLoading(true);

            try {
                // Check if we have papers
                if (!allPapers || allPapers.length === 0) {
                    console.warn("No papers data available for summarization");
                    setPapersSummary("No papers data available for summarization. Please try searching for papers first.");
                    setIsSummaryGenerated(true);
                    return;
                }

                // Filter and limit paper content
                const limitedPapers = allPapers
                    .filter(paper => paper && paper.content && typeof paper.content === 'string' && paper.content.trim().length > 0)
                    .map(paper => ({
                        ...paper,
                        content: paper.content.substring(0, 1000) // Limit to 1000 chars per paper
                    }));

                if (limitedPapers.length === 0) {
                    console.warn("Papers exist but contain no valid content");
                    setPapersSummary("Papers found, but they don't contain valid content for summarization.");
                    setIsSummaryGenerated(true);
                    return;
                }

                // Prepare content for summarization
                const paperContent = limitedPapers.map(p => p.content).join("\n").substring(0, 10000);

                // Use graphData directly (not memorizedGraphData)
                const summary = await APIService.SummarizePapers(paperContent, 300, graphData);
                setPapersSummary(summary?.summary || "No summary available.");
                setIsSummaryGenerated(true);
            } catch (error) {
                console.error('Error generating summary:', error);
                setPapersSummary(`Error generating summary: ${error.message}`);
                setIsSummaryGenerated(true);
            } finally {
                setIsSummaryLoading(false);
            }
        }, [allPapers, graphData, isSummaryGenerated]);  // Update dependencies to use graphData directly

        useEffect(() => {
            // Don't destructure unless we're sure graphData exists and has the right shape
            if (!graphData) {
                console.warn("graphData is undefined in useEffect");
                setGraphDescription("No graph data available for description.");
                return;
            }

            // Safe access
            const nodes = graphData.nodes || [];
            const links = graphData.links || [];

            if (!Array.isArray(nodes) || !Array.isArray(links)) {
                console.warn("Invalid graphData structure:", {graphData});
                setGraphDescription("No graph data available for description.");
                return;
            }

            if (nodes.length > 0 && links.length > 0) {
                try {
                    const description = generateGraphDescription(nodes, links);
                    setGraphDescription(description);
                } catch (error) {
                    console.error("Error generating graph description:", error);
                    setGraphDescription("Error generating graph description.");
                }
            } else {
                console.warn("Empty nodes or links arrays");
                setGraphDescription("No graph data available for description.");
            }
        }, [graphData, generateGraphDescription]);


        useEffect(() => {
            if (state && state.prompt) {
                const importantTokens = ['risk', 'detection', 'prevention', 'healthcare', 'system', 'data', 'technology', 'business'];
                const words = state.prompt.split(' ');
                const highlightedWords = words.map(word =>
                    importantTokens.some(token => word.toLowerCase().includes(token))
                        ? `<highlight>${word}</highlight>`
                        : word
                );
                setIsLoading(false);
                setHighlightedPrompt(highlightedWords.join(' '));
            }
        }, [state]);


        useEffect(() => {
            if (state && state.subgraphData) {
                console.log("[DEBUG] Received subgraphData:", state.subgraphData);

                const combined = state.subgraphData.combined_subgraph;
                console.log("[DEBUG] Combined Nodes (from backend):", combined?.nodes?.slice(0, 5));
                console.log("[DEBUG] Combined Links (from backend):", combined?.links?.slice(0, 5));

                // Optional: check for any unexpected predicates as nodes
                const predicateNodes = combined?.nodes?.filter(n =>
                    n.id.startsWith("ns0__") || n.id.startsWith("rdfs__")
                );
                if (predicateNodes.length > 0) {
                    console.warn("[WARN] Predicate-like nodes found in frontend graph:", predicateNodes);
                }
            }
        }, [state]);

        useEffect(() => {
            // Set initial loading state
            setIsLoading(true);

            if (!searchData) {
                setError("No search results available.");
                setIsLoading(false);
                return;
            }

            if (!searchData.Topic || typeof searchData.Topic !== 'object') {
                setError("Invalid search data structure.");
                setIsLoading(false);
                return;
            }

            try {
                // Process the papers data with better content extraction
                const papers = Object.entries(searchData.Topic).map(([id, paper]) => {
                    let content = '';
                    let score = 0;
                    let title = '';

                    if (typeof paper === 'string') {
                        content = paper;
                    } else if (typeof paper === 'object' && paper !== null) {
                        if (paper.content) content = paper.content;
                        else if (paper.Topics) content = paper.Topics;
                        else if (paper.text) content = paper.text;
                        else if (paper.body) content = paper.body;
                        else content = JSON.stringify(paper);

                        score = paper.score || paper._score || 0;
                        title = paper.title || (content.split('\n')[0]?.substring(0, 100) || 'Untitled');
                    }

                    let documentId = '';
                    if (Array.isArray(searchData['URL/PDF'])) {
                        const index = parseInt(id) - 1;
                        if (index >= 0 && index < searchData['URL/PDF'].length) {
                            documentId = searchData['URL/PDF'][index];
                        }
                    } else if (typeof searchData['URL/PDF'] === 'object') {
                        documentId = searchData['URL/PDF'][id] || '';
                    } else if (typeof searchData['URL/PDF'] === 'string') {
                        documentId = searchData['URL/PDF'];
                    }

                    if (!documentId && paper && paper.url) {
                        documentId = paper.url;
                    }

                    return {
                        id: parseInt(id),
                        documentId,
                        content,
                        title,
                        score: score || 0.5
                    };
                });

                const validPapers = papers.filter(paper =>
                    paper && paper.content && paper.content.trim().length > 0
                );

                if (validPapers.length === 0) {
                    console.warn("No valid papers found after filtering");
                    setError("No valid papers found in search results");
                } else {
                    validPapers.sort((a, b) => b.score - a.score);

                    console.log("Before setting state:", {
                        paperCount: papers.length,
                        validPaperCount: validPapers.length,
                        paperSample: papers[0],
                        validPaperSample: validPapers[0]
                    });

                    setAllPapers(validPapers);
                    setTopPapers(validPapers.slice(0, 3));
                    setRestPapers(validPapers.slice(3));

                    console.log("State setter functions called with:", {
                        allPapersCount: validPapers.length,
                        topPapersCount: validPapers.slice(0, 3).length,
                        restPapersCount: validPapers.slice(3).length
                    });
                }
            } catch (err) {
                console.error("Error processing paper data:", err);
                setError("Error processing paper data: " + err.message);
            } finally {
                setIsLoading(false);
            }

            // Only generate graph description if nodes and links exist
            if (graphData?.nodes?.length > 0 && graphData?.links?.length > 0) {
                const description = generateGraphDescription(graphData.nodes, graphData.links);
                setGraphDescription(description);
            } else {
                console.warn("Cannot generate graph description: missing nodes or links");
                setGraphDescription("No graph data available for description.");
            }
        }, [searchData, generateGraphDescription, graphData]);


        useEffect(() => {
            const fetchQueryResults = async () => {
                if (!state || !state.prompt) return;

                try {
                    const queryResults = await APIService.SearchQuery(state.prompt);

                    // Debug the search results structure
                    console.log("Search results structure:", {
                        hasResults: !!queryResults,
                        hasTopic: !!queryResults?.Topic,
                        topicType: queryResults?.Topic ? typeof queryResults.Topic : 'undefined',
                        topicKeys: queryResults?.Topic ? Object.keys(queryResults.Topic) : [],
                        topicLength: queryResults?.Topic ? Object.keys(queryResults.Topic).length : 0,
                        hasUrls: !!queryResults?.['URL/PDF'],
                        urlsType: queryResults?.['URL/PDF'] ? typeof queryResults['URL/PDF'] : 'undefined',
                        urlsLength: queryResults?.['URL/PDF'] ?
                            (Array.isArray(queryResults['URL/PDF']) ? queryResults['URL/PDF'].length :
                                typeof queryResults['URL/PDF'] === 'object' ? Object.keys(queryResults['URL/PDF']).length : 0) : 0,
                        firstTopic: queryResults?.Topic && Object.keys(queryResults.Topic).length > 0 ?
                            queryResults.Topic[Object.keys(queryResults.Topic)[0]] : null
                    });

                    // Explicitly add console logs for the first few papers
                    if (queryResults?.Topic) {
                        const keys = Object.keys(queryResults.Topic);
                        for (let i = 0; i < Math.min(3, keys.length); i++) {
                            const key = keys[i];
                            console.log(`Paper ${key} details:`, {
                                type: typeof queryResults.Topic[key],
                                stringValue: typeof queryResults.Topic[key] === 'string' ?
                                    queryResults.Topic[key].substring(0, 100) + '...' : 'not a string',
                                objectKeys: typeof queryResults.Topic[key] === 'object' ?
                                    Object.keys(queryResults.Topic[key]) : 'not an object',
                                correspondingUrl: queryResults['URL/PDF'] && queryResults['URL/PDF'][parseInt(key) - 1]
                            });
                        }
                    }

                    // Ensure state.search is correctly updated
                    if (state) {
                        // Create a deep copy to ensure the state update is recognized
                        setSearchData(queryResults);
                        console.log("Updated state with search results");
                    }

                    // Force re-processing the papers
                    const event = new CustomEvent('updateSearchData', {detail: queryResults});
                    window.dispatchEvent(event);
                } catch (error) {
                    console.error("Error fetching search query results:", error);
                    setError("Failed to fetch related papers. Please try again.");
                }
            };
            fetchQueryResults();
        }, [state?.prompt]);

        useEffect(() => {
            if (!state?.prompt) return;

            // Reset state before fetching new data
            setAllPapers([]);
            setTopPapers([]);
            setRestPapers([]);
            setIsSummaryGenerated(false);
            setPapersSummary("");
        }, [state?.prompt]);

        useEffect(() => {
            const handleSearchUpdate = (event) => {
                const updatedSearch = event.detail;
                setSearchData(updatedSearch); // ✅ This is the correct way to update it
            };

            window.addEventListener('updateSearchData', handleSearchUpdate);
            return () => {
                window.removeEventListener('updateSearchData', handleSearchUpdate);
            };
        }, []);

        useEffect(() => {
            if (state?.search) {
                setSearchData(state.search);
            }
        }, [state?.search]);

        useEffect(() => {
            console.log("Render cycle papers:", {
                topPapersLength: topPapers.length,
                allPapersLength: allPapers.length
            });
        }, [topPapers, allPapers]);


        const getDocumentUrl = useCallback((documentId, paper) => {
            if (!documentId || typeof documentId !== 'string') {
                // If no documentId but we have the paper, try to extract URL from content
                if (paper && paper.content) {
                    // Look for URLs in the content
                    const urlRegex = /(https?:\/\/[^\s]+)/g;
                    const matches = paper.content.match(urlRegex);
                    if (matches && matches.length > 0) {
                        return matches[0];
                    }

                    // If paper title starts with http, it might be the URL itself
                    if (paper.title && paper.title.startsWith('http')) {
                        return paper.title;
                    }
                }
                return '#';
            }

            if (!documentId && paper && paper.url && paper.url.includes('dev.to')) {
                return paper.url;
            }

            // For dev.to links, use the URL directly
            if (documentId.includes('dev.to')) {
                return documentId.startsWith('http') ? documentId : `https://${documentId}`;
            }

            // For URLs that are already complete, return as is
            if (documentId.startsWith('http://') || documentId.startsWith('https://')) {
                return documentId;
            }

            // Handle arXiv IDs
            if (documentId.toLowerCase().includes('arxiv') ||
                /\d{4}\.\d{4,5}/.test(documentId) ||
                documentId.includes('/')) {

                // Extract the arXiv ID
                let cleanId = documentId;

                // If it has a slash, remove prefix
                if (documentId.includes('/')) {
                    const parts = documentId.split('/');
                    cleanId = parts[1] || parts[0];  // fallback
                }

                cleanId = cleanId.split('v')[0]; // Remove version if present

                return `https://arxiv.org/abs/${cleanId}`;
            }

            // Default fallback - return as is
            return documentId;
        }, []);

        const parseSummary = (summary) => {
            if (typeof summary !== 'string') {
                console.error("parseSummary received non-string summary:", summary);
                return [{content: "Summary not available or invalid format."}];
            }

            const parts = summary.split(/(?=\d+\.\s*\*\*)/);
            return parts.map(part => {
                const [title, content] = part.split(':**');
                if (content) {
                    const bulletPoints = content.match(/\s*-\s*(.*?)(?=\s*-|\s*$)/g) || [];
                    return {
                        title: title.trim(),
                        bulletPoints: bulletPoints.map(point => point.replace(/^\s*-\s*/, '').trim())
                    };
                }
                return {content: part.trim()};
            });
        };

        return (
            <Container maxWidth="xl">
                {isLoading ? (
                    <CircularProgress/>
                ) : (
                    <>
                        {state && state.prompt && (
                            <Paper elevation={3}
                                   style={{padding: '15px', marginBottom: '20px', backgroundColor: '#f0f0f0'}}>
                                <Typography variant="h6" gutterBottom>Original Prompt:</Typography>
                                <Typography variant="body1" component="div" dangerouslySetInnerHTML={{
                                    __html: highlightedPrompt.replace(/<highlight>/g, '<span style="background-color: #fff176; padding: 0 4px; border-radius: 2px;">').replace(/<\/highlight>/g, '</span>')
                                }}/>
                            </Paper>
                        )}
                        <Typography variant="h4" gutterBottom>Solution Design Recommendations</Typography>

                        {error && (
                            <Paper elevation={3}
                                   style={{padding: '15px', marginBottom: '20px', backgroundColor: '#ffcccb'}}>
                                <Typography variant="h6" color="error">Error: {error}</Typography>
                            </Paper>
                        )
                        }

                        {/* Graph Legend */}
                        <Box display="flex" flexWrap="wrap" gap={1}>
                            {Object.entries(nodeColors).map(([type, color]) => (
                                <Chip
                                    key={type}
                                    label={type.charAt(0).toUpperCase() + type.slice(1)}
                                    style={{
                                        backgroundColor: color,
                                        color: ['domain', 'other'].includes(type) ? 'white' : 'black',
                                        fontWeight: 'bold'
                                    }}
                                />
                            ))}
                        </Box>
                        <Paper elevation={3} style={{padding: '15px', marginBottom: '20px'}}>
                            <Typography variant="h5" gutterBottom>Graph View</Typography>
                            <Typography
                                variant="body1"
                                paragraph
                                style={{
                                    maxWidth: '100%',
                                    overflowWrap: 'break-word',
                                    wordWrap: 'break-word',
                                    wordBreak: 'break-word',
                                    hyphens: 'auto'
                                }}
                            >
                                {graphDescription}
                            </Typography>
                            <div className="graph-container" style={{height: '70vh', width: '100%', position: 'relative'}}>
                                {graphData && Array.isArray(graphData.nodes) && graphData.nodes.length > 0 &&
                                Array.isArray(graphData.links) && graphData.links.length > 0 ? (
                                    <>
                                        <Graph
                                            nodes={graphData.nodes}
                                            links={graphData.links}
                                            initialZoom={0.2}
                                            highlightTraversal={highlightTraversal}
                                            colorScheme={activeSubgraph === "combined" ? "default" : "source"}
                                            isReasoningPath={activeSubgraph === "mcts"} // Pass this prop to indicate it's a reasoning path
                                        />
                                    </>
                                ) : (
                                    <Typography>No graph data available to display. Try selecting a different view or
                                        adjusting your search.</Typography>
                                )}
                            </div>
                            <ButtonGroup variant="outlined">
                                <Button onClick={() => setActiveSubgraph("combined")}>All</Button>
                                <Button onClick={() => setActiveSubgraph("user")}>User</Button>
                                <Button onClick={() => setActiveSubgraph("domain")}>Domain</Button>
                                <Button onClick={() => setActiveSubgraph("llm")}>LLM</Button>
                                <Button onClick={() => setActiveSubgraph("connection")}>Connections</Button>
                                <Button onClick={() => setActiveSubgraph("gnn")}>GNN</Button>
                                <Button onClick={() => setActiveSubgraph("traversal")}>Traversal</Button>
                                <Button
                                    onClick={() => setActiveSubgraph("mcts")}
                                    variant={activeSubgraph === "mcts" ? "contained" : "outlined"}
                                    color="secondary"
                                >
                                    MCTS Path
                                </Button>
                                <Button
                                    onClick={() => setUseConsolidated(!useConsolidated)}
                                    variant={useConsolidated ? "contained" : "outlined"}
                                >
                                    {useConsolidated ? "Consolidated" : "Detailed"}
                                </Button>
                                <Button
                                    onClick={() => setHighlightTraversal(!highlightTraversal)}
                                    variant={highlightTraversal ? "contained" : "outlined"}
                                    color="secondary"
                                >
                                    {highlightTraversal ? "Highlighting On" : "Highlight Traversal"}
                                </Button>
                                {activeSubgraph === "mcts" && (
                                    <Button
                                        onClick={animateReasoningPath}
                                        variant="contained"
                                        color="secondary"
                                    >
                                        Animate Path
                                    </Button>
                                )}
                            </ButtonGroup>
                            {activeSubgraph === "mcts" && (
                                <Box mt={2}>
                                    <Paper elevation={2} sx={{p: 2, backgroundColor: '#f5f5f5'}}>
                                        <Box
                                            sx={{
                                                backgroundColor: '#e3f2fd',
                                                padding: '10px',
                                                borderRadius: '4px',
                                                marginTop: '10px',
                                                marginBottom: '10px'
                                            }}
                                        >
                                            <Typography variant="body2">
                                                <strong>MCTS Reasoning Path:</strong> Viewing the optimized path generated
                                                with Monte
                                                Carlo Tree Search.
                                                This path
                                                is {state?.subgraphData?.mcts_reasoning_path?.nodes?.length || 0} nodes
                                                ({state?.subgraphData?.gnn_subgraph?.nodes?.length
                                                ? Math.round((state?.subgraphData?.mcts_reasoning_path?.nodes?.length / state?.subgraphData?.gnn_subgraph?.nodes?.length) * 100)
                                                : 0}% of GNN graph size)
                                            </Typography>
                                        </Box>
                                        <Typography variant="subtitle1" gutterBottom>MCTS Reasoning Path
                                            Metrics</Typography>

                                        <Grid container spacing={2}>
                                            <Grid item xs={4}>
                                                <Typography variant="body2">
                                                    <strong>Path Coherence:</strong> {
                                                    calculateCoherence(
                                                        state?.subgraphData?.gnn_subgraph,
                                                        state?.subgraphData?.mcts_reasoning_path
                                                    )
                                                }
                                                </Typography>
                                            </Grid>
                                            <Grid item xs={4}>
                                                <Typography variant="body2">
                                                    <strong>User Concept Coverage:</strong> {
                                                    calculateCoverage(
                                                        state?.subgraphData?.user_records,
                                                        state?.subgraphData?.mcts_reasoning_path
                                                    )
                                                }
                                                </Typography>
                                            </Grid>
                                            <Grid item xs={4}>
                                                <Typography variant="body2">
                                                    <strong>Path Quality Score:</strong> {
                                                    calculateOverallQuality(
                                                        state?.subgraphData?.gnn_subgraph,
                                                        state?.subgraphData?.mcts_reasoning_path,
                                                        state?.subgraphData?.user_records
                                                    )
                                                }/10
                                                </Typography>
                                            </Grid>
                                        </Grid>
                                    </Paper>
                                </Box>
                            )}
                        </Paper>
                        <Paper>
                            {/* Add instruction comparison section here */}
                            <Typography variant="h5" gutterBottom>Instructions Comparison</Typography>
                            <Typography variant="body2" color="textSecondary" paragraph>
                                Compare different instruction generation approaches to see which produces the most effective
                                guidance.
                            </Typography>
                            <Box mb={2}>
                                <ButtonGroup variant="outlined" size="small">
                                    <Button
                                        variant={useGraphForInstructions === 'mcts' ? "contained" : "outlined"}
                                        onClick={() => setUseGraphForInstructions('mcts')}
                                    >
                                        Use MCTS Graph
                                    </Button>
                                    <Button
                                        variant={useGraphForInstructions === 'combined' ? "contained" : "outlined"}
                                        onClick={() => setUseGraphForInstructions('combined')}
                                    >
                                        Use Combined Graph
                                    </Button>
                                </ButtonGroup>
                                <Typography variant="caption" display="block" color="textSecondary" mt={1}>
                                    Currently
                                    using: {useGraphForInstructions === 'mcts' ? 'MCTS Reasoning Path' : 'Combined Graph'} for
                                    instruction generation
                                </Typography>
                            </Box>
                            <InstructionComparison
                                prompt={state?.prompt || ""}
                                userRecords={state?.subgraphData?.user_records || []}
                                gnnSubgraph={state?.subgraphData?.gnn_subgraph || {}}
                                mctsSubgraph={
                                    useGraphForInstructions === 'mcts'
                                        ? state?.subgraphData?.mcts_reasoning_path || {}
                                        : state?.subgraphData?.combined_subgraph || {}
                                }
                            />
                        </Paper>

                        <Paper elevation={3} style={{padding: '15px', marginBottom: '20px'}}>
                            <Typography variant="h5" gutterBottom>Solution Summary</Typography>
                            {!isSummaryGenerated ? (
                                <Button
                                    variant="contained"
                                    color="primary"
                                    onClick={handleGenerateSummary}
                                    disabled={isSummaryLoading}
                                >
                                    {isSummaryLoading ? 'Generating...' : 'Generate'}
                                </Button>
                            ) : isSummaryLoading ? (
                                <CircularProgress/>
                            ) : papersSummary ? (
                                <>
                                    {parseSummary(papersSummary).map((section, index) => (
                                        <div key={index} className={classes.summarySection}>
                                            {section.title ? (
                                                <>
                                                    <Typography variant="h6"
                                                                className={classes.sectionTitle}>{section.title}</Typography>
                                                    <List className={classes.bulletList}>
                                                        {section.bulletPoints.map((point, pointIndex) => (
                                                            <ListItem key={pointIndex} className={classes.bulletItem}>
                                                                <ListItemIcon className={classes.bulletIcon}>
                                                                    <FiberManualRecordIcon style={{fontSize: 8}}/>
                                                                </ListItemIcon>
                                                                <ListItemText primary={point}
                                                                              className={classes.bulletText}/>
                                                            </ListItem>
                                                        ))}
                                                    </List>
                                                </>
                                            ) : (
                                                <Typography variant="body1"
                                                            className={classes.paragraph}>{section.content}</Typography>
                                            )}
                                            {index < parseSummary(papersSummary).length - 1 &&
                                                <Divider style={{margin: '10px 0'}}/>}
                                        </div>
                                    ))}
                                </>
                            ) : (
                                <Typography variant="body1">No summary available.</Typography>
                            )}
                        </Paper>
                        <Typography variant="h5" gutterBottom>
                            Top Papers {topPapers ? `(${topPapers.length})` : '(0)'}
                        </Typography>
                        <Grid container spacing={3}>
                            {topPapers && topPapers.length > 0 ? (
                                topPapers.map((paper) => (
                                    <Grid item xs={12} md={4} key={paper.id || Math.random()}>
                                        <Paper elevation={3} style={{padding: '15px', height: '100%'}}>
                                            <Typography variant="h6" style={{
                                                maxWidth: '100%',
                                                overflowWrap: 'break-word',
                                                wordWrap: 'break-word'
                                            }}>
                                                {paper.title && paper.title.startsWith('http')
                                                    ? 'Article: ' + paper.title.split('/').pop()
                                                    : (paper.title || 'Untitled')}
                                            </Typography>
                                            <Typography variant="body2" color="textSecondary">
                                                {paper.score !== undefined ? `Score: ${paper.score.toFixed(4)}` : ''}
                                            </Typography>
                                            <Typography variant="body1" style={{
                                                maxWidth: '100%',
                                                overflowWrap: 'break-word',
                                                wordWrap: 'break-word'
                                            }}>
                                                {paper.content && !paper.content.startsWith('http')
                                                    ? (paper.content.length > 200 ? `${paper.content.substring(0, 200)}...` : paper.content)
                                                    : 'Click View Source to read article content'}
                                            </Typography>
                                            {(paper.documentId || paper.content.startsWith('http')) && (
                                                <Button
                                                    variant="outlined"
                                                    color="primary"
                                                    href={paper.content.startsWith('http') ? paper.content : getDocumentUrl(paper.documentId, paper)}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    style={{marginTop: '10px'}}
                                                >
                                                    View Source
                                                </Button>
                                            )}
                                        </Paper>
                                    </Grid>
                                ))
                            ) : (
                                <Typography variant="body1" style={{padding: "20px"}}>
                                    No papers available to display. Papers data: {JSON.stringify(topPapers)}
                                </Typography>
                            )}
                        </Grid>

                        {
                            restPapers.length > 0 && (
                                <Accordion style={{marginTop: '20px'}}>
                                    <AccordionSummary expandIcon={<ExpandMoreIcon/>}>
                                        <Typography>View More Papers ({restPapers.length})</Typography>
                                    </AccordionSummary>
                                    <AccordionDetails>
                                        <Grid container spacing={3}>
                                            {restPapers.map((paper) => (
                                                <Grid item xs={12} md={4} key={paper.id}>
                                                    <Paper elevation={3} style={{padding: '15px', height: '100%'}}>
                                                        <Typography variant="h6" style={{
                                                            maxWidth: '100%',
                                                            overflowWrap: 'break-word',
                                                            wordWrap: 'break-word'
                                                        }}>
                                                            {paper.title && paper.title.startsWith('http')
                                                                ? 'Article: ' + paper.title.split('/').pop()
                                                                : (paper.title || 'Untitled')}
                                                        </Typography>
                                                        <Typography variant="body1">
                                                            {paper.content.toLowerCase().includes('withdrawn') ? (
                                                                <i>View content by button click.</i>
                                                            ) : (
                                                                paper.content.length > 200 ? `${paper.content.substring(0, 200)}...` : paper.content
                                                            )}
                                                        </Typography>
                                                        <Typography variant="body2" color="textSecondary">
                                                            {paper.score !== undefined ? `Score: ${paper.score.toFixed(4)}` : ''}
                                                        </Typography>
                                                        <Typography variant="body1" style={{
                                                            maxWidth: '100%',
                                                            overflowWrap: 'break-word',
                                                            wordWrap: 'break-word'
                                                        }}>
                                                            {paper.content && !paper.content.startsWith('http')
                                                                ? (paper.content.length > 200 ? `${paper.content.substring(0, 200)}...` : paper.content)
                                                                : 'Click View Source to read article content'}
                                                        </Typography>
                                                        {(paper.documentId || paper.content.startsWith('http')) && (
                                                            <Button
                                                                variant="outlined"
                                                                color="primary"
                                                                href={paper.content.startsWith('http') ? paper.content : getDocumentUrl(paper.documentId, paper)}
                                                                target="_blank"
                                                                rel="noopener noreferrer"
                                                                style={{marginTop: '10px'}}
                                                            >
                                                                View Source
                                                            </Button>
                                                        )}
                                                    </Paper>
                                                </Grid>
                                            ))}
                                        </Grid>
                                    </AccordionDetails>
                                </Accordion>
                            )
                        }

                        <Button variant="contained" color="primary" href="/app" style={{marginTop: '20px'}}>
                            Specify Your Problem Definition
                        </Button>
                    </>
                )}
            </Container>
        )
            ;
    }
;

export default CombinedResults;