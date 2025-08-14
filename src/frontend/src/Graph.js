import React, {useRef, useEffect, useState} from 'react';
import * as d3 from 'd3';
import {
    Button
} from '@mui/material';

const Graph = ({
                   nodes,
                   links,
                   initialZoom = 0.5,
                   showLabels = true,
                   colorScheme = 'default',
                   height = '70vh',
                   highlightTraversal = false,
                   activeSubgraph = '' // Add this prop
               }) => {
    const svgRef = useRef(null);
    const containerRef = useRef(null);
    const [dimensions, setDimensions] = useState({width: 0, height: 0});
    const [error, setError] = useState(null);

    const colorSchemes = {
        default: {
            'data': '#377eb8',
            'system': '#4daf4a',
            'business': '#984ea3',
            'technology': '#ff7f00',
            'other': '#999999'
        },
        source: {
            'user': '#4285F4',
            'domain': '#34A853',
            'query': '#FBBC05',
            'llm': '#EA4335',
            'gnn': '#E91E63',  // Added GNN
            'connection': '#FF9800',  // Added connection
            'other': '#757575'
        }
    };

    const activeColorScheme = colorSchemes[colorScheme] || colorSchemes.default;

    // Add special colors for traversal paths
    const traversalColors = {
        'business': '#FF5722',  // Business level
        'system': '#8BC34A',    // System level
        'data': '#03A9F4',      // Data level
        'technology': '#FF9800' // Technology level
    };

    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current) {
                const rect = containerRef.current.getBoundingClientRect();
                setDimensions({
                    width: rect.width > 0 ? rect.width : 800,
                    height: rect.height > 0 ? rect.height : 600
                });
            }
        };
        const timer = setTimeout(updateDimensions, 100);
        window.addEventListener('resize', updateDimensions);
        return () => {
            clearTimeout(timer);
            window.removeEventListener('resize', updateDimensions);
        };
    }, []);


    useEffect(() => {
        if (!nodes || !links || nodes.length === 0 || dimensions.width === 0) {
            console.warn("Missing data for graph:", {
                nodesLength: nodes?.length,
                linksLength: links?.length,
                dimensions
            });
            return;
        }

        // Debug data
        console.log("Graph data:", {
            nodes: nodes.slice(0, 5),
            links: links.slice(0, 5),
            totalNodes: nodes.length,
            totalLinks: links.length
        });

        // Check for source/target mismatches
        const nodeIds = new Set(nodes.map(n => n.id));
        const orphanedLinks = links.filter(link =>
            !nodeIds.has(typeof link.source === 'object' ? link.source.id : link.source) ||
            !nodeIds.has(typeof link.target === 'object' ? link.target.id : link.target)
        );

        if (orphanedLinks.length > 0) {
            console.error("Found orphaned links:", orphanedLinks);
        }

        // Rest of your existing code...
    }, [nodes, links, dimensions, initialZoom, showLabels, colorScheme, activeColorScheme, highlightTraversal]);

    // Get color based on node type and domain
    const getNodeColor = (node) => {
        // Always prioritize business/system/data/technology domain types for coloring
        if (node.domain_type && (
            node.domain_type.toLowerCase() === 'business' ||
            node.domain_type.toLowerCase() === 'system' ||
            node.domain_type.toLowerCase() === 'data' ||
            node.domain_type.toLowerCase() === 'technology')) {

            // For traversal highlighting
            if (highlightTraversal && (node.origin_label === 'gnn' || node.origin_label === 'connection')) {
                return traversalColors[node.domain_type.toLowerCase()];
            }

            // Standard coloring by domain type
            return colorSchemes.default[node.domain_type.toLowerCase()];
        }

        // Fallback to origin-based coloring if no valid domain type
        if (colorScheme === 'source' && node.origin_label) {
            return activeColorScheme[node.origin_label] || activeColorScheme.other;
        }

        // Ultimate fallback
        return activeColorScheme.other;
    };

    // Get stroke color for node
    const getNodeStroke = (node) => {
        // Special stroke for multi-origin nodes (e.g. both user and domain)
        if (node.origins && node.origins.length > 1) {
            return '#FFD700'; // Gold for multi-origin
        }

        // Special stroke for GNN traversal nodes
        if (node.origin_label === 'gnn' && node.predicate && node.predicate.includes('Is_implemented_by')) {
            return traversalColors[node.domain_type?.toLowerCase()] || '#E91E63';
        }

        return '#fff'; // Default
    };

    // Get edge colors
    const getLinkColor = (link) => {
        // Special color for GNN cross-level connections
        if (link.origin === 'gnn' && link.label && link.label.includes('implemented_by')) {
            return '#E91E63'; // Bright pink for traversal
        }

        // Connection links
        if (link.origin === 'connection') {
            return '#FF9800'; // Orange
        }

        // Default color
        return '#999';
    };

    // Shorten label for display
    const shortenLabel = (label) => {
        if (!label) return '';
        // Remove namespace prefixes for cleaner display
        const cleanLabel = label.replace(/^(ns0__|rdfs__|owl__)/, '');
        // Simplify traversal labels
        const simplifiedLabel = cleanLabel.replace('Is_implemented_by_', '→ ');
        // Truncate if still too long
        return simplifiedLabel.length > 15 ? simplifiedLabel.substring(0, 13) + '...' : simplifiedLabel;
    };

    useEffect(() => {
        if (!nodes || !links || nodes.length === 0 || dimensions.width === 0) return;
        setError(null);

        d3.select(svgRef.current).selectAll("*").remove();

        const graphNodes = nodes.map(node => ({...node}));
        const nodeIds = new Set(graphNodes.map(n => n.id));
        const graphLinks = links
            .map(link => ({...link}))
            .filter(link => nodeIds.has(link.source) && nodeIds.has(link.target));

        const svg = d3.select(svgRef.current)
            .attr('width', dimensions.width)
            .attr('height', dimensions.height)
            .attr('viewBox', [0, 0, dimensions.width, dimensions.height]);

        const g = svg.append('g');
        const zoom = d3.zoom().scaleExtent([0.1, 5]).on('zoom', event => {
            g.attr('transform', event.transform);
        });
        svg.call(zoom).call(zoom.transform, d3.zoomIdentity.scale(initialZoom));

        // Add arrow markers with different colors
        const markerTypes = ['default', 'gnn', 'connection', 'reasoning_path'];
        const markerColors = {
            'default': '#999',
            'gnn': '#E91E63',
            'connection': '#FF9800',
            'reasoning_path': '#4CAF50'  // Add distinct color for reasoning path
        };

        svg.append('defs').selectAll('marker')
            .data(markerTypes)
            .enter().append('marker')
            .attr('id', d => `marker-${d}`)
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 25)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('fill', d => markerColors[d])
            .attr('d', 'M0,-5L10,0L0,5');

        // Create force simulation
        const simulation = d3.forceSimulation(graphNodes)
            .force('link', d3.forceLink(graphLinks).id(d => d.id).distance(link => {
                // Longer distance for traversal links for better visibility
                if (link.origin === 'gnn' && link.label && link.label.includes('implemented_by')) {
                    return 300;
                }
                return 200;
            }))
            .force('charge', d3.forceManyBody().strength(-800))
            .force('center', d3.forceCenter(dimensions.width / 2, dimensions.height / 2))
            .force('collision', d3.forceCollide().radius(60))
            .force('x', d3.forceX(dimensions.width / 2).strength(0.05))
            .force('y', d3.forceY(dimensions.height / 2).strength(0.05));

        // Create links with appropriate colors and markers
        const link = g.append('g')
            .selectAll('line')
            .data(graphLinks)
            .join('line')
            .attr('stroke', d => getLinkColor(d))
            .attr('stroke-opacity', 0.6)
            .attr('stroke-width', d => {
                // Thicker lines for important connections
                if (d.origin === 'gnn' || d.origin === 'connection') {
                    return 2;
                }
                return 1;
            })
            .attr('stroke-dasharray', d => {
                // Dashed lines for certain connection types
                if (d.origin === 'gnn' && d.label && d.label.includes('semantic')) {
                    return '5,5';
                }
                return null;
            })
            .attr('id', d => `link-${typeof d.source === 'object' ? d.source.id : d.source}-${typeof d.target === 'object' ? d.target.id : d.target}`)
            .attr('marker-end', d => {
                // Use appropriate marker based on link type
                if (d.origin === 'gnn') {
                    return 'url(#marker-gnn)';
                } else if (d.origin === 'connection') {
                    return 'url(#marker-connection)';
                }
                return 'url(#marker-default)';
            });

        // Create link labels
        const linkText = g.append('g')
            .selectAll('text')
            .data(graphLinks)
            .join('text')
            .text(d => shortenLabel(d.label || ''))
            .attr('font-size', 10)
            .attr('font-weight', d => (d.origin === 'gnn' || d.origin === 'connection') ? 'bold' : 'normal')
            .attr('text-anchor', 'middle')
            .attr('dy', -2)
            .attr('fill', d => getLinkColor(d))
            .attr('opacity', showLabels ? 0.8 : 0)
            .attr('background', 'white');

        // Create a white background for link labels to improve readability
        const linkTextBackground = g.append('g')
            .selectAll('rect')
            .data(graphLinks)
            .join('rect')
            .attr('fill', 'white')
            .attr('opacity', 0.7)
            .attr('rx', 3)
            .attr('ry', 3)
            .attr('width', d => {
                const label = shortenLabel(d.label || '');
                return label.length * 5 + 10; // Approximate width based on text length
            })
            .attr('height', 14)
            .attr('display', showLabels ? 'block' : 'none');

        // Create nodes
        const node = g.append('g')
            .selectAll('circle')
            .data(graphNodes)
            .join('circle')
            .attr('r', d => {
                // Larger radius for important nodes
                if (d.origin_label === 'user') {
                    return 25; // User nodes are bigger
                } else if (d.origins && d.origins.length > 1) {
                    return 22; // Multi-origin nodes are bigger
                }
                return 20;
            })
            .attr('fill', d => getNodeColor(d))
            .attr('stroke', d => getNodeStroke(d))
            .attr('stroke-width', d => {
                // Thicker stroke for important nodes
                if (d.origin_label === 'user' ||
                    (d.origins && d.origins.length > 1) ||
                    (d.origin_label === 'gnn' && d.predicate && d.predicate.includes('Is_implemented_by'))) {
                    return 3;
                }
                return 1.5;
            })
            .call(drag(simulation));

        // Add hover title
        node.append('title')
            .text(d => {
                let text = `${d.label}\nSource: ${d.origin_label || 'N/A'}\nType: ${d.domain_type || 'N/A'}`;
                if (d.origins && d.origins.length > 1) {
                    text += `\nOrigins: ${d.origins.join(', ')}`;
                }
                if (d.path) {
                    text += `\nPath: ${d.path}`;
                }
                return text;
            });

        // Create tooltip
        const tooltip = d3.select(containerRef.current)
            .append('div')
            .attr('class', 'tooltip')
            .style('position', 'absolute')
            .style('padding', '8px 12px')
            .style('background', 'white')
            .style('border', '1px solid #ccc')
            .style('border-radius', '4px')
            .style('box-shadow', '0 2px 4px rgba(0,0,0,0.2)')
            .style('pointer-events', 'none')
            .style('opacity', 0)
            .style('font-size', '12px')
            .style('max-width', '250px');

        // Add hover effects
        node.on('mouseover', function (event, d) {
            // Build tooltip content with clear section for domain type and origin
            let tooltipContent = `<strong style="font-size:14px;">${d.label}</strong>`;

            // Domain Type (Layer) - prominently displayed with colored badge
            const domainType = d.domain_type?.toLowerCase() || 'other';
            const domainColor = colorSchemes.default[domainType] || '#999999';
            tooltipContent += `<br/><div style="margin:5px 0;">
                 <span style="background-color:${domainColor}; color:white; padding:2px 5px; border-radius:3px; font-weight:bold;">
                        ${domainType.charAt(0).toUpperCase() + domainType.slice(1)} Layer
                   </span>
                </div>`;

            // Source Information
            tooltipContent += `<br/><span style="color:#666;">Source: `;
            if (d.origins && d.origins.length > 1) {
                tooltipContent += d.origins.join(', ');
            } else {
                tooltipContent += d.origin_label || 'Unknown';
            }
            tooltipContent += `</span>`;

            // Additional information
            if (d.origin_label === 'gnn' && d.predicate && d.predicate.includes('implemented_by')) {
                tooltipContent += `<br/><span style="color:#E91E63;">Traversal: ${d.predicate.replace('ns0__Is_implemented_by_', '')}</span>`;
            }

            if (d.similarity) {
                tooltipContent += `<br/><span style="color:#666;">Similarity: ${(d.similarity * 100).toFixed(1)}%</span>`;
            }

            // Show tooltip
            tooltip.html(tooltipContent)
                .style('left', `${event.pageX + 10}px`)
                .style('top', `${event.pageY - 20}px`)
                .style('opacity', 1);

            // Highlight connected nodes and links
            const connectedNodeIds = new Set();
            link.each(l => {
                if (l.source.id === d.id || l.target.id === d.id) {
                    connectedNodeIds.add(l.source.id);
                    connectedNodeIds.add(l.target.id);
                }
            });

            node.attr('opacity', n => connectedNodeIds.has(n.id) || n.id === d.id ? 1 : 0.3);
            link.attr('opacity', l => (l.source.id === d.id || l.target.id === d.id) ? 1 : 0.1);
            linkText.attr('opacity', l => (l.source.id === d.id || l.target.id === d.id) ? 1 : 0);
            linkTextBackground.attr('opacity', l => (l.source.id === d.id || l.target.id === d.id) ? 0.7 : 0);
            nodeText.attr('opacity', n => connectedNodeIds.has(n.id) || n.id === d.id ? 1 : 0.3);

        }).on('mouseout', function () {
            // Hide tooltip
            tooltip.style('opacity', 0);

            // Reset opacity
            node.attr('opacity', 1);
            link.attr('opacity', 0.6);
            linkText.attr('opacity', showLabels ? 0.8 : 0);
            linkTextBackground.attr('opacity', showLabels ? 0.7 : 0);
            nodeText.attr('opacity', 1);
        });

        const getNodeLabel = (node) => {
            // Show path order if available
            const pathOrder = node.path_order !== undefined ? `[${node.path_order}] ` : '';
            return `${pathOrder}${node.label || node.id}`;
        };

        // Create node labels
        const nodeText = g.append('g')
            .selectAll('text')
            .data(graphNodes)
            .join('text')
            .text(d => getNodeLabel(d))
            .attr('font-size', 12)
            .attr('font-weight', d => d.origin_label === 'user' ? 'bold' : 'normal')
            .attr('text-anchor', 'middle')
            .attr('dy', 30)
            .attr('pointer-events', 'none'); // Prevent text from blocking node interaction

        // Update positions on simulation tick
        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            // Position link text and background at the middle of the link
            linkText
                .attr('x', d => (d.source.x + d.target.x) / 2)
                .attr('y', d => (d.source.y + d.target.y) / 2);

            linkTextBackground
                .attr('x', d => (d.source.x + d.target.x) / 2 - ((shortenLabel(d.label || '').length * 5 + 10) / 2))
                .attr('y', d => (d.source.y + d.target.y) / 2 - 7);

            nodeText
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });

        // Node drag behavior
        function drag(simulation) {
            return d3.drag()
                .on('start', (event, d) => {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                })
                .on('drag', (event, d) => {
                    d.fx = event.x;
                    d.fy = event.y;
                })
                .on('end', (event, d) => {
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                });
        }

        // Reset all when clicking in empty space
        svg.on('click', () => {
            node.attr('opacity', 1);
            link.attr('opacity', 0.6);
            linkText.attr('opacity', showLabels ? 0.8 : 0);
            linkTextBackground.attr('opacity', showLabels ? 0.7 : 0);
            nodeText.attr('opacity', 1);
        });

        return () => simulation.stop();

    }, [nodes, links, dimensions, initialZoom, showLabels, colorScheme, activeColorScheme, highlightTraversal]);

    return (
        <div ref={containerRef} style={{width: '100%', height: height}}>
            {error && <div style={{color: 'red'}}>Error: {error}</div>}
            <svg ref={svgRef} style={{width: '100%', height: '100%'}}></svg>
        </div>
    );
};

export default Graph;