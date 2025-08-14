// src/components/InstructionComparison.jsx - Updated with regenerate button
import React, { useState, useEffect } from 'react';
import {
    Box,
    Button,
    Chip,
    CircularProgress,
    Divider,
    Grid,
    Paper,
    Rating,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Tabs,
    Tab,
    Typography,
    Alert,
    AlertTitle,
    Stack
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import APIService from './APIService';

const InstructionComparison = ({ prompt, userRecords, gnnSubgraph, mctsSubgraph }) => {
    const [instructions, setInstructions] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState(0);
    const [modelStatus, setModelStatus] = useState({
        gnn: 'pending',
        mcts: 'pending',
        reasoning: 'pending'
    });

    // Validate props on component mount
    useEffect(() => {
        console.log("InstructionComparison mounted with props:", {
            hasPrompt: !!prompt,
            promptType: typeof prompt,
            promptLength: prompt?.length || 0,
            userRecordsCount: userRecords?.length || 0,
            gnnSubgraphNodeCount: gnnSubgraph?.nodes?.length || 0,
            mctsSubgraphNodeCount: mctsSubgraph?.nodes?.length || 0
        });
    }, [prompt, userRecords, gnnSubgraph, mctsSubgraph]);

    const generateInstructions = async () => {
        setLoading(true);
        setError(null);
        // Reset model statuses
        setModelStatus({
            gnn: 'pending',
            mcts: 'pending',
            reasoning: 'pending'
        });

        try {
            // Debug prompt value
            console.log("Prompt value check:", {
                prompt,
                type: typeof prompt,
                length: prompt?.length,
                isEmpty: !prompt || prompt.trim() === '',
                userRecordsCount: userRecords?.length || 0
            });

            // Ensure prompt is not empty before proceeding
            if (!prompt || typeof prompt !== 'string' || prompt.trim() === '') {
                throw new Error("Prompt is empty or invalid. Cannot generate instructions without a prompt.");
            }

            // Extract user nodes from records
            const userNodes = [];
            if (userRecords && userRecords.length > 0) {
                for (const record of userRecords) {
                    if (record && record.subject_label && record.object_label) {
                        userNodes.push({
                            id: record.subject_id,
                            label: record.subject_label
                        });
                        userNodes.push({
                            id: record.object_id,
                            label: record.object_label
                        });
                    }
                }
            }

            // Deduplicate user nodes by ID
            const uniqueUserNodes = Array.from(
                new Map(userNodes.map(node => [node.id, node])).values()
            );

            console.log("Sending data to API:", {
                promptLength: prompt.length,
                userNodesCount: uniqueUserNodes.length,
                gnnNodeCount: gnnSubgraph?.nodes?.length || 0,
                mctsNodeCount: mctsSubgraph?.nodes?.length || 0
            });

            const response = await APIService.GenerateInstructions({
                prompt,
                user_nodes: uniqueUserNodes,
                gnn_subgraph: gnnSubgraph || { nodes: [], links: [] },
                mcts_subgraph: mctsSubgraph || { nodes: [], links: [] }
            });

            // Update model statuses based on response
            if (response.model_statuses) {
                setModelStatus(response.model_statuses);
            } else {
                // If no specific statuses, check if instructions look like fallbacks
                const fallbackDetector = (text) => {
                    if (!text) return 'error';
                    if (text.includes("graph-based data structure") &&
                        text.includes("visualization tools") &&
                        text.includes("algorithms to analyze")) {
                        return 'fallback';
                    }
                    return 'success';
                };

                setModelStatus({
                    gnn: fallbackDetector(response.gnn_instructions),
                    mcts: fallbackDetector(response.mcts_instructions),
                    reasoning: fallbackDetector(response.rmodel_instructions)
                });
            }

            setInstructions(response);
        } catch (error) {
            console.error("Error generating instructions:", error);
            setError(error.message || "An error occurred while generating instructions");

            // Update model statuses to error state if we have a global error
            setModelStatus({
                gnn: 'error',
                mcts: 'error',
                reasoning: 'error'
            });
        } finally {
            setLoading(false);
        }
    };

    // Format instruction text with line breaks
    const formatInstructions = (text) => {
        if (!text) return "No instructions available";
        return text.split('\n').map((line, i) => (
            <Typography key={i} variant="body1" paragraph={line.trim().length > 0}>
                {line}
            </Typography>
        ));
    };

    // Handle tab change
    const handleTabChange = (event, newValue) => {
        setActiveTab(newValue);
    };

    // Get winner style
    const getWinnerStyle = (winner, approach) => {
        if (winner === approach) {
            return {
                fontWeight: 'bold',
                color: approach === 'gnn' ? '#4285F4' :
                        approach === 'mcts' ? '#E91E63' : '#34A853'
            };
        }
        return {};
    };

    // Render status indicator
    const renderStatusIndicator = (status) => {
        switch(status) {
            case 'success':
                return <Chip size="small" label="API" color="success" />;
            case 'fallback':
                return <Chip size="small" label="Fallback" color="warning" />;
            case 'error':
                return <Chip size="small" label="Error" color="error" />;
            default:
                return null;
        }
    };

    return (
        <Box mt={3}>
            {!instructions ? (
                <Box textAlign="center" my={3}>
                    {!prompt && (
                        <Alert severity="warning" sx={{ mb: 2 }}>
                            <AlertTitle>Missing Prompt</AlertTitle>
                            A prompt is required to generate instructions. Please ensure a prompt is specified.
                        </Alert>
                    )}

                    <Button
                        variant="contained"
                        color="primary"
                        onClick={generateInstructions}
                        disabled={loading || !prompt}
                        startIcon={loading ? <CircularProgress size={20} color="inherit" /> : null}
                    >
                        {loading ? "Generating..." : "Generate Implementation Instructions"}
                    </Button>

                    {error && (
                        <Alert severity="error" style={{marginTop: 10}}>
                            <AlertTitle>Error</AlertTitle>
                            {error}
                        </Alert>
                    )}
                </Box>
            ) : (
                <>
                    <Paper elevation={1} style={{marginBottom: '20px'}}>
                        <Box display="flex" justifyContent="space-between" alignItems="center" px={2} pt={1}>
                            <Tabs
                                value={activeTab}
                                onChange={handleTabChange}
                                variant="fullWidth"
                                textColor="secondary"
                                indicatorColor="secondary"
                                sx={{ flexGrow: 1 }}
                            >
                                <Tab label="Instructions" />
                                <Tab label="Metrics Comparison" />
                            </Tabs>

                            <Button
                                variant="outlined"
                                color="primary"
                                onClick={generateInstructions}
                                disabled={loading}
                                startIcon={loading ? <CircularProgress size={16} /> : <RefreshIcon />}
                                size="small"
                                sx={{ ml: 2 }}
                            >
                                {loading ? "Regenerating..." : "Regenerate"}
                            </Button>
                        </Box>

                        {/* Instructions Panel */}
                        {activeTab === 0 && (
                            <Box p={3}>
                                <Grid container spacing={3}>
                                    <Grid item xs={12} md={4}>
                                        <Paper elevation={2} style={{padding: '15px', height: '100%', backgroundColor: '#f8f9fa'}}>
                                            <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                                                <Typography variant="h6" color="primary">
                                                    GNN-Based Instructions
                                                </Typography>
                                                {renderStatusIndicator(modelStatus.gnn)}
                                            </Box>
                                            <Divider style={{marginBottom: '10px'}} />
                                            {formatInstructions(instructions.gnn_instructions)}
                                        </Paper>
                                    </Grid>

                                    <Grid item xs={12} md={4}>
                                        <Paper elevation={2} style={{padding: '15px', height: '100%', backgroundColor: '#fef5f9'}}>
                                            <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                                                <Typography variant="h6" color="secondary">
                                                    MCTS-Based Instructions
                                                </Typography>
                                                {renderStatusIndicator(modelStatus.mcts)}
                                            </Box>
                                            <Divider style={{marginBottom: '10px'}} />
                                            {formatInstructions(instructions.mcts_instructions)}
                                        </Paper>
                                    </Grid>

                                    <Grid item xs={12} md={4}>
                                        <Paper elevation={2} style={{padding: '15px', height: '100%', backgroundColor: '#f1f8e9'}}>
                                            <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                                                <Typography variant="h6" style={{color: '#33a853'}}>
                                                    AI Reasoning Instructions
                                                </Typography>
                                                {renderStatusIndicator(modelStatus.reasoning)}
                                            </Box>
                                            <Divider style={{marginBottom: '10px'}} />
                                            {formatInstructions(instructions.rmodel_instructions)}
                                        </Paper>
                                    </Grid>
                                </Grid>
                            </Box>
                        )}

                        {/* Metrics Panel */}
                        {activeTab === 1 && (
                            <Box p={3}>
                                <TableContainer component={Paper} variant="outlined">
                                    <Table>
                                        <TableHead>
                                            <TableRow>
                                                <TableCell><strong>Metric</strong></TableCell>
                                                <TableCell align="center">
                                                    <Stack direction="row" spacing={1} alignItems="center" justifyContent="center">
                                                        <Typography><strong>GNN-Based</strong></Typography>
                                                        {renderStatusIndicator(modelStatus.gnn)}
                                                    </Stack>
                                                </TableCell>
                                                <TableCell align="center">
                                                    <Stack direction="row" spacing={1} alignItems="center" justifyContent="center">
                                                        <Typography><strong>MCTS-Based</strong></Typography>
                                                        {renderStatusIndicator(modelStatus.mcts)}
                                                    </Stack>
                                                </TableCell>
                                                <TableCell align="center">
                                                    <Stack direction="row" spacing={1} alignItems="center" justifyContent="center">
                                                        <Typography><strong>AI Reasoning</strong></Typography>
                                                        {renderStatusIndicator(modelStatus.reasoning)}
                                                    </Stack>
                                                </TableCell>
                                                <TableCell align="center"><strong>Winner</strong></TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {instructions.metrics && Object.entries(instructions.metrics).map(([metric, scores]) => {
                                                // Skip "error" field if present
                                                if (metric === "error") return null;

                                                // Find winner
                                                const values = [
                                                    {approach: 'gnn', score: scores.gnn || 0},
                                                    {approach: 'mcts', score: scores.mcts || 0},
                                                    {approach: 'rmodel', score: scores.rmodel || 0}
                                                ];
                                                const winner = values.reduce((a, b) => a.score > b.score ? a : b).approach;

                                                // Format metric name
                                                const formattedMetric = metric
                                                    .replace(/_/g, ' ')
                                                    .replace(/\b\w/g, l => l.toUpperCase());

                                                return (
                                                    <TableRow key={metric}>
                                                        <TableCell>{formattedMetric}</TableCell>
                                                        <TableCell align="center" style={getWinnerStyle(winner, 'gnn')}>
                                                            <Box display="flex" flexDirection="column" alignItems="center">
                                                                <Typography>{scores.gnn?.toFixed(1) || 'N/A'}</Typography>
                                                                <Rating value={scores.gnn / 2 || 0} precision={0.5} readOnly size="small" />
                                                            </Box>
                                                        </TableCell>
                                                        <TableCell align="center" style={getWinnerStyle(winner, 'mcts')}>
                                                            <Box display="flex" flexDirection="column" alignItems="center">
                                                                <Typography>{scores.mcts?.toFixed(1) || 'N/A'}</Typography>
                                                                <Rating value={scores.mcts / 2 || 0} precision={0.5} readOnly size="small" />
                                                            </Box>
                                                        </TableCell>
                                                        <TableCell align="center" style={getWinnerStyle(winner, 'rmodel')}>
                                                            <Box display="flex" flexDirection="column" alignItems="center">
                                                                <Typography>{scores.rmodel?.toFixed(1) || 'N/A'}</Typography>
                                                                <Rating value={scores.rmodel / 2 || 0} precision={0.5} readOnly size="small" />
                                                            </Box>
                                                        </TableCell>
                                                        <TableCell align="center">
                                                            <Chip
                                                                label={winner === 'gnn' ? 'GNN' : winner === 'mcts' ? 'MCTS' : 'AI Reasoning'}
                                                                color={winner === 'gnn' ? 'primary' : winner === 'mcts' ? 'secondary' : 'default'}
                                                                style={winner === 'rmodel' ? {backgroundColor: '#4caf50', color: 'white'} : {}}
                                                                size="small"
                                                            />
                                                        </TableCell>
                                                    </TableRow>
                                                );
                                            })}

                                            {/* Overall Score Row */}
                                            {instructions.metrics && (
                                                <TableRow style={{backgroundColor: '#f5f5f5'}}>
                                                    <TableCell><strong>Overall Score</strong></TableCell>
                                                    <TableCell align="center">
                                                        <Typography variant="h6" color="primary">
                                                            {(Object.entries(instructions.metrics)
                                                                .filter(([key]) => key !== 'error')
                                                                .reduce((sum, [_, scores]) => sum + (scores.gnn || 0), 0) /
                                                                (Object.keys(instructions.metrics).length - (instructions.metrics.error ? 1 : 0))).toFixed(1)}
                                                        </Typography>
                                                    </TableCell>
                                                    <TableCell align="center">
                                                        <Typography variant="h6" color="secondary">
                                                            {(Object.entries(instructions.metrics)
                                                                .filter(([key]) => key !== 'error')
                                                                .reduce((sum, [_, scores]) => sum + (scores.mcts || 0), 0) /
                                                                (Object.keys(instructions.metrics).length - (instructions.metrics.error ? 1 : 0))).toFixed(1)}
                                                        </Typography>
                                                    </TableCell>
                                                    <TableCell align="center">
                                                        <Typography variant="h6" style={{color: '#33a853'}}>
                                                            {(Object.entries(instructions.metrics)
                                                                .filter(([key]) => key !== 'error')
                                                                .reduce((sum, [_, scores]) => sum + (scores.rmodel || 0), 0) /
                                                                (Object.keys(instructions.metrics).length - (instructions.metrics.error ? 1 : 0))).toFixed(1)}
                                                        </Typography>
                                                    </TableCell>
                                                    <TableCell align="center">
                                                        {/* Leave blank or add explanation */}
                                                    </TableCell>
                                                </TableRow>
                                            )}
                                        </TableBody>
                                    </Table>
                                </TableContainer>

                                <Box mt={2} p={2} bgcolor="#f0f7ff" borderRadius={1}>
                                    <Typography variant="subtitle2" gutterBottom>
                                        Metrics Explanation:
                                    </Typography>
                                    <Typography variant="body2">
                                        <strong>User Focus:</strong> How well instructions address specific user entities and needs
                                    </Typography>
                                    <Typography variant="body2">
                                        <strong>Technological Specificity:</strong> How clearly instructions specify concrete technologies
                                    </Typography>
                                    <Typography variant="body2">
                                        <strong>Actionability:</strong> How immediately actionable and clear the instructions are
                                    </Typography>
                                    <Typography variant="body2">
                                        <strong>Coherence:</strong> How logically instructions flow from concept to implementation
                                    </Typography>
                                    <Typography variant="body2">
                                        <strong>Overall Effectiveness:</strong> Combined assessment of instruction quality
                                    </Typography>
                                </Box>
                            </Box>
                        )}
                    </Paper>
                </>
            )}
        </Box>
    );
};

export default InstructionComparison;