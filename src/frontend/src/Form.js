import React, {useState, useEffect} from 'react';
import APIService from "./APIService";
import {
    Button, Box, TextField, Tooltip, FormControlLabel, Checkbox,
    Select, MenuItem, InputLabel, FormControl, Typography, Accordion,
    AccordionSummary, AccordionDetails, Chip, LinearProgress
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import {useNavigate} from "react-router-dom";
import './App.css';


const Form = ({graphData}) => {
        const navigate = useNavigate();
        const [text_input, setText_input] = useState('');
        const [keywords, setKeywords] = useState([]);
        const [error, setError] = useState(null);
        const [advancedOptions, setAdvancedOptions] = useState(false);
        const [isBuilding, setIsBuilding] = useState(false);
        const [jobProgress, setJobProgress] = useState(0);
        // Self-Determination
        const [isHumanDependent, setIsHumanDependent] = useState(false);
        const [isSelfGoalDefined, setIsSelfGoalDefined] = useState(false);
        const [isSelfGoverned, setIsSelfGoverned] = useState(false);
        const [isSelfProcessDefined, setIsSelfProcessDefined] = useState(false);

        // Organization
        const [classifier, setClassifier] = useState('Classification_nonlinear');

        // Self-Organization
        const [isReinforcing, setIsReinforcing] = useState(false);
        const [isSelfSelecting, setIsSelfSelecting] = useState(false);

        // Data and Process Management
        const [dataAccess, setDataAccess] = useState('fully');
        const [isDataRedundant, setIsDataRedundant] = useState(false);
        const [isDataHeterogenous, setIsDataHeterogenous] = useState(false);
        const [isProcessingIandK, setIsProcessingIandK] = useState(false);

        // Interaction
        const [performance, setPerformance] = useState('real-time');
        const [performanceQuality, setPerformanceQuality] = useState('productive_state');

        // Business Area
        const [businessArea, setBusinessArea] = useState('Healthcare');

        // System Orientation
        const [systemOrientation, setSystemOrientation] = useState('Centralized');

        // Approach
        const [approach, setApproach] = useState('Deep Learning');

        // onChange functions
        const handleKeywordsChange = (event) => {
            setKeywords(event.target.value.split(',').map(keyword => keyword.trim()));
        };

        useEffect(() => {
            console.log("graphData received in Form:", graphData);
        }, [graphData]);

        useEffect(() => {
            APIService.setProgressCallback(setJobProgress);
            return () => {
                APIService.setProgressCallback(null);
            };
        }, []);

        const handleCombinedAction = async (event) => {
            event.preventDefault();
            setError(null);
            setIsBuilding(true);
            setJobProgress(0);

            try {
                if (!text_input.trim()) {
                    throw new Error("Please provide a non-empty search query.");
                }

                // Start the processes sequentially to avoid overwhelming backend
                const result = await handleGetGraphDetails();
                console.log("API result returned:", result);

                // Log subgraph data structure to verify it exists
                console.log("Subgraph data:", {
                    exists: !!result.subgraphData,
                    structure: result.subgraphData ? Object.keys(result.subgraphData) : 'N/A',
                    combinedExists: !!result.subgraphData?.combined_subgraph,
                    nodesCount: result.subgraphData?.combined_subgraph?.nodes?.length || 0,
                    linksCount: result.subgraphData?.combined_subgraph?.links?.length || 0
                });

                // Get search results
                const search = await APIService.SearchQuery(text_input);

                // Navigate to results with all data
                navigate('/results', {
                    state: {
                        prompt: text_input,
                        search,
                        subgraphData: result.subgraphData, // Make sure this matches exactly what your backend is returning
                        approach,
                        classifier,
                        systemOrientation
                    }
                });
            } catch (error) {
                console.error("Error processing request:", error);
                setError(error.message || "An error occurred while processing your request");
            } finally {
                setIsBuilding(false);
            }
        };

        const handleGetGraphDetails = async () => {
            try {
                const requestData = {
                    text_input,
                    keywords,
                    isHumanDependent,
                    isSelfGoalDefined,
                    isSelfGoverned,
                    isSelfProcessDefined,
                    classifier,
                    isReinforcing,
                    isSelfSelecting,
                    dataAccess,
                    isDataRedundant,
                    isDataHeterogenous,
                    isProcessingIandK,
                    performance,
                    performanceQuality,
                    businessArea,
                    systemOrientation,
                    approach
                };

                console.log("Sending request data to GetGraphDetails:", requestData);

                // Get the result from the API
                const result = await APIService.GetGraphDetails(requestData);
                console.log("GetGraphDetails response:", result);

                return result;
            } catch (error) {
                console.error("Error getting graph details:", error);
                throw error;
            }
        };

        useEffect(() => {
            console.log("Graph data received in Form:", graphData);
        }, [graphData]);

        return (
            <form onSubmit={handleCombinedAction}>
                {error && (
                    <Typography color="error" sx={{mt: 2, mb: 2}}>
                        Error: {error}
                    </Typography>
                )}

                <Tooltip title="Provide a detailed description of the opportunity or problem you want to address">
                    <TextField
                        label="Query (provide a detailed opportunity description)"
                        multiline
                        rows={10}
                        value={text_input}
                        onChange={(e) => setText_input(e.target.value)}
                        fullWidth
                        required
                        sx={{mb: 2}}
                    />
                </Tooltip>

                <Tooltip title="Enter keywords related to your query, separated by commas">
                    <TextField
                        label="Keywords"
                        value={keywords.join(', ')}
                        onChange={handleKeywordsChange}
                        fullWidth
                        sx={{mb: 2}}
                    />
                </Tooltip>

                <Box sx={{mb: 2}}>
                    {keywords.map((keyword, index) => (
                        <Chip key={index} label={keyword} sx={{mr: 1, mb: 1}}/>
                    ))}
                </Box>

                <Accordion
                    expanded={advancedOptions}
                    onChange={() => setAdvancedOptions(!advancedOptions)}
                >
                    <AccordionSummary expandIcon={<ExpandMoreIcon/>}>
                        <Typography>Advanced Options</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <Typography variant="h6" gutterBottom>Self-Determination: the solution...</Typography>
                        <Box sx={{mb: 2}}>
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={isHumanDependent}
                                        onChange={(e) => setIsHumanDependent(e.target.checked)}
                                        name="humandependency"
                                    />
                                }
                                label="is not dependent on humans"
                            />
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={isSelfGoalDefined}
                                        onChange={(e) => setIsSelfGoalDefined(e.target.checked)}
                                        name="selfgoal"
                                    />
                                }
                                label="defines goals itself"
                            />
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={isSelfGoverned}
                                        onChange={(e) => setIsSelfGoverned(e.target.checked)}
                                        name="selfgov"
                                    />
                                }
                                label="defines its governance"
                            />
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={isSelfProcessDefined}
                                        onChange={(e) => setIsSelfProcessDefined(e.target.checked)}
                                        name="selfprocess"
                                    />
                                }
                                label="defines its processes"
                            />
                        </Box>

                        <Typography variant="h6" gutterBottom>Organization: the solution...</Typography>
                        <FormControl fullWidth sx={{mb: 2}}>
                            <InputLabel>is executing this task</InputLabel>
                            <Select
                                value={classifier}
                                onChange={(e) => setClassifier(e.target.value)}
                                label="is executing this task"
                            >
                                <MenuItem value="Classification_nonlinear">Classification_nonlinear</MenuItem>
                                <MenuItem value="Clustering_linear">Clustering_linear</MenuItem>
                                <MenuItem value="Regression">Regression</MenuItem>
                                <MenuItem value="None">None</MenuItem>
                            </Select>
                        </FormControl>

                        <Typography variant="h6" gutterBottom>Self-Organization: the solution...</Typography>
                        <Box sx={{mb: 2}}>
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={isReinforcing}
                                        onChange={(e) => setIsReinforcing(e.target.checked)}
                                        name="reinforce"
                                    />
                                }
                                label="is reinforcing itself"
                            />
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={isSelfSelecting}
                                        onChange={(e) => setIsSelfSelecting(e.target.checked)}
                                        name="selfselect"
                                    />
                                }
                                label="is selecting by itself"
                            />
                        </Box>

                        <Typography variant="h6" gutterBottom>Data and Process Management: the
                            solution...</Typography>
                        <FormControl fullWidth sx={{mb: 2}}>
                            <InputLabel>accessing data</InputLabel>
                            <Select
                                value={dataAccess}
                                onChange={(e) => setDataAccess(e.target.value)}
                                label="accessing data"
                            >
                                <MenuItem value="fully">fully</MenuItem>
                                <MenuItem value="partially">partially (no personal data)</MenuItem>
                                <MenuItem value="None">None</MenuItem>
                            </Select>
                        </FormControl>
                        <Box sx={{mb: 2}}>
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={isDataRedundant}
                                        onChange={(e) => setIsDataRedundant(e.target.checked)}
                                        name="dataredundant"
                                    />
                                }
                                label="requiring data redundancy"
                            />
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={isDataHeterogenous}
                                        onChange={(e) => setIsDataHeterogenous(e.target.checked)}
                                        name="dataheterogeneous"
                                    />
                                }
                                label="accessing heterogeneous data"
                            />
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={isProcessingIandK}
                                        onChange={(e) => setIsProcessingIandK(e.target.checked)}
                                        name="infknoprocessing"
                                    />
                                }
                                label="processing information and knowledge"
                            />
                        </Box>

                        <Typography variant="h6" gutterBottom>Interaction: the solution...</Typography>
                        <FormControl fullWidth sx={{mb: 2}}>
                            <InputLabel>Requiring performance of</InputLabel>
                            <Select
                                value={performance}
                                onChange={(e) => setPerformance(e.target.value)}
                                label="Requiring performance of"
                            >
                                <MenuItem value="real-time">Real-time</MenuItem>
                                <MenuItem value="near real-time">Near real-time</MenuItem>
                                <MenuItem value="batch">Batch</MenuItem>
                                <MenuItem value="none">None</MenuItem>
                            </Select>
                        </FormControl>
                        <FormControl fullWidth sx={{mb: 2}}>
                            <InputLabel>Requiring state of performance</InputLabel>
                            <Select
                                value={performanceQuality}
                                onChange={(e) => setPerformanceQuality(e.target.value)}
                                label="Requiring state of performance"
                            >
                                <MenuItem value="productive_state">Productive</MenuItem>
                                <MenuItem value="proof_of_concept">Proof-of-Concept</MenuItem>
                                <MenuItem value="prototype">Prototype</MenuItem>
                                <MenuItem value="none">None</MenuItem>
                            </Select>
                        </FormControl>

                        <Typography variant="h6" gutterBottom>Business Area: the solution...</Typography>
                        <FormControl fullWidth sx={{mb: 2}}>
                            <InputLabel>Applied in</InputLabel>
                            <Select
                                value={businessArea}
                                onChange={(e) => setBusinessArea(e.target.value)}
                                label="Applied in"
                            >
                                <MenuItem value="Healthcare">Healthcare</MenuItem>
                                <MenuItem value="Production">Production</MenuItem>
                                <MenuItem value="Risk-Management">Risk-Management</MenuItem>
                                <MenuItem value="Mobility">Mobility</MenuItem>
                                <MenuItem value="none">None</MenuItem>
                            </Select>
                        </FormControl>

                        <Typography variant="h6" gutterBottom>System Orientation: the solution...</Typography>
                        <FormControl fullWidth sx={{mb: 2}}>
                            <InputLabel>Applied based on an architecture that is</InputLabel>
                            <Select
                                value={systemOrientation}
                                onChange={(e) => setSystemOrientation(e.target.value)}
                                label="Applied based on an architecture that is"
                            >
                                <MenuItem value="Centralized">Centralized</MenuItem>
                                <MenuItem value="Decentralized">Decentralized</MenuItem>
                                <MenuItem value="Fragmented">Fragmented</MenuItem>
                                <MenuItem value="none">None</MenuItem>
                            </Select>
                        </FormControl>

                        <Typography variant="h6" gutterBottom>Approach: the solution...</Typography>
                        <FormControl fullWidth sx={{mb: 2}}>
                            <InputLabel>Applying</InputLabel>
                            <Select
                                value={approach}
                                onChange={(e) => setApproach(e.target.value)}
                                label="Applying"
                            >
                                <MenuItem value="Deep Learning">Deep Learning</MenuItem>
                                <MenuItem value="Machine Learning">Machine Learning</MenuItem>
                                <MenuItem value="Statistical Learning">Statistical Learning</MenuItem>
                                <MenuItem value="Text Analysis">Text Analysis</MenuItem>
                                <MenuItem value="Geometry">Geometry</MenuItem>
                                <MenuItem value="None">None</MenuItem>
                            </Select>
                        </FormControl>
                    </AccordionDetails>
                </Accordion>

                <Button
                    variant="contained"
                    color="primary"
                    type="submit"
                    disabled={isBuilding}
                    sx={{mb: 2}}
                >
                    {isBuilding ? 'Processing...' : 'Get Recommendations and Build Graph'}
                </Button>

                {
                    isBuilding && (
                        <Box sx={{width: '100%', mb: 2}}>
                            <Typography variant="body2" color="text.secondary">
                                Building the sub-graphs... This process might take 5-10 minutes.
                            </Typography>
                            <LinearProgress
                                variant="determinate"
                                value={jobProgress}
                                sx={{mt: 1}}
                            />
                            <Typography variant="caption" color="text.secondary" align="center"
                                        sx={{display: 'block', mt: 1}}>
                                {Math.round(jobProgress)}% Complete
                            </Typography>
                        </Box>
                    )
                }


                {error && (
                    <Typography color="error" sx={{mt: 2, mb: 2}}>
                        Error: {error}
                    </Typography>
                )}
            </form>
        )
            ;
    }
;
export default Form;