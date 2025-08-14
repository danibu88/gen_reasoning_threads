export default class APIService {
    static BASE_URL = process.env.REACT_APP_BACKEND || 'http://localhost:3002';

    static async fetchWithErrorHandling(endpoint, method, body) {
        try {
            console.log(`Attempting to connect to ${this.BASE_URL}${endpoint}`);

            const response = await fetch(`${this.BASE_URL}${endpoint}`, {
                method,
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(body)
            });

            const responseText = await response.text();
            let data;

            try {
                data = JSON.parse(responseText);
            } catch (e) {
                console.error("Failed to parse response as JSON:", responseText);
                throw new Error(`Invalid JSON response from server: ${responseText}`);
            }

            if (!response.ok) {
                console.error(`Server responded with status ${response.status}:`, data);
                throw new Error(`HTTP error! status: ${response.status}, message: ${JSON.stringify(data)}`);
            }

            if (data.error) {
                console.error("Error from server:", data.error);
                if (data.traceback) {
                    console.error("Traceback:", data.traceback);
                }
                throw new Error(data.error);
            }

            return data;
        } catch (error) {
            if (error.message === 'Failed to fetch') {
                console.error(`Connection to server failed (${this.BASE_URL}${endpoint}). Is the backend server running?`);
                throw new Error(`Connection to server failed. Is the backend server running at ${this.BASE_URL}?`);
            } else {
                console.error(`Error in ${endpoint}:`, error);
                throw error;
            }
        }
    }

    static async SearchQuery(text_input) {
        if (!text_input || typeof text_input !== 'string' || text_input.trim() === '') {
            throw new Error("Please provide a non-empty search query.");
        }
        const data = await this.fetchWithErrorHandling('/query/', 'POST', {text_input});
        return data;
    }

    static async GetGraphDetails(data) {
        if (!data || !data.text_input || typeof data.text_input !== 'string' || data.text_input.trim() === '') {
            throw new Error("Please provide a non-empty input for graph details.");
        }

        // Start the job
        const jobResponse = await this.fetchWithErrorHandling('/result/', 'POST', data);
        if (!jobResponse || !jobResponse.job_id || jobResponse.status === 'error') {
            throw new Error(`Job creation failed: ${jobResponse.error || 'No job ID returned'}`);
        }

        const jobId = jobResponse.job_id;

        if (!jobId) {
            throw new Error("Server did not return a valid job ID");
        }

        // Set up polling for job status
        return new Promise((resolve, reject) => {
            let attempts = 0;
            const maxAttempts = 240; // 20 minutes (with 5-second intervals)

            const checkStatus = async () => {
                const baseDelay = 2000; // Start with 2 seconds
                const maxDelay = 30000; // Max 30 seconds between polls
                const currentDelay = Math.min(baseDelay * Math.pow(1.5, attempts), maxDelay);

                try {
                    attempts++;
                    const statusResponse = await this.checkJobStatus(jobId);

                    console.log("Job status response:", statusResponse);

                    // Add defensive check before accessing the progress property
                    if (statusResponse === undefined || statusResponse === null) {
                        console.warn('Received undefined or null response from job status check');
                        // Use a default progress value
                        if (this.onProgressUpdate && typeof this.onProgressUpdate === 'function') {
                            this.onProgressUpdate(0);
                        }

                        // Wait and try again
                        setTimeout(checkStatus, currentDelay);
                        return;
                    }

                    // Safely access progress with a default value
                    const progress = statusResponse && typeof statusResponse.progress === 'number'
                        ? statusResponse.progress
                        : 0;

                    // Return progress information through a callback if provided
                    if (this.onProgressUpdate && typeof this.onProgressUpdate === 'function') {
                        this.onProgressUpdate(progress);
                    }

                    if (statusResponse.status === 'complete') {
                        // Check for subgraphData
                        console.log("Job completed. Checking for subgraphData:", {
                            hasSubgraphData: 'subgraphData' in statusResponse,
                            subgraphDataKeys: statusResponse.subgraphData ? Object.keys(statusResponse.subgraphData) : [],
                            subgraphDataType: typeof statusResponse.subgraphData
                        });

                        // Prepare a clean result object with consistent structure
                        const result = {
                            // Include the result field if present
                            result: statusResponse.result || {},

                            // Include the subgraphData, with fallback to an empty structure
                            subgraphData: statusResponse.subgraphData || {
                                user_subgraph: {nodes: [], links: []},
                                domain_subgraph: {nodes: [], links: []},
                                llm_subgraph: {nodes: [], links: []},
                                combined_subgraph: {nodes: [], links: []}
                            }
                        };

                        resolve(result);
                        return;
                    } else if (statusResponse.status === 'error') {
                        console.error("Job failed:", statusResponse.error);
                        reject(new Error(statusResponse.error || 'Job processing failed'));
                        return;
                    } else if (attempts >= maxAttempts) {
                        console.error("Job timed out after maximum attempts");
                        reject(new Error('Job processing timed out'));
                        return;
                    }

                    // Schedule next check with adaptive delay
                    setTimeout(checkStatus, currentDelay);
                } catch (error) {
                    console.error("Error checking job status:", error);

                    // On error, retry a few times with increasing delays
                    if (attempts < 5) {
                        setTimeout(checkStatus, 2000 * attempts);
                    } else {
                        reject(error);
                    }
                }
            };

            // Start the polling
            checkStatus();
        });
    }

    // Add a method to set the progress callback
    static setProgressCallback(callback) {
        this.onProgressUpdate = callback;
    }

    static async checkJobStatus(jobId) {
        if (!jobId) {
            console.error("Invalid job ID provided to checkJobStatus");
            return {
                status: 'error',
                error: 'Invalid job ID',
                progress: 0  // Provide a default progress value
            };
        }

        try {
            const response = await this.fetchWithErrorHandling(`/job_status/${jobId}`, 'GET');

            console.log("Job status response:", {
                status: response?.status,
                progress: response?.progress,
                hasResult: !!response?.result,
                hasSubgraphData: !!response?.subgraphData,
                subgraphDataKeys: response?.subgraphData ? Object.keys(response.subgraphData) : []
            });

            // Ensure the response always has a progress field
            if (response && !('progress' in response)) {
                console.warn("Response is missing progress field, adding default value");
                response.progress = 0;
            }

            return response;
        } catch (error) {
            console.error('Error checking job status:', error);

            // Return a valid error response with a progress field
            return {
                status: 'error',
                error: error.message || 'Error checking job status',
                progress: 0  // Always include a default progress value
            };
        }
    }

    static async SummarizePapers(papers, maxWords = 500, graphData) {
        console.log("SummarizePapers called with:", {
            papersType: typeof papers,
            papersIsArray: Array.isArray(papers),
            papersLength: typeof papers === 'string' ? papers.length : (Array.isArray(papers) ? papers.length : 'unknown'),
            maxWords,
            hasGraphData: !!graphData,
            graphDataNodeCount: graphData?.nodes?.length || 0
        });

        // Handle different paper formats
        let content = '';
        if (typeof papers === 'string') {
            content = papers;
        } else if (Array.isArray(papers)) {
            content = papers.map(p => {
                if (typeof p === 'string') return p;
                return p.content || p.text || p.body || JSON.stringify(p);
            }).join('\n\n');
        } else if (papers && typeof papers === 'object') {
            content = JSON.stringify(papers);
        }

        // Limit content length
        if (content.length > 200000) {
            content = content.substring(0, 200000) + '...';
        }

        try {
            const response = await this.fetchWithErrorHandling('/summarize/', 'POST', {
                papers: content,
                max_words: maxWords,
                graph_data: graphData
            });

            console.log("Summarize response:", response);

            // Handle both structured and simple string responses
            if (typeof response === 'string') {
                return {summary: response};
            } else if (response && response.summary) {
                return response;
            } else {
                console.warn("Unexpected summarize response format:", response);
                return {summary: "Unable to generate summary from the response."};
            }
        } catch (error) {
            console.error("Error in SummarizePapers:", error);
            return {summary: "Error generating summary: " + error.message};
        }
    }

    // Complete fixed implementation of GenerateInstructions in APIService.js

    static async GenerateInstructions(data) {
        try {
            // Validate input before sending
            if (!data || !data.prompt) {
                console.error("GenerateInstructions: Missing required parameter 'prompt'");
                throw new Error("Missing required parameter: prompt");
            }

            console.log("Sending instruction generation request:", {
                promptLength: data.prompt.length,
                userNodesCount: data.user_nodes?.length || 0,
                gnnSubgraphNodes: data.gnn_subgraph?.nodes?.length || 0,
                mctsSubgraphNodes: data.mcts_subgraph?.nodes?.length || 0
            });

            // Pass the data directly to fetchWithErrorHandling
            const response = await this.fetchWithErrorHandling('/generate_instructions/', 'POST', data);
            console.log("Instruction generation successful");
            return response; // fetchWithErrorHandling already parses JSON

        } catch (error) {
            console.error("Error in GenerateInstructions:", error);
            throw error;
        }
    }

    static async PollGraphDataCompletion(jobId, maxAttempts = 240, progressCallback = null) {
        return new Promise((resolve, reject) => {
            let attempts = 0;

            const checkStatus = async () => {
                if (attempts >= maxAttempts) {
                    reject(new Error('Graph data processing timed out'));
                    return;
                }

                attempts++;

                try {
                    const response = await this.CheckGraphDataStatus(jobId);

                    // Safely handle progress with a default value
                    const progress = response && typeof response.progress === 'number'
                        ? response.progress
                        : 0;

                    // Call progress callback if provided
                    if (progressCallback && typeof progressCallback === 'function') {
                        progressCallback(progress);
                    }

                    // Call the global progress callback if set
                    if (this.onProgressUpdate && typeof this.onProgressUpdate === 'function') {
                        this.onProgressUpdate(progress);
                    }

                    if (response.status === 'complete') {
                        resolve(response);
                    } else if (response.status === 'error') {
                        console.error('Graph data processing failed:', response.error);
                        reject(new Error(response.error || 'Graph data processing failed'));
                    } else {
                        // Calculate adaptive delay based on attempt number
                        const baseDelay = 1000; // Start with 1 second
                        const maxDelay = 10000; // Max 10 seconds
                        const currentDelay = Math.min(baseDelay * Math.pow(1.2, attempts / 10), maxDelay);

                        // Still processing, check again after delay
                        setTimeout(checkStatus, currentDelay);
                    }
                } catch (error) {
                    console.error('Error checking graph data status:', error);

                    // On error, retry a few times with increasing delays
                    if (attempts < 5) {
                        setTimeout(checkStatus, 2000 * attempts);
                    } else {
                        reject(error);
                    }
                }
            };

            // Start the polling
            checkStatus();
        });
    }
}