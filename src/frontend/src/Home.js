import React, {useEffect, useState, useRef, useCallback, Suspense, useMemo} from 'react';
import {ErrorBoundary} from 'react-error-boundary';
import './App.css';
import {useQuery, gql} from "@apollo/client";
import {Container, Typography, Box, Paper, CircularProgress, IconButton, useMediaQuery} from '@mui/material';
import {styled} from '@mui/system';
import PlayCircleOutlineIcon from '@mui/icons-material/PlayCircleOutline';
import {ThemeProvider, createTheme} from '@mui/material/styles';
import solutionApproachImage from './solution-approach.png';

const StyledPaper = styled(Paper)(({theme}) => ({
    padding: theme.spacing(3),
    marginBottom: theme.spacing(3),
    overflow: 'hidden',
}));

const sectionStyle = {
    marginBottom: '1.5rem'
};

const paragraphStyle = {
    marginBottom: '1rem',
    lineHeight: '1.6',
};

const listStyle = {
    paddingLeft: '1.5rem',
    marginBottom: '1rem'
};

const listItemStyle = {
    marginBottom: '0.5rem'
};

const LazyForceGraph3D = React.lazy(() => import('react-force-graph-3d'));

const ImageContainer = styled(Box)(({theme}) => ({
    width: '100%',
    height: '500px',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
    '& img': {
        maxWidth: '100%',
        maxHeight: '100%',
        objectFit: 'contain',
    },
    [theme.breakpoints.down('sm')]: {
        height: '300px',
    },
}));

const theme = createTheme();

const VideoLink = styled(Box)(({theme}) => ({
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    '&:hover': {
        backgroundColor: theme.palette.action.hover,
    },
    padding: theme.spacing(2),
    borderRadius: theme.shape.borderRadius,
}));

const GraphContainer = styled(Box)(({theme}) => ({
    width: '100%',
    height: '700px',
    position: 'relative',
    overflow: 'hidden',
    '& canvas': {
        width: '100% !important',
        height: '100% !important',
    },
    [theme.breakpoints.down('sm')]: {
        height: '400px',
    },
}));

const FullWidthBox = styled(Box)({
    width: '100vw',
    position: 'relative',
    left: '50%',
    right: '50%',
    marginLeft: '-50vw',
    marginRight: '-50vw',
});

const DisclaimerBox = styled(Box)(({theme}) => ({
    marginTop: theme.spacing(4),
    padding: theme.spacing(2),
    backgroundColor: theme.palette.grey[100],
    borderRadius: theme.shape.borderRadius,
}));

const NodeTooltip = styled('div')(({theme}) => ({
    position: 'absolute',
    top: '10px',
    left: '10px',
    padding: '10px',
    background: 'white',
    border: '1px solid #ccc',
    borderRadius: '4px',
    pointerEvents: 'none',
    zIndex: 1000,
}));
export default function Home({ graphData }) {
    const graphRef = useRef();
    const graphContainerRef = useRef(null);
    const [graphDimensions, setGraphDimensions] = useState({width: 600, height: 600});
    const [hoverNode, setHoverNode] = useState(null);
    const isSmallScreen = useMediaQuery(theme.breakpoints.down('sm'));
    const updateDimensions = useCallback(() => {
        if (graphContainerRef.current) {
            const width = window.innerWidth; // Use full window width
            const height = graphContainerRef.current.offsetHeight;
            setGraphDimensions({width, height});
        }
    }, []);

    useEffect(() => {
        updateDimensions();

        if (graphRef.current) {
            // Enable navigation controls
            graphRef.current.controls().enableDamping = true;
            graphRef.current.controls().enableZoom = true;
            graphRef.current.controls().enableRotate = true;
            graphRef.current.controls().enablePan = true;
        }
    }, [graphData, updateDimensions]);

    useEffect(() => {
        window.addEventListener('resize', updateDimensions);
        updateDimensions();
        return () => window.removeEventListener('resize', updateDimensions);
    }, [updateDimensions]);


    const handleNodeHover = useCallback((node) => {
        setHoverNode(node);
    }, []);

    return (
        <ThemeProvider theme={theme}>
            <Container maxWidth="lg" sx={{width: '100%', maxWidth: 'none', px: {xs: 2, sm: 3}}}>
                <Typography
                    variant="h4"
                    component="h1"
                    gutterBottom
                    align="center"
                    sx={{
                        mt: 4,
                        fontSize: isSmallScreen ? '2rem' : '3rem'
                    }}
                >
                    FindYourSolution WebApp
                </Typography>

                <Typography variant="h5" sx={{mt: 4, fontSize: {xs: '1rem', sm: '1.5rem'}}}>
                    The webapp helps you defining an opportunity for autonomization. Based on this,
                    the tool will identify corresponding data-driven solution components. Solution recommendations
                    structured in form of solution threads help you to understand a potential path of solution design
                    but also to identify new paths.
                </Typography>

                <StyledPaper elevation={3}>
                    <Typography variant="h5" gutterBottom>
                        Understanding the Problem
                    </Typography>
                    <Typography variant="body1" component="div">
                        <Box>
                            <Box style={sectionStyle}>
                                <Typography style={paragraphStyle}>
                                    Organizations today face a significant challenge in their pursuit of digital
                                    innovation and autonomization. The core issue lies in the gap between business
                                    language—driven by trends, hype, and lofty expectations—and the technical solution
                                    world, which is rapidly evolving due to constant technological advancements.
                                </Typography>
                            </Box>

                            <Box>
                                <Typography style={paragraphStyle}>
                                    This disconnect is further complicated by communication barriers between business
                                    and IT stakeholders, resulting in a lack of shared knowledge essential for
                                    innovation. This leads to significant challenges for solution architects:
                                </Typography>
                                <Box component="ul" style={listStyle}>
                                    <li style={listItemStyle}>Uncertainty due to insufficient business understanding
                                    </li>
                                    <li style={listItemStyle}>Incorrect decisions that increase costs</li>
                                    <li style={listItemStyle}>Difficulty in selecting the right technologies</li>
                                    <li style={listItemStyle}>A high organizational burden to manage the complexity</li>
                                    <li style={listItemStyle}>Without a clear bridge between business goals and
                                        technical execution, organizations struggle to innovate effectively
                                    </li>
                                </Box>
                            </Box>
                        </Box>
                        <VideoLink
                            onClick={() => window.open("https://videos.simpleshow.com/O0SuZzCI6H", "_blank", "noopener,noreferrer")}
                        >
                            <IconButton aria-label="play video" size="large">
                                <PlayCircleOutlineIcon sx={{fontSize: 60, color: 'primary.main'}}/>
                            </IconButton>
                            <Typography variant="body1" sx={{ml: 2}}>
                                Click to watch the video explanation
                            </Typography>
                        </VideoLink>
                    </Typography>
                </StyledPaper>

                <StyledPaper elevation={3}>
                    <Typography variant="h5" gutterBottom>
                        Our Solution Approach
                    </Typography>
                    <Typography variant="body1" component="div">
                        <Box style={sectionStyle}>
                            <Typography style={paragraphStyle}>
                                To address these challenges, the solution developed during the dissertation project
                                focuses on creating a model that demystifies business expectations and aligns them
                                with the building blocks of technical solutions. This alignment supports data-driven
                                design and helps realize business opportunities for autonomization. The key
                                innovation is an ontology that enables:
                            </Typography>
                            <Box component="ul" style={listStyle}>
                                <li style={listItemStyle}>Representation of the domain for better interoperability and
                                    reasoning
                                </li>
                                <li style={listItemStyle}>Shared understanding across corporate boundaries to ensure
                                    collaboration
                                </li>
                                <li style={listItemStyle}>Evaluation and guided development of conceptual models for
                                    structured decision-making
                                </li>
                                <li style={listItemStyle}>The addition of further knowledge components to enhance future
                                    solutions
                                </li>
                                <li style={listItemStyle}>By integrating business expectations with technical realities,
                                    the ontology serves as a conceptual bridge, enabling more effective communication
                                    and better solution design
                                </li>
                            </Box>
                        </Box>
                    </Typography>
                    <ImageContainer>
                        <img src={solutionApproachImage} alt="Solution Approach Visualization"/>
                    </ImageContainer>
                </StyledPaper>

                <StyledPaper elevation={3}>
                    <Typography variant="h5" gutterBottom>
                        Result and Benefits
                    </Typography>
                    <Typography variant="body1" component="div">
                        <Box style={sectionStyle}>
                            <Typography style={paragraphStyle}>
                                The implementation of this ontology has led to the development of an application
                                called "FindYourSolution". This tool helps organizations bridge the gap between
                                business opportunities and technical solutions. "FindYourSolution" enables
                                businesses to:
                            </Typography>
                            <Box component="ul" style={listStyle}>
                                <li style={listItemStyle}>Connect opportunities with solution concepts that perform
                                    complex tasks autonomously
                                </li>
                                <li style={listItemStyle}>Implement agents that self-optimize through environmental
                                    analysis
                                </li>
                                <li style={listItemStyle}>Minimize direct human intervention</li>
                            </Box>
                            <Typography style={paragraphStyle}>
                                Ultimately, the application supports businesses in driving higher efficiency and
                                flexibility through autonomization, enabling them to stay competitive in a rapidly
                                changing technological landscape.
                            </Typography>
                        </Box>
                    </Typography>

                </StyledPaper>

                <FullWidthBox>
                    <StyledPaper elevation={3} sx={{width: '100%', overflow: 'visible'}}>
                        <Typography variant="h5" gutterBottom>
                            Knowledge Graph for Data-driven Solution Design
                        </Typography>
                        <Typography variant="body1" paragraph component="div">
                            Explore our interactive knowledge graph. Hover over nodes to see details and drag to rotate
                            the view.
                        </Typography>
                        <GraphContainer ref={graphContainerRef}>
                            {graphData?.nodes?.length > 0 ? (
                                <>
                                    <ErrorBoundary fallback={<div>Error loading graph</div>}>
                                        <Suspense fallback={<div>Loading graph...</div>}>
                                            <LazyForceGraph3D
                                                ref={graphRef}
                                                graphData={graphData}
                                                height={graphDimensions.height}
                                                width={graphDimensions.width}
                                                nodeAutoColorBy="type"
                                                nodeLabel={node => `${node.type}: ${node.id.split('#').pop()}`}
                                                linkColor={() => 'rgba(255,255,255,0.2)'}
                                                linkWidth={1}
                                                linkDirectionalParticles={2}
                                                linkDirectionalParticleWidth={2}
                                                onNodeHover={handleNodeHover}
                                                onNodeClick={(node) => {
                                                    console.log('Clicked node:', node);
                                                }}
                                                d3Force={(d3) => {
                                                    d3.forceSimulation()
                                                        .force('link', d3.forceLink().id(d => d.id).distance(100))
                                                        .force('charge', d3.forceManyBody().strength(-200))
                                                        .force('center', d3.forceCenter())
                                                        .force('collision', d3.forceCollide(20));
                                                }}
                                                controlType="orbit"
                                                enableNodeDrag={false}
                                                enableNavigationControls={true}
                                                showNavInfo={true}
                                            />
                                        </Suspense>
                                    </ErrorBoundary>
                                    {hoverNode && (
                                        <NodeTooltip>
                                            <Typography variant="body2">
                                                <strong>Type:</strong> {hoverNode.type}
                                            </Typography>
                                            <Typography variant="body2">
                                                <strong>ID:</strong> {hoverNode.id.split('#').pop()}
                                            </Typography>
                                        </NodeTooltip>
                                    )}
                                </>
                            ) : (
                                <Typography>No graph data available.</Typography>
                            )}
                        </GraphContainer>
                    </StyledPaper>
                </FullWidthBox>
                <DisclaimerBox>
                    <Typography variant="body2" align="center">
                        The prototypical application was developed as part of the dissertation project of Daniel
                        Burkhardt at the Ferdinand Steinbeis Institute. Thus, this application does not serve in any way
                        commercial reasons.
                    </Typography>
                </DisclaimerBox>
            </Container>
        </ThemeProvider>
    )
}
