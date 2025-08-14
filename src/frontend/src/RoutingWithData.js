// RoutingWithData.js
import React, { useMemo, useEffect, useState } from "react";
import { Route, Routes, BrowserRouter } from 'react-router-dom';
import { styled, ThemeProvider, createTheme } from '@mui/material/styles';
import { AppBar, Toolbar, Typography, Button, CssBaseline } from '@mui/material';
import { useQuery, gql } from "@apollo/client";
import Home from './Home';
import App from './App';
import CombinedResults from "./CombinedResults";

const theme = createTheme();

const Root = styled('div')(({ theme }) => ({
    flexGrow: 1,
}));

const Title = styled(Typography)(({ theme }) => ({
    flexGrow: 1,
}));

const mostRecentQuery = gql`
    query mostRecentQuery {
        namedIndividual4s {
            uri
            namedIndividual3SIssolvedbyConnection {
                edges {
                    node {
                        uri
                        actsIsappliedinConnection {
                            edges {
                                node { uri }
                            }
                        }
                        analyticsDesignsIsImplementingConnection {
                            edges {
                                node {uri}
                            }
                        }
                        issolvedbyDataProcessingTasksConnection {
                            edges {
                                node {uri}
                            }
                        }
                        issolvedbyDataStoragesConnection {
                            edges {
                                node { uri}
                            }
                        }
                    }
                }
            }
        }
    }
`;

const formatData = (data) => {
    if (!data || !data.namedIndividual4s) {
        return { nodes: [], links: [] };
    }

    const nodes = [], links = [], added = new Set();

    const addNode = (uri, type) => {
        if (!added.has(uri)) {
            nodes.push({ id: uri, type });
            added.add(uri);
        }
    };

    const addLink = (source, target) => {
        if (added.has(source) && added.has(target)) {
            links.push({ source, target });
        }
    };

    data.namedIndividual4s.forEach(ni4 => {
        addNode(ni4.uri, 'NamedIndividual4');
        ni4.namedIndividual3SIssolvedbyConnection.edges.forEach(({ node: ni3 }) => {
            addNode(ni3.uri, 'NamedIndividual3');
            addLink(ni4.uri, ni3.uri);

            ni3.actsIsappliedinConnection.edges.forEach(({ node }) => {
                addNode(node.uri, 'Act');
                addLink(ni3.uri, node.uri);
            });

            ni3.analyticsDesignsIsImplementingConnection.edges.forEach(({ node }) => {
                addNode(node.uri, 'AnalyticsDesign');
                addLink(ni3.uri, node.uri);
            });

            ni3.issolvedbyDataProcessingTasksConnection.edges.forEach(({ node }) => {
                addNode(node.uri, 'DataProcessingTask');
                addLink(node.uri, ni3.uri);
            });

            ni3.issolvedbyDataStoragesConnection.edges.forEach(({ node }) => {
                addNode(node.uri, 'DataStorage');
                addLink(node.uri, ni3.uri);
            });
        });
    });

    return { nodes, links };
};

export default function RoutingWithData() {
    const { loading, error, data } = useQuery(mostRecentQuery);
    const [graphData, setGraphData] = useState({ nodes: [], links: [] });

    const formattedGraphData = useMemo(() => formatData(data), [data]);

    useEffect(() => {
        if (data) {
            setGraphData(formattedGraphData);
        }
    }, [data, formattedGraphData]);

    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error loading data.</div>;

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <BrowserRouter>
                <Root>
                    <AppBar position="static" style={{ background: '#595959' }}>
                        <Toolbar>
                            <Title variant="h6">Solution Finder</Title>
                            <Button color="inherit" href="/">Home</Button>
                            <Button color="inherit" href="/app">App</Button>
                        </Toolbar>
                    </AppBar>
                </Root>
                <Routes>
                    <Route exact path="/" element={<Home graphData={graphData} />} />
                    <Route exact path="/app" element={<App graphData={graphData} />} />
                    <Route exact path="/results" element={<CombinedResults />} />
                </Routes>
            </BrowserRouter>
        </ThemeProvider>
    );
}
