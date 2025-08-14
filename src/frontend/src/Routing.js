// Routing.js
import React from 'react';
import {ApolloProvider, ApolloClient, InMemoryCache, HttpLink} from "@apollo/client";
import RoutingWithData from './RoutingWithData';

const createApolloClient = () => {
    const graphqlUrl = process.env.REACT_APP_GRAPH_QL;
    const link = new HttpLink({
        uri: graphqlUrl.endsWith('/') ? graphqlUrl.slice(0, -1) : graphqlUrl
    });
    return new ApolloClient({
        link,
        cache: new InMemoryCache()
    });
};

export default function Routing() {
    return (
        <ApolloProvider client={createApolloClient()}>
            <RoutingWithData />
        </ApolloProvider>
    );
}
