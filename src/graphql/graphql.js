const { Neo4jGraphQL } = require("@neo4j/graphql");
const { ApolloServer, gql } = require("apollo-server");
const neo4j = require("neo4j-driver");
const { toGraphQLTypeDefs } = require("@neo4j/introspector");
const fs = require('fs');
const dotenv = require('dotenv');

// Load environment variables from appropriate file
if (process.env.NODE_ENV !== 'production') {
  dotenv.config({ path: '.env.development' });
  console.log('Running in development mode');
}

// Use the same variable names as production for simplicity
const AURA_ENDPOINT = process.env.NODE_ENV === 'production'
  ? process.env.NEO4J_URI
  : process.env.DEV_NEO4J_URI;
const USERNAME = process.env.NODE_ENV === 'production'
  ? process.env.NEO4J_USER
  : process.env.DEV_NEO4J_USER;
const PASSWORD = process.env.NODE_ENV === 'production'
  ? process.env.NEO4J_PASSWORD
  : process.env.DEV_NEO4J_PASSWORD;

console.log(`Connecting to Neo4j at: ${AURA_ENDPOINT}`);

// Create Neo4j driver
const driver = neo4j.driver(
  AURA_ENDPOINT,
  neo4j.auth.basic(USERNAME, PASSWORD)
);

const sessionFactory = () => driver.session({
  defaultAccessMode: neo4j.session.READ
});

async function main() {
  try {
    const readonly = true;
    const typeDefs = await toGraphQLTypeDefs(sessionFactory, readonly);
    const neoSchema = new Neo4jGraphQL({
      typeDefs,
      driver
    });

    neoSchema.getSchema().then((schema) => {
      const server = new ApolloServer({
        schema,
        cors: {
          origin: ['https://findyoursolution.ai', 'http://localhost:3000', 'http://localhost:3001'],
          credentials: true,
          methods: ['GET', 'POST', 'OPTIONS'],
          allowedHeaders: ['Content-Type', 'Authorization', 'Apollo-Require-Preflight']
        },
        cache: "bounded"
      });

      // Listen on all interfaces (0.0.0.0) instead of just localhost
      server.listen({
        host: '0.0.0.0',
        port: process.env.NODE_ENV === 'production' ? 4000 : 4001
      }).then(({url}) => {
        console.log(`🚀 GraphQL Server ready at ${url} (${process.env.NODE_ENV || 'development'} mode)`);
      });
    });
  } catch (error) {
    console.error("Error starting GraphQL server:", error);
  }
}

main();