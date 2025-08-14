from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain.tools.graph_query.cypher import CypherQueryTool
import os
import threading
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# Connect to Neo4j
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)

# Initialize LLM
llm = Ollama(model="llama3", temperature=0.0)

# --- Commented out the Cypher tool creation and agent creation ---
# Create Cypher tool
# cypher_tool = CypherQueryTool(graph=graph)

# Build the agent manually
# agent_executor = AgentExecutor.from_agent_and_tools(
#     agent=create_tool_calling_agent(llm, tools=[cypher_tool]),
#     tools=[cypher_tool],
#     verbose=True,
# )

# Comment out the run_graph_query function
# def run_graph_query(user_query):
#     """Run a user query through the Neo4j agent."""
#     return agent_executor.invoke({"input": user_query})

# --- You also commented this already, good ---
# from neo4j_verify_agent import run_graph_query

# Determine which .env file to load
env_file = '.env.production' if os.getenv('FLASK_ENV') == 'production' else '.env.development'
# Load environment variables from the appropriate file
load_dotenv(env_file)
