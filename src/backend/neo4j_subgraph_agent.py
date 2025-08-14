import logging
from sentence_transformers import SentenceTransformer, util
from neo4j import GraphDatabase


# Neo4j config
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "test1234"

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('frontend_subgraph_agent')

# Initialize embedding model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("SentenceTransformer model loaded successfully")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {str(e)}")
    embedding_model = None

# Job queue
subgraph_jobs = {}


def clean_string(s):
    if s is None:
        return ''
    cleaned = str(s).replace('\xa0', ' ').replace('\n', ' ').replace('\t', ' ').strip()
    return ' '.join(cleaned.split())


def get_label_from_node(node):
    props = dict(node)
    for key in ["localName", "name", "label", "title"]:
        if key in props and props[key]:
            return clean_string(props[key])
    return None


def fetch_full_graph_from_neo4j():
    with driver.session() as session:
        result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m")
        nodes = {}
        links = []

        for record in result:
            n, r, m = record["n"], record["r"], record["m"]

            nid = n.element_id if hasattr(n, "element_id") else str(n.id)
            mid = m.element_id if hasattr(m, "element_id") else str(m.id)

            source_label = get_label_from_node(n)
            target_label = get_label_from_node(m)
            relation_label = clean_string(r.type)

            # Add source node if not already in dict
            if nid not in nodes:
                nodes[nid] = {
                    "id": nid,
                    "label": get_label_from_node(n) or nid,
                    "type": list(n.labels)[0] if n.labels else "Node"
                }

            # Add target node
            if mid not in nodes:
                nodes[mid] = {
                    "id": mid,
                    "label": target_label,
                    "type": list(m.labels)[0] if m.labels else "Node"
                }

            # Add edge
            links.append({
                "source": nid,
                "target": mid,
                "label": relation_label
            })

        print(f"[DEBUG] Extracted {len(nodes)} nodes and {len(links)} links from Neo4j.")
        for link in links[:20]:
            source = nodes.get(link['source'], {"label": f"[MISSING:{link['source']}]"})["label"]
            target = nodes.get(link['target'], {"label": f"[MISSING:{link['target']}]"})["label"]
            relation = link['label']
            print(f"[Neo4J]  ({source}, {relation}, {target})")

        return {"nodes": list(nodes.values()), "links": links}
