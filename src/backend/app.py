from flask import Flask, request, send_file
import benepar
from spellchecker import SpellChecker
import spacy
import requests
import warnings
from gen_reasoning_threads.models.triple_extraction.triple_gen import *
from sentence_transformers import SentenceTransformer, util
from flask_cors import CORS
from top2vec import Top2Vec
import traceback
from flask import jsonify
from map_triple_to_ontology import *
from elasticsearch import Elasticsearch
import sys
import os
import time
import openai
from openai import OpenAI
from dotenv import load_dotenv
from neo4j import GraphDatabase
import uuid
from datetime import datetime, timedelta
import nltk
import ssl
import threading
from subgraph_enricher import subgraph_enricher
from functools import lru_cache

env_file = '.env.production' if os.getenv('FLASK_ENV') == 'production' else '.env.development'
load_dotenv(env_file)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create a job queue (in-memory for simplicity)
job_queue = {}

app = Flask(__name__)
app.config.from_object(__name__)

# Define allowed origins
CORS_ALLOWED_ORIGIN = os.getenv('CORS_ALLOWED_ORIGIN',
                                'https://findyoursolution.ai,https://www.findyoursolution.ai,null')
origins = list(set(origin.strip() for origin in CORS_ALLOWED_ORIGIN.split(',')))

# Add development origins if needed
if os.getenv('FLASK_ENV') == 'development':
    origins.append('http://localhost:3002')

# Apply CORS with explicit origins
CORS(app,
     resources={r"/*": {
         "origins": origins,
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
     }},
     supports_credentials=True)

warnings.filterwarnings("ignore", category=UserWarning)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def download_models():
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')


threading.Thread(target=download_models, daemon=True).start()

openai.api_key = os.getenv('OPENAI_API_KEY')
ASSISTANT_ID = os.getenv('OPENAI_ASSISTANT_ID')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Neo4j Aura connection detail
NEO4J_URI = os.getenv('NEO4J_URI_GRAPHQL')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

elasticsearch_host = os.getenv('ES')
elasticsearch_user = os.getenv('ELASTICSEARCH_USER')
elasticsearch_password = os.getenv('ELASTICSEARCH_PASSWORD')
use_cert = os.getenv('CERTS', 'False').lower() == 'true'  # Default to False if not set

# Check if the host already includes the protocol
if not elasticsearch_host.startswith(('http://', 'https://')):
    # Add the appropriate prefix based on CERT setting
    protocol = "https://" if use_cert else "http://"
    elasticsearch_host = f"{protocol}{elasticsearch_host}"

# Initialize Elasticsearch client with authentication
if use_cert:
    es = Elasticsearch(
        [elasticsearch_host],
        basic_auth=(elasticsearch_user, elasticsearch_password),
        verify_certs=True,
        ca_certs="/root/config/certs/ca.crt",
        ssl_show_warn=False,
        ssl_assert_hostname=False,
        request_timeout=90,  # Higher timeout
        retry_on_timeout=True,
        max_retries=5  # Increase retries
    )
else:
    # Simplified connection without SSL/TLS options
    es = Elasticsearch(
        [elasticsearch_host],
        basic_auth=(elasticsearch_user, elasticsearch_password)
    )

var_1 = None
search = None
concepts = None
relations = None
targets = None
img = None
sent = None
key = None


# Add the cleanup function
def cleanup_old_jobs():
    while True:
        try:
            threshold = datetime.now() - timedelta(hours=4)
            removed_count = 0

            for job_id in list(job_queue.keys()):
                if job_queue[job_id]['created_at'] < threshold:
                    del job_queue[job_id]
                    removed_count += 1

            # Sleep for 15 minutes before next cleanup
            time.sleep(900)
        except Exception as e:
            print(f"Error in cleanup task: {str(e)}")
            time.sleep(60)  # If there's an error, wait a minute and try again


# Start the cleanup thread when the app starts
def start_cleanup_thread():
    cleanup_thread = threading.Thread(target=cleanup_old_jobs)
    cleanup_thread.daemon = True  # This ensures the thread will exit when the main process exits
    cleanup_thread.start()


def update_elasticsearch_mapping(max_retries=5, retry_delay=5):
    es_url = elasticsearch_host

    if es_url.startswith("http://") or es_url.startswith("https://"):
        pass
    else:
        es_url = f"https://{es_url}"
    if es_url.endswith("/"):
        es_url = es_url[:-1]

    indices = ["invocations_1", "invocations_2", "invocations_3", "invocations_4", "invocations_5"]

    for index in indices:
        mapping = {
            "properties": {
                "Topics": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "URLs": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                }
            }
        }

        # Construct the full URL correctly
        url = f"{es_url}/{index}/_mapping"

        # Try to update the mapping
        try:
            response = requests.put(
                url,
                json=mapping,
                auth=(elasticsearch_user, elasticsearch_password),
                verify=False  # Disable certificate verification
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to update mapping for {index}. Error: {str(e)}")


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

model_ = Word2Vec.load(os.getenv('TRIPLESMODEL'))
classifier_model = SentenceTransformer("all-MiniLM-L6-v2")
top2vecmodel = Top2Vec.load(os.getenv('TOP2VECMODEL'))

spell = SpellChecker()

category_prototypes = {
    "Business": ["customer", "market", "revenue", "cost", "value proposition"],
    "System": ["architecture", "module", "process", "workflow", "interface"],
    "Data": ["database", "dataset", "schema", "data pipeline", "ETL"],
    "Technology": ["API", "TensorFlow", "cloud", "Docker", "GPU"]
}

category_embeddings = {
    category: classifier_model.encode(examples, convert_to_tensor=True).mean(dim=0)
    for category, examples in category_prototypes.items()
}


def classify_term(term):
    term_embedding = classifier_model.encode(term, convert_to_tensor=True)
    similarities = {
        category: float(util.cos_sim(term_embedding, prototype_embedding))
        for category, prototype_embedding in category_embeddings.items()
    }
    return max(similarities, key=similarities.get)


def spellCheck(query):
    if not query or not isinstance(query, str):
        return ""

    misspelled = spell.unknown(query.split())
    for word in misspelled:
        correction = spell.correction(word)
        if correction is not None:
            query = query.replace(word, correction)
        else:
            print(f"Warning: No correction found for '{word}'")
    return query


def get_keywords(query):
    nlp = load_spacy_model()
    if nlp is None:
        print("Failed to load the spaCy model. Exiting.")
        sys.exit(1)

    doc = nlp(query)
    keywords = [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN')]
    return keywords


def load_spacy_model(model_name='en_core_web_sm'):
    try:
        nlp = spacy.load(model_name)
        return nlp
    except IOError:
        print(f"Model '{model_name}' not found. Attempting to download...")
        spacy.cli.download(model_name)
        return spacy.load(model_name)
    except Exception as e:
        print(f"An unexpected error occurred while loading {model_name}: {e}")
        if "Unknown function registry" in str(e):
            print("This might be due to a version mismatch or corrupted installation.")
            print("Try reinstalling spaCy and the model:")
            print(f"pip uninstall -y spacy")
            print(f"pip uninstall -y {model_name}")
            print("pip install spacy")
            print(f"python -m spacy download {model_name}")
        else:
            print("Please check your spaCy installation and model compatibility.")
        return None


def filter_keywords(keywords, top2vecmodel):
    filtered_keywords = []
    for keyword in keywords:
        # Split multi-word phrases
        words = keyword.split()
        for word in words:
            try:
                if word in top2vecmodel.model.wv.key_to_index:
                    filtered_keywords.append(word)
            except Exception as e:
                pass
    return filtered_keywords


def truncate_content(text, max_chars=100000):
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... (truncated)"


###################################
# Handle user prompt and generate subgraphs
###################################
def process_result_job(job_id, data):
    job = job_queue[job_id]
    global var_1, search, concepts, relations, targets, img, sent, key
    try:
        final_json = {}
        job['progress'] = 10

        # ==================================================
        # User Input Processing
        # ==================================================

        required_fields = ['text_input', 'approach', 'classifier', 'systemOrientation']
        for field in required_fields:
            if field not in data:
                job['status'] = 'error'
                job['error'] = f"Missing required field: {field}"
                return

        var_1 = data['text_input']

        # Update progress after initial checks
        job['progress'] = 15
        nlp = spacy.load('en_core_web_md')
        benepar.download('benepar_en3')
        nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

        # Update progress after loading models
        job['progress'] = 25

        # Clean the input text
        var_1 = clean_text(var_1)

        # Check for empty input
        if not var_1.strip():
            job['status'] = 'error'
            job['error'] = "Empty input provided"
            return

        # Split long input into sentences
        sentences = nltk.sent_tokenize(var_1)

        # Update progress
        job['progress'] = 35

        all_triples = []
        sent2_list = []

        for sentence in sentences:
            try:
                doc = nlp(sentence)
                sent2_list.append(doc)
                triples = triple_gen(doc)
                if triples:
                    all_triples.extend(triples)
                else:
                    print(f"No triples generated for sentence: {sentence[:50]}...")
            except Exception as e:
                print(f"Processing sentence: {sentence[:50]}...")
                print(f"Error details: {str(e)}")
                continue

        # Check for no triples
        if not all_triples:
            job['status'] = 'error'
            job['error'] = "No triples generated from input"
            final_json = {}
            return

        # Update progress after generating triples
        job['progress'] = 50

        # Create the 'sent' list
        sent = [token.text for sentence in sent2_list for token in sentence]

        all_triples = [(str(t[0]), str(t[1]), str(t[2])) for t in all_triples]

        # Search triples in fixed Ontology (csv.)
        mapped_triples = map_triple_to_ontology(all_triples)
        all_triples.extend(mapped_triples)
        triple_df = pd.DataFrame(all_triples, columns=['source', 'target', 'edge'])

        # Process text sentence by sentence
        try:
            doc = nlp(var_1)
        except AssertionError as e:
            print(f"AssertionError during tokenization: {str(e)}")
            job['status'] = 'error'
            job['error'] = f"Tokenization error: {str(e)}"
            return

        sent2_list = list(doc.sents)

        if not sent2_list:
            print("No sentences found in input")
            job['status'] = 'error'
            job['error'] = "No sentences found in input"
            return

        job['progress'] = 70

        # Add candidates from the input fields
        approach = data["approach"]
        suggestion = []
        empty = []

        if approach == 'Deep Learning':
            suggestion.append('deep learning')
        elif approach == 'Machine Learning':
            suggestion.append('machine learning')
        elif approach == 'Statistical Learning':
            suggestion.append('statistical learning')
        elif approach == 'None':
            sent = sent + empty
        sent.extend(suggestion)

        suggestion = []
        operation = data["classifier"]
        if operation == 'Classification_nonlinear':
            suggestion.append('classifier')
        elif operation == 'Regression':
            suggestion.append('regression')
        elif operation == 'None':
            sent = sent + empty
        sent.extend(suggestion)

        orientation = data["systemOrientation"]
        suggestion = []
        if orientation == 'Decentralized':
            suggestion.append('decentralized')
        elif orientation == 'Centralized':
            suggestion.append('centralized')
        elif orientation == 'None':
            sent = sent + empty
        sent.extend(suggestion)

        # prep data from user input
        concepts = triple_df.source.tolist()
        relations = triple_df.edge.tolist()
        targets = triple_df.target.tolist()
        Concepts = concepts
        Concepts.extend(sent)
        Concepts = [ele.replace('_', ' ') for ele in Concepts]

        # ==================================================
        # INTEGRATION POINT: Use Subgraph Agents
        # ==================================================
        try:
            job['status_detail'] = "Running subgraph enrichment pipeline..."
            job['progress'] = 72

            debug_mode = os.getenv('DEBUG', False)  # Get DEBUG env var, default to False

            # If you want to make sure it's a proper boolean
            if isinstance(debug_mode, str):
                debug_mode = debug_mode.lower()

            # Call the subgraph_enricher to get graph data
            subgraph_result = subgraph_enricher(
                concepts=Concepts,
                targets=targets,
                relations=relations,
                debug_mode=debug_mode
            )

            # Store the entire result in the job
            job["subgraphData"] = subgraph_result

            # No need to try to unpack values that don't exist
            # The frontend should extract them from the subgraphData structure

        except Exception as e:
            print(f"Error during subgraph enrichment: {str(e)}")
            traceback.print_exc()

            # Provide default empty graph data
            job["subgraphData"] = {
                "user_subgraph": {"nodes": [], "links": []},
                "domain_subgraph": {"nodes": [], "links": []},
                "llm_subgraph": {"nodes": [], "links": []},
                "combined_subgraph": {"nodes": [], "links": []},
                "matches": [],
                "similarity": 0,
                "llm_records": [],
                "domain_records": [],
                "user_records": [],
                "mcts_records": []
            }

        job['progress'] = 95
        job['status'] = 'complete'
        job['progress'] = 100

    except Exception as e:
        print(f"Error in job processing: {str(e)}")
        print(traceback.format_exc())
        job['status'] = 'error'
        job['error'] = str(e)
        job['progress'] = job.get('progress', 0)
        job['traceback'] = traceback.format_exc()

        # Provide default results even on error
        job['result'] = {
            "error_result": {
                "RS_Triple": f"Error processing {data.get('text_input', 'input')}",
                "Ontology_Triple": f"Error: {str(e)[:50]}...",
                "Score": "0.1"
            }
        }

        # Ensure subgraphData exists even on error
        if "subgraphData" not in job:
            job["subgraphData"] = {
                "user_subgraph": {"nodes": [], "links": []},
                "domain_subgraph": {"nodes": [], "links": []},
                "llm_subgraph": {"nodes": [], "links": []},
                "combined_subgraph": {"nodes": [], "links": []},
                "matches": [],
                "similarity": 0,
                "llm_records": [],
                "domain_records": [],
                "user_records": []
            }


def update_elasticsearch_mapping():
    try:
        es_url = elasticsearch_host

        # Remove protocol if it exists in the host string
        if es_url.startswith("http://") or es_url.startswith("https://"):
            # Use the URL as-is, but remove any trailing slash
            if es_url.endswith("/"):
                es_url = es_url[:-1]
        else:
            # No protocol in the host, add http:// by default
            es_url = f"https://{es_url}"

        # If port is not in the URL and we need to add it
        if ":9200" not in es_url and not es_url.endswith("/"):
            es_url = f"{es_url}:9200"

        indices = ["invocations_1", "invocations_2", "invocations_3", "invocations_4", "invocations_5"]

        for index in indices:
            # Construct the full endpoint URL
            try:
                url = f"{es_url}/{index}/_mapping"
                mapping = {
                    "properties": {
                        "Topics": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "URLs": {
                            "type": "text",
                            "analyzer": "standard"
                        }
                    }
                }
                # Make the request
                response = requests.put(
                    url,
                    json=mapping,
                    auth=(elasticsearch_user, elasticsearch_password),
                    verify=False  # For development; use True with proper certs in production
                )
            except Exception as e:
                print(f"Warning: Failed to update mapping for {index}. Error: {str(e)}")
                # Continue with the next index

    except Exception as e:
        print(f"Warning: Elasticsearch configuration failed: {str(e)}")
        print("Application will continue without Elasticsearch functionality.")


###################################
# Generate Solution Summary with OpenAI API
###################################

def generate_summary(paper_contents, graph_description, max_words):
    try:
        # Create a thread
        thread = openai.beta.threads.create()

        # Construct a more specific prompt including graph data
        prompt = f"""Please provide a concise solution summary based on the following information, focusing on the key technological components and their relationships. The summary should:

    1. Be approximately {max_words} words long.
    2. Focus on the main technological concepts and their interactions.
    3. Highlight any innovative approaches or unique combinations of technologies.
    4. Incorporate insights from the knowledge graph triples if provided.
    5. Use clear, technical language without unnecessary elaboration.

    Information to summarize:

    {paper_contents}

    Knowledge Graph Triples (if available):

    {graph_description}

    Please structure the summary as follows:
    1. Brief overview (1-2 sentences)
    2. Key technological components (bullet points)
    3. Relationships between components (1-2 sentences, incorporating graph insights if available)
    4. Innovative aspects (if any, 1-2 sentences)
    5. Potential applications or impact (1 sentence)
    6. Start the text with "The solution recommendation consists of..."
    7. Do not speak about papers or graphs in the text - rather use: the solution suggests...
    """

        # Add a message to the thread
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )

        # Run the assistant
        run = openai.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID
        )

        # Wait for the run to complete
        while True:
            run_status = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run_status.status == 'completed':
                break

        # Retrieve the messages
        messages = openai.beta.threads.messages.list(thread_id=thread.id)

        # Get the assistant's response
        assistant_message = next((msg for msg in messages if msg.role == 'assistant'), None)
        summary = assistant_message.content[0].text.value if assistant_message else "No summary generated."

        return summary

    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        return f"Failed to generate summary: {str(e)}"


##################################
# generate instructions comparisons with Hugging Face API
###################################
# Create cached functions to interact with Hugging Face API
@lru_cache(maxsize=32)
def get_huggingface_api_key():
    """Get Hugging Face API key from environment variables with fallback."""
    api_key = os.getenv('HUGGINGFACE_API_KEY')
    if not api_key:
        print("WARNING: HUGGINGFACE_API_KEY not found in environment variables.")
        return None
    return api_key


def get_model_endpoint(model_type):
    """Get the model endpoint based on model type."""
    base_url = os.getenv('HUGGINGFACE_API_URL', 'https://api-inference.huggingface.co/models')

    if model_type == 'gnn':
        model = os.getenv('HUGGINGFACE_GNN_MODEL')
    elif model_type == 'mcts':
        model = os.getenv('HUGGINGFACE_MCTS_MODEL')
    elif model_type == 'reasoning':
        model = os.getenv('REASONING_MODEL')

    return f"{base_url}/{model}"


def query_huggingface_model(prompt, model_type, max_retries=3, timeout=120):
    """
    Query a model on Hugging Face for text generation.

    Args:
        prompt: The text prompt to send to the model
        model_type: 'gnn', 'mcts', or 'reasoning'
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds

    Returns:
        Generated text response
    """
    api_key = get_huggingface_api_key()
    endpoint = get_model_endpoint(model_type)
    headers = {"Authorization": f"Bearer {api_key}"}

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 600,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False  # Only return the generated text, not the prompt
        }
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()

            # Parse response - format depends on the model
            result = response.json()

            # Extract generated text - handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if 'generated_text' in result[0]:
                    return result[0]['generated_text']
                else:
                    return str(result[0])
            elif isinstance(result, dict) and 'generated_text' in result:
                return result['generated_text']
            else:
                return str(result)

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                # If the error is 503 (Model is loading), wait longer
                if hasattr(e, 'response') and e.response is not None and e.response.status_code == 503:
                    wait_time = min(2 ** (attempt + 2), 60)  # Exponential backoff, max 60 seconds
                    print(f"Model is loading. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    # For other errors, use a shorter delay
                    time.sleep(5)
            else:
                print(f"Failed to query Hugging Face API after {max_retries} attempts: {str(e)}")


################################################
# GNN instruction
################################################
def generate_gnn_based_instructions(prompt, user_nodes, gnn_subgraph):
    """Generate instructions based on GNN subgraph using Hugging Face models"""
    try:
        # Use gnn_subgraph
        nodes = gnn_subgraph.get('nodes', [])
        user_entity_names = [node.get('label', '') for node in user_nodes if node.get('label', '')]

        # Extract meaningful entities (simplified)
        meaningful_entities = []
        for node in nodes:
            label = node.get('label', '').strip()
            if (label and
                    len(label) > 3 and
                    not label.startswith('ns0__') and
                    'cutting edge' not in label.lower() and
                    'edge ai' not in label.lower()):
                meaningful_entities.append(label)

        print(f"DEBUG - GNN meaningful entities: {meaningful_entities[:8]}")

        # Simple, focused prompt
        instruction_prompt = f"""Create 3 technical implementation instructions for: "{prompt}"

Available components from GNN analysis: 
{', '.join(meaningful_entities[:30]) if meaningful_entities else 'core system components'}

User entities: {', '.join(user_entity_names) if user_entity_names else 'user requirements'}

Requirements:
- Use specific component names from the available list
- Address the user's request directly
- Be practical and actionable
- Avoid generic marketing terms, be very technical and solution focused
- Focus on GNN-discovered relationships and entities

Generate 3 specific GNN-based instructions.
FORMAT EACH INSTRUCTION AS:
"INSTRUCTION X: [Your instruction here]"

PROVIDE ONLY THE THREE NUMBERED INSTRUCTIONS, NO INTRODUCTION OR EXPLANATION."""

        # Try Hugging Face API first
        instructions = query_huggingface_model(instruction_prompt, 'gnn')

        # If API fails or returns empty/null, use contextual fallback
        if not instructions or instructions.strip() == "" or instructions.lower() in ['null', 'none', 'error']:
            print("GNN API failed or returned empty, using contextual fallback")

        print(f"Final GNN instructions: {instructions}")
        return instructions

    except Exception as e:
        print(f"Error in GNN-based instruction generation: {str(e)}")


################################################
# MCTS instruction
################################################
def generate_mcts_based_instructions(prompt, user_nodes, mcts_subgraph):
    """Generate instructions based on MCTS subgraph using Hugging Face models"""
    try:
        # Use mcts_subgraph
        nodes = mcts_subgraph.get('nodes', [])
        user_entity_names = [node.get('label', '') for node in user_nodes if node.get('label', '')]

        # Extract meaningful entities (simplified)
        meaningful_entities = []
        for node in nodes:
            label = node.get('label', '').strip()
            if (label and
                    len(label) > 3 and
                    not label.startswith('ns0__') and
                    'cutting edge' not in label.lower() and
                    'edge ai' not in label.lower()):
                meaningful_entities.append(label)

        print(f"DEBUG - MCTS meaningful entities: {meaningful_entities[:8]}")

        # Simple, focused prompt
        instruction_prompt = f"""Create 3 precise technical instructions for: "{prompt}"

        MCTS-IDENTIFIED COMPONENTS: {', '.join(meaningful_entities[:20]) if meaningful_entities else 'system elements'}
        USER GOALS: {', '.join(user_entity_names[:3]) if user_entity_names else 'primary objectives'}

        INSTRUCTION CRITERIA:
        ✓ Clear Structure: Use action verbs + specific components + clear outcomes
        ✓ User-Focused: Directly address the original request
        ✓ Technical Depth: Include specific tools, APIs, configurations
        ✓ Readable Format: Short sentences, logical sequence  
        ✓ Actionable Details: Implementable immediately
        ✓ MCTS Integration: Use the discovered components above

        FORMAT:
        INSTRUCTION 1: [Action Verb] [Specific Component] to [Clear Purpose]
        - Implementation detail with specific technology
        - Configuration or setup requirement
        - Expected outcome or validation step

        INSTRUCTION 2: [Action Verb] [Specific Component] to [Clear Purpose]
        - Build logically on previous instruction  
        - Include technical specifications
        - Reference integration points

        INSTRUCTION 3: [Action Verb] [Specific Component] to [Clear Purpose]
        - Complete the implementation sequence
        - Include testing or deployment steps
        - Validate against user requirements

        Generate clear, actionable instructions that balance technical precision with readability."""

        # Try Hugging Face API first
        instructions = query_huggingface_model(instruction_prompt, 'mcts')

        # If API fails or returns empty/null, use contextual fallback
        if not instructions or instructions.strip() == "" or instructions.lower() in ['null', 'none', 'error']:
            print("MCTS API failed or returned empty, using contextual fallback")

        print(f"Final MCTS instructions: {instructions}")
        return instructions

    except Exception as e:
        print(f"Error in MCTS-based instruction generation: {str(e)}")


################################################
# Large Reasoning Model instruction
################################################

def generate_reasoning_model_instructions(prompt, return_reasoning=False):
    """Generate instructions using a large reasoning model via Hugging Face"""
    try:
        # Construct a prompt for the LLM that mimics advanced reasoning
        instruction_prompt = f"""
                You are an expert system that creates clear, actionable instructions for users.

                USER REQUEST:
                {prompt}

                Based on this information, create THREE clear, specific, actionable, and coherent solution instructions that 
                would help a user implement this solution. 

                The instructions should:
                1. Start with action verbs
                2. Be specific and technically accurate
                3. Follow logically
                4. Address the user's core request and experience directly
                5. Specify technologies or approaches where relevant

                Format:
                REASONING: [step-by-step reasoning here]
                INSTRUCTION 1: ...
                INSTRUCTION 2: ...
                INSTRUCTION 3: ...

                PROVIDE ONLY THE THREE NUMBERED INSTRUCTIONS, NO INTRODUCTION OR EXPLANATION.
                """

        # Make the API call - you need to add this part
        client = openai.OpenAI()  # Make sure to set your API key
        response = client.chat.completions.create(
            model="gpt-4",  # or whatever model you're using
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that generates clear, actionable instructions."},
                {"role": "user", "content": instruction_prompt}
            ],
            max_tokens=400,
            temperature=0.7
        )

        output = response.choices[0].message.content.strip()
        reasoning = ""
        instructions = ""

        if "REASONING:" in output:
            parts = output.split("REASONING:", 1)[1].strip().split("INSTRUCTION 1:", 1)
            reasoning = parts[0].strip()
            instructions = "INSTRUCTION 1:" + parts[1].strip() if len(parts) > 1 else ""
        else:
            instructions = output

        result = {"instructions": instructions}
        if return_reasoning:
            result["reasoning"] = reasoning

        return result

    except Exception as e:
        print(f"OpenAI GPT-4o error: {str(e)}")
        if return_reasoning:
            return {"instructions": f"Failed to generate instructions: {str(e)}", "reasoning": ""}
        else:
            return {"instructions": f"Failed to generate instructions: {str(e)}"}


def calculate_instruction_metrics(prompt, user_nodes, gnn_instructions, mcts_instructions, rmodel_instructions):
    try:
        # Import required modules at the top of the function
        import spacy
        from sentence_transformers import SentenceTransformer, util

        # Initialize models (you might want to do this globally to avoid reloading)
        nlp = spacy.load("en_core_web_sm")
        classifier_model = SentenceTransformer("all-MiniLM-L6-v2")

        metrics = {
            "user_focus": {},
            "technological_specificity": {},
            "actionability": {},
            "coherence": {},
            "semantic_similarity": {},
            "domain_specificity": {},
            "overall_effectiveness": {}
        }

        user_entities = [node.get('label', '').lower() for node in user_nodes]
        domain_terms = set(
            ["health", "finance", "education", "legal", "retail", "agriculture", "energy", "logistics", "security",
             "manufacturing"])

        doc = nlp(prompt)
        prompt_entities = {ent.label_.lower() for ent in doc.ents}
        domain_overlap = len(domain_terms.intersection(set(token.lemma_.lower() for token in doc)))

        for method, instructions in [
            ("gnn", gnn_instructions),
            ("mcts", mcts_instructions),
            ("rmodel", rmodel_instructions)
        ]:
            # Handle case where instructions might be a dict (from the fixed function above)
            if isinstance(instructions, dict):
                instructions = instructions.get("instructions", "")

            instructions_lower = instructions.lower()

            # User focus score
            user_focus_score = sum(2 for entity in user_entities if entity and entity in instructions_lower)
            metrics["user_focus"][method] = min(10, user_focus_score)

            # Technological specificity
            tech_terms = ["api", "database", "framework", "algorithm", "python", "javascript",
                          "system", "interface", "model", "cloud", "server", "client", "docker",
                          "machine learning", "ai", "analytics", "neural", "transform"]
            tech_score = sum(1.5 for term in tech_terms if term in instructions_lower)
            metrics["technological_specificity"][method] = min(10, tech_score)

            # Actionability
            action_verbs = ["implement", "create", "develop", "build", "establish", "design",
                            "configure", "set up", "deploy", "integrate", "incorporate",
                            "analyze", "measure", "test", "validate"]
            action_score = sum(2 for verb in action_verbs if verb in instructions_lower)
            metrics["actionability"][method] = min(10, action_score)

            # Coherence
            coherence_score = 5
            if "instruction 1" in instructions_lower and "instruction 2" in instructions_lower:
                coherence_score += 2
            if len(instructions.split()) > 30:
                coherence_score += 1
            if len(instructions.split("\n")) > 2:  # Fixed: was "/n" should be "\n"
                coherence_score += 1
            metrics["coherence"][method] = min(10, coherence_score)

            # Semantic similarity
            prompt_emb = classifier_model.encode(prompt, convert_to_tensor=True)
            instructions_emb = classifier_model.encode(instructions, convert_to_tensor=True)
            similarity = util.cos_sim(prompt_emb, instructions_emb).item()  # Fixed: use util.cos_sim
            metrics["semantic_similarity"][method] = round(similarity, 4)

            # Domain specificity
            domain_spec_score = domain_overlap + len(prompt_entities)
            metrics["domain_specificity"][method] = min(10, domain_spec_score)

        # Calculate overall effectiveness (moved outside the loop)
        weights = {
            "user_focus": 0.25,
            "technological_specificity": 0.15,
            "actionability": 0.2,
            "coherence": 0.15,
            "semantic_similarity": 0.1,
            "domain_specificity": 0.15
        }

        for method in ["gnn", "mcts", "rmodel"]:
            score = sum(weights[key] * metrics[key][method] for key in weights)
            metrics["overall_effectiveness"][method] = round(score, 4)

        return metrics

    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            "user_focus": {"gnn": 0.0, "mcts": 0.0, "rmodel": 0.0},
            "technological_specificity": {"gnn": 0.0, "mcts": 0.0, "rmodel": 0.0},
            "actionability": {"gnn": 0.0, "mcts": 0.0, "rmodel": 0.0},
            "coherence": {"gnn": 0.0, "mcts": 0.0, "rmodel": 0.0},
            "semantic_similarity": {"gnn": 0.0, "mcts": 0.0, "rmodel": 0.0},
            "domain_specificity": {"gnn": 0.0, "mcts": 0.0, "rmodel": 0.0},
            "overall_effectiveness": {"gnn": 0.5, "mcts": 0.5, "rmodel": 0.5}
        }


@app.route('/query/', methods=['POST'])
def query():
    try:
        content = request.json
        user_input = content.get('text_input')

        if not user_input or not isinstance(user_input, str):
            return jsonify({"error": "Invalid or missing text_input"}), 400
        query = spellCheck(user_input)

        # If query is empty after spell check, return an error
        if not query:
            return jsonify({"error": "Invalid input after spell check"}), 400

        indices = "invocations_1,invocations_2,invocations_3,invocations_4,invocations_5"

        es_query = {
            "query": {
                "bool": {
                    "should": [
                        {"more_like_this": {
                            "fields": ["Topics"],
                            "like": query,
                            "min_term_freq": 1,
                            "max_query_terms": 12
                        }},
                        {"match": {"URLs": query}}
                    ]
                }
            }
        }

        try:
            response = es.search(index=indices, body=es_query)
        except Exception as e:
            print(f"Elasticsearch error: {str(e)}")
            return jsonify({"error": "Elasticsearch query failed", "details": str(e)}), 500

        hits = response.get('hits', {}).get('hits', [])
        es_documents = [hit['_source'] for hit in hits]
        es_document_ids = [hit['_id'] for hit in hits]

        try:
            triples = triple_gen(query)
            if not triples:
                print("No triples generated. Using original query as keyword.")
                keywords = [query.lower()]
            else:
                keywords = [triple[0].lower() for triple in triples] + [triple[1].lower() for triple in triples]

            buzz_words_to_remove = ["our", "we"]
            keywords = [kw for kw in keywords if kw not in buzz_words_to_remove]
            filtered_keywords = filter_keywords(keywords, top2vecmodel)

            if filtered_keywords:
                try:
                    top2vec_documents, scores, top2vec_document_ids = top2vecmodel.search_documents_by_keywords(
                        keywords=filtered_keywords, num_docs=15)

                    # Create a dictionary of documents
                    top2vec_documents_dict = {i: {"content": doc, "id": doc_id, "score": score}
                                              for i, (doc, doc_id, score) in
                                              enumerate(zip(top2vec_documents, top2vec_document_ids, scores),
                                                        start=1)}

                    # Generate URLs based on document IDs
                    top2vec_document_urls = [f"document/{doc_id}" for doc_id in top2vec_document_ids]
                except Exception as e:
                    print(f"Error in Top2Vec document processing: {str(e)}")
                    top2vec_documents_dict = {}
                    top2vec_document_urls = []
            else:
                print("No valid keywords found for Top2Vec search")
                top2vec_documents_dict = {}
                top2vec_document_urls = []
        except Exception as e:
            print(f"Error in Top2Vec processing: {str(e)}")
            top2vec_documents_dict = {}
            top2vec_document_urls = []

        def get_url(doc):
            return doc.get('URLs', '') or doc.get('url', '') or f"document/{doc.get('id', 'unknown')}"

        # Combine Elasticsearch and Top2Vec results
        combined_documents = es_documents + list(top2vec_documents_dict.values())
        combined_urls = [get_url(doc) for doc in combined_documents]

        result = {
            "Topic": {i: doc for i, doc in enumerate(combined_documents, start=1)},
            "URL/PDF": combined_urls
        }
        print(f"Final result has {len(combined_documents)} documents")
        return jsonify(result)

    except Exception as e:
        print(f"Unexpected error in query function: {str(e)}")
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500


@app.route('/suggestions/', methods=['POST'])
def suggestions():
    content = request.json
    text_input = content['text_input']


@app.route('/')
def welcome():
    return jsonify({"message": "API Service is running"}), 200


def has_numbers(inputString):
    return bool(regex.search(r'\d', inputString))


def clean_text(text):
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters except punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()


@app.route('/result/', methods=['POST'])
def result():
    try:
        data = request.json

        # Create a job ID
        job_id = str(uuid.uuid4())

        # Store initial job information
        job_queue[job_id] = {
            'status': 'processing',
            'data': data,
            'created_at': datetime.now(),
            'progress': 0
        }

        # Start processing in a background thread
        thread = threading.Thread(target=process_result_job, args=(job_id, data))
        thread.daemon = True
        thread.start()

        # Return immediately with the job ID
        return jsonify({"job_id": job_id, "status": "processing"})

    except Exception as e:
        print(f"Error creating job: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# Modified job_status route to ensure the progress field is always present
@app.route('/job_status/<job_id>', methods=['GET'])
def check_job_status(job_id):
    if job_id not in job_queue:
        return jsonify({
            'status': 'error',
            'error': 'Job not found',
            'progress': 0
        }), 404

    job = job_queue[job_id]

    # Build the response object
    response = {
        'status': job.get('status', 'unknown'),
        'progress': job.get('progress', 0)
    }

    # Add the result field if present
    if job['status'] == 'complete' and 'result' in job:
        response['result'] = job['result']

    # Add the subgraphData field if present - make sure this is camelCase to match frontend
    if 'subgraphData' in job:
        print(
            f"Including subgraphData in response: {len(job['subgraphData'].get('combined_subgraph', {}).get('nodes', []))} nodes")
        response['subgraphData'] = job['subgraphData']
    elif 'subgraph_data' in job:  # Handle snake_case version too for compatibility
        print(
            f"Including subgraph_data in response: {len(job['subgraph_data'].get('combined_subgraph', {}).get('nodes', []))} nodes")
        response['subgraphData'] = job['subgraph_data']

    # Add error information if present
    if job['status'] == 'error':
        response['error'] = job.get('error', 'Unknown error')

    return jsonify(response)


@app.route('/summarize/', methods=['POST'])
def summarize_papers():
    try:
        content = request.json
        papers = content.get('papers')
        graph_data = content.get('graph_data', {})
        max_words = content.get('max_words', 500)

        if not papers or not isinstance(papers, str):
            return jsonify({"error": "Invalid or missing papers data"}), 400

        # Truncate paper contents
        truncated_papers = truncate_content(papers)

        # Process graph data
        nodes = graph_data.get('nodes', [])
        links = graph_data.get('links', [])
        graph_triples = [f"{link['source']} {link['label']} {link['target']}" for link in links]
        graph_description = "\n".join(graph_triples)

        # Truncate graph description
        truncated_graph_description = truncate_content(graph_description, max_chars=50000)

        # Use the cached function
        summary = generate_summary(truncated_papers, truncated_graph_description, max_words)

        if summary.startswith("Failed to generate summary"):
            return jsonify({"error": summary}), 500

        return jsonify({"summary": summary})

    except Exception as e:
        print(f"Unexpected error in summarize_papers function: {str(e)}")
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500


# Better debugging in the Flask route

@app.route('/generate_instructions/', methods=['POST'])
def generate_instructions():
    try:

        # Get the content and log it
        content = request.json
        print(f"Parsed request content keys: {list(content.keys()) if content else 'None'}")

        # Track model statuses
        model_statuses = {
            'gnn': 'pending',
            'mcts': 'pending',
            'reasoning': 'pending'
        }

        # Better validation with detailed errors
        if not content:
            return jsonify({"error": "Empty request body"}), 400

        prompt = content.get('prompt', '')
        user_nodes = content.get('user_nodes', [])
        gnn_subgraph = content.get('gnn_subgraph', {})
        mcts_subgraph = content.get('mcts_subgraph', {})

        # Detailed validation logging
        validation_issues = []
        if not prompt:
            validation_issues.append("Missing prompt")
        if not isinstance(prompt, str):
            validation_issues.append(f"Prompt is not a string (type: {type(prompt)})")

        # Log validation issues and return if any
        if validation_issues:
            print(f"Validation issues: {validation_issues}")
            return jsonify({"error": ", ".join(validation_issues)}), 400

        print(f"Processing prompt: {prompt[:100]}...")
        print(f"User nodes count: {len(user_nodes)}")
        print(f"GNN subgraph nodes: {len(gnn_subgraph.get('nodes', []))}")
        print(f"MCTS subgraph nodes: {len(mcts_subgraph.get('nodes', []))}")

        # Initialize results with defaults
        gnn_instructions = ""
        mcts_instructions = ""
        rmodel_instructions = ""

        # Generate the three types of instructions with status tracking
        print(f"Generating GNN instructions...")
        try:
            gnn_instructions = generate_gnn_based_instructions(prompt, user_nodes, gnn_subgraph)
            if gnn_instructions and len(gnn_instructions.strip()) > 0:
                # Check if it looks like a fallback response
                if "graph-based analysis system" in gnn_instructions:
                    model_statuses['gnn'] = 'fallback'
                else:
                    model_statuses['gnn'] = 'success'
            else:
                print("GNN instructions are empty, marking as error")
                model_statuses['gnn'] = 'error'
        except Exception as e:
            print(f"Error generating GNN instructions: {str(e)}")
            model_statuses['gnn'] = 'error'

        print(f"Generating MCTS instructions...")
        try:
            mcts_instructions = generate_mcts_based_instructions(prompt, user_nodes, mcts_subgraph)
            if mcts_instructions and len(mcts_instructions.strip()) > 0:
                # Check if it looks like a fallback response
                if "decision-tree analysis system" in mcts_instructions:
                    model_statuses['mcts'] = 'fallback'
                else:
                    model_statuses['mcts'] = 'success'
            else:
                print("MCTS instructions are empty, marking as error")
                model_statuses['mcts'] = 'error'
        except Exception as e:
            print(f"Error generating MCTS instructions: {str(e)}")
            model_statuses['mcts'] = 'error'

        print(f"Generating reasoning model instructions...")
        try:
            result = generate_reasoning_model_instructions(prompt)
            if isinstance(result, dict):
                rmodel_instructions = result.get("instructions", "")
            else:
                rmodel_instructions = str(result)

            if rmodel_instructions and "Failed to generate instructions" not in rmodel_instructions:
                model_statuses['reasoning'] = 'success'
            else:
                model_statuses['reasoning'] = 'error'
        except Exception as e:
            print(f"Error generating reasoning model instructions: {str(e)}")
            model_statuses['reasoning'] = 'error'

        # Validate that all instructions are non-empty
        if not gnn_instructions or gnn_instructions.strip() == "":
            model_statuses['gnn'] = 'fallback_final'

        if not mcts_instructions or mcts_instructions.strip() == "":
            model_statuses['mcts'] = 'fallback_final'

        if not rmodel_instructions or rmodel_instructions.strip() == "":
            model_statuses['reasoning'] = 'fallback_final'

        # Calculate effectiveness metrics
        print("Calculating metrics...")
        try:
            simple_user_nodes = [
                {"label": node.get("label", ""), "id": node.get("id", "")}
                for node in user_nodes
                if isinstance(node, dict)
            ]

            metrics = calculate_instruction_metrics(
                prompt,
                simple_user_nodes,
                gnn_instructions,
                mcts_instructions,
                rmodel_instructions
            )
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            # Provide default metrics on error
            metrics = {
                "user_focus": {"gnn": 0.0, "mcts": 0.0, "rmodel": 0.0},
                "technological_specificity": {"gnn": 0.0, "mcts": 0.0, "rmodel": 0.0},
                "actionability": {"gnn": 0.0, "mcts": 0.0, "rmodel": 0.0},
                "coherence": {"gnn": 0.0, "mcts": 0.0, "rmodel": 0.0},
                "overall_effectiveness": {"gnn": 0.0, "mcts": 0.5, "rmodel": 0.5}
            }

        # Log final results
        print(f"Final GNN instructions length: {len(gnn_instructions)}")
        print(f"Final MCTS instructions length: {len(mcts_instructions)}")
        print(f"Final Reasoning instructions length: {len(rmodel_instructions)}")
        print(f"Model statuses: {model_statuses}")

        print("Instructions generation complete")
        return jsonify({
            "gnn_instructions": gnn_instructions,
            "mcts_instructions": mcts_instructions,
            "rmodel_instructions": rmodel_instructions,
            "metrics": metrics,
            "model_statuses": model_statuses,
            "gnn_subgraph": gnn_subgraph,  # Include subgraphs for debugging
            "mcts_subgraph": mcts_subgraph
        })

    except Exception as e:
        import traceback
        error_message = str(e)
        trace = traceback.format_exc()
        print(f"Error generating instructions: {error_message}")
        print(trace)


@app.route('/graph', methods=['POST', 'GET'])
def graph_plot():
    global key
    file = "%s.jpg" % key  # change location as required
    # return send_file(img, mimetype='image/png')
    return send_file(file, mimetype='image/jpg')


@app.errorhandler(Exception)
def handle_error(e):
    """Global error handler to provide proper error responses with CORS headers"""
    response = jsonify({
        "error": str(e),
        "status": "error"
    })
    response.status_code = 500

    # Ensure CORS headers are applied even to error responses
    origin = request.headers.get('Origin')
    if origin in origins:
        response.headers.set('Access-Control-Allow-Origin', origin)
        response.headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
        response.headers.set('Access-Control-Allow-Credentials', 'true')

    return response


@app.errorhandler(400)
def bad_request(e):
    return jsonify(error=str(e)), 400


@app.errorhandler(500)
def server_error(e):
    return jsonify(error=str(e)), 500


@app.after_request
def apply_cors_headers(response):
    # Get the origin from the request
    origin = request.headers.get('Origin')

    # Add CORS headers for null origin or allowed origins
    if origin in origins:
        response.headers.set('Access-Control-Allow-Origin', origin)
        response.headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
        response.headers.set('Access-Control-Allow-Credentials', 'true')

    return response


@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    response = app.make_default_options_response()
    origin = request.headers.get('Origin')
    if origin in origins:
        response.headers.set('Access-Control-Allow-Origin', origin)
        response.headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
        response.headers.set('Access-Control-Allow-Credentials', 'true')
    return response


@app.route('/test-cors', methods=['GET', 'OPTIONS'])
def test_cors():
    """
    Test endpoint to verify CORS is working properly
    """
    # Log request details
    print(f"*** TEST-CORS: Request received from origin: {request.headers.get('Origin')}")
    print(f"*** TEST-CORS: Method: {request.method}")
    print(f"*** TEST-CORS: Headers: {request.headers}")

    # Return a simple response
    return jsonify({
        "status": "ok",
        "message": "CORS test successful",
        "request_details": {
            "method": request.method,
            "path": request.path,
            "origin": request.headers.get('Origin'),
            "origin_allowed": request.headers.get('Origin') in origins or request.headers.get('Origin') == 'null'
        },
        "server_config": {
            "allowed_origins": origins,
            "flask_env": os.getenv('FLASK_ENV', 'not set')
        },
        "headers_set": "Check response headers in browser dev tools"
    })


@app.errorhandler(405)
def method_not_allowed(e):
    """Handle method not allowed errors specifically"""
    print(f"*** 405 Error: {request.method} {request.path}")
    if request.method == 'OPTIONS':
        response = jsonify({
            'status': 'ok',
            'message': 'OPTIONS request handled by error handler'
        })
        # Manually add CORS headers
        origin = request.headers.get('Origin')
        if origin in origins:
            response.headers.set('Access-Control-Allow-Origin', origin)
            response.headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
            response.headers.set('Access-Control-Allow-Credentials', 'true')
        return response, 200

    return jsonify(error=str(e)), 405


HOST = '0.0.0.0'
PORT = 3002

if __name__ == '__main__':
    try:
        update_elasticsearch_mapping()
    except Exception as e:
        print(f"Warning: Elasticsearch initialization failed: {str(e)}")
    start_cleanup_thread()
    app.run(debug=os.getenv('DEBUG', False), host=HOST, port=PORT)
