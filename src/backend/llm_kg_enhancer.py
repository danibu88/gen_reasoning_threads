import os
import traceback
import re
import time
import pandas as pd
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from openai import OpenAI
import uuid

# Load environment variables
env_file = '.env.production' if os.getenv('FLASK_ENV') == 'production' else '.env.development'
load_dotenv(env_file)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASSISTANT_ID = os.getenv('OPENAI_ASSISTANT_ID')
MAX_WAIT_SECONDS = 60

def clean_string(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    return ' '.join(s.replace('\xa0', ' ').strip().split())

def analyze_graph_for_llm_prompt(graph_df: pd.DataFrame) -> Dict:
    subjects = graph_df['subject'].value_counts()
    objects = graph_df['object'].value_counts()
    predicates = graph_df['predicate'].value_counts()

    all_entities = pd.concat([subjects, objects])
    top_entities = all_entities.groupby(all_entities.index).sum().sort_values(ascending=False).head(10)
    top_predicates = predicates.head(10)

    all_nodes = set(graph_df['subject']).union(set(graph_df['object']))
    connected_nodes = set(graph_df['subject']).union(set(graph_df['object']))
    isolated_nodes = all_nodes - connected_nodes

    return {
        "top_entities": top_entities.to_dict(),
        "top_predicates": top_predicates.to_dict(),
        "isolated_nodes": list(isolated_nodes),
        "total_triples": len(graph_df),
        "total_entities": len(all_nodes)
    }


def enhance_subgraph_with_llm(pruned_df, max_triples=15, debug_mode=True, node_dict=None, label_to_ids=None,
                              user_entities=None):
    """
    Enhance the subgraph with LLM-generated triples.
    Includes connections to user entities and focuses on domain concepts.
    """
    if debug_mode:
        if isinstance(pruned_df, dict):
            print(
                f"[LLM Debug] Starting enhancement with dict input ({len(pruned_df.get('nodes', []))} nodes, {len(pruned_df.get('links', []))} links)")
        elif isinstance(pruned_df, pd.DataFrame):
            print(f"[LLM Debug] Starting enhancement with {len(pruned_df)} input triples")
        else:
            print(f"[LLM Debug] Starting enhancement with {type(pruned_df)} input")

        if node_dict:
            print(f"[LLM Debug] node_dict sample (first 5): {dict(list(node_dict.items())[:5])}")

    node_dict = node_dict or {}
    label_to_ids = label_to_ids or {}

    # Convert dictionary to DataFrame if necessary
    if isinstance(pruned_df, dict):
        try:
            # Create a DataFrame from the graph dictionary
            example_triples = []
            for link in pruned_df.get('links', []):
                source_id = link.get('source')
                target_id = link.get('target')
                predicate = link.get('label', '')

                # Find node labels
                source_label = None
                target_label = None
                for node in pruned_df.get('nodes', []):
                    if str(node.get('id')) == str(source_id):
                        source_label = node.get('label', source_id)
                    if str(node.get('id')) == str(target_id):
                        target_label = node.get('label', target_id)

                if source_label and target_label:
                    example_triples.append({
                        'subject': source_label,
                        'predicate': predicate,
                        'object': target_label
                    })

            df = pd.DataFrame(example_triples)
            if debug_mode:
                print(f"[LLM Debug] Converted dict to DataFrame with {len(df)} rows")
        except Exception as e:
            print(f"[LLM Debug] Error converting dict to DataFrame: {e}")
            # Create a minimal valid DataFrame as fallback
            df = pd.DataFrame([{'subject': 'Entity1', 'predicate': 'relatedTo', 'object': 'Entity2'}])
    else:
        df = pruned_df

    # Handle empty DataFrame case
    if len(df) == 0:
        if user_entities and len(user_entities) >= 2:
            # Create a basic DataFrame with user entities
            df = pd.DataFrame([{
                'subject': user_entities[0],
                'predicate': 'relatedTo',
                'object': user_entities[1]
            }])
        else:
            # Absolute fallback
            df = pd.DataFrame([{'subject': 'Entity1', 'predicate': 'relatedTo', 'object': 'Entity2'}])

    # Generate example triples with LABELS instead of IDs
    example_triples = []
    for _, row in df.head(50).iterrows():
        try:
            # Convert subject and object to text labels if they're IDs
            subject_label = node_dict.get(row.subject, row.subject) if isinstance(row.subject, (int, str)) else str(
                row.subject)
            object_label = node_dict.get(row.object, row.object) if isinstance(row.object, (int, str)) else str(
                row.object)
            predicate = row.predicate if hasattr(row, 'predicate') else 'relatedTo'

            example_triples.append(f"({subject_label}, {predicate}, {object_label})")
        except Exception as e:
            if debug_mode:
                print(f"[LLM Debug] Error processing row {row}: {e}")
            continue

    triple_text = "\n".join(example_triples)

    # Extract entities from DataFrame
    try:
        entities = list(set(
            [node_dict.get(str(row.subject), str(row.subject)) for _, row in df.iterrows()] +
            [node_dict.get(str(row.object), str(row.object)) for _, row in df.iterrows()]
        ))
    except Exception as e:
        if debug_mode:
            print(f"[LLM Debug] Error extracting entities: {e}")
        entities = []

    # Get top entity labels
    top_entity_labels = entities[:10]

    # Get the top predicates
    try:
        top_predicates = df['predicate'].value_counts().head(10).index.tolist()
    except Exception as e:
        if debug_mode:
            print(f"[LLM Debug] Error getting top predicates: {e}")
        top_predicates = ['relatedTo', 'implements', 'enables']

    # User entity specific prompt
    user_entity_text = ""
    if user_entities:
        user_entity_text = "User-provided entities (CONNECT TO THESE):\n" + "\n".join(
            f"- {entity}" for entity in user_entities[:10])

    top_entities_text = "\n".join(f"- {label}" for label in top_entity_labels)
    top_predicates_text = "\n".join(f"- {p}" for p in top_predicates)

    prompt = f"""You are a knowledge graph expert. I'll provide you with existing triples from a knowledge graph about technology and machine learning.
    Your task is to generate {max_triples} new knowledge triples that would enhance the graph with meaningful technological relationships.

    Here are some key entities in the graph:
    {top_entities_text}

    {user_entity_text}

    Here are common relationship types (predicates) in the graph:
    {top_predicates_text}

    Here's a sample of the existing knowledge graph triples:
    {triple_text}

    IMPORTANT RESPONSE GUIDELINES:
    1. Only generate triples with the predicate types seen above
    2. Format each triple EXACTLY as: (Subject, Predicate, Object)
    3. Return ONLY the list of triples - no explanations, no numbering, no other text
    4. Each triple should be on its own line
    5. Focus on TECHNICAL and solution-oriented relationships - at least 70% of triples should involve technical concepts
    6. Make sure subjects and objects are CONCEPT NAMES, not numeric IDs
    7. CREATE CONNECTIONS between graph entities and user entities, NOT between user entities themselves
    8. Use SPECIFIC technical terminology - prefer terms like "neural networks", "machine learning", "data processing", "sensors", "APIs", etc.
    9. AVOID generic terms like "this innovative solution" or "groundbreaking opportunity"
    10. Focus on DEEP TECHNICAL CONNECTIONS rather than surface-level relationships

    Generate {max_triples} DIVERSE, TECHNICAL triples following these guidelines:"""

    if debug_mode:
        print(f"[LLM Debug] Sending prompt to LLM: {prompt[:200]}...")

    # Rest of your original function remains the same...
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(thread_id=thread.id, role="user", content=prompt)
        run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=ASSISTANT_ID)

        start_time = time.time()
        while True:
            status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if status.status == 'completed':
                break
            if time.time() - start_time > MAX_WAIT_SECONDS:
                print("[LLM] Timeout")
                return []
            time.sleep(2)

        messages = client.beta.threads.messages.list(thread_id=thread.id)
        assistant_msg = next((m for m in messages.data if m.role == "assistant"), None)
        if not assistant_msg or not assistant_msg.content:
            return []

        content = assistant_msg.content[0].text.value.strip()

        if debug_mode:
            print(f"[LLM Debug] Raw response from LLM:\n{content}")

        triple_lines = [re.sub(r'^\s*\d+\.\s*', '', line.strip()) for line in content.split('\n') if '(' in line]

        if debug_mode:
            print(f"[LLM Debug] Extracted triple lines ({len(triple_lines)}):\n{triple_lines}")

        valid = []

        for i, line in enumerate(triple_lines):
            if debug_mode:
                print(f"\n[LLM Debug] Processing line {i + 1}: {line}")

            match = re.search(r'\(([^)]+)\)', line)
            if not match:
                if debug_mode:
                    print(f"[LLM Debug] Line {i + 1}: No match found")
                continue

            parts = [clean_string(x) for x in match.group(1).split(',')]
            if len(parts) != 3:
                if debug_mode:
                    print(f"[LLM Debug] Line {i + 1}: Wrong number of parts: {len(parts)}, expected 3")
                continue

            s, p, o = parts

            if debug_mode:
                print(f"[LLM Debug] Line {i + 1}: Parsed (s,p,o) = ({s}, {p}, {o})")

            # Generate LLM IDs
            llm_s_id = f"llm_{uuid.uuid4().hex[:8]}"
            llm_o_id = f"llm_{uuid.uuid4().hex[:8]}"

            if debug_mode:
                print(f"[LLM Debug] Line {i + 1}: Generated LLM IDs: {llm_s_id}, {llm_o_id}")

            # Check if the entities are numeric IDs
            if s.isdigit():
                # Try to look up the label for this ID
                if s in node_dict:
                    s = node_dict[s]
                    if debug_mode:
                        print(f"[LLM Debug] Line {i + 1}: Replaced subject ID {s} with label {s}")

            if o.isdigit():
                # Try to look up the label for this ID
                if o in node_dict:
                    o = node_dict[o]
                    if debug_mode:
                        print(f"[LLM Debug] Line {i + 1}: Replaced object ID {o} with label {o}")

            new_triple = {
                "subject_id": llm_s_id,
                "subject_label": s,
                "predicate": p,
                "object_id": llm_o_id,
                "object_label": o,
                "origin": "llm"
            }

            if debug_mode:
                print(f"[LLM Debug] Line {i + 1}: Created triple: {new_triple}")

            valid.append(new_triple)

        # Add post-processing to filter redundant triples
        seen_entities = set()
        filtered_triples = []

        for triple in valid:
            s = triple['subject_label']
            o = triple['object_label']

            # Skip very generic terms
            generic_terms = ['this innovative solution', 'groundbreaking opportunity', 'solution', 'opportunity']
            if s.lower() in [term.lower() for term in generic_terms] or o.lower() in [term.lower() for term in
                                                                                      generic_terms]:
                if debug_mode:
                    print(f"[LLM Debug] Skipping generic triple: {s}, {triple['predicate']}, {o}")
                continue

            # Skip if both subject and object are already in previous triples (redundant)
            if (s, o) in seen_entities or (o, s) in seen_entities:
                if debug_mode:
                    print(f"[LLM Debug] Skipping redundant triple: {s}, {triple['predicate']}, {o}")
                continue

            seen_entities.add((s, o))
            filtered_triples.append(triple)

        if debug_mode:
            print(f"[LLM Debug] Filtered {len(valid)} to {len(filtered_triples)} non-redundant triples")

        return filtered_triples

    except Exception as e:
        print(f"[LLM] Error: {e}")
        if debug_mode:
            import traceback
            traceback.print_exc()
        return []
