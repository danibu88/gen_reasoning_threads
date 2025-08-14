from dotenv import load_dotenv
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from gensim.models import Word2Vec
from collections import deque
import os


# Load appropriate .env file
env_file = '.env.production' if os.getenv('FLASK_ENV') == 'production' else '.env.development'
load_dotenv(env_file)

# Load pre-trained word vectors (you'll need to download these)
word2vec_model = Word2Vec.load(os.getenv('W2VMODEL'))

def cosine_sim(v1, v2):
    l1 = []; l2 = []
    rvector = set(v1).union(set(v2)) 
    for w in rvector:
        if w in v1: l1.append(1) # create a vector
        else: l1.append(0)
        if w in v2: l2.append(1)
        else: l2.append(0)
    c = 0
    
    # cosine formula 
    for i in range(len(rvector)):
        c+= l1[i]*l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    return cosine

def get_vector(word):
    try:
        return word2vec_model.wv[word]
    except KeyError:
        return None

def cosine_sim_embeddings(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_hierarchical_concepts(concept, triples_generated):
    parents = triples_generated[triples_generated['obj'] == concept]['sub'].tolist()
    children = triples_generated[triples_generated['sub'] == concept]['obj'].tolist()
    return parents + children

def get_related_concepts(concept, triples_generated):
    related = triples_generated[
        (triples_generated['sub'] == concept) |
        (triples_generated['obj'] == concept)
    ]['relation'].unique().tolist()
    return related

def bfs_ontology(start_concept, triples_generated, max_depth=2):
    queue = deque([(start_concept, 0)])
    visited = set()
    related_concepts = []

    while queue:
        concept, depth = queue.popleft()
        if concept not in visited and depth <= max_depth:
            visited.add(concept)
            related_concepts.append(concept)

            neighbors = triples_generated[
                            (triples_generated['sub'] == concept) |
                            (triples_generated['obj'] == concept)
                            ]['sub'].tolist() + triples_generated[
                            (triples_generated['sub'] == concept) |
                            (triples_generated['obj'] == concept)
                            ]['obj'].tolist()

            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

    return related_concepts


def get_keywords_to_find(triples, unique_topics):
    topics_in_triples = set([i[0] for i in triples]) | set([i[1] for i in triples])
    topics_in_triples_dict = {topic: word_tokenize(topic) for topic in topics_in_triples if
                              word_tokenize(topic) not in stopwords.words('english')}

    cosine_threshold = 0.3  # Adjusted threshold
    potential_topics = []

    print("Generated vectors for the query. Now finding similar triples")
    for unique_topic in unique_topics:
        if isinstance(unique_topic, (float, int)):
            continue  # Skip numeric values
        try:
            unique_topic_vector = word_tokenize(unique_topic)
            if unique_topic_vector in stopwords.words('english'):
                continue

            unique_topic_embedding = get_vector(unique_topic)
            if unique_topic_embedding is None:
                continue

            for topic, topic_vector in topics_in_triples_dict.items():
                topic_embedding = get_vector(topic)
                if topic_embedding is not None:
                    cosine_score = cosine_sim_embeddings(topic_embedding, unique_topic_embedding)
                    if cosine_score > cosine_threshold:
                        potential_topics.append(topic)
                else:
                    # Fallback to token-based similarity if embedding is not found
                    cosine_score = cosine_sim(topic_vector, unique_topic_vector)
                    if cosine_score > cosine_threshold:
                        potential_topics.append(topic)
        except Exception as e:
            print(f"Error processing topic '{unique_topic}': {str(e)}")
            continue

    return list(set(potential_topics))
        

def get_triples_from_onto(triples_generated, topics):
    topics = set(topics)

    # Expand topics with hierarchical and related concepts
    expanded_topics = set()
    for topic in topics:
        expanded_topics.add(topic)
        expanded_topics.update(get_hierarchical_concepts(topic, triples_generated))
        expanded_topics.update(get_related_concepts(topic, triples_generated))
        expanded_topics.update(bfs_ontology(topic, triples_generated))

    topics = expanded_topics

    doc_ids_to_return = []
    for doc_no in range(len(triples_generated)):
        if triples_generated.loc[doc_no, "sub"] and triples_generated.loc[doc_no, "obj"]:
            if triples_generated.loc[doc_no, "sub"] in topics or triples_generated.loc[doc_no, "obj"] in topics:
                doc_ids_to_return.append(triples_generated.loc[doc_no, "doc_id"])

    triples_to_return = []
    for _, row in triples_generated.iterrows():
        if row["doc_id"] in doc_ids_to_return:
            triples_to_return.append([
                row["sub"],
                row["obj"],
                row["relation"]
            ])

    return triples_to_return

def map_triple_to_ontology(triples):
    if not triples:
        return []

    try:
        ontology_csv_path = os.getenv('ONTTRIPLES')
        triples_generated = pd.read_csv(ontology_csv_path)
        unique_topics = set(triples_generated["sub"].tolist()) | set(triples_generated["obj"].tolist())

        topics = get_keywords_to_find(triples, list(unique_topics)[1:])
        triples_from_onto = get_triples_from_onto(triples_generated, topics)

        return triples_from_onto
    except Exception as e:
        print(f"Error in map_triple_to_ontology: {str(e)}")
        return []




