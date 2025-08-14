import pandas as pd
from elasticsearch import Elasticsearch, helpers
import json
import os
from dotenv import load_dotenv
from time import sleep

# Determine which .env file to load
env = os.getenv('FLASK_ENV', 'development')
env_file = '.env.production' if env == 'production' else '.env.development'

# Load environment variables from the appropriate file
load_dotenv(env_file)

# Get the Elasticsearch connection details from the environment
elasticsearch_host = os.getenv('ES')
elasticsearch_user = os.getenv('ELASTICSEARCH_USER')
elasticsearch_password = os.getenv('ELASTICSEARCH_PASSWORD')
use_cert = os.getenv('CERT', 'False').lower() == 'true'  # Default to False if not set

# Check if the host already includes the protocol
if not elasticsearch_host.startswith(('http://', 'https://')):
    # Add the appropriate prefix based on CERT setting
    protocol = "https://" if use_cert else "http://"
    elasticsearch_host = f"{protocol}{elasticsearch_host}"

# Connect to Elasticsearch with conditional SSL settings
if use_cert:
    es = Elasticsearch(
        [elasticsearch_host],
        basic_auth=(elasticsearch_user, elasticsearch_password),
        verify_certs=True,
        ca_certs="/root/config/certs/ca.crt",
        ssl_show_warn=False,
        ssl_assert_hostname=False
    )
else:
    # Simplified connection without SSL/TLS options
    es = Elasticsearch(
        [elasticsearch_host],
        basic_auth=(elasticsearch_user, elasticsearch_password)
    )

def bulk_index_with_retry(es, actions, max_retries=3):
    for attempt in range(max_retries):
        try:
            return helpers.bulk(es, actions, raise_on_error=False)
        except Exception as e:
            print(f"Retry {attempt + 1}/{max_retries} failed: {e}")
            sleep(2 ** attempt)  # exponential backoff
    return (0, [{"error": "Max retries exceeded"}])

# Create indices with proper mappings if they don't exist
def create_indices():
    indices = ["invocations_1", "invocations_2", "invocations_3", "invocations_4"]
    mapping = {
        "mappings": {
            "properties": {
                "Topics": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "URLs": {"type": "text", "fields": {"keyword": {"type": "keyword"}}}
            }
        }
    }

    for index in indices:
        if not es.indices.exists(index=index):
            es.indices.create(index=index, body=mapping)
        else:
            print(f"Index {index} already exists")


# Function to create actions for bulk indexing
def create_actions(df, index_name):
    """
    Create actions for bulk indexing from a dataframe
    """
    # Clean dataframe - handle nulls
    df = df.replace({pd.NA: None, float('nan'): None, 'NaN': None})
    df = df.where(pd.notnull(df), None)

    # Print columns for debugging
    print(f"Processing file with columns: {df.columns.tolist()}")

    # Map columns appropriately
    if 'description' in df.columns and 'url' in df.columns:
        df = df.rename(columns={'description': 'Topics', 'url': 'URLs'})
    elif 'text' in df.columns and 'url' in df.columns:
        df = df.rename(columns={'text': 'Topics', 'url': 'URLs'})
    elif 'Content' in df.columns and 'Query' in df.columns:
        df = df.rename(columns={'Content': 'Topics', 'Query': 'URLs'})
    else:
        print(f"Warning: CSV does not have required columns. Using default mapping.")
        # Try to determine which columns might be Topics and URLs based on content
        if 'Topics' not in df.columns and len(df.columns) > 0:
            # Use first non-id column as Topics
            potential_topics = [col for col in df.columns if not col.lower() in ['id', 'unnamed: 0']]
            if potential_topics:
                df['Topics'] = df[potential_topics[0]]
            else:
                df['Topics'] = ''

        if 'URLs' not in df.columns and len(df.columns) > 1:
            # Use last column as URLs if not already assigned
            potential_urls = [col for col in df.columns if 'url' in col.lower()]
            if potential_urls:
                df['URLs'] = df[potential_urls[0]]
            else:
                df['URLs'] = ''

    # Truncate long strings to avoid Elasticsearch limits
    max_topic_length = 1000
    max_url_length = 8000  # Well under the 32766 byte limit

    # Truncate Topics and add ellipsis if needed
    df['Topics'] = df['Topics'].astype(str).apply(
        lambda x: x.strip()[:max_topic_length] + '...' if len(x.strip()) > max_topic_length else x.strip()
    )

    # Truncate URLs and add ellipsis if needed
    df['URLs'] = df['URLs'].astype(str).apply(
        lambda x: x.strip()[:max_url_length] + '...' if len(x.strip()) > max_url_length else x.strip()
    )

    # Count truncated fields
    topics_truncated = sum(df['Topics'].str.endswith('...'))
    urls_truncated = sum(df['URLs'].str.endswith('...'))
    if topics_truncated > 0:
        print(f"Warning: Truncated {topics_truncated} Topics fields that exceeded {max_topic_length} characters")
    if urls_truncated > 0:
        print(f"Warning: Truncated {urls_truncated} URLs fields that exceeded {max_url_length} characters")

    # Generate actions
    for i, row in df.iterrows():
        doc = {
            "Topics": row.get('Topics', ''),
            "URLs": row.get('URLs', '')
        }

        # Add other fields (excluding very long values)
        for col in df.columns:
            if col not in ['Topics', 'URLs'] and pd.notnull(row[col]):
                value = str(row[col])
                if len(value) <= max_url_length:  # Same limit for other fields
                    doc[col] = value

        yield {
            "_index": index_name,
            "_source": doc
        }


# Create indices
create_indices()

# Define data sources based on environment
if env == 'production':
    data_sources = {
        '/root/opt/findyoursolution/data/devTo_updated_2023.csv': 'invocations_1',
        '/root/opt/findyoursolution/data/medium_articles_with_url_indexed.csv': 'invocations_2',
        '/root/opt/findyoursolution/data/Data_extracted/towardsdatascience.csv': 'invocations_3',
        '/root/opt/findyoursolution/data/Data_extracted/wikipedia.csv': 'invocations_4'
    }
else:  # development
    data_sources = {
        'data/devTo_updated_2023.csv': 'invocations_1',
        'data/medium_articles_with_url_indexed.csv': 'invocations_2',
        'data/towardsdatascience.csv': 'invocations_3',
        'data/wikipedia.csv': 'invocations_4',
        'data/autonomizationRPs.csv': 'invocations_5'
    }

# Read and index the CSV files in chunks
chunksize = 1000  # Reduced chunk size
total_indexed = 0

for file_name, index_name in data_sources.items():
    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        continue

    try:
        file_indexed = 0
        for chunk_num, chunk in enumerate(pd.read_csv(file_name, chunksize=chunksize)):
            chunk = chunk.dropna(how='all')
            # Create actions list (convert generator to list to avoid consumption issues with retries)
            actions = list(create_actions(chunk, index_name))
            if not actions:
                print(f"  Chunk {chunk_num + 1}: No valid actions created, skipping")
                continue

            success, failed = bulk_index_with_retry(es, actions, max_retries=3)
            file_indexed += success
            total_indexed += success
            if failed:
                print(f"  Failed to index {len(failed)} documents")
                for error in failed[:3]:
                    print(json.dumps(error, indent=2))

    except Exception as e:
        print(f"Error processing file {file_name}: {e}")

# Check final index stats
for index in ["invocations_1", "invocations_2", "invocations_3", "invocations_4"]:
    try:
        stats = es.indices.stats(index=index)
        doc_count = stats['_all']['primaries']['docs']['count']
    except Exception as e:
        print(f"Couldn't get stats for {index}: {e}")
