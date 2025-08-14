import re

def clean_owl_prefixes(input_path: str, output_path: str, base_iris: list, remove_all_resource_uris=False):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove all specified base IRIs
    for iri in base_iris:
        content = content.replace(iri, '')

    # Optionally remove all full URIs in rdf:resource attributes
    if remove_all_resource_uris:
        # Keep standard schema URIs unless opted to remove
        content = re.sub(r'rdf:resource="https?://[^"]+[#/]([a-zA-Z0-9_-]+)"', r'rdf:resource="\1"', content)

    # Remove any leading colon if left dangling
    content = re.sub(r'([^a-zA-Z0-9])\:([a-zA-Z_][\w\-]*)', r'\1\2', content)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ Cleaned OWL file saved to {output_path}")


# Example usage
if __name__ == "__main__":
    base_iris = [
        "http://www.semanticweb.org/danielburkhardt/ontologies/2021/1/untitled-ontology-10#",
        "http://www.semanticweb.org/danielburkhardt/ontologies/2021/1/untitled-ontology-13#"
    ]
    input_file = "kg-data/ontology.owl"
    output_file = "kg-data/ontology_cleaned.owl"
    clean_owl_prefixes(input_file, output_file, base_iris, remove_all_resource_uris=True)
