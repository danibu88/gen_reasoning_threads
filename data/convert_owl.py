from rdflib import Graph, URIRef, Literal, BNode


BASE_IRI = "http://www.semanticweb.org/danielburkhardt/ontologies/2021/1/untitled-ontology-10#"

input_file = "kg-data/ontology.owl"
output_file = "kg-data/ontology_cleaned.ttl"

g = Graph()
g.parse(input_file, format="xml")  # Parse original OWL/XML

def strip_base_iri(term):
    if isinstance(term, URIRef) and str(term).startswith(BASE_IRI):
        return URIRef(str(term).replace(BASE_IRI, ''))
    return term  # Leave Literal, BNode, or other URIs untouched

new_graph = Graph()

for s, p, o in g:
    new_s = strip_base_iri(s)
    new_p = strip_base_iri(p)
    new_o = strip_base_iri(o)
    new_graph.add((new_s, new_p, new_o))

new_graph.serialize(destination=output_file, format="turtle")
print(f"✅ Cleaned Turtle written to {output_file}")
