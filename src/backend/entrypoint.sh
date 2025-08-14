#!/bin/bash

# Re-create symbolic links just to be sure
ln -sf /root/opt/findyoursolution/models/word2vec.model /app/word2vec.model
ln -sf /root/opt/findyoursolution/models/top2vecmodel.model /app/top2vecmodel.model
ln -sf /root/opt/findyoursolution/models/triples_ontology.model /app/triples_ontology.model
ln -sf /root/opt/findyoursolution/models/triples12122022.csv /app/triples12122022.csv

# Print environment information for debugging
echo "Starting Flask application with:"
echo "FLASK_ENV: ${FLASK_ENV:-development}"
echo "DEBUG: ${DEBUG:-False}"
echo "CORS_ALLOWED_ORIGIN: ${CORS_ALLOWED_ORIGIN:-https://findyoursolution.ai}"

# Download NLTK data if needed
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger

# Run the application
exec python app.py