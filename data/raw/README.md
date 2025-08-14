# Dataset Documentation

This directory contains the raw datasets used in the ReT-Eval framework research.

## Available Datasets

### ✅ Included Files:
- **`medium_articles_with_url_indexed.csv`** - Medium articles with URL indexing and metadata
- **`towardsdatascience.csv`** - Towards Data Science articles and content
- **`updated2.csv`** - Updated dataset version 2 with additional annotations
- **`wikipedia.csv`** - Wikipedia articles subset focused on technical topics

### 📁 Large Files (External Storage Required):
Due to GitHub file size limitations, the following datasets need to be downloaded separately:

- **`autonomizationRPs.csv`** (223MB) - Autonomization research papers dataset
  - Contains research papers related to automation and autonomization
  - Used for domain-specific knowledge extraction in ReT-Eval

- **`devTo_updated_2023.csv`** (106MB) - Dev.to articles dataset from 2023
  - Technical articles and tutorials from the Dev.to platform
  - Provides current technical knowledge for reasoning thread generation

## Dataset Usage in ReT-Eval

These datasets are used for:
1. **Knowledge Graph Construction**: Extracting domain-specific entities and relationships
2. **Triple Generation**: Creating semantic triples for the knowledge base
3. **Evaluation**: Providing ground truth data for reasoning thread assessment
4. **Domain Coverage**: Ensuring comprehensive coverage across technical domains

## Download Instructions

```bash
# The large files are available through:
# 1. Contact the authors directly
# 2. Check the paper's supplementary materials
# 3. Repository releases section (if available)

.
