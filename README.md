# Lead Measure Embedding
# Improving Leadership Measurement with Embeddings

## Project Overview
This repository contains research materials, code, and analyses for improving leadership measurement through embedding-based approaches. The project explores the relationships between various leadership constructs by analyzing the semantic similarity in their measurement scales using natural language processing techniques.

Leadership research has seen a proliferation of constructs (ethical leadership, abusive leadership, transformational leadership, etc.) with potential redundancy among them. Traditional psychometric approaches have limitations in detecting semantic similarities between these measures. This project leverages recent advancements in NLP and embedding models to provide new insights into construct relationships.

## Repository Structure
- [`data/`](data/README.md): Contains raw and processed leadership measurement scales and related datasets
  - [`data/raw/`](data/raw/README.md): Original, unmodified data sources
  - [`data/processed/`](data/processed/README.md): Cleaned and prepared datasets
  - [`data/metadata/`](data/metadata/README.md): Information about the datasets and measures
- [`analyses/`](analyses/README.md): Code for data analysis and modeling
  - [`analyses/python/`](analyses/python/README.md): Python scripts and notebooks for embedding generation and analysis
  - [`analyses/r/`](analyses/r/README.md): R scripts for statistical analysis and visualization
- [`docs/`](docs/README.md): Documentation and research outputs
  - [`docs/quarto/`](docs/quarto/README.md): Quarto documents for reproducible research reports

## Getting Started
1. Clone this repository
2. Set up the Python environment (see requirements.txt)
3. Explore the data directory for available leadership measures
4. Run analyses to explore semantic relationships between measures

## Research Questions
- How semantically similar are items across different leadership construct measures?
- Which leadership constructs show redundancy based on their linguistic content?
- Can embedding-based approaches improve how we understand and measure leadership constructs?
- How do different embedding models (commercial APIs vs. open-source models) compare in capturing leadership construct relationships?

## Contributing
This is a research project led by [Your Name/Institution]. Contributions are welcome through discussions and pull requests.

## License
[Specify license information here]
