# Supreme Court Knowledge Graph

This module provides tools to build and query a knowledge graph of U.S. Supreme Court cases using [Data Commons](https://datacommons.org/).

## Features

- Query Supreme Court cases by year
- Retrieve detailed information about specific cases
- Save and load knowledge graphs to/from JSON
- Command-line interface for building knowledge graphs

2. (Optional) Get a Data Commons API key from [Google Cloud Console](https://console.cloud.google.com/apis/credentials) if you need higher rate limits.

## Usage

### Command Line Interface

The `build_supreme_court_kg.py` script provides a command-line interface to build a knowledge graph:

```bash
# Build a knowledge graph with cases from 2020-2022 (default)
python scripts/build_supreme_court_kg.py --output data/supreme_court_kg.json

# Specify custom years and limit
python scripts/build_supreme_court_kg.py --years 2018 2019 2020 --limit 20 --output data/custom_kg.json

# Use an API key
python scripts/build_supreme_court_kg.py --api-key YOUR_API_KEY
```

### Python API

```python
from pathlib import Path
from src.kg.supreme_court_kg import SupremeCourtKG, save_knowledge_graph

# Initialize the knowledge graph client
kg_client = SupremeCourtKG(api_key="YOUR_API_KEY")  # API key is optional

# Query cases from a specific year
cases = kg_client.query_cases_by_year(year=2022, limit=5)

# Get detailed information about a specific case
if cases:
    case_details = kg_client.get_case_details(cases[0].dcid)
    print(f"Case details: {case_details}")

# Save the knowledge graph to a file
output_path = Path("data/supreme_court_cases.json")
save_knowledge_graph(cases, output_path)
```

## Data Model

The knowledge graph represents Supreme Court cases with the following attributes:

- `dcid`: Data Commons ID (unique identifier)
- `name`: Name/title of the case
- `description`: Brief description of the case
- `date_decided`: Date when the case was decided
- `citation`: Legal citation for the case
- `parties`: Names of the parties involved (e.g., "Roe v. Wade")
- `decision_direction`: How the case was decided (e.g., "affirmed", "reversed")
- `opinion_author`: Name of the justice who wrote the majority opinion

## Rate Limiting

The Data Commons API has rate limits for unauthenticated requests. To avoid hitting these limits:

1. Use an API key for higher rate limits
2. Add delays between requests when processing many cases
3. Cache results locally when possible

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
