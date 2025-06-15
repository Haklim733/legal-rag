#!/bin/bash

YEAR=2023

# Create ind_zipcode table in DuckDB
# uv run python scripts/create_tables.py \
#     --backend duckdb \
#     --table-name ind_zipcode \
#     --ddl ddl/ind_zipcode_${YEAR}.sql \
#     --duckdb-path db.db \
#     --recreate

# Create ind_county table in DuckDB
uv run python -m src.irs.create_tables \
    --backend duckdb \
    --ddl ./ddl/individual_county.ddl \
    --table-name individual_county \
    --duckdb-path db.db \
    --recreate

# insert values
uv run python -m src.irs.extract \
    --backend duckdb \
    --table-name individual_county \
    --duckdb-path db.db \
    --year $YEAR \
    --data-type county \
    --file-path ./assets/2022/22incyallagi.csv