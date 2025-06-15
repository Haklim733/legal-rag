import argparse
from enum import Enum
import io
import logging
from pathlib import Path
import sys
from typing import Literal, Optional 

import duckdb
import requests
import pyarrow as pa
import pyarrow.csv as csv
from pathlib import Path

logger = logging.getLogger(__name__)

class IRSType(Enum):
    ZIPCODE = "zipcode"
    COUNTY = "county"

def get_irs_data(year: int, data_type: IRSType, file_path: Optional[str] = None) -> pa.Table:
    """
    Download IRS CSV file for the given year and data type, and return as a PyArrow Table.
        logger.info(f"Dropping table {table_name}")

    Args:
        year (int): The year to download (e.g., 2022).
        data_type (IRSType): The type of data to download (zipcode or county).

    Raises:
        Exception: If the file cannot be downloaded.
    """
    base_url = "https://www.irs.gov/pub/irs-soi/"
    yy = str(year)[-2:]

    filename = f"{yy}zpallagi.csv"
    logger.info(data_type)
    if data_type == IRSType.COUNTY:
        filename = f"{yy}incyallagi.csv"

    logger.info(file_path)

    if file_path:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}") 
        try:
            table = csv.read_csv(file_path)
            table = table.append_column("year", pa.repeat(pa.scalar(year, type=pa.int16()), len(table)))
            return table
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            raise

    url = f"{base_url}{filename}"

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        table = csv.read_csv(io.BytesIO(response.content))
        table = table.append_column("year", pa.repeat(pa.scalar(year, type=pa.int16()), len(table)))
        return table
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        raise

def insert_csv_to_table(
    table: pa.Table,
    table_name: str,
    backend: Literal["duckdb", "iceberg"],
    duckdb_path: Optional[str] = ":memory:",
    iceberg_append_args: Optional[dict] = None,
    catalog_type: Optional[str] = None,
    namespace: Optional[str] = None,
    role_name: Optional[str] = None,
):
    """
    Insert a CSV file into a DuckDB or Iceberg table.

    Args:
        table (pa.Table): PyArrow Table to insert.
        table_name (str): Name of the target table.
        backend (str): "duckdb" or "iceberg".
        duckdb_path (str, optional): Path to DuckDB database (default: ":memory:").
        iceberg_table (object, optional): PyIceberg Table object (required for Iceberg backend).
        iceberg_append_args (dict, optional): Extra arguments for Iceberg append (if needed).

    Raises:
        ValueError: If backend is not recognized or required arguments are missing.
    """
    if backend == "duckdb":
        con = duckdb.connect(duckdb_path)
        con.register("arrow_table", table)

        con.execute(f"INSERT INTO {table_name} SELECT * FROM arrow_table")
        logger.debug(f"Inserted rows into existing DuckDB table {table_name}")
        return con
    elif backend == "iceberg":
        # Initialize Iceberg catalog
        catalog = load_catalog(
            "default",  # Catalog name
            **{
                "uri": catalog_kwargs.get("uri"),  # e.g., "http://rest-catalog:8181"
                "s3.endpoint": catalog_kwargs.get("s3_endpoint"),
                "py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO",
                **catalog_kwargs
            }
        )
          # Get the table
        iceberg_table = catalog.load_table(f"{namespace}.{table_name}")
        iceberg_table.append(table) 
        logger.info(f"Successfully inserted {len(table)} rows into Iceberg table {namespace}.{table_name}")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="INsert values in DuckDB or AWS Glue using Iceberg format")
    parser.add_argument("--backend", required=True, choices=["duckdb", "iceberg"], help="Backend/engine type: duckdb or iceberg")
    parser.add_argument("--year", required=True, type=int, help="Provide a year")
    parser.add_argument("--data-type", required=True, type=str, choices=[e.value for e in IRSType], help="Provide a data type")
    parser.add_argument("--table-name", required=True, help="Provide a Table name")
    parser.add_argument("--file-path", type=str, default=None, required=False, help="Path to CSV file or download from IRS")

    # DuckDB-specific arguments (conditionally required)
    parser.add_argument("--duckdb-path", type=str, default=":memory:", required=False, help="Path to DuckDB database (default: ':memory:')")

    # Iceberg-specific arguments (conditionally required)
    parser.add_argument("--catalog-type", type=str, default="aws_rest", required=False, help="Type of Iceberg catalog (required for Iceberg backend)")
    parser.add_argument("--namespace", type=str, default="irs", required=False, help="AWS Glue database name (schema, required for Iceberg backend)")
    parser.add_argument("--role-name", type=str, required=False, help="AWS Glue role name (required for Iceberg backend)")

    args = parser.parse_args()

    # Custom validation for Iceberg backend
    if args.backend == "iceberg":
        missing = []
        if not args.catalog_type:
            missing.append("--catalog-type")
        if not args.namespace:
            missing.append("--namespace")
        if not args.role_name:
            missing.append("--role-name")
        if missing:
            parser.error(f"When --backend=iceberg, the following arguments are required: {', '.join(missing)}")

    table= get_irs_data(
        year=args.year,
        data_type=args.data_type,
        file_path=args.file_path,
    )

    success = insert_csv_to_table(
        table=table,
        table_name=args.table_name,
        backend=args.backend,
        duckdb_path=args.duckdb_path,
        catalog_type=args.catalog_type,
        namespace=args.namespace,
        role_name=args.role_name,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
