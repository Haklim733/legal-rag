#!/usr/bin/env python
"""
Script to create the EOD (End of Day) table in AWS Glue using Iceberg format.
This script demonstrates how to use the GlueIceberg class to create a table
based on the predefined EOD schema.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import boto3
import duckdb
from irs.db import PyIceberg, CatalogType
from irs.models import TableSchema

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def assume_role(role_name: str, session_name="DataOpsSession") -> tuple[dict[str, str], str]:
    sts = boto3.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
    response = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName=session_name,
    )
    creds = response["Credentials"]
    return creds, account_id

def create_table_duckdb(
    table_name: str,
    ddl_path: str,
    duckdb_path: Optional[str] = None,
    recreate: Optional[bool] = False,
    **kwargs
) -> bool:
    db_path = duckdb_path if duckdb_path else ":memory:"
    conn = duckdb.connect(db_path)

    logger.info(f"Creating table {table_name} in DuckDB")
    path = Path(ddl_path).resolve()
    logger.info(f"Resolved path: {path}")
    with open(ddl_path, "r") as ddl_file:
        ddl_sql = ddl_file.read()

    if recreate:
        logger.info(f"Dropping table {table_name}")
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(ddl_sql)
    conn.close()
    return True

def create_table_pyiceberg(
    bucket: str,
    namespace: str,
    role_name: str,
    ddl_path: str = None,
    catalog_type: CatalogType = CatalogType.AWS_REST,
    table_name: Optional[str] = None,
    key: Optional[str] = None,
    recreate: Optional[bool] = False,
    partition_col: Optional[str] = None
) -> bool:
    """
    Create the tables in AWS Glue using Iceberg format.
    
    Args:
        bucket: S3 bucket name
        key: Optional S3 key path
        namespace: AWS Glue database name (schema)
        table_name: Optional custom table name (defaults to schema-defined name)
        recreate: Optional flag to recreate the table if it exists
        
    Returns:
        bool: True if table creation was successful, False otherwise
    """

    try:
        # Initialize the GlueIceberg client with explicit credentials

        credentials, account_id = assume_role(role_name)
        location = f"s3://{bucket}/{key}" #doesn't matter for s3 tables
        logger.info(bucket)
        schema = TableSchema(name=table_name, ddl_path=ddl_path)

        with PyIceberg(
            catalog_name="s3tablescatalog",
            catalog_type=catalog_type,
            account=account_id,
            bucket=bucket,
            key=key,
            credentials=credentials,
            
        ) as iceberg_db:
            # Get the EOD schema
            ns = iceberg_db.catalog.list_namespaces()
            if (namespace,) not in ns:
                iceberg_db.catalog.create_namespace(namespace)

            if (namespace, table_name) not in iceberg_db.catalog.list_tables(namespace): 
                logger.info(f"actual {table_name}, namespace {namespace}")
            
                # Prepare table properties if S3 location is provided
                properties = {
                    "write.target-file-size-bytes": str(256 * 1024 * 1024),  # 256MB
                    "history.expire.max-snapshot-age-ms": str(720 * 60 * 60 * 1000),  # 720 hours in ms
                    "history.expire.min-snapshots-to-keep": "5",
                }
                if partition_col:
                    properties["partition"] = partition_col

                logger.info(f"Creating table {namespace}.{table_name} with properties: {properties}") 
                if recreate:
                    iceberg_db.catalog.purge_table((namespace, table_name))

                result = iceberg_db.create_table(
                    table_name=table_name,
                    namespace=namespace,
                    fields=schema.fields,
                    properties=properties,
                    location=location
                )

                logger.info(iceberg_db.catalog.list_tables(namespace))
            
            if result:
                logger.info(f"Successfully created table {namespace}.{table_name}")
                
                # Verify the table was created correctly
                if iceberg_db.verify(table_name=table_name, namespace=namespace):
                    logger.info(f"Table verification successful")
                else:
                    logger.warning(f"Table verification failed")
                    
                return True
            else:
                logger.error(f"Failed to create table {namespace}.{actual_table_name}")
                return False
                
    except Exception as e:
        logger.error(f"Error: {e}")
        return False

def create_table(
    backend: str,
    **kwargs
) -> bool:
    if backend == "duckdb":
        return create_table_duckdb(**kwargs)
    elif backend == "iceberg":
        return create_table_pyiceberg(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}") 

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Create table in DuckDB or AWS Glue using Iceberg format")

    parser.add_argument("--backend", required=True, choices=["duckdb", "iceberg"], help="Backend/engine type: duckdb or iceberg")
    parser.add_argument("--table-name", required=True, help="Provide a Table name")
    parser.add_argument("--ddl", required=True, help="Path to DDL file for table schema")
    parser.add_argument("--recreate", action="store_true", required=False, help="Set to True to recreate the table if it exists")

    # DuckDB-specific arguments (conditionally required)
    parser.add_argument("--duckdb-path", type=str, required=False, help="Path to DuckDB database (default: ':memory:')")

    # Iceberg-specific arguments (conditionally required)
    parser.add_argument("--bucket", required=False, help="S3 bucket name (Iceberg only)")
    parser.add_argument("--key", required=False, help="S3 key path (Iceberg only)")
    parser.add_argument("--catalog-type", type=str, help="Type of Iceberg catalog (required for Iceberg backend)")
    parser.add_argument("--namespace", type=str, help="AWS Glue database name (schema, required for Iceberg backend)")
    parser.add_argument("--role-name", type=str, help="AWS Glue role name (required for Iceberg backend)")

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

    success = create_table(
        backend=args.backend,
        namespace=args.namespace,
        table_name=args.table_name,
        ddl_path=args.ddl,
        recreate=args.recreate,
        catalog_type=args.catalog_type,
        bucket=args.bucket,
        key=args.key,
        role_name=args.role_name,
        duckdb_path=args.duckdb_path,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
