import logging
from abc import ABC, abstractmethod
from enum import Enum
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
import uuid

import duckdb
import psutil
import pyarrow as pa
import sqlglot.expressions as exp
from pyiceberg.catalog import load_catalog
from pyiceberg.table import Table
from pyiceberg.partitioning import PartitionSpec, PartitionField
import pyiceberg.transforms as transforms

from irs.models import TableSchema, SchemaField, TableOptions
from irs.file_reader import FileReader
from irs.sql_gen import SqlGenerator, DataFormat, SqlAction

logger = logging.getLogger(__name__)


class CatalogType(Enum):
    AWS_GLUE = "glue"
    AWS_REST = "rest"

class DbEngine(ABC):
    """Abstract base class for database connections"""
    
    sql_dialect: str = "" # Default dialect, should be overridden by subclasses
    
    def __init__(self, file_reader: Optional[FileReader] = None, **kwargs):
        """Initialize the DB class with an optional file reader.
        
        Args:
            file_reader: Optional FileReader instance for reading files
            **kwargs: Additional arguments for initialization
        """
        self.file_reader = file_reader or FileReader()
        self.sql_generator = SqlGenerator(sql_dialect=self.sql_dialect) 
        self.profile = False

    @abstractmethod
    def configure(self, **kwargs):
        """Configure the database connection"""
        raise NotImplementedError
      
    @abstractmethod
    def close(self):
        """Close the database connection"""
        raise NotImplementedError
         
    @abstractmethod
    def verify(self, table: str, **kwargs) -> bool:
        """Verify a table exists and has expected schema"""
        raise NotImplementedError
     
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Explicitly close the connection if not using with statement"""
        raise NotImplementedError

     
    @abstractmethod
    def _get_column_info(self, data, data_format: str) -> Tuple[List[str], str]:
        """Helper method to get column information from various data sources.
        
        Args:
            data: The data source (table name, file path, etc.)
            data_format: Format of the data ('dict', 'table', or 'file')
            
        Returns:
            Tuple containing:
                - List of column names
                - Source table name or expression for SQL
        """
        raise NotImplementedError

    def _check_table_exists(self, table_name: str, schema_name: Optional[str] = None) -> Tuple[bool, str]:
        """Check if a table exists
        
        Args:
            table_name: Table name to check
            schema_name: Optional schema name
            
        Returns:
            Tuple[bool, str]: (table_exists, full_table_name)
        """
        # Use the SqlGenerator to build a properly quoted table identifier
        full_table_name = self.sql_generator.build_table_identifier(table_name, schema_name, self.sql_dialect)
        
        try:
            # Build a safer SQL query using sqlglot
            where_conditions = [f"table_name = '{table_name}'"]
            if schema_name:
                where_conditions.append(f"table_schema = '{schema_name}'")
                
            query = f"""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE {' AND '.join(where_conditions)}
            """
            
            table_exists = self.con.execute(query).fetchone()[0] > 0
            
            return table_exists, full_table_name
        except Exception as e:
            logger.warning(f"Error checking if table exists: {e}")
            return False, full_table_name
    
    def extract_schema_and_rows(self, data: List[Dict], table_name: str) -> Tuple[TableSchema, List[List[Any]]]:
        """Infer a TableSchema from dictionary data and return (schema, parameterized values)
        
        Args:
            data: List of dictionaries to infer schema from
            table_name: Name for the table
            
        Returns:
            Tuple[TableSchema, List[List[Any]]]: (Inferred schema, parameterized 2D list of values)
        """
        if not data or not isinstance(data, list) or not isinstance(data[0], dict):
            raise ValueError("Data must be a non-empty list of dictionaries")
        
        # Get a sample record to infer types
        sample = data[0]
        fields = []
        for key, value in sample.items():
            # Simple type inference; you may want to expand this
            if isinstance(value, int):
                dtype = "INTEGER"
            elif isinstance(value, float):
                dtype = "FLOAT"
            elif isinstance(value, str):
                dtype = "VARCHAR"
            else:
                dtype = "VARCHAR"
            fields.append(SchemaField(name=key, data_type=dtype))
        schema = TableSchema(name=table_name, fields=fields)
        columns = [field.name for field in fields]
        param_values = [[row.get(col) for col in columns] for row in data]
        return schema, param_values

    def create_table(self, 
        table_name: str,
        fields: List[SchemaField],
        schema_name: Optional[str] = None,
        custom_field_mapping: Optional[Dict[str, Any]] = None,
        table_options: Optional[TableOptions] = None,
        temporary: bool = False,
        recreate: bool = False) -> bool:
        """Create a table using the schema definition.
            
            Args:
                name: Name of the table to create
                fields: List of SchemaField objects defining the table structure
                schema_name: Schema (namespace) name
                custom_field_mapping: Optional mapping to customize field properties
                table_options: Additional options for the table creation
                temporary: Whether to create a temporary table
                recreate: Whether to drop and recreate the table if it exists
                
            Returns:
                bool: True if successful
            """
            # Check if table exists
        if not schema_name and not temporary:
            raise ValueError("schema_name must be provided for non-temporary tables")
        table_exists, full_table_name = self._check_table_exists(table_name, schema_name)
        if table_exists and not recreate:
            logger.debug("Table {} already exists. Use recreate=True to recreate.".format(full_table_name))
            return False
            
        # Create the schema if it doesn't exist
        self.con.execute("CREATE SCHEMA IF NOT EXISTS {}".format(schema_name))
        
        # Drop table if recreate is True and table exists
        if recreate and table_exists:
            self.con.execute("DROP TABLE IF EXISTS {}".format(full_table_name))
            
        create_sql = self.sql_generator.build_create_table_sql(
                name=table_name,
                fields=fields,
                schema_name=schema_name,
                temporary=temporary,
                table_options=table_options
            )
        
        # Execute the create table statement
        logger.debug("executing {}".format(create_sql))
        self.con.execute(create_sql)
        return True

    def _prepare_data_for_operation(self, data: Any, operation: SqlAction, data_format: DataFormat, **kwargs) -> Tuple[Any, str, bool]:
        """Helper function to prepare data for database operations by creating a temporary table unless the data_format is a file or temporary table and sql dialect is duckdb. If the data_format is a file and the dialect is duckdb, then the duckdb file will be used.
        
        Args:
            data: Input data in various formats (dict, list, table, file path, etc.)
            operation: Name of the operation (e.g. SqlAction)
            table_name: Name of the target table (used for naming temp tables)
            data_type: Explicit type of data being provided ('dict', 'file', 'table', 'list')
                      If None, will attempt to auto-detect the data type
            
        Returns:
            Tuple containing:
                - source: The data to use in the operation (might be a temp table name)
                - data_format: The detected format ('dict', 'table', 'file', etc.)
                - is_temp_table: Boolean indicating if a temporary table was created
        """ 
        if not data: 
            raise Exception(f"No data for {operation.value} operation on {table_name}")
    
        source = data
        is_temp_table = False
        temp_table_name = f"temp_{operation.value}_{id(data)}"

        try:
            if data_format == DataFormat.VALUES:
                schema, rows = self.extract_schema_and_rows(data, temp_table_name)
                # Create the temporary table with the inferred schema
                self.create_table(
                    name=temp_table_name,
                    fields=schema.fields,
                    temporary=True,
                    recreate=True
                )    
                insert_sql = self.sql_generator.build_sql(SqlAction.INSERT, table_name=temp_table_name, columns=[field.name for field in schema.fields]) 

                self.con.executemany(insert_sql, rows) 
                logger.debug(f"Created temporary table {temp_table_name} with schema for {len(rows)} records") 
                source = temp_table_name
                is_temp_table = True
                data_format = DataFormat.TABLE
                
            elif data_format == DataFormat.FILE:
                if not isinstance(data, (str, Path)):
                    raise ValueError(f"Data type 'file' specified but data is not a string or Path: {type(data)}")
                    
                file_path = str(data)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                    
                source = file_path
                is_temp_table = False
                
            elif data_format == DataFormat.TABLE:
                pass

            elif data_format == DataFormat.PYARROW:
                pass
            else:
                pass

        except Exception as e:
            logger.error(f"Failed to create temporary table from data: {e}")
            raise

        return source, data_format, is_temp_table        
    
        
    def execute(self, query: str, parameters=None):
        """Execute a SQL query
        
        Args:
            query: SQL query to execute
            parameters: Optional parameters for the query. If a list of lists/tuples, executemany will be used
            
        Returns:
            list: Query results for regular queries, None for bulk operations
        """
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        explain_query = query
        
        profile = self.profile
         
        if profile and query.strip().upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'COPY')):
            explain_query = f"EXPLAIN ANALYZE {query}"
        
        # Validate that the query is not empty or only whitespace
        if not explain_query or not explain_query.strip():
            raise ValueError("SQL query is empty or only whitespace. Cannot execute.")

        is_bulk = parameters is not None and isinstance(parameters, list) and len(parameters) > 0 and \
                    (isinstance(parameters[0], list) or isinstance(parameters[0], tuple))
        try:
            if is_bulk:
                logger.debug(f"Executing query with {len(parameters)} parameter sets: {query}...") 
                results = self.con.executemany(explain_query, parameters).fetchall()
            else:
                logger.debug(f"Executing query: {query}...")
                results = self.con.execute(explain_query, parameters).fetchall()
        except Exception as e:
            logger.error(f"SQL execution failed. Query: '{explain_query}' Parameters: {parameters}")
            raise

        
        if parameters:
            logger.debug(f"Executed query with {len(parameters)} parameter sets")
        else:
            logger.debug("Executed query with no parameters")
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_change = mem_after - mem_before
        logger.debug(f"Memory change: {mem_change:.2f} MB ({mem_before:.2f} MB → {mem_after:.2f} MB)")

        plan_lines = self.format_profile_output(results) 
        plan_text = "\n".join(line for line in plan_lines if line.strip())
        logger.debug(f"Query execution plan for: {query[:100]}...\n{plan_text}")
            
        return results

    def insert_data(self, table_name: str, schema_name: str, data: Union[List[Dict], Dict, str, pa.Table, Any], 
                   data_format: DataFormat, 
                   operation: SqlAction = SqlAction.INSERT, key_columns: Optional[List[str]] = None, **kwargs) -> bool:
        """Insert, update, or merge data into a DuckDB table.
        This is a convenience method that delegates to the specific operation methods.
        
        Args:
            table_name: Name of the table to insert into
            data: Data or source of data to insert
            data_type: Type of data being provided, one of:
                      - 'dict': Dictionary
                      - 'list' list of dictionaries, tuples
                      - 'file': Path to a file (CSV, Parquet, etc.)
                      - 'table': PyArrow table or database table name
                      If None, will attempt to auto-detect the data type
            schema_name: Optional schema name
            operation: The operation to perform, one of 'insert', 'update', 'merge'
            key_columns: Primary key column(s) to use for update/merge operations
            **kwargs: Additional arguments
            
        Returns:
            bool: True if successful
        """
        if operation not in [SqlAction.INSERT, SqlAction.UPDATE, SqlAction.MERGE]:
            raise ValueError(f"Unsupported operation: {operation}. Must be one of 'insert', 'update', 'merge'")

        if operation == SqlAction.INSERT and len(data) == 0:
            raise ValueError("No data provided for INSERT operation")

        if operation in [SqlAction.UPDATE, SqlAction.MERGE] and not key_columns:
            raise ValueError(f"{operation.capitalize()} operation requires key_columns parameter")
 
        source, data_format, is_temp_table = self._prepare_data_for_operation(data, operation, data_format=data_format)

        logger.debug(f"Source: {source}, Data format: {data_format}, Is temp table: {is_temp_table}")
 
        if source is None:
            raise ValueError("No data provided for operation")
        
        try:
            if operation == SqlAction.INSERT:
                if data_format == DataFormat.PYARROW:
                    record = data.to_batches()
                    reader = pa.ipc.RecordBatchReader.from_batches(data.schema, record) 
                    stmt = self.sql_generator.build_sql(SqlAction.INSERT, 
                                                    table_name=table_name, 
                                                    schema_name=schema_name, 
                                                    source_table_name='reader', 
                                                    data_format=data_format)
                    self.con.execute(stmt)
                    return True
                else:
                    stmt = self.sql_generator.build_sql(SqlAction.INSERT, 
                                                    table_name=table_name, 
                                                    schema_name=schema_name, 
                                                    source_table_name=source, 
                                                    data_format=data_format)
            elif operation == SqlAction.UPDATE:
                
                if data_format == DataFormat.VALUES:
                    # Create a unique temporary table name
                    temp_table_name = f"temp_update_{uuid.uuid4().hex[:16]}"
                    
                    # Create a temporary table with the update data
                    self.insert_data(temp_table_name, data, operation=SqlAction.INSERT, is_temp=True)
                    
                    # Use the temporary table as the source for the update
                    source = temp_table_name 
                        
                    # Generate and execute the update SQL with the temporary table
                    stmt = self.sql_generator.build_sql(SqlAction.UPDATE,
                                                        table_name=table_name,
                                                        schema_name=schema_name,
                                                        source_table_name=source,
                                                        set_values=data,
                                                        data_format=DataFormat.TABLE,
                                                        key_columns=key_columns)
                else:
                    # For updates with an existing source (table or file), use standard approach
                    stmt = self.sql_generator.build_sql(SqlAction.UPDATE,
                                                      table_name=table_name,
                                                      schema_name=schema_name,
                                                      source_table_name= source,
                                                      set_values=data,
                                                      data_format=data_format,
                                                      key_columns=key_columns)
            elif operation == SqlAction.MERGE:
                result = self.con.execute(f"SELECT * FROM {schema_name}.{table_name} LIMIT 0").description
                all_columns = [col[0] for col in result]
                    
                stmt = self.sql_generator.build_sql(SqlAction.MERGE, 
                                                 table_name=table_name, 
                                                 schema_name=schema_name, 
                                                 source_table_name=source, 
                                                 key_columns=key_columns, 
                                                 all_columns=all_columns,
                                                 data_format=data_format)
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            logger.debug(f"Executing {stmt}")
            self.execute(stmt)
            return True

        finally:
            # Clean up temporary table if one was created
            if is_temp_table and source:
                try:
                    self.con.execute(f"DROP TABLE IF EXISTS {source}")
                    logger.debug(f"Dropped temporary table {source}")
                except Exception as e:
                    logger.warning(f"Failed to drop temporary table {source}: {e}")
                    
    def verify(self, table: str, **kwargs) -> bool:
        """Verify a table exists and contains expected data
        
        Args:
            table: Table name to verify
            **kwargs: Additional verification options
            
        Returns:
            bool: True if verification passed
        """
        if not self.STOCK_COLUMNS:
            logger.debug(f"Skipping verification for table {table}: no stock columns defined")
            return True
            
        stock_columns = self.STOCK_COLUMNS
        verification_query = f"""
        SELECT 
            COUNT(*) as total_rows,
            {', '.join(f'SUM(CASE WHEN {stock} IS NULL THEN 1 ELSE 0 END) as {stock}_null_count' for stock in stock_columns)}
        FROM {table} 
        WHERE date > (SELECT MIN(date) FROM price_wide)
        """
        result = self.con.execute(verification_query).fetchone()
        total_rows = result[0]
        logger.debug(f"Verification: Table has {total_rows} rows")
        for i, stock in enumerate(stock_columns):
            null_count = result[i+1]
            if null_count == total_rows:
                logger.error(f"Column {stock} has all NULL values in the table")
            elif null_count > 0:
                logger.warning(f"Column {stock} has {null_count} NULL values out of {total_rows} in the table")
        return True

class DuckDB(DbEngine):
    """DuckDB database connection implementation"""
    
    sql_dialect = "duckdb"
    
    def __init__(self, db_path: str = ':memory:', cpu_count: int = 4, memory_limit: int = 8, profile: bool = False, file_reader: Optional[FileReader] = None, **kwargs):
        """Initialize a DuckDB connection
        
        Args:
            db_path: Path to database file or :memory: for in-memory database
            cpu_count: Number of CPU threads to use
            memory_limit: Memory limit in GB
            profile: Whether to enable query profiling
            file_reader: Optional FileReader instance for reading files
            **kwargs: Additional arguments
        """
        super().__init__(file_reader=file_reader, **kwargs)
        self.con = duckdb.connect(db_path)
        self.memory_limit = memory_limit
        self.cpu_count = cpu_count
        self.profile = profile
        self.process = psutil.Process(os.getpid())
        self.STOCK_COLUMNS = kwargs.get('stock_columns', [])
        
    def __enter__(self):
        """Enter context manager"""
        return self

    def configure(self, libraries: list[str] = None, repo: Literal["core", "core_nightly"] = "core", credentials: Optional[dict[str, str]] = None, **kwargs):
        """Configure the DuckDB connection
        
        Args:
            libraries: Optional list of libraries to install
            repo: Repository to install libraries from
            **kwargs: Additional configuration options
        """
        if libraries:
            self.con.execute(" UPDATE EXTENSIONS;")
            for lib in libraries:
                try:
                    self.con.execute(f""" FORCE INSTALL {lib} FROM {repo}; LOAD {lib}; """)
                except Exception as e:
                    logger.warning(f"Failed to install library {lib}: {e}")

        self.con.execute(f"PRAGMA memory_limit='{self.memory_limit}GB'")
        self.con.execute(f"PRAGMA threads={self.cpu_count}")

        if self.profile:
            stmt = f"""
            PRAGMA profiling_mode='detailed';
            PRAGMA custom_profiling_settings = '{"CPU_TIME": "true", "EXTRA_INFO": "true", "OPERATOR_CARDINALITY": "true", "OPERATOR_TIMING": "true"}';
            PRAGMA enable_object_cache=false;
            """ 
            self.con.execute(stmt)

        stmt = f"""
        CREATE SECRET (
            TYPE s3,
            PROVIDER credential_chain
        );
        """
        self.con.execute(stmt)
        if credentials:
            if 's3' in credentials:
                """ create secret for s3 using sts credentials """
                stmt = f"""
                CREATE SECRET (
                    TYPE s3,
                    KEY_ID '{credentials['s3']['AccessKeyId']}',
                    SECRET '{credentials['s3']['SecretAccessKey']}',
                    REGION '{credentials['s3']['Region']}'
                );
                """
                stmt = f"""
                CREATE SECRET (
                    TYPE s3,
                    PROVIDER credential_chain
                );
                """
                self.con.execute(stmt)

    def format_profile_output(self, explain_output: list[Any]):
        """Format the output from an EXPLAIN ANALYZE query (specific to duckdb)
        
        Args:
            explain_output: Output from EXPLAIN ANALYZE query
            
        Returns:
            list: Formatted plan lines
        """
        plan_lines = []
        for row in explain_output:
            if isinstance(row, (tuple, list)) and len(row) > 0:
                for item in row:
                    if item is not None:
                        if isinstance(item, str) and '\n' in item:
                            plan_lines.extend(item.split('\n'))
                        else:
                            plan_lines.append(str(item))
        return plan_lines
 
    def _read_csv(self, file_path: str, **kwargs) -> str:
        """Read a CSV file into a format suitable for SQL operations (DuckDB implementation)
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments for reading CSV
            
        Returns:
            str: SQL expression that can be used in queries
        """
        return f"read_csv_auto('{file_path}')"  # DuckDB-specific CSV reader function
        
    def _read_parquet(self, file_path: str, **kwargs) -> str:
        """Read a Parquet file into a format suitable for SQL operations (DuckDB implementation)
        
        Args:
            file_path: Path to the Parquet file
            **kwargs: Additional arguments for reading Parquet
            
        Returns:
            str: SQL expression that can be used in queries
        """
        return f"read_parquet('{file_path}')"  # DuckDB-specific Parquet reader function
        
    def _get_column_info(self, data, data_format: str) -> Tuple[List[str], str]:
        """Helper method to get column information from various data sources (specific to duckdb).
        
        Args:
            data: The data source (table name, file path, etc.)
            data_format: Format of the data ('dict', 'table', or 'file')
            
        Returns:
            Tuple containing:
                - List of column names
                - Source table name or expression for SQL
        """
        all_columns = None
        source_table_name = None
        
        if data_format == 'table':
            # For table format, data is already a table name in the database
            source_table_name = data
            
            # Get column information from the table using sqlglot
            source_table = exp.Table(this=exp.Identifier(this=source_table_name))
            limit_query_expr = exp.Select(
                expressions=[exp.Star()]
            ).from_(source_table).limit(0)
            limit_query = limit_query_expr.sql(dialect=self.sql_dialect)
            
            # Execute the query and get column names
            logger.debug(f"Columns in table {source_table_name}: {self.con.execute(limit_query).description}")
            all_columns = [col[0] for col in self.con.execute(limit_query).description]
            
            logger.debug(f"Using table {source_table_name} as source")
            
        elif data_format == 'file':
            # File path - use direct SQL with proper escaping
            file_path = str(data)  # Ensure it's a string
            file_ext = file_path.lower()
            
            if file_ext.endswith('.csv'):
                # For CSV files, use read_csv_auto directly
                csv_reader = exp.Func(this="read_csv_auto", expressions=[exp.Literal(this=file_path, is_string=True)])
                limit_query_expr = exp.Select(
                    expressions=[exp.Star()]
                ).from_(csv_reader).limit(0)
                limit_query = limit_query_expr.sql(dialect=self.sql_dialect)
                
                # Execute the query and get column names
                all_columns = [col[0] for col in self.con.execute(limit_query).description]
                
                # Set the source table name for the SQL query
                source_table_name = f"read_csv_auto('{file_path}')"
                
                logger.debug(f"Using CSV file {file_path} as source")
                
            elif file_ext.endswith('.parquet'):
                # For Parquet files, use read_parquet directly
                parquet_reader = exp.Func(this="read_parquet", expressions=[exp.Literal(this=file_path, is_string=True)])
                limit_query_expr = exp.Select(
                    expressions=[exp.Star()]
                ).from_(parquet_reader).limit(0)
                limit_query = limit_query_expr.sql(dialect=self.sql_dialect)
                
                # Execute the query and get column names
                all_columns = [col[0] for col in self.con.execute (limit_query).description]
                
                # Set the source table name for the SQL query
                source_table_name = f"read_parquet('{file_path}')"
                
                logger.debug(f"Using Parquet file {file_path} as source")
            else:
                raise ValueError(f"Unsupported file extension: {file_ext}. Must be .csv or .parquet.")
        else:
            raise ValueError(f"Unsupported data_format in _get_column_info: {data_format}. Must be 'table' or 'file'.")
            
        return all_columns, source_table_name

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        if hasattr(self, 'con') and self.con:
            try:
                self.con.close()
                logger.debug("Database connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
        return False
    
    def close(self):
        """Explicitly close the connection if not using with statement"""
        if hasattr(self, 'con') and self.con:
            self.con.close()
            logger.debug("Database connection closed successfully") 
             

class PyIceberg(DbEngine):
    """Class for working with AWS Glue/Iceberg tables; 
    utilizes pyiceberg for catalog management"""
    
    # Set the SQL dialect for AWS Iceberg
    sql_dialect = "athena"
    
    def __init__(self, catalog_name: str, bucket: str, account: str, key: Optional[str] = None, catalog_type: CatalogType = CatalogType.AWS_REST,file_reader: Optional[FileReader] = None, credentials: Optional[Dict[str, str]] = None, region: Optional[str] = 'us-east-1', **kwargs):
        super().__init__(file_reader=file_reader, **kwargs)
        self.catalog_name = catalog_name
        self.bucket = bucket
        self.key = key
        self.region = region
        self.catalog_type = catalog_type
        self.account = account
        self.configure(credentials=credentials, **kwargs)
        
    def __enter__(self):
        """Enter context manager"""
        return self
        
    def configure(self, credentials: Optional[Dict[str, str]] = None, **kwargs):
        """Configure the Iceberg connection"""
        if credentials:
            """ """
            os.environ["AWS_ACCESS_KEY_ID"] = credentials["AccessKeyId"]
            os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["SecretAccessKey"]
            os.environ["AWS_SESSION_TOKEN"] = credentials["SessionToken"]
            os.environ["AWS_DEFAULT_REGION"] = self.region

        if self.catalog_type == CatalogType.AWS_REST:
            warehouse = f"arn:aws:s3tables:{self.region}:{self.account}:bucket/{self.bucket}"
            self.catalog = load_catalog(
                name=self.catalog_name,
                **{
                "type": "rest", 
                "warehouse": warehouse, 
                "rest.sigv4-enabled": "true",
                "rest.signing-name": "s3tables",
                "rest.signing-region": self.region
                }
            )
        elif self.catalog_type == CatalogType.AWS_GLUE:
            location = f"s3://{self.bucket}"
            if self.key:
                location += f'/{self.key}'

            self.catalog = load_catalog(
                name=self.catalog_name,
                location=location
                **kwargs
            ) # assumes config will be in env variables
    
        
    def format_profile_output(self, results):
        """Format profile output for a query
        
        Args:
            results: Query results to format
            
        Returns:
            list: Formatted profile output lines
        """
        # No profiling support for PyIceberg
        return []
         
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
     
    def close(self):
        """Close the connection"""
        # Any cleanup needed
        pass
        

    def _build_partition_spec(self, schema: TableSchema) -> Optional[PartitionSpec]:
        """Build a partition spec based on the schema's partition_by field"""
        if not schema.partition_by:
            return PartitionSpec()
            
        partition_fields = []
        for field_name in schema.partition_by:
            # Find the field in the schema
            field_idx = None
            field_type = None
            for i, field in enumerate(schema.fields):
                if field.name == field_name:
                    field_idx = i + 1  # PyIceberg field IDs start at 1
                    field_type = field.type.lower()
                    break
            
            if field_idx is None:
                raise ValueError(f"Partition field {field_name} not found in schema")
            
            # Determine the appropriate transform based on field type
            if field_type == 'date':
                transform = transforms.day()
                name = f"{field_name}_day"
            elif field_type in ('timestamp', 'timestamptz'):
                transform = transforms.month()
                name = f"{field_name}_month"
            else:
                transform = transforms.identity()
                name = field_name
                
            partition_fields.append(
                PartitionField(source_id=field_idx, transform=transform, name=name)
            )
            
        return PartitionSpec(*partition_fields) if partition_fields else PartitionSpec()
    
        
    def _check_table_exists(self, namespace: str, table_name: str) -> Optional[Table]:
        """Check if a table exists in the catalog
        
        Args:
            namespace: AWS Glue database name
            table_name: Table name
            
        Returns:
            Optional[Table]: The existing table if found, None otherwise
        """
        try:
            table = self.catalog.load_table((namespace, table_name))
            logger.debug(f"Table {namespace}.{table_name} already exists")
            return table
        except Exception:
            return None
            
    def _prepare_table_properties(self, schema: TableSchema, properties: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Prepare table properties for Iceberg table creation
        
        Args:
            schema: Schema with potential comment
            properties: Additional properties provided by user
            
        Returns:
            Dict[str, str]: Combined properties for table creation
        """
        props = properties or {}
        
        # Add schema comment if available
        if schema.comment:
            props["comment"] = schema.comment
            
        # Default to format version 2 for better performance
        if "format-version" not in props:
            props["format-version"] = "2"
            
        # Add primary key information if available
        primary_key_fields = [field.name for field in schema.fields if getattr(field, 'primary_key', False)]
        if primary_key_fields:
            props["primary-key"] = ",".join(primary_key_fields)
            
        return props
    
    def create_table(self, *, table_name: str, fields: List[SchemaField], namespace: str = "default", 
                     location: Optional[str] = None,
                     properties: Optional[Dict[str, str]] = None,
                     custom_field_mapping: Optional[Dict[str, Any]] = None,
                     **kwargs) -> Table:
        """Create an Iceberg table using the provided schema definition.
        
        Args:
            table_name: Name of the table to create
            fields: List of SchemaField objects defining the table structure
            namespace: AWS Glue database name (Iceberg namespace)
            properties: Additional properties for the table
            custom_field_mapping: Optional mapping to customize field properties
            **kwargs: Additional arguments
            
        Returns:
            Table: PyIceberg Table object for the created table
        """
        # Ensure we have a valid table name
        actual_table_name = table_name
        full_name = f"{namespace}.{actual_table_name}"
         
        # Create a TableSchema from the fields
        table_schema = TableSchema(name=table_name, fields=fields)
        
        # Convert to Iceberg schema
        iceberg_schema = table_schema.to_pyiceberg_schema(custom_field_mapping=custom_field_mapping)
        logger.debug(iceberg_schema)
        
        # Build the partition spec
        partition_spec = self._build_partition_spec(table_schema)
        
        # Prepare table properties
        props = self._prepare_table_properties(table_schema, properties)

        logger.debug(f"Creating table {full_name}, partition spec: {partition_spec}, properties: {props}") 

        # Create the table
        self.catalog.create_table(
            identifier=full_name,
            schema=iceberg_schema,
            partition_spec=partition_spec,
            properties=props,
            location=location
        )
        
        logger.debug(f"Created table {full_name}")
        return True
        
    
    def verify(self, table_name: str, namespace: str = "default"):
        """Verify a table exists and has expected schema"""
        try:
            table = self.catalog.load_table((namespace, table_name))
            table.schema()
            return True
        except Exception as e:
            logger.error(f"Error verifying table {namespace}.{table_name}: {e}")
            return False
            
    def _get_column_info(self, data, data_format: DataFormat) -> Tuple[List[str], str]:
        """Helper method to get column information from various data sources.
        
        Args:
            data: The data source (table name, file path, etc.)
            data_format: Format of the data ('dict', 'table', or 'file')
            
        Returns:
            Tuple containing:
                - List of column names
                - Source table name or expression for SQL
        """
        all_columns = None
        source_table_name = None
        
        if data_format == DataFormat.TABLE:
            # For table format, data is already a table name in the database
            source_table_name = data
            
            # Get column information from the table
            try:
                # Try to load the table from the catalog
                namespace, table = self._parse_table_name(source_table_name)
                iceberg_table = self.catalog.load_table((namespace, table))
                
                # Extract column names from the schema
                all_columns = [field.name for field in iceberg_table.schema().fields]
            except Exception as e:
                logger.error(f"Error getting column info from table {source_table_name}: {e}")
                raise
                
        elif data_format == DataFormat.FILE:
            # For file format, data is a file path
            file_path = data
            file_type = self._get_file_type(file_path)
            
            if file_type == 'csv':
                # For CSV files, read the header to get column names
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path, nrows=0)
                    all_columns = df.columns.tolist()
                    source_table_name = f"read_csv('{file_path}')"
                except Exception as e:
                    logger.error(f"Error reading CSV file {file_path}: {e}")
                    raise
            elif file_type == 'parquet':
                # For Parquet files, use PyArrow to read the schema
                try:
                    import pyarrow.parquet as pq
                    parquet_schema = pq.read_schema(file_path)
                    all_columns = parquet_schema.names
                    source_table_name = f"read_parquet('{file_path}')"
                except Exception as e:
                    logger.error(f"Error reading Parquet file {file_path}: {e}")
                    raise
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        return all_columns, source_table_name
    
    def _parse_table_name(self, table_name: str) -> Tuple[str, str]:
        """Parse a table name into namespace and table components
        
        Args:
            table_name: Table name, possibly including namespace (e.g., 'namespace.table')
            
        Returns:
            Tuple[str, str]: (namespace, table)
        """
        parts = table_name.split('.')
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            return "default", table_name
            
    def _get_file_type(self, file_path: str) -> str:
        """Determine the file type from the file path
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: File type ('csv', 'parquet', etc.)
        """
        if file_path.endswith('.csv'):
            return 'csv'
        elif file_path.endswith('.parquet'):
            return 'parquet'
        else:
            # Try to infer from content or raise error
            raise ValueError(f"Unable to determine file type for {file_path}")
    
    def insert_data(self, table_name: str, schema_name: str, data: Union[List[Dict], Dict, str, pa.Table, Any], 
                   data_format: DataFormat, 
                   operation: SqlAction = SqlAction.INSERT, key_columns: Optional[List[str]] = None, **kwargs) -> bool:
        """Insert data into an Iceberg table; currently only supports PyArrow
        
        Args:
            table_name: Name of the target table
            schema_name: Schema/namespace name
            data: Data to insert (can be a list of dictionaries, a PyArrow table, or a file path)
            data_format: Type of data (DataFormat)
            operation: Operation type ('insert', 'update', 'merge')
            key_columns: List of key columns for update/merge operations
            **kwargs: Additional arguments
            
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        try:
            # Load the target table
            target_table = self.catalog.load_table((schema_name, table_name))
              
            # Perform the operation
            if operation == SqlAction.INSERT:
                # Simple append operation
                if data_format == DataFormat.PYARROW:
                    target_table.append(data)
                logger.debug(f"Successfully inserted {len(data)} rows into {schema_name}.{table_name}")
            elif operation in [SqlAction.UPDATE, SqlAction.MERGE]:
                if not key_columns:
                    raise ValueError("Key columns must be provided for update/merge operations")
                    
                if data_format == DataFormat.PYARROW:
                    target_table.upsert(data)
                logger.debug(f"Successfully merged data into {schema_name}.{table_name}")
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            return True
        except Exception as e:
            logger.error(f"Error inserting data into {schema_name}.{table_name}: {e}")
            raise e


class ModuleFileHandlerFilter(logging.Filter):
    """
    A filter that adds file handlers to loggers when they're first used.
    Each module gets its own log file based on its __name__.
    """
    def __init__(self, log_dir: Path, formatter: logging.Formatter):
        super().__init__()
        self.log_dir = log_dir
        self.formatter = formatter
        self.configured_loggers: Set[str] = set()
        
    def filter(self, record):
        logger_name = record.name
        
        if logger_name == "root" or logger_name in self.configured_loggers:
            return True
            
        logger = logging.getLogger(logger_name)
        
        has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        
        if not has_file_handler:
            # Create a log file path based on the module name
            log_filename = f"{logger_name.replace('.', '_')}.log"
            log_file_path = self.log_dir / log_filename
            
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(self.formatter)
            logger.addHandler(file_handler)
            
            self.configured_loggers.add(logger_name)
        
        return True