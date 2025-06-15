"""
SQL Generation Module

This module centralizes all SQL generation logic for creating tables and schemas
across different SQL dialects. It leverages sqlglot for safe SQL generation and
transpilation between dialects.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import re
import sqlglot
import sqlglot.expressions as exp
from dataclasses import asdict
from enum import Enum

from irs.models import SchemaField, TableOptions, DUCKDB_TO_ICEBERG_MAPPING

# Enum for SQL actions
class SqlAction(Enum):
    ALTER_TABLE_ADD_COLUMN = "alter_table_add_column"
    CREATE_TABLE = "create_table"
    DELETE = "delete"
    DROP_TABLE = "drop_table"
    INSERT = "insert"
    MERGE = "merge"
    SELECT = "select"
    TABLE_IDENTIFIER = "table_identifier"
    UPDATE = "update"

# Enum for data formats
class DataFormat(Enum):
    VALUES = "values"
    TABLE = "table"
    FILE = "file"
    DICT = "dict"
    PYARROW = "pyarrow"
    CSV = "csv"
    PARQUET = "parquet"


logger = logging.getLogger(__name__)

class SqlGenerator:
    """SQL Generator class for safely building SQL statements across different dialects."""

    def __init__(self, sql_dialect="duckdb"):
        """Initialize the SQL Generator

        Args:
            sql_dialect: SQL dialect to use for generation
        """
        self.sql_dialect = sql_dialect

    def build_table_identifier(
        self,
        table_name: str,
        schema_name: Optional[str] = None,
        dialect: str = "duckdb",
    ) -> str:
        """Build a properly quoted table identifier for the specified dialect

        Args:
            table_name: Name of the table
            schema_name: Optional schema/database name
            dialect: SQL dialect to use for quoting

        Returns:
            str: Properly quoted table identifier
        """
        table_expr = exp.Table(this=exp.Identifier(this=table_name, quote=True))

        if schema_name:
            table_expr = exp.Table(
                this=exp.Identifier(this=table_name, quote=True),
                db=exp.Identifier(this=schema_name, quote=True),
            )
            logger.debug(f"Table identifier built: {table_expr.sql(dialect=dialect)}")
        return table_expr.sql(dialect=dialect)

    def build_sql(
        self,
        sql_action: SqlAction,
        table_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        columns: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        data_format: DataFormat = DataFormat.VALUES,
        source_table_name: Optional[str] = None,
        data_type: Optional[str] = None,
        default_value: Optional[Any] = None,
        file_path: Optional[str] = None,
        if_not_exists: bool = True,
        where: Optional[Any] = None,
        limit: Optional[int] = None,
        set_values: Optional[list[str]] = None,
        key_columns: Optional[list[str]] = None,
        all_columns: Optional[list[str]] = None,
    ) -> str:
        """Generic SQL builder that can construct various types of SQL statements safely.

        Args:
            sql_action: Type of SQL statement to build ('table_identifier', 'alter_table_add_column', etc.)
            **kwargs: Arguments specific to the SQL action being built

        Returns:
            str: Safe SQL statement or expression object when to_sql=False
        """

        if sql_action == SqlAction.TABLE_IDENTIFIER:
            # Use the class method instead of the standalone function
            expr = self.build_table_identifier(table_name, schema_name)
            logger.info(f"Table identifier built: {expr.sql(dialect=self.sql_dialect)}")
            return expr.sql(dialect=self.sql_dialect) if kwargs.get("to_sql", True) else expr

        elif sql_action == SqlAction.ALTER_TABLE_ADD_COLUMN:
            return self._build_alter_table_add_column(
                table_name,
                columns[0],
                data_type,
                default_value,
                schema_name,
                if_not_exists,
            )

        elif sql_action == SqlAction.SELECT:
            # Build a SELECT statement
            return self._build_select(columns, table_name, schema_name, where, limit)

        elif sql_action == SqlAction.INSERT:
            # Get source type and related parameters

            if data_format == DataFormat.VALUES:
                return self._build_insert(
                    table_name, schema_name, columns
                )

            elif data_format in [DataFormat.TABLE, DataFormat.PYARROW]:
                logger.debug(f"Inserting from table: {source_table_name}")
                target_table_expr = exp.Table(this=exp.Identifier(this=table_name, quote=True))
                if schema_name:
                    target_table_expr = exp.Table(
                        this=exp.Identifier(this=table_name, quote=True),
                        db=exp.Identifier(this=schema_name, quote=True),
                    )
                
                source_table_expr = exp.Table(this=exp.Identifier(this=source_table_name, quote=True))
                select_expr = exp.Select(expressions=[exp.Star()]).from_(source_table_expr)
                logger.debug(f"Generated SELECT: {select_expr.sql(dialect=self.sql_dialect)}")
                insert_expr = exp.Insert(this=target_table_expr, expression=select_expr)
                logger.debug(f"Generated INSERT: {insert_expr.sql(dialect=self.sql_dialect)}")
                return insert_expr.sql(dialect=self.sql_dialect)

            elif data_format == DataFormat.FILE:
                # Insert from a file
                file_ext = str(source_table_name).lower()

                logger.debug(f"Inserting from file: {file_path}")

                # Use the class method instead of the standalone function
                target_table = self.build_table_identifier(
                    table_name, schema_name
                )

                # Dialect-specific file handling
                if self.sql_dialect == "duckdb":
                    if file_ext.endswith(".csv"):
                        source_table_str = f"read_csv_auto('{source_table_name}')"
                    elif file_ext.endswith(".parquet"):
                        source_table_str = f"read_parquet('{source_table_name}')"
                    else:
                        raise ValueError(
                            f"Unsupported file extension for {self.sql_dialect}: {file_ext}. Must be .csv or .parquet."
                        )

                    select_expr = exp.Select(expressions=[exp.Star()]).from_(source_table_str)
                    insert_expr = exp.Insert(this=target_table, expression=select_expr)

                    # Generate the SQL
                    logger.debug(f"Generated INSERT from file: {insert_expr.sql(dialect=self.sql_dialect)}")
                    return insert_expr.sql(dialect=self.sql_dialect)
                else:
                    # For other dialects, use a more generic approach or dialect-specific handling
                    raise ValueError(
                        f"File-based insert not yet implemented for dialect: {self.sql_dialect}"
                    )

        elif sql_action == SqlAction.UPDATE:
        
            return self._build_update(
                table_name=table_name,
                set_values=set_values,
                schema_name=schema_name,
                where=where,
                source_table_name=source_table_name,
                key_columns=key_columns
            )

        elif sql_action == SqlAction.DELETE:
            return self._build_delete(table_name=table_name, schema_name=schema_name, where=where)

        elif sql_action == SqlAction.CREATE_TABLE:

            return self.build_create_table_sql(
                name=table_name, schema_name=schema_name, fields=columns, constraints=constraints
            )

        elif sql_action == SqlAction.DROP_TABLE:

            return self._build_drop_table(table_name, schema_name, if_exists)

        elif sql_action == SqlAction.MERGE:
            return self._build_merge(
                table_name=table_name,
                schema_name=schema_name,
                source_table_name=source_table_name,
                key_columns=key_columns,
                all_columns=all_columns,
            )

    def convert_duckdb_to_iceberg_type(self, duckdb_type: str) -> str:
        """Convert a DuckDB type to an Iceberg type

        Args:
            duckdb_type: DuckDB data type

        Returns:
            str: Corresponding Iceberg type
        """
        # Handle DECIMAL type with precision and scale
        if duckdb_type.startswith("DECIMAL"):
            # Extract precision and scale if provided
            if "(" in duckdb_type and ")" in duckdb_type:
                params = duckdb_type[duckdb_type.find("(") + 1 : duckdb_type.find(")")]
                if "," in params:
                    precision, scale = params.split(",")
                    return f"decimal({precision.strip()},{scale.strip()})"
                else:
                    precision = params.strip()
                    return f"decimal({precision})"
            return "decimal(38,6)"  # Default precision and scale

        # Handle VARCHAR with length
        if duckdb_type.startswith("VARCHAR"):
            return "string"

        # Use the mapping for other types
        base_type = duckdb_type.split("(")[0].upper()
        return DUCKDB_TO_ICEBERG_MAPPING.get(base_type, "string")

    def build_create_table_sql(
        self,
        *,
        name: str,
        schema_name: Optional[str] = None,
        fields: List[SchemaField],
        custom_field_mapping: Optional[Dict[str, Any]] = None,
        table_options: Optional[TableOptions] = None,
        temporary: bool = False,
    ) -> Union[str, List[str]]:
        """Generate CREATE SQL for the specified dialect using sqlglot expression API

        Args:
            name: Table name
            schema_name: Optional schema/database name
            fields: List of field definitions with properties (name, data_type, etc.)
            custom_field_mapping: Optional mapping to customize field properties
            table_options: Additional options for the table creation including location, partitioning, and Iceberg properties
            temporary: If True, creates a TEMPORARY table (if supported by the dialect)

        Returns:
            Union[str, List[str]]: TABLE schema for the specified dialect. For DuckDB with comments,
                                returns a list of statements [create_table, comment_on_table]
        """
        # Normalize dialect name
        sql_dialect = self.sql_dialect.lower()

        # Special handling for PyIceberg - use custom function
        if self.sql_dialect == "pyiceberg":
            return self._generate_pyiceberg_sql(
                name=name,
                fields=fields,
                custom_field_mapping=custom_field_mapping,
                comment=comment,
            )

        # For all other dialects, use sqlglot's expression API
        # Build column definitions using sqlglot expressions
        columns = []
        primary_key_fields = [field.name for field in fields if field.primary_key]

        for field in fields:
            data_type = field.data_type
            # Apply custom field mapping if provided
            if custom_field_mapping and field.name in custom_field_mapping:
                custom_mapping = custom_field_mapping[field.name]
                if "data_type" in custom_mapping:
                    data_type = custom_mapping["data_type"]

            # Create constraints list
            column_constraints = []

            if field.constraints:
                for constraint_str in field.constraints:
                    # Skip PRIMARY KEY constraints for Athena
                    if (
                        self.sql_dialect == "athena"
                        and constraint_str.upper() == "PRIMARY KEY"
                    ):
                        logger.debug(
                            f"Skipping PRIMARY KEY constraint for {field.name} in Athena dialect"
                        )
                        continue
                    # Skip REFERENCES constraints for Athena
                    if (
                        self.sql_dialect == "athena"
                        and constraint_str.upper().startswith("REFERENCES")
                    ):
                        logger.debug(
                            f"Skipping REFERENCES constraint for {field.name} in Athena dialect"
                        )
                        continue

                    constraint = self._map_constraint_to_sqlglot(constraint_str)
                    column_constraints.append(constraint)

            # Add column comment as a ColumnConstraint if supported
            if field.comment and sql_dialect in ["mysql"]:
                column_constraints.append(
                    sqlglot.exp.ColumnConstraint(
                        this=sqlglot.exp.Identifier(this="COMMENT"),
                        expression=sqlglot.exp.Literal.string(field.comment),
                    )
                )

            # Add foreign key reference as a ColumnConstraint if supported
            if field.foreign_key and self.sql_dialect not in [
                "duckdb",
                "pyiceberg",
                "athena",
            ]:
                if "." in field.foreign_key:
                    ref_table, ref_column = field.foreign_key.split(".")
                    column_constraints.append(
                        sqlglot.exp.ColumnConstraint(
                            this=sqlglot.exp.Identifier(this="REFERENCES"),
                            expression=sqlglot.exp.Identifier(this=ref_table),
                        )
                    )

            column = sqlglot.exp.ColumnDef(
                this=sqlglot.exp.Identifier(this=field.name, quoted=True),
                kind=sqlglot.exp.DataType(this=data_type),
                constraints=column_constraints,
            )
            columns.append(column)

        # Build table-level constraints
        constraints = []

        # Add PRIMARY KEY constraint if we have primary key fields
        # Athena doesn't support PRIMARY KEY constraints
        if primary_key_fields and self.sql_dialect != "athena":
            # For DuckDB, use PRIMARY KEY (col1, col2) syntax
            # For other dialects, use CONSTRAINT PRIMARY KEY (col1, col2) syntax
            pk_columns = [
                sqlglot.exp.Identifier(this=field, quoted=True)
                for field in primary_key_fields
            ]

            if self.sql_dialect == "duckdb":
                # DuckDB uses PRIMARY KEY (col1, col2) syntax
                constraints.append(sqlglot.exp.PrimaryKey(expressions=pk_columns))
            else:
                # Other dialects use CONSTRAINT PRIMARY KEY (col1, col2) syntax
                constraints.append(
                    sqlglot.exp.Constraint(
                        this=sqlglot.exp.Identifier(this="PRIMARY KEY"),
                        expressions=pk_columns,
                    )
                )

        # Add FOREIGN KEY constraints for DuckDB
        # Athena doesn't support FOREIGN KEY constraints
        if self.sql_dialect == "duckdb":
            for field in fields:
                if field.foreign_key:
                    if "." in field.foreign_key:
                        ref_table, ref_column = field.foreign_key.split(".")
                        constraints.append(
                            sqlglot.exp.ForeignKey(
                                expressions=[
                                    sqlglot.exp.Identifier(this=field.name, quoted=True)
                                ],
                                reference=sqlglot.exp.Reference(
                                    this=sqlglot.exp.Identifier(
                                        this=ref_table, quoted=True
                                    ),
                                    expressions=[
                                        sqlglot.exp.Identifier(
                                            this=ref_column, quoted=True
                                        )
                                    ],
                                ),
                            )
                        )
        # Compose table expressions: columns + constraints + properties (if any)
        table_expressions = columns + constraints

        properties_list = []
        # Store Athena options separately for post-processing

        if table_options:
            if self.sql_dialect == "athena":
                # Get Athena options as a tuple: (properties_list, special_clauses_dict)
                properties_from_athena, special_clauses_athena = (
                    self._table_options_athena(table_options)
                )
                properties_list.extend(properties_from_athena)

        if properties_list:
            table_expressions.append(
                sqlglot.exp.Properties(expressions=properties_list)
            )

        schema = sqlglot.exp.Schema(expressions=table_expressions)
        # Use schema-qualified table name if schema_name is provided
        if schema_name:
            table_expr = sqlglot.exp.Table(
                this=sqlglot.exp.Identifier(this=name, quoted=True),
                schema=sqlglot.exp.Identifier(this="main", quoted=True),
                db=sqlglot.exp.Identifier(this=schema_name, quoted=True),
                columns=columns,
            )
        else:
            table_expr = sqlglot.exp.Table(
                this=sqlglot.exp.Identifier(this=name, quoted=True),
                db=sqlglot.exp.Identifier(this="main", quoted=True),
                schema=sqlglot.exp.Identifier(this="main", quoted=True),
                columns=columns,
            )
        # Create the CREATE TABLE statement with the schema
        create_expr = sqlglot.exp.Create(
            this=table_expr,
            kind="table",
            exists=False,
            temporary=temporary,
            expression=schema
        )

        # Generate the SQL for the CREATE TABLE statement
        create_sql = create_expr.sql(dialect=self.sql_dialect)
        
        # Remove the AS clause using regex
        create_sql = re.sub(r'\s+AS\s+\(', ' (', create_sql)

        # If temporary, ensure CREATE TEMPORARY TABLE is present (for dialects that support it)
        if temporary and not create_sql.upper().startswith("CREATE TEMPORARY TABLE"):
            create_sql = re.sub(
                r"(?i)^CREATE\s+TABLE", "CREATE TEMPORARY TABLE", create_sql, count=1
            )

        create_sql = self.uppercase_keywords(create_sql)

        # For Athena, append special clauses that aren't supported by sqlglot's expression API
        if self.sql_dialect == "athena" and table_options:
            # Use special_clauses_athena from above
            special_clauses = special_clauses_athena
            # Append PARTITIONED BY clause if present
            if special_clauses.get("partitioned_by"):
                partition_cols = special_clauses["partitioned_by"]
                if isinstance(partition_cols, (list, tuple)):
                    partition_cols_str = ", ".join(str(col) for col in partition_cols)
                else:
                    partition_cols_str = str(partition_cols)
                create_sql += f"\nPARTITIONED BY ({partition_cols_str})"

            # Append ROW FORMAT clause if present
            if special_clauses.get("row_format"):
                create_sql += f"\nROW FORMAT {special_clauses['row_format']}"

            # Append STORED AS clause if present
            if special_clauses.get("stored_as"):
                create_sql += f"\nSTORED AS {special_clauses['stored_as']}"

            # Append LOCATION clause if present
            if special_clauses.get("location"):
                create_sql += f"\nLOCATION '{special_clauses['location']}'"

            if special_clauses.get("comment"):
                create_sql += f"\nCOMMENT '{special_clauses['comment']}'"
            # Append TBLPROPERTIES clause if present
            if special_clauses.get("tblproperties"):
                tbl_props = special_clauses["tblproperties"]
                # If tblproperties is a dictionary, format it as key-value pairs
                if isinstance(tbl_props, dict):
                    prop_pairs = [f"'{k}' = '{v}'" for k, v in tbl_props.items()]
                    create_sql += f"\nTBLPROPERTIES ({', '.join(prop_pairs)})"
                else:
                    # If it's already a formatted string, use it directly
                    create_sql += f"\nTBLPROPERTIES {tbl_props}"

        # DuckDB: handle table options and comments as extra statements
        if self.sql_dialect == "duckdb" and table_options:
            extra_statements = self._table_options_duckdb(
                name,
                table_options,
            )
            if extra_statements:
                create_sql = create_sql + "; " + "; ".join(extra_statements)
        logger.debug("CREATE SQL: %s", create_sql)

        return create_sql

    def _map_constraint_to_sqlglot(self, constraint_str: str) -> sqlglot.Expression:
        """
        Map a string constraint to a sqlglot constraint expression.

        Args:
            constraint_str: String representation of the constraint

        Returns:
            sqlglot.exp.Expression: The corresponding sqlglot constraint expression
        """
        # Map string constraints to sqlglot constraint objects
        if constraint_str == "NOT NULL":
            return sqlglot.exp.NotNullColumnConstraint()
        elif constraint_str.startswith("DEFAULT "):
            # Extract default value
            default_value = constraint_str[8:].strip()
            return sqlglot.exp.DefaultColumnConstraint(
                this=sqlglot.parse_one(default_value)
            )
        elif constraint_str == "UNIQUE":
            return sqlglot.exp.UniqueColumnConstraint()
        elif constraint_str.startswith("CHECK "):
            # Extract check condition
            check_condition = constraint_str[6:].strip()
            return sqlglot.exp.CheckColumnConstraint(
                this=sqlglot.parse_one(check_condition)
            )
        else:
            # For other constraints, use a generic approach
            return sqlglot.exp.ColumnConstraint(
                this=sqlglot.exp.Identifier(this=constraint_str)
            )

    def _generate_pyiceberg_sql(
        self,
        name: str,
        fields: List[SchemaField],
        custom_field_mapping: Optional[Dict[str, Any]] = None,
        comment: Optional[str] = None,
    ) -> str:
        """Generate PyIceberg-specific SQL with proper handling of constraints and options"""
        # Use sqlglot to build the CREATE TABLE statement
        columns = []
        primary_key_fields = [field.name for field in fields if field.primary_key]

        for field in fields:
            data_type = field.data_type

            # Apply custom field mapping if provided
            if custom_field_mapping and field.name in custom_field_mapping:
                custom_mapping = custom_field_mapping[field.name]
                if "data_type" in custom_mapping:
                    data_type = custom_mapping["data_type"]

            # Track primary key fields
            if field.primary_key and field.name not in primary_key_fields:
                primary_key_fields.append(field.name)

            # Parse the data type
            try:
                data_type_expr = sqlglot.parse_one(data_type)
            except:
                data_type_expr = sqlglot.exp.DataType.build(this=data_type)

            # Build column definition with sqlglot
            column = sqlglot.exp.ColumnDef(
                this=sqlglot.exp.Identifier(
                    this=field.name, quoted=True
                ),  # Use backtick quoting
                kind=data_type_expr,
                constraints=[],
            )

            # Add NOT NULL constraint by default unless explicitly specified in constraints
            if not field.constraints or "NOT NULL" not in field.constraints:
                column.constraints.append(
                    sqlglot.exp.ColumnConstraint(
                        this=sqlglot.exp.Identifier(this="NOT NULL")
                    )
                )

            columns.append(column)

        # Build table-level constraints
        constraints = []

        # Add PRIMARY KEY constraint if we have primary key fields
        if primary_key_fields:
            pk_columns = [
                sqlglot.exp.Identifier(this=field, quoted=True)
                for field in primary_key_fields
            ]
            # PyIceberg uses NOT ENFORCED for primary keys
            pk_constraint = sqlglot.exp.Constraint(
                this=sqlglot.exp.Identifier(this="PRIMARY KEY"), expressions=pk_columns
            )
            # Add NOT ENFORCED as a property
            pk_constraint.properties = [
                sqlglot.exp.Property(this=sqlglot.exp.Identifier(this="NOT ENFORCED"))
            ]
            constraints.append(pk_constraint)

        # Create the CREATE TABLE expression
        create_expr = sqlglot.exp.Create(
            this=sqlglot.exp.Table(this=sqlglot.exp.Identifier(this=name, quoted=True)),
            kind="TABLE",
            expressions=columns,
            constraints=constraints,
        )

        # Add table comment if provided
        if comment:
            create_expr.properties.append(
                sqlglot.exp.Property(
                    this=sqlglot.exp.Identifier(this="COMMENT"),
                    value=sqlglot.exp.Literal.string(comment),
                )
            )

        # Generate SQL with proper PyIceberg syntax
        # Use 'spark' dialect which is closest to PyIceberg
        create_sql = create_expr.sql(dialect="spark")

        # Append WITH (format = 'parquet') for PyIceberg compatibility
        create_sql += " WITH (format = 'parquet')"

        # Return the complete CREATE TABLE statement for PyIceberg
        return create_sql

    def _build_insert(
        self,
        table_name: str,
        schema_name: Optional[str] = None,
        columns: List[str] = None,
        select: Optional[str] = None,
    ) -> str:
        """Build an INSERT statement using sqlglot for safety.

        Args:
            table_name: Name of the table to insert into
            schema_name: Optional schema/database name
            columns: List of column names for the insert
            values: List of values to insert
            select: Optional SELECT statement to use as the source

        Returns:
            str: Safe SQL INSERT statement
        """
        # Build the table identifier
        table_expr = self.build_table_identifier(table_name, schema_name)

        # Build the INSERT statement
        insert_expr = exp.Insert(this=table_expr)

        # Add column names if provided
        column_exprs = None
        if columns:
            column_exprs = [exp.Identifier(this=col, quoted=True) for col in columns]

        # Add values or select statement
        if select:
            # Parse the select statement and add it to the insert
            select_expr = sqlglot.parse_one(select, dialect=self.sql_dialect)
            # Create a new Insert with columns and expression
            insert_expr = exp.Insert(
                this=table_expr,
                columns=column_exprs,
                expression=select_expr
            )
            # Log the SQL for debugging
            logger.debug(f"Generated INSERT: {insert_expr.sql(dialect=self.sql_dialect)}")
        else:
            placeholders = ["?" for _ in columns]
            table_sql = self.build_table_identifier(table_name, schema_name)
            sql = f"INSERT INTO {table_sql}  VALUES ({', '.join(placeholders)})"
            logger.debug(f"Generated parameterized INSERT: {sql}")
            return sql 

        return insert_expr.sql(dialect=self.sql_dialect)

    def _build_update(
        self,
        table_name: str,
        set_values: list[Dict[str, Any]],
        schema_name: str,
        where: Optional[Dict] = None,
        source_table_name: Optional[str] = None,
        key_columns: Optional[List[str]] = None,
    ) -> str:
        """Build an UPDATE statement using sqlglot for safety.

        Args:
            table_name: Name of the table to update
            set_values: List of dictionaries with column-value pairs for SET clause
            schema_name: Optional schema/database name
            where: Optional dictionary of column-value pairs for WHERE clause
            source_table_name: Optional source table name for DuckDB's FROM clause
            key_columns: Optional list of column names to use as join keys

        Returns:
            str: Safe SQL UPDATE statement
        """
        target_table = self.build_table_identifier(table_name, schema_name)

        # If we have a source table and key columns, use DuckDB's UPDATE...FROM syntax
        if source_table_name and key_columns:
            # This is a batch update from another table
            # Use DuckDB's UPDATE...FROM syntax which is more efficient
            
            # For the source table, check if it's a function call like read_parquet()
            # If so, don't quote it as it's not a simple table name
            if source_table_name and ('(' in source_table_name and ')' in source_table_name):
                source = source_table_name
            else:
                source = f'"{source_table_name}"'
            
            # Build the SET clause
            set_clauses = []
            # Get all columns from the first dictionary in set_values
            # If key_columns is provided, exclude them as they're used for joining
            # Otherwise, update all columns from the source table
            all_columns = list(set_values[0].keys())
            update_columns = [col for col in all_columns if not key_columns or col not in key_columns]
            
            for col in update_columns:
                # Use the source table's column value
                set_clauses.append(f'"{col}" = {source}."{col}"')
            
            set_clause = ", ".join(set_clauses)
            
            # Build the WHERE clause for joining tables
            join_conditions = []
            
            # If key_columns is provided, use them for joining
            # Otherwise, we'll do a cross join (all rows) and rely on additional WHERE conditions
            if key_columns:
                for key in key_columns:
                    join_conditions.append(f'{target_table}."{key}" = {source}."{key}"')
                
                join_clause = " AND ".join(join_conditions)
            else:
                # If no key columns, use a condition that's always true for a cross join
                join_clause = "1=1"
            
            # Add additional WHERE conditions if provided
            additional_where = ""
            if where:
                where_conditions = []
                for col, val in where.items():
                    if val is None:
                        where_conditions.append(f'{target_table}."{col}" IS NULL')
                    elif isinstance(val, (int, float)):
                        where_conditions.append(f'{target_table}."{col}" = {val}')
                    else:
                        # Escape single quotes in string values
                        escaped_val = str(val).replace("'", "''")
                        where_conditions.append(f'{target_table}."{col}" = \'{escaped_val}\'')
                
                if where_conditions:
                    additional_where = " AND " + " AND ".join(where_conditions)
            
            # Construct the final SQL
            sql = f"UPDATE {target_table} SET {set_clause} FROM {source} WHERE {join_clause}{additional_where}"
            logger.debug(f"Generated SQL: {sql}")
            return sql
        
        # For simple updates, use a more direct approach to avoid sqlglot expression issues
        # Build the table identifier
        
        # Build the UPDATE statement manually
        sql = f"UPDATE {target_table} SET "
        
        # Use the first dictionary in the list for single-row updates
        if set_values and len(set_values) > 0:
            update_data = set_values[0]  # Take the first row for single updates
            
            # Build SET clause
            set_clauses = []
            for col, val in update_data.items():
                if val is None:
                    set_clauses.append(f'"{col}" = NULL')
                elif isinstance(val, (int, float)):
                    set_clauses.append(f'"{col}" = {val}')
                else:
                    # Escape single quotes in string values
                    escaped_val = str(val).replace("'", "''")
                    set_clauses.append(f'"{col}" = \'{escaped_val}\'')
            
            sql += ", ".join(set_clauses)
        else:
            # No values to update
            return f"-- No values to update for {table_sql}"
        
        # Add WHERE clause if provided
        if where:
            where_clauses = []
            for col, val in where.items():
                if val is None:
                    where_clauses.append(f'"{col}" IS NULL')
                elif isinstance(val, (int, float)):
                    where_clauses.append(f'"{col}" = {val}')
                else:
                    # Escape single quotes in string values
                    escaped_val = str(val).replace("'", "''")
                    where_clauses.append(f'"{col}" = \'{escaped_val}\'')
            
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)
        
        return sql

    def _build_delete(
        self,
        table_name: str,
        schema_name: Optional[str] = None,
        where: Optional[Dict] = None,
    ) -> str:
        """Build a DELETE statement using sqlglot for safety.

        Args:
            table_name: Name of the table to delete from
            schema_name: Optional schema/database name
            where: Optional dictionary of column-value pairs for WHERE clause

        Returns:
            str: Safe SQL DELETE statement
        """
        # Build the table identifier
        table_expr = self.build_table_identifier(table_name, schema_name)

        # Build the DELETE statement
        delete_expr = exp.Delete(this=table_expr)

        # Add WHERE clause if provided
        if where:
            where_conditions = []
            for col, val in where.items():
                if val is None:
                    where_conditions.append(exp.IsNull(this=exp.Identifier(this=col)))
                else:
                    if isinstance(val, (int, float)):
                        val_expr = exp.Literal(this=val)
                    else:
                        val_expr = exp.Literal(this=val, is_string=True)
                    where_conditions.append(
                        exp.EQ(this=exp.Identifier(this=col), expression=val_expr)
                    )

            # Combine conditions with AND
            if where_conditions:
                where_expr = where_conditions[0]
                for condition in where_conditions[1:]:
                    where_expr = exp.And(this=where_expr, expression=condition)

                delete_expr = delete_expr.where(where_expr)

        # Generate the SQL
        return delete_expr.sql(dialect=self.sql_dialect)

    def _build_drop_table(
        self, table_name: str, schema_name: Optional[str] = None, if_exists: bool = True
    ) -> str:
        """Build a DROP TABLE statement using sqlglot for safety.

        Args:
            table_name: Name of the table to drop
            schema_name: Optional schema/database name
            if_exists: Whether to add IF EXISTS clause

        Returns:
            str: Safe SQL DROP TABLE statement
        """
        # Build the table identifier
        table_expr = self.build_table_identifier(table_name, schema_name)

        # Build the DROP TABLE statement
        drop_expr = exp.Drop(this=exp.Table(this=table_expr), exists=if_exists)

        # Generate the SQL
        return drop_expr.sql(dialect=self.sql_dialect)

    def _build_merge(
        self,
        table_name: str,
        schema_name: str,
        source_table_name: str = None,
        key_columns: List[str] = None,
        all_columns: List[str] = None,
    ) -> Union[Dict[str, str], str]:
        """Build a MERGE statement (upsert) using sqlglot for safety.

        Args:
            table_name: Name of the target table
            schema_name: Optional schema/database name
            source_table_name: Name of the source table
            key_columns: List of key columns for matching records
            all_columns: List of all columns in the tables

        Returns:
            Union[Dict[str, str], str]: Either a dictionary of SQL statements for transaction-based merge
                                        or a single SQL statement for native MERGE support
        """
        # Build the table identifier
        target_table = self.build_table_identifier(table_name, schema_name)
         
        # Determine which columns to update (all except key columns)
        logger.debug(f"All columns for merge: {all_columns}")
        update_columns = [col for col in all_columns if col not in key_columns]

        logger.debug(f"building merge statement for {target_table} with key columns {key_columns} and update columns {update_columns}")

        # DuckDB doesn't support MERGE syntax directly, use transaction-based approach
        if self.sql_dialect == "duckdb":
            # Build match conditions for the WHERE clause
            match_parts = []
            for col in key_columns:
                col_expr = exp.EQ(
                    this=exp.Column(
                        this=exp.Identifier(this=col),
                        table=exp.Identifier(this="target"),
                    ),
                    expression=exp.Column(
                        this=exp.Identifier(this=col),
                        table=exp.Identifier(this="source"),
                    ),
                )
                match_parts.append(col_expr.sql(dialect=self.sql_dialect))

            # Build SET expressions for UPDATE
            set_parts = []
            for col in update_columns:
                set_expr = exp.EQ(
                    this=exp.Identifier(this=col),
                    expression=exp.Column(
                        this=exp.Identifier(this=col),
                        table=exp.Identifier(this="source"),
                    ),
                )
                set_parts.append(set_expr.sql(dialect=self.sql_dialect))

            # Build column lists for INSERT
            column_list = ", ".join(
                [
                    exp.Identifier(this=col).sql(dialect=self.sql_dialect)
                    for col in all_columns
                ]
            )
            source_column_list = ", ".join(
                [
                    exp.Column(
                        this=exp.Identifier(this=col),
                        table=exp.Identifier(this="source"),
                    ).sql(dialect=self.sql_dialect)
                    for col in all_columns
                ]
            )

            # Build the WHERE clause to find records that don't exist in the target
            where_not_exists = []
            for col in key_columns:
                col_expr = exp.Column(
                    this=exp.Identifier(this=col), table=exp.Identifier(this="target")
                )
                where_not_exists.append(f"{col_expr.sql(dialect=self.sql_dialect)} IS NULL")

            # Build the UPDATE statement
            update_sql = f"UPDATE {target_table} AS target SET {', '.join(set_parts)} FROM {source_table_name} AS source WHERE {' AND '.join(match_parts)}"

            # Build the INSERT statement for new records
            insert_sql = f"""
            INSERT INTO {target_table} ({column_list})
            SELECT {source_column_list}
            FROM {source_table_name} AS source
            LEFT JOIN {target_table} AS target
            ON {' AND '.join(match_parts)}
            """

            if where_not_exists:
                insert_sql += f"WHERE {' OR '.join(where_not_exists)}"
            # Return a dictionary with all SQL statements for the merge operation
            return insert_sql + '; ' + update_sql
        else:
            # Generic MERGE syntax (ANSI SQL)
            merge_expr = exp.Merge(
                this=target_table,
                using=exp.Table(
                    this=exp.Identifier(this=source_table_name),
                    alias=exp.TableAlias(this=exp.Identifier(this="source")),
                ),
                on=self._build_merge_on_condition(key_columns),
            )

            # WHEN MATCHED THEN UPDATE
            set_items = []
            for col in update_columns:
                set_items.append(
                    exp.EQ(
                        this=exp.Identifier(this=col),
                        expression=exp.Column(
                            this=exp.Identifier(this=col),
                            table=exp.Identifier(this="source"),
                        ),
                    )
                )

            # WHEN NOT MATCHED THEN INSERT
            insert_cols = [exp.Identifier(this=col) for col in all_columns]
            insert_vals = [
                exp.Column(
                    this=exp.Identifier(this=col),
                    table=exp.Identifier(this="source"),
                )
                for col in all_columns
            ]

            # Add the clauses to the MERGE statement
            merge_expr = merge_expr.when(
                matched=True, then=exp.Update(expressions=set_items)
            ).when(
                matched=False,
                then=exp.Insert(this=insert_cols, expressions=insert_vals),
            )

            # Generate the SQL
            try:
                logger.debug(f"generating merge SQL: {merge_expr.sql(dialect=self.sql_dialect)}")
                return merge_expr.sql(dialect=self.sql_dialect)
            except Exception as e:
                # Fallback to string-based SQL if sqlglot doesn't support this dialect's MERGE syntax
                raise ValueError(
                    f"Merge operation not yet implemented for dialect: {self.sql_dialect}. Error: {e}"
                )

    def _build_merge_on_condition(self, key_columns: List[str]) -> exp.Expression:
        """Build the ON condition for a MERGE statement.

        Args:
            key_columns: List of key columns for matching records

        Returns:
            exp.Expression: sqlglot expression for the ON condition
        """
        conditions = []
        for col in key_columns:
            conditions.append(
                exp.EQ(
                    this=exp.Column(
                        this=exp.Identifier(this=col),
                        table=exp.Identifier(this="target"),
                    ),
                    expression=exp.Column(
                        this=exp.Identifier(this=col),
                        table=exp.Identifier(this="source"),
                    ),
                )
            )

        # Combine conditions with AND
        if not conditions:
            raise ValueError("At least one key column is required for MERGE operation")

        result = conditions[0]
        for condition in conditions[1:]:
            result = exp.And(this=result, expression=condition)

        return result

    def _build_select(
        self,
        columns: List[str],
        table_name: str,
        schema_name: Optional[str] = None,
        where: Optional[Dict] = None,
        limit: Optional[int] = None,
    ) -> str:
        """Build a SELECT statement using sqlglot for safety.

        Args:
            columns: List of columns to select
            table_name: Name of the table to select from
            schema_name: Optional schema/database name
            where: Optional dictionary of column-value pairs for WHERE clause
            limit: Optional limit for the number of rows to return

        Returns:
            str: Safe SQL SELECT statement
        """
        # Build the table identifier
        table_expr = self.build_table_identifier(table_name, schema_name)

        # Build the column expressions
        column_exprs = []
        for col in columns:
            if col == "*":
                column_exprs.append(exp.Star())
            else:
                column_exprs.append(exp.Identifier(this=col))

        # Build the SELECT statement
        select_expr = exp.Select(expressions=column_exprs).from_(table_expr)

        # Add WHERE clause if provided
        if where:
            where_conditions = []
            for col, val in where.items():
                if val is None:
                    where_conditions.append(exp.IsNull(this=exp.Identifier(this=col)))
                else:
                    if isinstance(val, (int, float)):
                        val_expr = exp.Literal(this=val)
                    else:
                        val_expr = exp.Literal(this=val, is_string=True)
                    where_conditions.append(
                        exp.EQ(this=exp.Identifier(this=col), expression=val_expr)
                    )

            # Combine conditions with AND
            if where_conditions:
                where_expr = where_conditions[0]
                for condition in where_conditions[1:]:
                    where_expr = exp.And(this=where_expr, expression=condition)

                select_expr = select_expr.where(where_expr)

        # Add LIMIT clause if provided
        if limit is not None:
            select_expr = select_expr.limit(exp.Literal(this=limit))

        # Generate the SQL
        return select_expr.sql(dialect=self.sql_dialect)

    # Post-process to uppercase SQL keywords (CREATE, TABLE, etc.)
    def uppercase_keywords(self, sql):
        # Uppercase common SQL keywords only (not identifiers)
        keywords = [
            "create",
            "temporary",
            "table",
            "as",
            "primary key",
            "not null",
            "unique",
            "constraint",
            "references",
            "comment",
            "on",
            "is",
            "partition by",
            "with",
            "location",
            "if not exists",
            "exists",
        ]
        # Sort by length to avoid partial replacements
        keywords = sorted(keywords, key=len, reverse=True)
        for kw in keywords:
            # Only match whole words (case-insensitive)
            sql = re.sub(rf"\b{kw}\b", kw.upper(), sql, flags=re.IGNORECASE)
        return sql

    def _table_options_athena(self, table_options: TableOptions):
        """
        Construct Athena table options as a tuple of (properties_list, special_clauses_dict).
        For Athena, we need to handle special clauses like PARTITIONED BY and LOCATION
        separately from regular table properties.

        Args:
            table_options (dict or TableOptions): Dictionary or TableOptions object with Athena table options.
        Returns:
            tuple: (properties_list, special_clauses_dict)
        """
        properties_list = []
        special_clauses = {
            "partitioned_by": None,
            "location": None,
            "tblproperties": None,
            "row_format": None,
            "stored_as": None,
            "comment": None,
        }

        # Handle ROW FORMAT SERDE - separate clause in Athena
        if table_options.row_format_serde:
            special_clauses["row_format"] = table_options.row_format_serde

        # Handle STORED AS - separate clause in Athena
        if table_options.stored_format:
            special_clauses["stored_as"] = table_options.stored_format

        # Handle PARTITIONED BY - separate clause in Athena
        if table_options.partition_by:
            special_clauses["partitioned_by"] = table_options.partition_by

        # Handle CLUSTERED BY - goes in WITH clause
        if table_options.clustered_by:
            clustered_by = table_options.clustered_by
            properties_list.append(
                exp.Property(
                    this=exp.Identifier(this="CLUSTERED BY"),
                    value=exp.Literal.string(clustered_by),
                )
            )

        # Handle LOCATION - separate clause in Athena
        if table_options.location:
            special_clauses["location"] = table_options.location

        # Handle table comment - separate clause in Athena
        if table_options.comment:
            special_clauses["comment"] = table_options.comment

        # Handle SERDEPROPERTIES - goes in WITH clause
        if table_options.serde_properties:
            serde_properties = table_options.serde_properties
            serde_prop_pairs = []
            for key, value in serde_properties.items():
                serde_prop_pairs.append(f"{key} '{value}'")
            properties_list.append(
                exp.Property(
                    this=exp.Identifier(this="WITH SERDEPROPERTIES"),
                    value=exp.Literal.string(f"({', '.join(serde_prop_pairs)})"),
                )
            )

        # Handle TBLPROPERTIES - separate clause in Athena
        if table_options.tblproperties:
            special_clauses["tblproperties"] = table_options.tblproperties

        logger.debug(f"Athena table options (properties): {properties_list}")
        logger.debug(f"Athena table options (special clauses): {special_clauses}")
        return properties_list, special_clauses

    def _table_options_duckdb(self, table_name: str, table_options: TableOptions):
        """
        Validate DuckDB table options. Raises if unsupported options are present.
        Args:
            table_name (str): Name of the table for generating COMMENT ON TABLE if needed.
            table_options (dict or TableOptions): Table options for DuckDB.
        Returns:
            Tuple[List, List[str]]: Always an empty list for options, and a list of extra SQL statements (e.g., COMMENT ON TABLE ...).
        Raises:
            ValueError: If any unsupported options are present.
        """
        unsupported = [
            "partitioned_by",
            "row_format_serde",
            "stored_format",
            "location",
            "tblproperties",
        ]
        _options = asdict(table_options)
        extra_statements = []
        for key, value in _options.items():
            if value and key in unsupported:
                raise ValueError(
                    f"DuckDB does not support the following table options: {', '.join(unsupported)}"
                )
            if key == "comment" and value:
                comment_val = str(value).replace("'", "''")
                extra_statements.append(
                    f"COMMENT ON TABLE \"{table_name}\" IS '{comment_val}';"
                )
        return extra_statements

    def convert_duckdb_to_iceberg_type(self, duckdb_type: str) -> str:
        """Convert a DuckDB type to an Iceberg type

        Args:
            duckdb_type: DuckDB data type

        Returns:
            str: Corresponding Iceberg type
        """
        # Handle DECIMAL type with precision and scale
        if duckdb_type.startswith("DECIMAL"):
            # Extract precision and scale if provided
            if "(" in duckdb_type and ")" in duckdb_type:
                params = duckdb_type[duckdb_type.find("(") + 1 : duckdb_type.find(")")]
                if "," in params:
                    precision, scale = params.split(",")
                    return f"decimal({precision.strip()},{scale.strip()})"
                else:
                    precision = params.strip()
                    return f"decimal({precision})"
            return "decimal(38,6)"  # Default precision and scale

        # Handle VARCHAR with length
        if duckdb_type.startswith("VARCHAR"):
            return "string"

        # Use the mapping for other types
        base_type = duckdb_type.split("(")[0].upper()
        return DUCKDB_TO_ICEBERG_MAPPING.get(base_type, "string")

    # This standalone function has been consolidated into the SqlGenerator class
    def _build_alter_table_add_column(
        self,
        table_name: str,
        column_name: str,
        data_type: str = "VARCHAR",
        default_value: Optional[str] = None,
        schema_name: Optional[str] = None,
        if_not_exists: bool = True,
    ):
        table_id = self.build_table_identifier(table_name, schema_name)
        col_def = f'"{column_name}" {data_type}'
        if default_value is not None:
            if isinstance(default_value, str):
                col_def += f" DEFAULT '{default_value}'"
            else:
                col_def += f" DEFAULT {default_value}"
        if if_not_exists:
            sql = f'ALTER TABLE {table_id} ADD COLUMN IF NOT EXISTS {col_def};'
        else:
            sql = f'ALTER TABLE {table_id} ADD COLUMN {col_def};'
        return sql
