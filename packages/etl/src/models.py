from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
import pyarrow as pa
import pyiceberg.types as iceberg_types
from pyiceberg.schema import Schema
from pyiceberg.types import NestedField


DUCKDB_TO_PYICEBERG_MAPPING = {
    'VARCHAR': iceberg_types.StringType,
    'DATE': iceberg_types.DateType,
    'TIMESTAMP': iceberg_types.TimestampType,
    'DOUBLE': iceberg_types.DoubleType,
    'FLOAT': iceberg_types.FloatType,
    'INTEGER': iceberg_types.IntegerType,
    'BIGINT': iceberg_types.LongType,
    'SMALLINT': iceberg_types.IntegerType,
    'BOOLEAN': iceberg_types.BooleanType,
    'DECIMAL': iceberg_types.DecimalType,
    'UBIGINT': iceberg_types.LongType,
}

DUCKDB_TO_ICEBERG_MAPPING = {
    "VARCHAR": "string",
    "DATE": "date",
    "TIMESTAMP": "timestamp",
    "DOUBLE": "double",
    "FLOAT": "float",
    "INTEGER": "int",
    "BIGINT": "long",
    "SMALLINT": "int",
    "BOOLEAN": "boolean",
    "DECIMAL": "decimal",
}

class EngineType(Enum):
    ATHENA = "athena"
    DUCKDB = "duckdb"
    PYICEBERG = "pyiceberg"
    PYARROW = "pyarrow"

@dataclass
class SchemaField:
    """Field definition that can be converted to both DuckDB and Iceberg types (default is DuckDB)"""
    name: str
    data_type: str  # Type string in the format expected by our converter function
    comment: Optional[str] = None
    primary_key: bool = False
    foreign_key: Optional[str] = None  # Reference in the format 'table_name.column_name'
    constraints: Optional[List[str]] = field(default_factory=list)  # SQL constraints like NOT NULL, UNIQUE, CHECK, etc.
      
    def get_pyiceberg_type(self) -> Any:
        """Convert field type to Iceberg type by mapping from DuckDB type"""
        # Get the base type (without parameters)
        base_type = self.data_type.split('(')[0]
        
        # Handle special cases first
        if base_type == 'DECIMAL':
            try:
                # Extract precision and scale if specified
                if '(' in self.data_type:
                    params = self.data_type.split('(')[1].split(')')[0].split(',')
                    precision = int(params[0])
                    scale = int(params[1]) if len(params) > 1 else 0
                else:
                    # Default precision and scale
                    precision, scale = 18, 8
                return iceberg_types.DecimalType(precision, scale)
            except (IndexError, ValueError):
                return iceberg_types.DecimalType(18, 8)
        
        # For all other types
        iceberg_type_class = DUCKDB_TO_PYICEBERG_MAPPING.get(base_type, iceberg_types.StringType)
        return iceberg_type_class()

class TableSchema:
    def __init__(self, name: str, ddl_path: Optional[str] = None, fields: Optional[List[SchemaField]] = None, 
                 partition_by: Optional[List[str]] = None,
                 primary_key: Optional[List[str]] = None,
                 foreign_keys: Optional[Dict[str, str]] = None,
                 comment: Optional[str] = None):
        self.name = name
        self.fields = fields
        self.partition_by = partition_by or []
        self.primary_key = primary_key or []
        self.foreign_keys = foreign_keys or {}
        self.comment = comment
        
        if not self.fields:
            self.parse_ddl_schema(ddl_path)
        
        if not self.primary_key:
            self.primary_key = [field.name for field in self.fields if field.primary_key]
        
        for field in self.fields:
            if field.foreign_key and field.name not in self.foreign_keys:
                self.foreign_keys[field.name] = field.foreign_key
    
    def to_field_dicts(self) -> List[dict]:
        """Convert SchemaField objects to dictionaries for use with SqlGenerator
        
        Returns:
            List[dict]: List of field dictionaries with properties
        """
        field_dicts = []
        for field in self.fields:
            field_dict = {
                'name': field.name,
                'data_type': field.data_type,
                'primary_key': field.primary_key,
                'comment': field.comment,
                'foreign_key': field.foreign_key
            }
            if field.constraints:
                field_dict['constraints'] = field.constraints
            field_dicts.append(field_dict)
        return field_dicts
        
    def to_pyiceberg_schema(self, custom_field_mapping: Optional[Dict[str, Any]] = None) -> Any:
        """Convert TableSchema to PyIceberg Schema
        
        Uses each SchemaField's get_pyiceberg_type method to convert the schema to PyIceberg format.
        
        Args:
            custom_field_mapping: Optional mapping to customize field properties
            
        Returns:
            PyIceberg Schema object
        """
        
        # Create a list of PyIceberg NestedField objects
        iceberg_fields = []
        
        for i, field in enumerate(self.fields):
            # Get the PyIceberg type for this field
            iceberg_type = field.get_pyiceberg_type()
            
            # Apply custom field mapping if provided
            if custom_field_mapping and field.name in custom_field_mapping:
                custom_props = custom_field_mapping[field.name]
                if 'type' in custom_props:
                    # Override the type if specified in the custom mapping
                    iceberg_type = custom_props['type']
            
            # Determine if the field is required (not nullable)
            # A field is required if it has a NOT NULL constraint
            required = any(constraint.upper() == 'NOT NULL' for constraint in field.constraints) if field.constraints else False 
            
            # Create the NestedField
            iceberg_field = NestedField(
                field_id=i + 1,  # Field IDs start at 1
                name=field.name,
                field_type=iceberg_type,
                required=required,
                doc=field.comment if hasattr(field, 'comment') else None
            )
            
            iceberg_fields.append(iceberg_field)
        
        # Create and return the Schema
        return Schema(*iceberg_fields)
        
    def get_fields(self) -> List[SchemaField]:
        """Return the list of SchemaField objects
        
        Returns:
            List[SchemaField]: List of SchemaField objects
        """
        return self.fields

    def to_pyarrow_schema(self):
        arrow_fields = []
        for field in self.fields:
            dtype = field.data_type.lower()
            if dtype in ('string', 'varchar', 'text'):
                pa_type = pa.string()
            elif dtype in ('float', 'float32'):
                pa_type = pa.float32()
            elif dtype in ('double', 'float64'):
                pa_type = pa.float64()
            elif dtype in ('int', 'int32', 'integer'):
                pa_type = pa.int32()
            elif dtype in ('bigint', 'int64', 'long'):
                pa_type = pa.int64()
            elif dtype in ('ubigint', 'uint64', 'ulong'):
                pa_type = pa.uint64()
            elif dtype in ('date',):
                pa_type = pa.date32()
            elif dtype in ('timestamp',):
                pa_type = pa.timestamp()
            else:
                raise ValueError(f"Unknown type: {field.data_type}")

            constraints = field.constraints or []
            nullable = True
            for c in constraints:
                if "not null" in c.lower():
                    nullable = False
                    break

            arrow_fields.append(pa.field(field.name, pa_type, nullable=nullable))
        return pa.schema(arrow_fields)

    def schema(self, engine_type: EngineType = EngineType.DUCKDB):
        if engine_type == EngineType.DUCKDB:
            return self
        elif engine_type == EngineType.PYICEBERG:
            return self.to_pyiceberg_schema()
        elif engine_type == EngineType.PYARROW:
            return self.to_pyarrow_schema()
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")
    
    def parse_ddl_schema(self, ddl_path: str):
        """
        Parse a simple CREATE TABLE DDL file and return a list of SchemaField objects.
        Assumes each line in the CREATE TABLE (...) block is of the form:
            column_name DATA_TYPE,
        """
        fields = []
        ddl_text = Path(ddl_path).read_text()
        # Extract the column definitions block
        match = re.search(r'CREATE TABLE\s+\w+\s*\((.*?)\);', ddl_text, re.DOTALL | re.IGNORECASE)
        if not match:
            raise ValueError("Could not find CREATE TABLE block in DDL.")
        columns_block = match.group(1)
        # Split lines and parse each column
        for line in columns_block.splitlines():
            line = line.strip().rstrip(',')
            if not line or line.startswith('--'):
                continue
            # Match: name type (ignore constraints/comments)
            col_match = re.match(r'([A-Za-z0-9_]+)\s+([A-Za-z0-9_()]+)', line)
            if col_match:
                name, sql_type = col_match.groups()
                # Map SQL type to your expected string
                if sql_type.startswith('VARCHAR'):
                    dtype = 'VARCHAR'
                elif sql_type in ('INT', 'INTEGER'):
                    dtype = 'INTEGER'
                elif sql_type == 'NUMERIC':
                    dtype = 'DOUBLE'
                else:
                    dtype = sql_type  # fallback
                fields.append(SchemaField(name=name, data_type=dtype))
        self.fields = fields  

    @classmethod
    def _pyarrow_type_to_str(cls, pa_type):
        if pa.types.is_string(pa_type):
            return "VARCHAR"
        elif pa.types.is_float32(pa_type):
            return "FLOAT"
        elif pa.types.is_float64(pa_type):
            return "DOUBLE"
        elif pa.types.is_int32(pa_type):
            return "INTEGER"
        elif pa.types.is_int64(pa_type):
            return "BIGINT"
        elif pa.types.is_uint64(pa_type):
            return "UBIGINT"
        elif pa.types.is_date32(pa_type):
            return "DATE"
        elif pa.types.is_timestamp(pa_type):
            return "TIMESTAMP"
        else:
            raise ValueError(f"Unsupported pyarrow type: {pa_type}")

    @classmethod
    def from_pyarrow(cls, pa_schema) -> List[SchemaField]:
        fields = []
        for field in pa_schema:
            dtype = cls._pyarrow_type_to_str(field.type)
            constraints = []
            if not field.nullable:
                constraints.append("NOT NULL")
            # You can add more logic for other constraints if needed
            fields.append(SchemaField(name=field.name, data_type=dtype, constraints=constraints))
        return fields

@dataclass
class TableOptions:
    partition_by: Optional[List[Dict[str, str]]] = None  # e.g., [{"year": "string"}, ...]
    row_format_serde: Optional[str] = None
    stored_format: Optional[str] = None
    location: Optional[str] = None
    comment: Optional[str] = None
    clustered_by: Optional[List[str]] = None
    serde_properties: Dict[str, str] = field(default_factory=dict)
    tblproperties: Dict[str, str] = field(default_factory=dict)