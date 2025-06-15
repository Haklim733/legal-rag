"""File reader module for reading various file formats using PyArrow."""

import os
import logging
from typing import Dict, List

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

class FileReader:
    """Class for reading various file formats using PyArrow.
    
    This class provides methods to read CSV and Parquet files into PyArrow tables,
    which can then be used by database classes.
    """
    
    def __init__(self, **kwargs):
        """Initialize the FileReader.
        
        Args:
            **kwargs: Additional arguments for file reading
        """
        self.kwargs = kwargs
    
    def read_csv(self, file_path: str, **kwargs) -> pa.Table:
        """Read a CSV file into a PyArrow table.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments for CSV reading
            
        Returns:
            pa.Table: PyArrow table containing the CSV data
        """
        logger.debug(f"Reading CSV file: {file_path}")
        
        read_options = {**self.kwargs, **kwargs}
        
        try:
            return csv.read_csv(file_path, **read_options)
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            raise
    
    def read_parquet(self, file_path: str, **kwargs) -> pa.Table:
        """Read a Parquet file into a PyArrow table.
        
        Args:
            file_path: Path to the Parquet file
            **kwargs: Additional arguments for Parquet reading
            
        Returns:
            pa.Table: PyArrow table containing the Parquet data
        """
        logger.debug(f"Reading Parquet file: {file_path}")
        
        read_options = {**self.kwargs, **kwargs}
        
        try:
            return pq.read_table(file_path, **read_options)
        except Exception as e:
            logger.error(f"Error reading Parquet file {file_path}: {e}")
            raise
    
    def read_file(self, file_path: str, **kwargs) -> pa.Table:
        """Read a file into a PyArrow table based on its extension.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional arguments for file reading
            
        Returns:
            pa.Table: PyArrow table containing the file data
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            return self.read_csv(file_path, **kwargs)
        elif file_ext == '.parquet':
            return self.read_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
    
    def convert_to_dict(self, table: pa.Table) -> List[Dict]:
        """Convert a PyArrow table to a list of dictionaries.
        
        Args:
            table: PyArrow table to convert
            
        Returns:
            List[Dict]: List of dictionaries, one per row
        """
        return table.to_pylist()
    
    def get_column_names(self, table: pa.Table) -> List[str]:
        """Get the column names from a PyArrow table.
        
        Args:
            table: PyArrow table
            
        Returns:
            List[str]: List of column names
        """
        return table.column_names
