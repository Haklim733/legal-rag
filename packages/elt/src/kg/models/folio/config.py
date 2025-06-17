"""
Configuration for the FOLIO (Federated Open Legal Information Ontology) Python library.
"""

# annotations
from __future__ import annotations

# imports
import json
from pathlib import Path
from typing import Literal, Optional

# packages
from pydantic import BaseModel, ConfigDict, Field

# project imports
from folio.logger import get_logger

# Default configuration path
DEFAULT_CONFIG_PATH: Path = Path.home() / ".folio/config.json"

# Default GitHub API URL
DEFAULT_GITHUB_API_URL: str = "https://api.github.com"
DEFAULT_GITHUB_OBJECT_URL: str = "https://raw.githubusercontent.com"

# Default source type, which determines how the ontology is loaded.
DEFAULT_SOURCE_TYPE: Literal["github", "http", "file"] = "file"

# Default HTTP URL for the ontology
DEFAULT_HTTP_URL: Optional[str] = None

# Default local OWL filename (used when source_type is 'file')
DEFAULT_OWL_FILENAME: str = "FOLIO.owl"

# Default GitHub owner, repo, and branch for the ontology
DEFAULT_GITHUB_REPO_OWNER: str = "alea-institute"
DEFAULT_GITHUB_REPO_NAME: str = "FOLIO"
DEFAULT_GITHUB_REPO_BRANCH: str = "2.0.0"

# set up the logger
LOGGER = get_logger(__name__)


class FOLIOConfiguration(BaseModel):
    """
    Configuration for the FOLIO (Federated Open Legal Information Ontology) Python library.
    """

    source: Literal["github", "http", "file"] = Field(
        ...,
        description="The source of the FOLIO configuration. Must be 'github', 'http', or 'file'.",
    )
    url: Optional[str] = Field(
        default=DEFAULT_HTTP_URL,
        description="The URL of the FOLIO ontology if source is 'http'.",
    )
    github_repo_owner: Optional[str] = Field(
        default=DEFAULT_GITHUB_REPO_OWNER,
        description="The owner of the GitHub repository for FOLIO if source is 'github'.",
    )
    github_repo_name: Optional[str] = Field(
        default=DEFAULT_GITHUB_REPO_NAME,
        description="The name of the GitHub repository for FOLIO if source is 'github'.",
    )
    github_repo_branch: Optional[str] = Field(
        default=DEFAULT_GITHUB_REPO_BRANCH,
        description="The branch of the GitHub repository for FOLIO if source is 'github'.",
    )
    owl_filename: Optional[str] = Field(
        default=DEFAULT_OWL_FILENAME,
        description="The filename of the local OWL ontology if source is 'file'. Path is resolved relative to usage.",
    )
    use_cache: bool = Field(
        True, description="Whether to use caching for the FOLIO configuration."
    )

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    @staticmethod
    def load_config(
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> FOLIOConfiguration:
        """
        Load the configuration from a JSON file.

        Args:
            config_path (str | Path): The path to the configuration file.

        Returns:
            dict: The configuration dictionary.
        """
        # determine the configuration file path
        if isinstance(config_path, str):
            config_file_path = Path(config_path)
        else:
            config_file_path = config_path

        # check if the configuration file exists
        if config_file_path.exists():
            with config_file_path.open("rt", encoding="utf-8") as input_file:
                config_data = json.load(input_file)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

        LOGGER.info("Loaded configuration from %s", config_file_path)

        # return the configuration dictionary
        return FOLIOConfiguration(
            source=config_data.get("folio", {}).get("source", DEFAULT_SOURCE_TYPE),
            url=config_data.get("folio", {}).get("url", DEFAULT_HTTP_URL),
            github_repo_owner=config_data.get("folio", {}).get(
                "github_repo_owner", DEFAULT_GITHUB_REPO_OWNER
            ),
            github_repo_name=config_data.get("folio", {}).get(
                "github_repo_name", DEFAULT_GITHUB_REPO_NAME
            ),
            github_repo_branch=config_data.get("folio", {}).get(
                "github_repo_branch", DEFAULT_GITHUB_REPO_BRANCH
            ),
            owl_filename=config_data.get("folio", {}).get(
                "owl_filename", DEFAULT_OWL_FILENAME
            ),
        )