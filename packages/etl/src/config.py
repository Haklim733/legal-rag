"""a class to store all the variables needed for all tests and functions"""
from typing import Optional

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """a class to store all the environment variables needed for all tests and function
    in this repository

    Note: environment variables are not case sensitive
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    options_bucket: str = Field(default="bhsl-test", alias="BUCKET")
    key_options_data: str = Field(default="data/options/raw/", alias="OPTIONS_DATA_KEY")
    website_bucket: str = Field(alias="BHSL_BUCKET", default="")
    api_url: Optional[str] = Field(alias="API_URL", default=None)
    aws_profile: str = Field(alias="AWS_PROFILE", default="")
    environment: str = Field(alias="ENVIRONMENT", default="dev")
    td_consumer_key: SecretStr = Field(alias="TD_CONSUMER_KEY", default="")

    @model_validator(mode="after")  # type: ignore
    def toggle_params(self) -> None:
        """sets the api_url and bucket if prod env file is used"""
        if self.api_url:
            if self.environment == "prod":
                self.options_bucket = "bhsl-prod"
            else:
                self.options_bucket = "bhsl-dev"

    @property
    def headers(self) -> dict[str, str]:
        """return the headers for the api call"""
        return {
            "Content-Type": "application/json",
        }

    @property
    def raw_data_path(self) -> str:
        """returnm the full base s3 path except"""
        return f"s3://{str(self.options_bucket)}/{self.key_options_data}"
