from pyiceberg.catalog import load_catalog
import logging
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import os

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def assume_role(role_arn, session_name="DataOpsSession"):
    sts = boto3.client("sts")
    response = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName=session_name,
    )
    creds = response["Credentials"]
    return {
        "aws_access_key_id": creds["AccessKeyId"],
        "aws_secret_access_key": creds["SecretAccessKey"],
        "aws_session_token": creds["SessionToken"],
    }

def check_glue_connection(region_name='us-east-1'):
    try:
        glue = boto3.client("glue", region_name=region_name)
        response = glue.get_databases(MaxResults=1)
        logger.info(f"Successfully connected to AWS Glue. Found {len(response.get('DatabaseList', []))} database(s).")
        return True
    except (BotoCoreError, ClientError) as e:
        logger.error(f"Failed to connect to AWS Glue: {e}")
        return False

def check_iceberg_connection():
    creds = assume_role("arn:aws:iam::529282641471:role/bhsl-DataOps")
    _session = boto3.session.Session(**creds)
    sts = _session.client("sts")
    print(sts.get_caller_identity())

    os.environ["AWS_ACCESS_KEY_ID"] = creds["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds["aws_secret_access_key"]
    os.environ["AWS_SESSION_TOKEN"] = creds["aws_session_token"]
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    warehouse = os.environ["REST_WAREHOUSE"]

    try:
        catalog = load_catalog("s3tablescatalog", 
        **{
        "type": "rest", 
        "warehouse": warehouse, 
        "rest.sigv4-enabled": "true",
        "rest.signing-name": "s3tables",
        "rest.signing-region": "us-east-1"}
        )
        if catalog.table_exists("test.test"):
            catalog.purge_table("test.test")
        if catalog.table_exists("default.eod"):
            catalog.purge_table("default.eod")
        eod_schema = StockSchemas.eod.schema() 
        catalog.create_table(
            identifier="test.eod",
            schema=eod_schema.to_iceberg_schema(),
            location="test"
        )
        logger.info("Successfully connected to Iceberg catalog.")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Iceberg catalog: {e}")
        return False

if __name__ == "__main__":
    # check_glue_connection()
    check_iceberg_connection()
