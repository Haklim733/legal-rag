# athena_maintenance.py
import boto3
import os

def lambda_handler(event, context):
    athena = boto3.client("athena")
    workgroup = os.environ["WORKGROUP"]
    database = os.environ["DATABASE"]
    query = os.environ["VACUUM_QUERY"]
    output = os.environ["OUTPUT_LOCATION"]
    resp = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output},
        WorkGroup=workgroup,
    )
    return {"QueryExecutionId": resp["QueryExecutionId"]}