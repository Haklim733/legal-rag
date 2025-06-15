""" module for custom aws functions
"""
from typing import Optional

import boto3
from mypy_boto3_s3.type_defs import ListObjectsV2OutputTypeDef, ObjectTypeDef


def get_parameter(
    parameter_path: str, boto3_session: Optional[boto3.Session] = None
) -> str:
    """function retrieves value from systems manager parameter store"""
    ssm = boto3.Session().client("ssm")
    if boto3_session:
        ssm = boto3_session.client("ssm")

    resp = ssm.get_parameter(Name=parameter_path, WithDecryption=True)
    value: str = resp["Parameter"]["Value"]
    return value


def get_s3_files(
    bucket: str,
    key: str,
    boto3_session: Optional[boto3.Session] = None,
) -> ListObjectsV2OutputTypeDef:
    """a helper function to retrieve a list of files form s3

    Returns:
        _type_: a list of files
    """
    s3_c = boto3.Session().client("s3")
    if boto3_session:
        s3_c = boto3_session.client("s3")

    files: ListObjectsV2OutputTypeDef = s3_c.list_objects_v2(Bucket=bucket, Prefix=key)
    if "Contents" in files:
        return files
    return "No files found"


def get_latest_s3_path(
    s3_contents: ListObjectsV2OutputTypeDef,
) -> ObjectTypeDef:
    """retrieves latest file modified in list of s3 Contents key form boto33 clientsj
    Args:
        s3_contents (list[str]): _description_
    """
    latest = max([x["LastModified"] for x in s3_contents["Contents"]])  #
    file = [x for x in s3_contents["Contents"] if x["LastModified"] == latest][0]
    return file

def list_s3_objects(bucket: str, prefix: str, suffix: Optional[str]=None) -> list[str]:
    """List all objects in the specified S3 path."""
    s3_client = boto3.client('s3')
    objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                if suffix:
                    if obj['Key'].endswith('.parquet'):
                        objects.append(obj['Key'])
                else:
                    objects.append(obj['Key'])
    return objects