""" module to upload parameter vlaues to systems manager parameter store
"""
import boto3
from botocore.exceptions import ClientError


def upload_to_ssmps(
    *,
    client: boto3.client,
    parameters: list[dict[str, str]],
    overwrite: bool = False,
    **kwargs: dict[str, str],
) -> None:
    """helper function to upload systems manager parameter store parameters

    Args:
        client (boto3.session.Session.client): boto3 client with properly set profilea
        name and region
        parameters (List[Dict[str, str]]): a list of dictionary with key,, value, and
         type overwrite (bool, optional): True or False
    """

    for item in parameters:
        try:
            client.put_parameter(
                Name=item["KEY"],
                Value=item["VALUE"],
                Type=item["TYPE"],
                Overwrite=overwrite,
                **kwargs,
            )
        except ClientError as err:
            print(err.args)
            raise


if __name__ == "__main__":
    import json
    from pathlib import Path

    file_dir = Path(__file__).resolve()
    with open(
        file_dir.parents[1].joinpath("secrets.json"), "r", encoding="utf-8"
    ) as fp:
        params = json.load(fp)

    for env in ["cicd", "dev", "prod"]:
        _profile = f"my{env}"

        SESSION = boto3.Session(profile_name=_profile)
        CLIENT = SESSION.client("ssm", region_name="us-east-1")
        upload_to_ssmps(client=CLIENT, parameters=params, overwrite=True)
