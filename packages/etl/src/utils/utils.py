from datetime import datetime, timedelta
from io import BytesIO
import logging
import sys
import pandas as pd
import requests

# Configure logging to write to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def transform_to_df(
    *, res: requests.Response, ticker: str, csv: str = "false"
) -> pd.DataFrame:
    """returns response from thetadata api as a pandas dataframe"""
    if csv == "true":
        data = BytesIO(res.content)
        df = pd.read_csv(data)
    else:
        data = res.json()
        cols = data["header"]["format"]
        df = pd.DataFrame(data["response"], columns=cols)
    df["symbol"] = ticker
    return df