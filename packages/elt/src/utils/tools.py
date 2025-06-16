"""_ this module contains functions and classes that can be used across projects
"""
import gzip
import sys
from datetime import datetime
from functools import singledispatch
from gc import get_referents
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any, Optional
from zoneinfo import ZoneInfo

BLACKLIST = type, ModuleType, FunctionType


def merge_dicts(d1: dict[str, Any], d2: dict[str, Any]) -> dict[str, Any]:
    """a function to merge nested dictionaries together

    Args:
        d1 (_type_): python dictionary
        d2 (_type_): python dictionary

    Returns:
        _type_: _description_
    """
    if isinstance(d1, dict) and isinstance(d2, dict):
        # Unwrap d1 and d2 in new dictionary to keep non-shared keys with **d1, **d2
        # Next unwrap a dict that treats shared keys
        # If two keys have an equal value, we take that value as new value
        # If the values are not equal, we recursively merge them
        return {
            **d1,
            **d2,
            **{
                k: d1[k] if d1[k] == d2[k] else merge_dicts(d1[k], d2[k])
                for k in {*d1} & {*d2}
            },
        }
    return d1  # the first dict value with the same ky as the second is kept


@singledispatch
def convert_path(value: Any) -> Path:
    """converts input to pathlib.Path"""
    raise NotImplementedError("Implement process function.")


@convert_path.register(str)
def convert_path_str(value: str) -> Path:
    """converts string input types to Path variable"""
    return Path(value)


@convert_path.register(Path)
def convert_path_path(value: Path) -> Path:
    """converts string input types to Path variable"""
    return value


def get_files(path: Path | str, suffix: Optional[list[str]] = None) -> list[Path]:
    """a helper function to retrieve a list of files

    Returns:
        _type_: _description_
    """
    converted = convert_path(path)
    if converted.is_dir() is False:
        raise ValueError("path must be a directory not a file")

    if suffix:
        files: list[Path] = []
        for ext in suffix:
            if not ext.startswith("."):
                raise ValueError("suffix must start with a .")
            files.extend(list(converted.glob(f"*{ext}")))
        return files

    files = list(converted.iterdir())
    return files


@singledispatch
def convert_to_datetime(value: Any) -> datetime:
    """a function to convert to datetime"""
    raise NotImplementedError("Implement process function.")


@convert_to_datetime.register(datetime)
def convert_to_datetime_datetime(value: datetime) -> datetime:
    """a function to return datetime"""
    return value


def check_datetime_len(value: int | float) -> int | float:
    """checks proper date time length"""
    int_len = len(str(value).split(".", maxsplit=1)[0])
    if int_len >= 13:
        value = value / 1000
    elif int_len < 10:
        raise ValueError("the length of the integer value is too small")
    return value


@convert_to_datetime.register(int)
def convert_datetime_int(value: int) -> datetime:
    """function to convert integer datetime"""
    return datetime.fromtimestamp(check_datetime_len(value))


@convert_to_datetime.register(float)
def convert_datetime_(value: float) -> datetime:
    """function to convert float to datetime"""
    return datetime.fromtimestamp(check_datetime_len(value))


def convert_tz(value: int | datetime, source_tz: str, local_tz: str) -> datetime:
    """a function to convert to local time
    Args:
        value (str): _description_

    Returns:
        _type_: _description_
    """
    dt_value = convert_to_datetime(value)
    from_zone = ZoneInfo(source_tz)
    to_zone = ZoneInfo(local_tz)
    source_value = dt_value.replace(tzinfo=from_zone)
    return source_value.astimezone(to_zone)


@singledispatch
def check_filename(value: Any) -> str:
    """function to check filename value"""
    raise NotImplementedError("value must be a string")


@check_filename.register(str)
def check_filename_str(value: str) -> str:
    """function to check filename string values"""
    EXTENSIONS = ["parquet", "csv", "json"]
    split = value.split(".")
    if len(split) < 2:
        raise ValueError("filename must include an extension")
    if split[-1] not in EXTENSIONS:
        raise ValueError(f"extensions must be one of {EXTENSIONS}")
    if split[0] == "":
        raise ValueError("must supply a filename not just the extension")
    return value


def open_gzip_file(file_path: Path | str) -> bytes:
    """function to unzip gzip files"""
    with gzip.open(file_path, "rb") as fp:
        file_content = fp.read()
        return file_content


def get_size(obj: Any) -> int:
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError("getsize() does not take argument of type: " + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for item in objects:
            if not isinstance(item, BLACKLIST) and id(item) not in seen_ids:
                seen_ids.add(id(item))
                size += sys.getsizeof(item)
                need_referents.append(item)
        objects = get_referents(*need_referents)
    return size
