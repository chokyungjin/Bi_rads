import os

import yaml
import json
import pickle
import pandas


def get_file_name(filepath: str, extension=False):
    """Returns the file name from a file path."""
    if extension:
        return filepath.split(os.sep)[-1]
    return filepath.split(os.sep)[-1].split(".")[0]


def read_yaml(filepath: str):
    """Reads a yaml file."""
    with open(filepath, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def write_yaml(filepath: str, obj):
    """Writes a yaml file."""
    with open(filepath, "w") as f:
        yaml.dump(obj, f)


def read_pickle(filepath: str):
    """Reads a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def write_pickle(filepath: str, obj):
    """Writes a pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def read_csv(filepath: str):
    """Reads a csv file."""
    return pandas.read_csv(filepath)


def write_csv(filepath: str, obj):
    """Writes a csv file."""
    if isinstance(obj, dict):
        obj = pandas.DataFrame(obj)
        obj.to_csv(filepath, index=False)

def write_json(filepath: str, obj):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent="\t")
