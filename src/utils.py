import os
import sys
import pickle
import numpy as np

from src.exception import CustomException


def save_object(file_path, obj):
    """Serialize and save a Python object to disk."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a serialized Python object from disk."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def get_project_root():
    """Returns the absolute path to the proctor_project directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_artifact_path(filename):
    """Returns the absolute path to a file inside the artifacts/ directory."""
    return os.path.join(get_project_root(), "artifacts", filename)
