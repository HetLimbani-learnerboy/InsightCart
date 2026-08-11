"""
Project : InsightCart
File    : common.py
Purpose : Common reusable utility functions
"""

import os
import pickle
import joblib

from src.logger import logger


def create_directories(paths):

    for path in paths:

        os.makedirs(
            path,
            exist_ok=True
        )

        logger.info(
            f"Directory Created : {path}"
        )


def save_object(
    file_path,
    obj
):

    joblib.dump(
        obj,
        file_path
    )

    logger.info(
        f"Object Saved : {file_path}"
    )


def load_object(
    file_path
):

    logger.info(
        f"Loading Object : {file_path}"
    )

    return joblib.load(
        file_path
    )


def save_pickle(
    file_path,
    obj
):

    with open(
        file_path,
        "wb"
    ) as file:

        pickle.dump(
            obj,
            file
        )

    logger.info(
        f"Pickle Saved : {file_path}"
    )


def load_pickle(
    file_path
):

    with open(
        file_path,
        "rb"
    ) as file:

        return pickle.load(
            file
        )