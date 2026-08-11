"""
Project : InsightCart

File : data_validation.py

Purpose :
Validate the dataset before preprocessing.
"""

import sys

from dataclasses import dataclass

from src.logger import logger
from src.exception import CustomException
from src.constants import (

    TARGET_COLUMN,

    TEXT_COLUMN,

    CATEGORY_COLUMN,

    RATING_COLUMN

)


@dataclass
class DataValidationConfig:

    required_columns = [

        CATEGORY_COLUMN,

        RATING_COLUMN,

        TARGET_COLUMN,

        TEXT_COLUMN

    ]


class DataValidation:

    def __init__(self):

        self.config = DataValidationConfig()

    def validate_dataset(
        self,
        dataframe
    ):

        try:

            logger.info(
                "Dataset Validation Started"
            )

            missing_columns = []

            for column in self.config.required_columns:

                if column not in dataframe.columns:

                    missing_columns.append(column)

            if missing_columns:

                raise ValueError(

                    f"Missing Columns : {missing_columns}"

                )

            logger.info(
                "All Required Columns Found"
            )

            logger.info(
                "Checking Missing Values"
            )

            missing_values = dataframe.isnull().sum()

            logger.info(

                f"\n{missing_values}"

            )

            logger.info(
                "Checking Duplicate Records"
            )

            duplicate_count = dataframe.duplicated().sum()

            logger.info(

                f"Duplicate Rows : {duplicate_count}"

            )

            logger.info(
                "Checking Target Labels"
            )

            labels = dataframe[TARGET_COLUMN].unique()

            logger.info(

                f"Labels : {labels}"

            )

            if not set(labels).issubset({"CG", "OR"}):

                raise ValueError(

                    "Unexpected Label Found"

                )

            logger.info(
                "Dataset Validation Completed"
            )

            return dataframe

        except Exception as e:

            raise CustomException(
                e,
                sys
            )