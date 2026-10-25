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
from src.constants import TARGET_COLUMN, TEXT_COLUMN, CATEGORY_COLUMN, RATING_COLUMN


@dataclass
class DataValidationConfig:

    required_columns = [CATEGORY_COLUMN, RATING_COLUMN, TARGET_COLUMN, TEXT_COLUMN]


class DataValidation:

    def __init__(self):

        self.config = DataValidationConfig()

    def validate_dataset(self, dataframe):

        try:

            logger.info("Dataset Validation Started")

            missing_columns = []

            for column in self.config.required_columns:

                if column not in dataframe.columns:

                    missing_columns.append(column)

            if missing_columns:

                raise ValueError(f"Missing Columns : {missing_columns}")

            logger.info("All Required Columns Found")

            original_count = len(dataframe)
            dataframe = dataframe.dropna(subset=self.config.required_columns).copy()
            logger.info(
                "Dropped incomplete required rows: %d", original_count - len(dataframe)
            )

            before_deduplication = len(dataframe)
            dataframe = dataframe.drop_duplicates().reset_index(drop=True)
            logger.info(
                "Dropped duplicate rows: %d", before_deduplication - len(dataframe)
            )
            if dataframe.empty:
                raise ValueError("Dataset has no complete records after validation")

            if not dataframe[RATING_COLUMN].between(1, 5).all():
                raise ValueError("Ratings must be between 1 and 5")

            logger.info("Checking Target Labels")

            labels = dataframe[TARGET_COLUMN].unique()

            logger.info(f"Labels : {labels}")

            observed = set(labels)
            valid_binary_labels = (
                observed == {"CG", "OR"} or observed == {0, 1} or observed == {0.0, 1.0}
            )
            if not valid_binary_labels:

                raise ValueError("Expected both target classes as CG/OR or binary 0/1")

            logger.info("Dataset Validation Completed")

            return dataframe

        except Exception as e:

            raise CustomException(e, sys)
