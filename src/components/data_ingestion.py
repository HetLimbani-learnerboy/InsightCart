"""
Project : InsightCart

File : data_ingestion.py

Purpose :
Load raw dataset from the data/raw directory.
"""

import os
import sys

import pandas as pd

from dataclasses import dataclass

from src.logger import logger
from src.exception import CustomException
from src.constants import RAW_DATA_PATH


@dataclass
class DataIngestionConfig:

    raw_data_path: str = RAW_DATA_PATH


class DataIngestion:

    def __init__(self):

        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        try:

            logger.info("Data Ingestion Started")

            if not os.path.exists(self.config.raw_data_path):

                raise FileNotFoundError(
                    f"Dataset not found : {self.config.raw_data_path}"
                )

            dataframe = pd.read_csv(self.config.raw_data_path)

            logger.info("Dataset Loaded Successfully")

            logger.info(f"Dataset Shape : {dataframe.shape}")

            return dataframe

        except Exception as e:

            raise CustomException(e, sys)
