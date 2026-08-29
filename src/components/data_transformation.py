"""
Project : InsightCart

File : data_transformation.py

Purpose :
Perform data preprocessing and feature engineering.
"""

import sys
from dataclasses import dataclass
from typing import Any

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.constants import (
    CATEGORY_COLUMN,
    CLEAN_TEXT_COLUMN,
    ENCODER_PATH,
    RATING_COLUMN,
    SCALER_PATH,
    TARGET_COLUMN,
    TEXT_COLUMN,
    TFIDF_PATH,
)
from src.exception import CustomException
from src.logger import logger
from src.utils.common import save_object
from src.utils.preprocessing import clean_text


@dataclass
class DataTransformationConfig:
    max_features: int = 10000
    ngram_range: tuple = (1, 2)
    min_df: int = 5
    max_df: float = 0.90
    tfidf_path: str = TFIDF_PATH
    encoder_path: str = ENCODER_PATH
    scaler_path: str = SCALER_PATH


@dataclass
class DataTransformationArtifacts:
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    tfidf: TfidfVectorizer
    category_encoder: OneHotEncoder
    rating_scaler: MinMaxScaler
    transformed_dataframe: pd.DataFrame


class DataTransformation:

    def __init__(self):
        self.config = DataTransformationConfig()

    def initiate_data_transformation(
        self, dataframe: pd.DataFrame
    ) -> DataTransformationArtifacts:
        try:
            logger.info("Data Transformation Started")

            transformed_df = dataframe.copy()

            logger.info("Removing Missing Values")
            transformed_df.dropna(inplace=True)

            logger.info("Encoding Target Labels (CG -> 1, OR -> 0)")
            transformed_df[TARGET_COLUMN] = transformed_df[TARGET_COLUMN].map(
                {"CG": 1, "OR": 0}
            )

            logger.info("Cleaning Review Text")
            transformed_df[CLEAN_TEXT_COLUMN] = transformed_df[TEXT_COLUMN].apply(
                clean_text
            )

            logger.info("Performing Train-Test Split")
            train_df, test_df = train_test_split(
                transformed_df,
                test_size=0.20,
                random_state=42,
                stratify=transformed_df[TARGET_COLUMN],
            )

            logger.info("Generating TF-IDF Features")
            tfidf = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                min_df=self.config.min_df,
                max_df=self.config.max_df,
            )

            review_features_train = tfidf.fit_transform(train_df[CLEAN_TEXT_COLUMN])
            review_features_test = tfidf.transform(test_df[CLEAN_TEXT_COLUMN])

            save_object(self.config.tfidf_path, tfidf)
            logger.info("TF-IDF Vectorizer Saved Successfully")

            logger.info("Encoding Product Categories")
            encoder = OneHotEncoder(handle_unknown="ignore")

            category_features_train = encoder.fit_transform(train_df[[CATEGORY_COLUMN]])
            category_features_test = encoder.transform(test_df[[CATEGORY_COLUMN]])

            save_object(self.config.encoder_path, encoder)
            logger.info("Category Encoder Saved Successfully")

            logger.info("Scaling Rating Feature (0 to 1)")
            scaler = MinMaxScaler()
            rating_features_train = scaler.fit_transform(train_df[[RATING_COLUMN]])
            rating_features_test = scaler.transform(test_df[[RATING_COLUMN]])

            save_object(self.config.scaler_path, scaler)
            logger.info("Rating Scaler Saved Successfully")

            logger.info("Combining All Features")
            X_train = hstack(
                [
                    review_features_train,
                    category_features_train,
                    rating_features_train,
                ]
            )

            X_test = hstack(
                [
                    review_features_test,
                    category_features_test,
                    rating_features_test,
                ]
            )

            y_train = train_df[TARGET_COLUMN]
            y_test = test_df[TARGET_COLUMN]

            logger.info(f"Training Feature Matrix Shape : {X_train.shape}")
            logger.info(f"Testing Feature Matrix Shape  : {X_test.shape}")

            return DataTransformationArtifacts(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                tfidf=tfidf,
                category_encoder=encoder,
                rating_scaler=scaler,
                transformed_dataframe=transformed_df,
            )

        except Exception as e:
            logger.error("Data Transformation Failed")
            raise CustomException(e, sys)
