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
    validation_size: float = 0.20
    test_size: float = 0.20
    random_state: int = 42


@dataclass
class DataTransformationArtifacts:
    X_train: Any
    X_validation: Any
    X_test: Any
    y_train: Any
    y_validation: Any
    y_test: Any
    tfidf: TfidfVectorizer
    category_encoder: OneHotEncoder
    rating_scaler: MinMaxScaler
    transformed_dataframe: pd.DataFrame


class DataTransformation:

    def __init__(self, config: DataTransformationConfig = None):
        self.config = config or DataTransformationConfig()

    def initiate_data_transformation(
        self, dataframe: pd.DataFrame
    ) -> DataTransformationArtifacts:
        try:
            logger.info("Data Transformation Started")

            transformed_df = dataframe.copy()

            logger.info("Removing Missing Values")
            original_row_count = len(transformed_df)
            transformed_df.dropna(inplace=True)
            logger.info(
                "Dropped %s incomplete rows", original_row_count - len(transformed_df)
            )

            logger.info("Normalizing verified target labels")
            labels = transformed_df[TARGET_COLUMN]
            if set(labels.unique()).issubset({"CG", "OR"}):
                transformed_df[TARGET_COLUMN] = labels.map({"CG": 1, "OR": 0})
            elif set(labels.unique()).issubset({0, 1, 0.0, 1.0}):
                transformed_df[TARGET_COLUMN] = labels.astype(int)
            else:
                raise ValueError("Unsupported target labels after dataset validation")

            logger.info("Cleaning Review Text")
            transformed_df[CLEAN_TEXT_COLUMN] = transformed_df[TEXT_COLUMN].apply(
                clean_text
            )

            logger.info("Performing stratified train-validation-test split")
            train_validation_df, test_df = train_test_split(
                transformed_df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=transformed_df[TARGET_COLUMN],
            )
            validation_fraction = self.config.validation_size / (
                1 - self.config.test_size
            )
            train_df, validation_df = train_test_split(
                train_validation_df,
                test_size=validation_fraction,
                random_state=self.config.random_state,
                stratify=train_validation_df[TARGET_COLUMN],
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
            review_features_validation = tfidf.transform(
                validation_df[CLEAN_TEXT_COLUMN]
            )

            save_object(self.config.tfidf_path, tfidf)
            logger.info("TF-IDF Vectorizer Saved Successfully")

            logger.info("Encoding Product Categories")
            encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")

            category_features_train = encoder.fit_transform(train_df[[CATEGORY_COLUMN]])
            category_features_test = encoder.transform(test_df[[CATEGORY_COLUMN]])
            category_features_validation = encoder.transform(
                validation_df[[CATEGORY_COLUMN]]
            )

            save_object(self.config.encoder_path, encoder)
            logger.info("Category Encoder Saved Successfully")

            logger.info("Scaling Rating Feature (0 to 1)")
            scaler = MinMaxScaler()
            rating_features_train = scaler.fit_transform(train_df[[RATING_COLUMN]])
            rating_features_test = scaler.transform(test_df[[RATING_COLUMN]])
            rating_features_validation = scaler.transform(
                validation_df[[RATING_COLUMN]]
            )

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
            X_validation = hstack(
                [
                    review_features_validation,
                    category_features_validation,
                    rating_features_validation,
                ]
            )

            y_train = train_df[TARGET_COLUMN]
            y_validation = validation_df[TARGET_COLUMN]
            y_test = test_df[TARGET_COLUMN]

            logger.info(f"Training Feature Matrix Shape : {X_train.shape}")
            logger.info(f"Testing Feature Matrix Shape  : {X_test.shape}")

            return DataTransformationArtifacts(
                X_train=X_train,
                X_validation=X_validation,
                X_test=X_test,
                y_train=y_train,
                y_validation=y_validation,
                y_test=y_test,
                tfidf=tfidf,
                category_encoder=encoder,
                rating_scaler=scaler,
                transformed_dataframe=transformed_df,
            )

        except Exception as e:
            logger.error("Data Transformation Failed")
            raise CustomException(e, sys)
