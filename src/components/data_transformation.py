import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import DataFrameUtils
import joblib

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, df):
        try:
            logging.info("Data Transformation initiated")


            # Kategorik ve sayısal sütunları bul
            df_utils = DataFrameUtils(df)
            logging.info("Data checked for basic statistics and information")

            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
            logging.info("Data types converted and target variable mapped")

            # Kategorik ve sayısal sütun isimlerini al
            logging.info("Grabbing column names for categorical and numerical features")
            cat_cols, num_cols = df_utils.grab_col_names(cat_th=10, car_th=20)
            logging.info(f"Categorical columns: {cat_cols}")
            logging.info(f"Numerical columns: {num_cols}")

            # Outlier işlemleri
            for col in num_cols:
                if df_utils.check_outlier(col):
                    df_utils.replace_with_thresholds(col)
                    logging.info(f"Outliers in {col} capped")

            # Rare encoding işlemleri
            for col in cat_cols:
                rare_categories = df_utils.rare_analysis(col)
                if len(rare_categories) > 0:
                    df_utils.rare_encoder(col)

            # Pipeline'lar
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer([
                ('num', num_pipeline, num_cols),
                ('cat', cat_pipeline, cat_cols)
            ])

            logging.info("Preprocessor pipeline created successfully")
            return preprocessor, cat_cols, num_cols

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        logging.info("Data Transformation started")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Boş hücreleri ve sadece boşluk içeren hücreleri NaN olarak işaretle
            train_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
            test_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

            # customerID gibi anlamsız sütunları düşür
            drop_cols = ['customerID']
            train_df.drop(columns=drop_cols, inplace=True, errors='ignore')
            test_df.drop(columns=drop_cols, inplace=True, errors='ignore')

            logging.info("Train and Test datasets loaded successfully")

            target_column = 'Churn'

            # Preprocessor ve feature listelerini al
            preprocessor, cat_cols, num_cols = self.get_data_transformer_object(train_df)

            # Target sütunu kategoriklerde varsa çıkar
            if target_column in cat_cols:
                cat_cols.remove(target_column)

            input_feature_train_df = train_df.drop(columns=[target_column])
            input_feature_test_df = test_df.drop(columns=[target_column])
            logging.info("Input features for train and test set prepared")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Input features for train set transformed")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Input features for test set transformed")

            train_arr = np.c_[input_feature_train_arr, np.array(train_df[target_column])]
            logging.info("Train array created")
            test_arr = np.c_[input_feature_test_arr, np.array(test_df[target_column])]
            logging.info("Test array created")

            joblib.dump(preprocessor, self.data_transformation_config.preprocessor_obj_file_path)
            logging.info(f"Preprocessor object saved at {self.data_transformation_config.preprocessor_obj_file_path}")
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)