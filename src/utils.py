import numpy as np
import pandas as pd

class DataFrameUtils:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def check_data(self, head=5):
        """
        Check the dataframe for basic statistics and information.
        """
        print(self.dataframe.shape)
        print(self.dataframe.columns)
        print(self.dataframe.dtypes)
        print(self.dataframe.head(head))
        print(self.dataframe.tail(head))
        print(self.dataframe.isnull().sum())
        print(self.dataframe.describe().T)

    def grab_col_names(self, cat_th=10, car_th=20):
        """Return (cat_cols, num_cols) with typical kaggle-style thresholds.

        - cat_cols: object dtype + numeric columns with low cardinality (< cat_th), excluding high-cardinality categoricals (> car_th)
        - num_cols: numeric dtype columns excluding those treated as categorical (num_but_cat)
        """
        df = self.dataframe
        categorical_cols = [col for col in df.columns if df[col].dtypes == 'O']
        numerical_cols = [col for col in df.columns if df[col].dtypes != 'O']

        num_but_cat = [col for col in numerical_cols if df[col].nunique() < cat_th]
        cat_but_car = [col for col in categorical_cols if df[col].nunique() > car_th]

        cat_cols = [col for col in (categorical_cols + num_but_cat) if col not in cat_but_car]
        num_cols = [col for col in numerical_cols if col not in num_but_cat]

        return cat_cols, num_cols

    def outlier_thresholds(self, variable):
        """
        Calculate the lower and upper bounds for outliers in a given variable.
        """
        Q1 = self.dataframe[variable].quantile(0.25)
        Q3 = self.dataframe[variable].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return lower_bound, upper_bound

    def check_outlier(self, variable):
        """
        Check if there are outliers in the given variable of the dataframe.
        """
        lower_bound, upper_bound = self.outlier_thresholds(variable)
        if self.dataframe[(self.dataframe[variable] < lower_bound) | (self.dataframe[variable] > upper_bound)].any(axis=None):
            print(f"Outliers detected in {variable}")
            return True
        else:
            print(f"No outliers detected in {variable}")
            return False

    def replace_with_thresholds(self, variable):
        """
        Replace outliers in the given variable with the lower and upper bounds.
        """
        lower_bound, upper_bound = self.outlier_thresholds(variable)
        self.dataframe[variable] = np.where(
            self.dataframe[variable] < lower_bound, lower_bound,
            np.where(self.dataframe[variable] > upper_bound, upper_bound, self.dataframe[variable])
        )
    def rare_analysis(self, variable):
        """ Analyze the rare categories in a categorical variable. """
        temp = self.dataframe[variable].value_counts() / len(self.dataframe)
        rare_categories = temp[temp < 0.01].index
        print(f"Rare categories in {variable}: {rare_categories.tolist()}")
        print(f"Number of rare categories: {len(rare_categories)}")
        return rare_categories

    def rare_encoder(self, variable):
        """ Encode rare categories in a categorical variable. """
        temp = self.dataframe[variable].value_counts() / len(self.dataframe)
        rare_categories = temp[temp < 0.01].index
        self.dataframe[variable] = np.where(self.dataframe[variable].isin(rare_categories), 'Rare', self.dataframe[variable])       
        print(f"Rare categories in {variable} encoded as 'Rare'")       

        return self.dataframe
