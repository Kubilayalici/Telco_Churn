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
        """
        Grab categorical and numerical column names based on thresholds.
        """
        categorical_cols = [col for col in self.dataframe.columns if self.dataframe[col].dtypes == 'O']
        numerical_cols = [col for col in self.dataframe.columns if self.dataframe[col].dtypes in ['int64', 'float64']]

        cat_cols = [col for col in categorical_cols if self.dataframe[col].nunique() < cat_th]
        num_cols = [col for col in numerical_cols if self.dataframe[col].nunique() > car_th]

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