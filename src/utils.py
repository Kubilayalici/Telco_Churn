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