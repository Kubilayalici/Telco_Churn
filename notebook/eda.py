# İŞ PROBLEMİ:
    # Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirmek

# Gerekli Kütüphaneler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from src.utils import DataFrameUtils

# Veri Yükleme
def load_data():
    df = pd.read_csv(r"notebook\data\Telco-Customer-Churn.csv")
    return df


df = load_data()
df.head()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')  # Convert TotalCharges to numeric, handling errors
df["SeniorCitizen"] = df["SeniorCitizen"].astype('object')  # Convert SeniorCitizen to object type
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})  # Convert Churn to binary


# EDA Fonksiyonu
DataFrameUtils(df).check_data()
DataFrameUtils(df).grab_col_names(cat_th=10, car_th=20)

cat_cols, num_cols = DataFrameUtils(df).grab_col_names(cat_th=10, car_th=20)

# Kategorik Değişkenlerin Analizi

def cat_summary(dataframe, col_name, plot=False):
    """
    Categorical variable summary.
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)

# Kategorik Değişkenlerin Hedef Değişken ile İlişkisi
def target_summary_with_cat(dataframe, target, categorical_col, plot=False):
    """
    Categorical variable analysis with respect to the target variable.
    """
    print(pd.DataFrame({categorical_col: dataframe[categorical_col].value_counts(),
                        "Target Mean": dataframe.groupby(categorical_col)[target].mean()}))
    if plot:
        sns.barplot(x=categorical_col, y=target, data=dataframe)
        plt.title(f"{categorical_col} vs {target}")
        plt.show()

df["Churn"].head()
for col in cat_cols:
    target_summary_with_cat(df, "Churn", col, plot=True)

# Sayısal Değişkenlerin Analizi

def num_summary(dataframe, numerical_col, plot=False):
    """
    Numerical variable summary.
    """
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        sns.histplot(dataframe[numerical_col], kde=True)
        plt.title(f"Distribution of {numerical_col}")
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

# Sayısal Değişkenlerin Hedef Değişken ile İlişkisi
def target_summary_with_num(dataframe, target, numerical_col, plot=False):
    """
    Numerical variable analysis with respect to the target variable.
    """
    print(dataframe.groupby(target).agg({numerical_col: ['mean', 'std', 'min', 'max', 'median']}))
    print("####################################################################################")
    if plot:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=target, y=numerical_col, data=dataframe)
        plt.title(f"{numerical_col} vs {target}")
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        
for col in num_cols:
    target_summary_with_num(df, "Churn", col, plot=True)


# Korelasyon Analizi


def corr_analysis(dataframe,numerical_cols=None):
    """
    Correlation analysis.
    """
    corr = dataframe[numerical_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix")
    plt.show()

corr_analysis(df, numerical_cols=num_cols)

#######################
# Yüksek Korelasyonlu Değişkenlerin Silinmesi
#######################

def high_correlated_cols(dataframe, numerical_columns, plot=False, corr_th=0.80):
    corr = dataframe[numerical_columns].corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (12, 10)})
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()
    return drop_list

drop_list = high_correlated_cols(df, numerical_columns=num_cols, plot=True)

df = df.drop(drop_list, axis=1)

num_cols = [col for col in num_cols if col not in drop_list]
_ = high_correlated_cols(df, numerical_columns=num_cols, plot=True)



