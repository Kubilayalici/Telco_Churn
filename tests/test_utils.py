import pandas as pd
from src.utils import DataFrameUtils


def test_grab_col_names_basic():
    df = pd.DataFrame({
        'A': ['x','y','z','x'],        # cat
        'B': [1, 2, 3, 4],             # num
        'C': [1, 1, 1, 1],             # num low-card -> cat by threshold
        'D': ['a']*4                    # cat low-card
    })
    cat_cols, num_cols = DataFrameUtils(df).grab_col_names(cat_th=3, car_th=10)
    assert 'A' in cat_cols
    assert 'D' in cat_cols
    assert 'C' in cat_cols  # numeric but low cardinality
    assert 'B' in num_cols

