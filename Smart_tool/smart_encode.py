import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def preprocess_data(df, target_column):
    encoder = OrdinalEncoder()
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
    target = df[target_column]
    problem_type = "classification" if target.nunique() <= 2 else "regression"
    
    return df, problem_type