# Analytics_sm.py
import streamlit as st
import pandas as pd
import pickle
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import is_classifier
import numpy as np

# ========================================
# 1. Detect Model Type
# ========================================
def detect_model_type(model):
    return model.__class__.__name__, model.__class__

# ========================================
# 2. Auto Encode Non-Numeric Columns
# ========================================
def encode_non_numeric(df):
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

# ========================================
# 3. Parameter Space
# ========================================
def get_param_space(model_cls):
    name = model_cls.__name__

    spaces = {
        # Classification models
        "LogisticRegression": {
            "C": (0.01, 10, "float_log"),
            "solver": (["lbfgs", "liblinear", "newton-cg", "sag", "saga"], "categorical"),
            "max_iter": (100, 500, "int"),
            "penalty": (["l1", "l2", "elasticnet", "none"], "categorical")
        },
        "RandomForestClassifier": {
            "n_estimators": (50, 300, "int"),
            "max_depth": (2, 20, "int_or_none"),
            "min_samples_split": (2, 10, "int"),
            "min_samples_leaf": (1, 5, "int"),
            "max_features": (["sqrt", "log2", None], "categorical"),
            "bootstrap": ([True, False], "categorical")
        },
        "GradientBoostingClassifier": {
            "n_estimators": (50, 300, "int"),
            "learning_rate": (0.01, 0.2, "float_log"),
            "max_depth": (2, 10, "int"),
            "min_samples_split": (2, 10, "int"),
            "min_samples_leaf": (1, 5, "int"),
            "max_features": (["sqrt", "log2", None], "categorical")
        },
        "AdaBoostClassifier": {
            "n_estimators": (50, 300, "int"),
            "learning_rate": (0.01, 1.0, "float"),
            "algorithm": (["SAMME", "SAMME.R"], "categorical")
        },
        "DecisionTreeClassifier": {
            "max_depth": (2, 20, "int_or_none"),
            "min_samples_split": (2, 10, "int"),
            "min_samples_leaf": (1, 5, "int"),
            "max_features": (["sqrt", "log2", None], "categorical"),
            "criterion": (["gini", "entropy"], "categorical")
        },
        "KNeighborsClassifier": {
            "n_neighbors": (3, 15, "int"),
            "weights": (["uniform", "distance"], "categorical"),
            "p": (1, 2, "int")
        },
        "SVC": {
            "C": (0.01, 10, "float_log"),
            "kernel": (["linear", "rbf", "poly", "sigmoid"], "categorical"),
            "gamma": (["scale", "auto"], "categorical"),
            "degree": (2, 5, "int")
        },
        "XGBClassifier": {
            "n_estimators": (50, 300, "int"),
            "max_depth": (2, 10, "int"),
            "learning_rate": (0.01, 0.3, "float_log"),
            "subsample": (0.5, 1.0, "float"),
            "colsample_bytree": (0.5, 1.0, "float"),
            "gamma": (0, 5, "float"),
            "min_child_weight": (1, 10, "int")
        },

        # Regression models
        "LinearRegression": {
            "fit_intercept": ([True, False], "categorical")
        },
        "RandomForestRegressor": {
            "n_estimators": (50, 300, "int"),
            "max_depth": (2, 20, "int_or_none"),
            "min_samples_split": (2, 10, "int"),
            "min_samples_leaf": (1, 5, "int"),
            "max_features": (["sqrt", "log2", None], "categorical"),
            "bootstrap": ([True, False], "categorical")
        },
        "GradientBoostingRegressor": {
            "n_estimators": (50, 300, "int"),
            "learning_rate": (0.01, 0.2, "float_log"),
            "max_depth": (2, 10, "int"),
            "min_samples_split": (2, 10, "int"),
            "min_samples_leaf": (1, 5, "int"),
            "max_features": (["sqrt", "log2", None], "categorical"),
            "loss": (["squared_error", "absolute_error", "huber", "quantile"], "categorical")
        },
        "KNeighborsRegressor": {
            "n_neighbors": (3, 15, "int"),
            "weights": (["uniform", "distance"], "categorical"),
            "p": (1, 2, "int")
        },
        "SVR": {
            "C": (0.01, 10, "float_log"),
            "kernel": (["linear", "rbf", "poly", "sigmoid"], "categorical"),
            "gamma": (["scale", "auto"], "categorical"),
            "degree": (2, 5, "int"),
            "epsilon": (0.01, 0.2, "float")
        },
        "XGBRegressor": {
            "n_estimators": (50, 300, "int"),
            "max_depth": (2, 10, "int"),
            "learning_rate": (0.01, 0.3, "float_log"),
            "subsample": (0.5, 1.0, "float"),
            "colsample_bytree": (0.5, 1.0, "float"),
            "gamma": (0, 5, "float"),
            "min_child_weight": (1, 10, "int")
        },
        "AdaBoostRegressor": {
            "n_estimators": (50, 300, "int"),
            "learning_rate": (0.01, 1.0, "float"),
            "loss": (["linear", "square", "exponential"], "categorical")
        },
        "DecisionTreeRegressor": {
            "max_depth": (2, 20, "int_or_none"),
            "min_samples_split": (2, 10, "int"),
            "min_samples_leaf": (1, 5, "int"),
            "max_features": (["sqrt", "log2", None], "categorical"),
            "criterion": (["squared_error", "friedman_mse", "absolute_error"], "categorical")
        }
    }
    return spaces.get(name, {})

# ========================================
# 4. Suggest Param Helper (fixed order)
# ========================================
def suggest_param(trial, name, value):
    if isinstance(value, tuple) and value[-1] == "categorical":
        return trial.suggest_categorical(name, value[0])
    elif isinstance(value, tuple):
        if value[-1] == "int":
            return trial.suggest_int(name, value[0], value[1])
        elif value[-1] == "float":
            return trial.suggest_float(name, value[0], value[1])
        elif value[-1] == "float_log":
            return trial.suggest_float(name, value[0], value[1], log=True)
        elif value[-1] == "int_or_none":
            return trial.suggest_categorical(name, [None] + list(range(value[0], value[1] + 1)))
    elif isinstance(value, list):
        return trial.suggest_categorical(name, value)
    return None

# ========================================
# 5. Objective Function with auto scoring
# ========================================
from sklearn.base import is_classifier

def tune_model(model_cls, param_space, X, y, n_trials=20):
    def objective(trial):
        params = {pname: suggest_param(trial, pname, pval) for pname, pval in param_space.items()}
        model = model_cls(**params)

        if is_classifier(model):
            scoring = "accuracy"
        else:
            scoring = "neg_root_mean_squared_error"  # RMSE but negative

        score = cross_val_score(model, X, y, cv=3, scoring=scoring).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # Adjust score for regression so RMSE is positive
    best_score = study.best_value
    if not is_classifier(model_cls()):
        best_score = -best_score  # convert negative RMSE to positive

    return study.best_params, best_score

# ========================================
# 6. Save Model
# ========================================
def save_model(model, filename="tuned_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    return filename

# ========================================
# 7. Streamlit UI
# ========================================
st.title("📈 Auto-Tuning hyperparameters for ML Models : ")
st.write("Increase the performance of your machine learning model using Auto-Tuning Tailored for your dataset")

uploaded_model_file = st.file_uploader("Upload trained model (.pkl)", type=["pkl"])
uploaded_data_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_data_file:
    df = pd.read_csv(uploaded_data_file)
    target_col = st.selectbox("Select the target on which the model was trained:", df.columns, index=0)
    n_trials = st.number_input("Number of tuning trials", min_value=5, max_value=100, value=20)

    if st.button(":material/network_intelligence: AI Auto Tune"):
        if uploaded_model_file and target_col:
            try:
                model = pickle.load(uploaded_model_file)

                X = df.drop(columns=[target_col])
                y = df[target_col]

                # Encode non-numeric features
                X = encode_non_numeric(X)

                # Detect model
                model_name, model_cls = detect_model_type(model)
                st.write(f"**Detected Model:** {model_name}")

                # Get parameter space
                param_space = get_param_space(model_cls)
                if not param_space:
                    st.error("❌ This model type is not supported yet for auto-tuning.")
                else:
                    with st.spinner("Tuning in progress..."):
                        best_params, best_score = tune_model(model_cls, param_space, X, y, n_trials=n_trials)

                    st.success(f"✅ Best Score: {best_score:.4f}")
                    st.write("**Best Parameters:**", best_params)

                    tuned_model = model_cls(**best_params)
                    tuned_model.fit(X, y)

                    tuned_filename = save_model(tuned_model)
                    with open(tuned_filename, "rb") as f:
                        st.download_button("⬇️ Download Tuned Model", f, file_name="tuned_model.pkl")

            except Exception as e:
                st.error(f"Error: {str(e)}")
