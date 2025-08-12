import pickle
import plotly.express as px
import streamlit as st
import altair as alt
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, precision_score,
                             r2_score,mean_squared_error,confusion_matrix,classification_report, recall_score, explained_variance_score)
import pandas as pd
import numpy as np

def train_model(X_train, y_train, model_type):
    # Check if y is continuous (regression) or categorical (classification)
    is_classification = np.issubdtype(y_train.dtype, np.integer) or len(np.unique(y_train)) < 10
    
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Logistic Regression":
        if not is_classification:
            raise ValueError("Logistic Regression requires categorical target (y). Use Linear Regression instead.")
        model = LogisticRegression()
    elif model_type == "KNN Classifier":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == "KNN Regressor":
        model = KNeighborsRegressor(n_neighbors=5)
    elif model_type == "Naive_Bayes Classifier":
        model = GaussianNB()
    elif model_type == "Decision Tree Classifier":
        model = DecisionTreeClassifier()
    elif model_type == "Decision Tree Regressor":
        model = DecisionTreeRegressor(max_depth=7)
    elif model_type == "Random Forest Classifier":
        model = RandomForestClassifier()
    elif model_type == "Random Forest Regressor":
        model = RandomForestRegressor()
    elif model_type == "AdaBoost Classifier":
        base_estimator = DecisionTreeClassifier(max_depth=9)
        model = AdaBoostClassifier(estimator=base_estimator)
    elif model_type == "AdaBoost Regressor":
        base_estimator = DecisionTreeRegressor(max_depth=9)
        model = AdaBoostRegressor(estimator=base_estimator)
    elif model_type == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier()
    elif model_type == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor()
    elif model_type == "XGBoost Classifier":
        model = XGBClassifier()
    elif model_type == "XGBoost Regressor":
        model = XGBRegressor()
    elif model_type == "Support Vector Machine (C)":
        model = SVC()
    elif model_type == "Support Vector Machine (R)":
        model = SVR()
    else:
        raise ValueError("Unknown model type")
    
    model.fit(X_train, y_train)
    return model

def evaluate_Multitarget_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    st.markdown("## :green[Model Evaluation:]")
    a, b = st.columns(2)
    c, d = st.columns(2)
    # Regression metrics only
    a.metric(
    ":green[R² Score]", 
    f"{r2_score(y_test, y_pred):.4f}", 
    help="Measures how well the model explains variance in data (1=perfect fit, 0=no better than mean).",
    border=True
    )

    b.metric(
        ":red[Mean Absolute Error]", 
        f"{mean_absolute_error(y_test, y_pred):.4f}", 
        help="Average absolute error between predictions and actual values (lower=better).",
        border=True
    )

    c.metric(
        ":blue[Variance Score]", 
        f"{explained_variance_score(y_test, y_pred):.4f}", 
        help="Proportion of variance explained by the model (1=perfect, <0=worse than random).",
        border=True
    )

    d.metric(
        ":orange[RMSE]", 
        f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}", 
        help="Root mean squared error (penalizes large errors more than MAE).",
        border=True
    )
    df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
    }).reset_index().rename(columns={"index": "idx"})

    # long form for Altair
    df_long = df.melt(id_vars="idx", value_vars=["Actual", "Predicted"],
                    var_name="Series", value_name="Value")

    scatter = alt.Chart(df_long).mark_point(filled=True, size=65).encode(
        x=alt.X("idx:Q", title="Index"),
        y=alt.Y("Value:Q"),
        color=alt.Color("Series:N", title=None)
    )

    '''line = alt.Chart(df_long).mark_line().encode(
        x="idx:Q",
        y="Value:Q",
        color=alt.Color("Series:N", title=None)
    )'''

    chart = (scatter).properties(
        title="Actual vs Predicted over Index",
        height=650,          # set height here
    ).interactive()

    st.altair_chart(chart, use_container_width=True)
    
    
def evaluate_binary_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    st.markdown("## :green[Model Evaluation:]")
    a, b = st.columns(2)
    c, d = st.columns(2)

    # Classification metrics
    a.metric(":green[Accuracy]", f"{accuracy_score(y_test, y_pred):.4f}",help="Accuracy is the most straightforward metric and measures " \
            "the proportion of all classifications that were correct. Evaluates the performance of the model",border=True)
    b.metric(":orange[Precision]", f"{precision_score(y_test, y_pred):.4f}", help="Precision measures the accuracy of the positive predictions" 
            "It answers the question: Of all the items the model labeled as positive, how many were actually positive?", border=True)
    c.metric(":blue[Recall]", f"{recall_score(y_test, y_pred):.4f}", border=True, help="Recall, also known as sensitivity or the true positive rate, " \
            "measures the model's ability to find all the actual positive instances. "
            "It answers the question: Of all the actual positives, how many did the model correctly identify?")
    d.metric(":violet[F1 Score]", f"{f1_score(y_test, y_pred):.4f}", border=True, help="The F1 score is the harmonic mean of precision and recall. " \
            "It combines both metrics into a single number, providing a balance between them.")
    
    with st.expander("View detailed: "):
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("#### :violet-background[Confusion Matrix:]")
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="GnBu")
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.markdown("#### :blue-background[Classification Report:]")
            st.container(height=40, border=False)
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            df = pd.DataFrame(report_dict).transpose()
            st.dataframe(df)
            with st.expander("How to read confusion matrix:"):
                st.markdown("▶ Top-left → True Negatives (TN) Model predicted 0 and actual was 0 (correct negative predictions).")
                st.markdown("▶ Top-Right → False Positives (FP) Model predicted 1 but actual was 0 (Type I error).")
                st.markdown("▶ Bottom-left → False Negatives (FN) Model predicted 0 but actual was 1 (Type II error).")
                st.markdown("▶ Bottom-right → True Positives (TP) Model predicted 1 and actual was 1 (correct positive predictions).")

def download_model(model):
    pkl_file = "Trained_Model.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(model, f)

    # 5. Download Button
    with open(pkl_file, "rb") as f:
        st.download_button(
            label="Download Model (PKL)",
            data=f,
            file_name=pkl_file,
            mime="application/octet-stream"
        )