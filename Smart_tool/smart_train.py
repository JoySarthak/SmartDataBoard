import pickle
import streamlit as st
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from Smart_tool import smart_encode, getModels
import time

def train_and_evaluate(models, X, y, typeT, progress_bar, status):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = {}
    total_models = len(models)
    
    for i, (name, model) in enumerate(models.items()):
        with st.spinner(f"Training {name}..."):
            status.write(f":material/group_work: Training model {i+1}/{total_models}: {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if typeT.lower() == "classification":
                score = accuracy_score(y_test, y_pred)
                score_f1 = f1_score(y_test, y_pred)
                results[name] = {"accuracy": score, "Model": model, "F1" : score_f1}
            else:  # Regression
                score = np.sqrt(mean_squared_error(y_test, y_pred))
                score_r2 = r2_score(y_test, y_pred)
                results[name] = {"rmse": score, "Model": model, "R2": score_r2}
            
            progress = (i + 1) / total_models
            progress_bar.progress(min(progress, 1.0))
            time.sleep(0.5)  # Small delay for better UX
    
    return results

def get_best_model(results, problem_type):
    if problem_type.lower() == "classification":
        best_model_name = max(results, key=lambda x: results[x]["accuracy"])
        best_metric = results[best_model_name]["accuracy"]
        metric_name = "accuracy"
    else:  # Regression
        best_model_name = min(results, key=lambda x: results[x]["rmse"])
        best_metric = results[best_model_name]["rmse"]
        metric_name = "rmse"
    
    return best_model_name, best_metric, metric_name

def display_results(results, best_model_name, metric_name, best_metric):
    st.code(f"Best Model found is {best_model_name} yielding : {metric_name} : {best_metric}")
    metrics_df = pd.DataFrame(results).T
    with st.container(border=True):
        col1,col2 = st.columns(2)
        col1.markdown(f"#### :green[{metric_name}] comparison visualisation:")
        col1.line_chart(metrics_df[[metric_name]], height=600, color="#57e140")
        if metric_name == "rmse":
            col2.markdown(f"#### :violet[R2] comparison visualisation:")
            col2.line_chart(metrics_df["R2"], height=600, color="#9743e6")
            st.markdown("##### Scaled R2 with RMSE Visualisation", help="Scaling helps to avoid distortion and provides a general understanding")
            scale_factor = metrics_df["rmse"].max() / 2  
            metrics_df["R2_scaled"] = metrics_df["R2"] * scale_factor
            st.markdown(f"Scale factor : :green[{scale_factor}]", help="Divide scaled value with scale_factor to get actual") 
            st.bar_chart(metrics_df.set_index("Model")[[metric_name, "R2_scaled"]], height=750, stack=False)
        else:
            col2.markdown(f"#### :violet[F1] comparison visualisation:")
            col2.line_chart(metrics_df["F1"], height=600, color="#9743e6")
            st.bar_chart(metrics_df[[metric_name, "F1"]], stack=False, height=650)

    st.success(f"🏆 **Best Model**: {best_model_name} ({metric_name}: {results[best_model_name][metric_name]:.3f})")
    # 4. Download button for best model
    best_model = results[best_model_name]["Model"]
    model_bytes = pickle.dumps(best_model)
    st.download_button(
        label="Download Best Model",
        data=model_bytes,
        file_name="best_model.pkl",
        mime="application/octet-stream"
    )

def smart_training(df, target_column):
    with st.status(":material/network_intelligence: Smart Training in Progress...", expanded=True) as status:
        # Initialize progress bar
        progress_bar = st.progress(0, text="Starting smart training process...")
        
        # Step 1: Starting
        status.write(":material/progress_activity: Starting smart model training...")
        progress_bar.progress(5, text="Initializing...")
        time.sleep(0.5)
        
        # Step 2: Preprocess
        status.write(":material/cycle: Encoding and preprocessing data...")
        progress_bar.progress(20, text="Preprocessing data...")
        df, problem_type = smart_encode.preprocess_data(df, target_column)
        status.write(f":material/token: Detected Target Prediction type: **{problem_type}**")
        time.sleep(0.5)
        
        # Step 3: Get models
        status.write(":material/data_thresholding: Preparing machine learning models...")
        progress_bar.progress(40, text="Loading models...")
        models = getModels.get_models(problem_type)
        model_count = len(models)
        status.write(f":material/search_insights: Testing {model_count} different models...")
        time.sleep(0.5)
        
        # Step 4: Train & evaluate
        status.write(":material/model_training: Training models (this may take a while)...")
        progress_bar.progress(60, text="Training models...")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        results = train_and_evaluate(models, X, y, problem_type, progress_bar, status)
        
        # Step 5: Find best model
        status.write(":material/compare_arrows: Comparing Models & Evaluating results...")
        progress_bar.progress(90, text="Finding best model...")
        best_model, best_metric, metric_name = get_best_model(results, problem_type)
        status.write(f":material/celebration: Best model found: **{best_model}**!")
        time.sleep(0.5)
        
        # Complete progress
        progress_bar.progress(100, text="Training complete!")
        time.sleep(0.5)
    
    # Step 6: Display results
    display_results(results, best_model, metric_name, best_metric)