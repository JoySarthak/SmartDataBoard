import pickle
import streamlit as st
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from Smart_tool import smart_encode, getModels
import time

def train_and_evaluate(models, X, y, typeT, progress_bar, status):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    results = {}
    total_models = len(models)
    
    for i, (name, model) in enumerate(models.items()):
        with st.spinner(f"Training {name}..."):
            status.write(f"⚙️ Training model {i+1}/{total_models}: {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if typeT.lower() == "classification":
                score = accuracy_score(y_test, y_pred)
                results[name] = {"accuracy": score, "Model": model}
            else:  # Regression
                score = np.sqrt(mean_squared_error(y_test, y_pred))
                results[name] = {"rmse": score, "Model": model}
            
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

def display_results(results, best_model_name, metric_name):
    # 1. Show metrics table
    col1, col2 = st.columns([0.6, 0.4])
    with col2:
        st.subheader("📊 Model Comparison")
        metrics_df = pd.DataFrame(results).T
        st.dataframe(metrics_df.iloc[:, 0:1]) 
    
    # 2. Plot metrics
    with col1:
        st.bar_chart(metrics_df[metric_name], height=550)
    
    # 3. Show best model
    st.success(f"🏆 **Best Model**: {best_model_name} ({metric_name}: {results[best_model_name][metric_name]:.3f})")
    
    # 4. Download button for best model
    best_model = results[best_model_name]["Model"]
    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    st.download_button("Download Best Model", "best_model.pkl")

def smart_training(df, target_column):
    with st.status("🚀 Smart Training in Progress...", expanded=True) as status:
        # Initialize progress bar
        progress_bar = st.progress(0, text="Starting smart training process...")
        
        # Step 1: Starting
        status.write("🔍 Starting smart model training...")
        progress_bar.progress(5, text="Initializing...")
        time.sleep(0.5)
        
        # Step 2: Preprocess
        status.write("🔄 Encoding and preprocessing data...")
        progress_bar.progress(20, text="Preprocessing data...")
        df, problem_type = smart_encode.preprocess_data(df, target_column)
        status.write(f"✅ Detected problem type: **{problem_type}**")
        time.sleep(0.5)
        
        # Step 3: Get models
        status.write("📦 Preparing machine learning models...")
        progress_bar.progress(40, text="Loading models...")
        models = getModels.get_models(problem_type)
        model_count = len(models)
        status.write(f"🧠 Testing {model_count} different models...")
        time.sleep(0.5)
        
        # Step 4: Train & evaluate
        status.write("⚙️ Training models (this may take a while)...")
        progress_bar.progress(60, text="Training models...")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        results = train_and_evaluate(models, X, y, problem_type, progress_bar, status)
        
        # Step 5: Find best model
        status.write("🏁 Evaluating results...")
        progress_bar.progress(90, text="Finding best model...")
        best_model, best_metric, metric_name = get_best_model(results, problem_type)
        status.write(f"🎉 Best model found: **{best_model}**!")
        time.sleep(0.5)
        
        # Complete progress
        progress_bar.progress(100, text="Training complete!")
        time.sleep(0.5)
    
    # Step 6: Display results
    st.balloons()
    display_results(results, best_model, metric_name)