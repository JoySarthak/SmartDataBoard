import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from PIL import Image # <-- Import the Image module from Pillow

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
)

# Replace with your actual model fetching function
from Smart_tool import getModels 

# PDF Generation
from fpdf import FPDF

# --- 1. Best Model Identification Function ---
def get_best_model(results, problem_type):
    """
    Identifies the best model from a dictionary of results based on a primary metric.
    """
    if not results:
        return None, None, None

    if problem_type.lower() == "classification":
        best_model_name = max(results, key=lambda x: results[x]["Accuracy"])
        best_metric = results[best_model_name]["Accuracy"]
        metric_name = "Accuracy"
    else:
        best_model_name = min(results, key=lambda x: results[x]["Root Mean Squared Error"])
        best_metric = results[best_model_name]["Root Mean Squared Error"]
        metric_name = "Root Mean Squared Error"
    
    return best_model_name, best_metric, metric_name

# --- 2. Professional PDF Report Generation Function ---
class PDF(FPDF):
    def header(self):
        try:
            self.set_font('Cascadia', 'B', 12)
        except RuntimeError:
            self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'Automated Machine Learning Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        try:
            self.set_font('Cascadia', 'I', 8)
        except RuntimeError:
            self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_professional_report(results, problem_type, best_model_info, target_column, file_name):
    """
    Generates a professional PDF report with custom fonts, charts, and analysis.
    """
    best_model_name, best_metric, metric_name = best_model_info
    
    pdf = PDF()
    
    # --- Font Setup (Simplified) ---
    try:
        # Looking for fonts in the main project directory
        pdf.add_font('Cascadia', '', 'CascadiaCode.ttf', uni=True)
        pdf.add_font('Cascadia', 'B', 'CascadiaCodeB.ttf', uni=True)
        pdf.add_font('Cascadia', 'I', 'CascadiaCodeItalic.ttf', uni=True)
        font_family = 'Cascadia'
    except FileNotFoundError:
        st.warning("Cascadia Code font files not found in the main project directory. Falling back to Helvetica.")
        font_family = 'Helvetica'
    
    pdf.add_page()
    
    # --- Title Page ---
    pdf.set_font(font_family, 'B', 24)
    pdf.cell(0, 20, 'Model Performance Analysis', ln=1, align='C')
    pdf.set_font(font_family, '', 12)
    pdf.cell(0, 10, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align='C')
    pdf.cell(0, 10, f"Dataset: '{file_name}'", ln=1, align='C')
    pdf.cell(0, 10, f"Target Variable: '{target_column}'", ln=1, align='C')
    pdf.cell(0, 10, f"Analysis Type: {problem_type.title()}", ln=1, align='C')
    pdf.ln(20)

    # --- Summary Section ---
    pdf.set_font(font_family, 'B', 16)
    pdf.cell(0, 10, '1. Executive Summary', ln=1)
    pdf.set_font(font_family, '', 11)
    summary_text = (
        f"This report details the performance of several machine learning models. "
        f"Based on the analysis, the best performing model is '{best_model_name}' "
        f"with a top {metric_name} of {best_metric:.4f}."
    )
    pdf.multi_cell(0, 8, summary_text)
    pdf.ln(10)

    # --- Comparison Chart Section ---
    pdf.set_font(font_family, 'B', 16)
    pdf.cell(0, 10, '2. Model Performance Comparison', ln=1)
    
    plot_data = []
    for model, metrics in results.items():
        plot_data.append({'Model': model, metric_name: metrics[metric_name]})
    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    bar_plot = sns.barplot(x='Model', y=metric_name, data=df_plot, palette='viridis')
    plt.title(f'Model Comparison by {metric_name}', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.4f'), 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha = 'center', va = 'center', 
                           xytext = (0, 9), 
                           textcoords = 'offset points')

    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    
    # --- THE ROBUST FIX using Pillow ---
    # Open the image from the buffer using Pillow
    chart_image = Image.open(buf)
    # Pass the Pillow image object to fpdf2, which is more reliable
    pdf.image(chart_image, x=10, w=pdf.w - 20)
    
    plt.close()
    buf.close() # Good practice to close the buffer
    pdf.ln(5)

    # --- Detailed Metrics Section ---
    pdf.add_page()
    pdf.set_font(font_family, 'B', 16)
    pdf.cell(0, 10, '3. Detailed Model Metrics', ln=1)
    
    for model_name, metrics in results.items():
        pdf.set_font(font_family, 'B', 14)
        pdf.cell(0, 10, f"-> {model_name}", ln=1)
        pdf.set_font(font_family, '', 11)
        for metric, value in metrics.items():
            pdf.cell(0, 8, f"{'':<5}- {metric.replace('_', ' ').title()}: {value:.4f}", ln=1)
        pdf.ln(5)

    return bytes(pdf.output())

# --- Your Main Streamlit App Logic (Remains the same) ---

st.title("Generate reports")
st.write("Generate pdf report with proper metrics information and comparison charts by default all models are selected unless mentioned specifically")
mode = st.radio("Select Mode : ",["regression","classification"], horizontal=True)

prompt = st.chat_input(
    "Upload csv / Type Generate report along target column name pls adhere to case",
    accept_file=True,
    file_type=["csv"],
)

if prompt and "files" in prompt and prompt["files"]:
    uploaded_file = prompt["files"][0]
    csv_filename = uploaded_file.name
    df_uploaded = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    target_column = df_uploaded.columns[-1]
    
    prompt_text_lower = prompt['text'].lower()
    for col_name in df_uploaded.columns:
        if col_name.lower() in prompt_text_lower:
            target_column = col_name
            break

    with st.chat_message("user"):
        st.markdown(f"Preview of **{csv_filename}**")
        st.dataframe(df_uploaded.head(5))
        st.markdown(f"Encoding dataset... Target column is **{target_column}**")
        encoder = OrdinalEncoder()
        categorical_cols = df_uploaded.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            df_uploaded[categorical_cols] = encoder.fit_transform(df_uploaded[categorical_cols])
        
        target = df_uploaded[target_column]
        def_mode = "binary" if target.nunique() <= 2 else "macro"

    models = getModels.get_models(mode)

    if "generate" in prompt['text'].lower():
        with st.spinner(f"Training {len(models)} models..."):
            x = df_uploaded.drop(columns=[target_column])
            y = df_uploaded[target_column]
            results = {} 
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            progress_bar = st.progress(0, "Initializing...")
            for i, (name, model) in enumerate(models.items()):
                progress_bar.progress((i) / len(models), text=f"Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if mode == "classification":
                    results[name] = {
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "F1 Score": f1_score(y_test, y_pred, average=def_mode),
                        "Precision": precision_score(y_test, y_pred, average=def_mode),
                        "Recall": recall_score(y_test, y_pred, average=def_mode)
                    }
                else: # Regression
                    results[name] = {
                        "R2 Score": r2_score(y_test, y_pred),
                        "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
                        "Root Mean Squared Error": np.sqrt(mean_squared_error(y_test, y_pred)),
                        "Explained Variance Score": explained_variance_score(y_test, y_pred)
                    }
            progress_bar.progress(1.0, "Evaluation complete!")
        
        st.success("All models have been trained and evaluated!")
        
        st.markdown("### Download Your Report")
        
        best_model_info = get_best_model(results, mode)
        
        pdf_data = create_professional_report(results, mode, best_model_info, target_column, csv_filename)
        
        st.download_button(
            label="⬇️ Download Professional Report",
            data=pdf_data,
            file_name=f"professional_report_{mode}.pdf",
            mime="application/pdf"
        )

