# 📊 Smartboard: Dataset Analyser & Model Trainer

> 🚀 An interactive **Streamlit** platform for **Datascience** and **machine learning** model training — all without writing code.

> Smartboard is an interactive data analysis and machine learning platform built with [Streamlit](https://streamlit.io/). It enables users to upload datasets, perform exploratory data analysis, clean data, encode features, visualize trends, and train machine learning models—all through a user-friendly web interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-ff4b4b?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Category | Features |
|----------|----------|
| 📄 **Data Overview** | Upload CSV, view dataframe & stats, visualize missing values (bar, pie, line), download processed datasets |
| 🧹 **Data Cleaning** | Drop/replace NaN, remove duplicates, drop columns, remove outliers |
| 🔍 **Subjective Analysis** | Search/filter data, value counts, group-by, replace values selectively |
| 📈 **Visualization** | Bar, line, area, pie charts with custom colors |
| 🏷 **Data Encoding** | OrdinalEncoder, LabelEncoder, download encoded data & encoder |
| 🤖 **Model Training** | Multiple ML models (LR, Logistic, KNN, Tree, RF, Boosting, XGBoost, Naive Bayes, SVM), train/test split, accuracy, F1, MAE, R², download trained models |

---

## 📚 Libraries Used

| Library | Purpose |
|---------|---------|
| 🖥 [**Streamlit**](https://streamlit.io/) | Interactive UI framework |
| 🐼 [**Pandas**](https://pandas.pydata.org/) | Data manipulation |
| 🔢 [**NumPy**](https://numpy.org/) | Numerical computing |
| 📊 [**Plotly**](https://plotly.com/python/) | Interactive visualizations |
| 🎨 [**Seaborn**](https://seaborn.pydata.org/) | Statistical visualization |
| 📉 [**Matplotlib**](https://matplotlib.org/) | Plotting library |
| 🤖 [**scikit-learn**](https://scikit-learn.org/) | ML toolkit |
| ⚡ [**XGBoost**](https://xgboost.readthedocs.io/) | Gradient boosting |

**Navigate through the sidebar to access different modules:**
   - Data Analysis (Overview, Subjective Analysis, Visualise Data)
   - Model Training (Data Encoding, Data Training, Prediction Model)

## Project Structure

- `app.py` — Main entry point, sets up navigation.
- `pages/` — Streamlit multipage app modules.
- `src/` — Core data processing, cleaning, visualization, and ML logic.
- `data/` — Sample datasets and images.
- `output/` — Generated outputs (e.g., processed files). #future updates


Developed using streamlit. More updates soon
live at : https://datascience-smart-dashboard.streamlit.app/

> Streamlit free community cloud only provides 1gb of processing power so please kindly be patient with the processing.

> AI features and smart elements will soon be available in a different repository since the community cloud cannot provide such processing capacities
