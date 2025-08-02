# 📊 Smartboard: Dataset Analyser & Model Trainer

> 🚀 An interactive **Streamlit** platform for dataset analysis, cleaning, visualization, encoding, and **machine learning** model training — all without writing code.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-ff4b4b?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green)
![GitHub last commit](https://img.shields.io/github/last-commit/your-username/DatasetAnalyser?logo=github)

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

DatasetAnalyser/
│
├── app.py          # Main entry point
├── pages/          # Streamlit multipage modules
├── src/            # Core processing & ML logic
├── data/           # Sample datasets & images
├── output/         # Generated processed files
└── requirements.txt

Developed using streamlit. More updates soon
live at : https://datascience-smart-dashboard.streamlit.app/
