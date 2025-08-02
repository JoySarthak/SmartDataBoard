# Smartboard: Dataset Analyser & Model Trainer

Smartboard is an interactive data analysis and machine learning platform built with [Streamlit](https://streamlit.io/). It enables users to upload datasets, perform exploratory data analysis, clean data, encode features, visualize trends, and train machine learning models—all through a user-friendly web interface.

## Features

- **Data Overview:**  
  - Upload CSV datasets.
  - View dataframe, summary statistics, and data types.
  - Visualize missing values (bar, pie, line charts).
  - Download cleaned or processed datasets.

- **Data Cleaning:**  
  - Drop or replace NaN values.
  - Remove duplicate rows.
  - Drop unwanted features (columns).
  - Remove outliers.

- **Subjective Analysis:**  
  - Search and filter data by column.
  - Value counts and group-by operations.
  - Replace values selectively.

- **Data Visualization:**  
  - Bar, line, area, and pie charts for feature distributions.
  - Customizable color schemes.

- **Data Encoding:**  
  - Encode categorical features using OrdinalEncoder or LabelEncoder.
  - Download encoded datasets and encoder objects for reuse.

- **Model Training:**  
  - Select features and target variable.
  - Choose from multiple ML models:  
    - Linear Regression, Logistic Regression  
    - KNN, Decision Tree, Random Forest  
    - AdaBoost, Gradient Boosting  
    - XGBoost  
    - Naive Bayes, SVM
  - Train/test split, model evaluation (accuracy, F1, MAE, R2, etc.).
  - Download trained models.

## Libraries Used

- [Streamlit](https://streamlit.io/) (version 1.32+ recommended)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Plotly](https://plotly.com/python/)
- [Seaborn](https://seaborn.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)

## Getting Started

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd DatasetAnalyser
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```sh
   streamlit run app.py
   ```

4. **Navigate through the sidebar to access different modules:**
   - Data Analysis (Overview, Subjective Analysis, Visualise Data)
   - Model Training (Data Encoding, Data Training, Prediction Model)

## Project Structure

- `app.py` — Main entry point, sets up navigation.
- `pages/` — Streamlit multipage app modules.
- `src/` — Core data processing, cleaning, visualization, and ML logic.
- `data/` — Sample datasets and images.
- `output/` — Generated outputs (e.g., processed files).

## Screenshots

*(Add screenshots of the UI here)*

## License

MIT License

---

*Developed with ❤️ using
