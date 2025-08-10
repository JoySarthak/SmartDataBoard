from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier,XGBRegressor

def get_models(problem_type):
    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "ADA Boosting Classifier": AdaBoostClassifier(),
            "Decision Tree Classifier": DecisionTreeClassifier(),
            "KNeighbors Classifier": KNeighborsClassifier(n_neighbors=5, weights="distance"),
            "SVC": SVC(kernel='poly'),
            "XGB Classifier": XGBClassifier(),
        }
    else:  # Regression
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "KNeighbors Regressor": KNeighborsRegressor(),
            "SVR": SVR(),
            "XGB Regressor": XGBRegressor(),
            "ADA Boosting Regressor": AdaBoostRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
        }