from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier,XGBRegressor

def get_models(problem_type):
    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "ADA Boosting Classifier": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=7)),
            "Decision Tree Classifier": DecisionTreeClassifier(),
            "KNeighbors Classifier": KNeighborsClassifier(n_neighbors=5, weights="distance"),
            "SVC": SVC(kernel='poly'), 
        }
    else:
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "KNeighbors Regressor": KNeighborsRegressor(),
            "SVR": SVR(kernel='poly'),
            "ADA Boosting Regressor": AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=7)),
            "Decision Tree Regressor": DecisionTreeRegressor(),
        }
        #"XGB Regressor": XGBRegressor(),
        #"XGB Classifier": XGBClassifier(),