from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

def get_ridge():
    return Ridge(alpha=1.0)

def get_knn():
    return KNeighborsRegressor(n_neighbors=5, n_jobs=-1)

def get_svr():
    return SVR(kernel='rbf', C=100.0, gamma='scale')

def get_random_forest():
    return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

def get_xgboost():
    return xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        random_state=42, 
        n_jobs=-1
    )

def get_all_baselines():
    return {
        'Ridge': get_ridge(),
        'KNN': get_knn(),
        'SVR': get_svr(),
        'Random Forest': get_random_forest(),
        'XGBoost': get_xgboost()
    }