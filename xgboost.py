import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import time
import xgboost as xgb

df = pd.read_csv('embeddings_scaled_final.csv')

# seperate x labels from y label, y label can be num_subscribers or num_reviews
X = df.drop(columns=['num_subscribers'])
y = df['num_subscribers']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# set hyperparameters
param_dist = {
    'n_estimators': [20, 40, 50, 80, 100],
    'max_depth': [3, 4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 2, 3, 4]
}

# set up the model
xg_reg = XGBRegressor(objective='reg:squarederror', random_state=42)
random_search = RandomizedSearchCV(estimator=xg_reg, param_distributions=param_dist, n_iter=10, cv=5,
                                   verbose=2, random_state=42, n_jobs=-1, scoring='neg_mean_squared_error')

best_params = random_search.fit(X_train_scaled, y_train).best_params_
xg_reg.set_params(**best_params)
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# Set early stopping and evaluation parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': best_params['max_depth'],
    'learning_rate': best_params['learning_rate'],
    'subsample': best_params['subsample'],
    'colsample_bytree': best_params['colsample_bytree'],
    'gamma': best_params['gamma'],
    'min_child_weight': best_params['min_child_weight'],
}

# perform cross-validation with early stopping using xgboost.cv
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    nfold=3,
    early_stopping_rounds=10,
    metrics="rmse",
    as_pandas=True,
    seed=42
)

# train the final model
best_num_rounds = cv_results['test-rmse-mean'].idxmin()
final_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=best_num_rounds
)
y_pred = final_model.predict(dtest)

# calculate RMSE and R²
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Best Hyperparameters: {best_params}")
print(f"Best number of boosting rounds: {best_num_rounds}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")

