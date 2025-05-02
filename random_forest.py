import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import time

df = pd.read_csv('embeddings_scaled_final.csv')

# seperate x labels from y label, y label can be num_subscribers or num_reviews
X = df.drop(columns=['num_subscribers'])
y = df['num_subscribers']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# set hyperparameters for RandomizedSearchCV
param_dist = {
    'n_estimators': [20, 50, 80, 100, 200],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],  # Corrected max_features
    'bootstrap': [True, False]
}

# find best hyperparameters and train the model
rf_reg = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(estimator=rf_reg, param_distributions=param_dist, n_iter=10, cv=5,
                                   verbose=2, random_state=42, n_jobs=-1, scoring='neg_mean_squared_error')

random_search.fit(X_train, y_train)
best_params = random_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# evaluation
y_pred = random_search.best_estimator_.predict(X_test)

# calculate R^2 and RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")
