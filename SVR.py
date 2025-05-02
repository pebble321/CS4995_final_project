from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

df = pd.read_csv('embeddings_scaled_final.csv')

# seperate x labels from y label, y label can be num_subscribers or num_reviews
X = df.drop(columns=['num_subscribers'])
y = df['num_subscribers']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# set up model
svr = SVR()

# set up hyperparameters and randomnized search cv
param_dist = {
    'C': np.logspace(-3, 3, 7),
    'gamma': ['scale', 'auto'] + np.logspace(-3, 3, 7).tolist(),
    'epsilon': [0.01, 0.1, 0.2, 0.5, 1.0],
}

random_search = RandomizedSearchCV(svr, param_distributions=param_dist, n_iter=50, cv=5,
                                   verbose=2, random_state=42, n_jobs=-1, scoring='neg_mean_squared_error')

# find the best model and predict
random_search.fit(X_train, y_train.ravel())
print("Best parameters found by RandomizedSearchCV:", random_search.best_params_)
best_svr = random_search.best_estimator_
y_pred = best_svr.predict(X_test)


# print R^2 and RMSE score
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print("R^2:", r2)
print("RMSE", rmse)
