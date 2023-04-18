import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import data_cleaning


def fit_grid_search(model):
    grid = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    return grid


# Import data
data = pd.read_csv("archive/AB_NYC_2019.csv")
pd.set_option('display.max_columns', None)

# Operations with data
data_scaled = data_cleaning.clean_data(data)

# Splitting the data
X = data_scaled.drop('price', axis=1)
y = data_scaled['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Parameters for models
param_grid = {'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 1, 5, 10, 50, 100],
              "max_iter": [1000, 5000, 10000]
              }
cv = KFold(n_splits=5, shuffle=True, random_state=1)

# ------- create the ElasticNet model
elasticnet = ElasticNet()
enet_pred = fit_grid_search(elasticnet).predict(X_test)
# Evaluate the model
enet_mse = mean_squared_error(y_test, enet_pred)
enet_r2 = r2_score(y_test, enet_pred)

# ------- create the Ridge model
ridge = Ridge()
ridge_pred = fit_grid_search(ridge).predict(X_test)
# Evaluate the model
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

# ------- create the Lasso model
lasso = Lasso()
lasso_pred = fit_grid_search(lasso).predict(X_test)
# Evaluate the model
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

# CSV report
models = {
    'ElasticNet': [np.sqrt(enet_mse), enet_mse, enet_r2],
    'Ridge': [np.sqrt(ridge_mse), ridge_mse, ridge_r2],
    'Lasso': [np.sqrt(lasso_mse), lasso_mse, lasso_r2]
}
report = []

for model, metrics in models.items():
    print("\n%s RMSE on test set: %.4f" % (model, metrics[0]))
    print("%s MSE on test set: %.4f" % (model, metrics[1]))
    print("%s R2 on test set: %.4f" % (model, metrics[2]))
    report.append((model, metrics[0], metrics[1], metrics[2]))

# Best model
max_ev = 0
best_model = ""
for model, value, tmp, tmpp in report:
    if value > max_ev:
        max_ev = value
        best_model = model
print(f"\nBest model is: {best_model}")

perf_report = pd.DataFrame(report, columns=['Model', 'RMSE', 'MSE', 'R2'])
perf_report.to_csv("Perf_report.csv")

# Importance analyse
best_model = best_model.lower()
best_model = fit_grid_search(globals()[best_model]).best_estimator_.coef_

feature_imp = pd.DataFrame({'feature': X_train.columns, 'importance': best_model})
feature_imp = feature_imp.reindex(feature_imp['importance'].abs().sort_values(ascending=False).index)

sns.barplot(x='importance', y='feature', data=feature_imp)
plt.title('Feature Importance')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.show()

