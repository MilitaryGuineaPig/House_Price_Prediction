import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import csv

data = pd.read_csv("/Users/monkey/Public/Python/Jop_prepar/Task_1/archive/AB_NYC_2019.csv")
pd.set_option('display.max_columns', None)
"""
print(data.columns) 
print(data.describe())
print(data.tail())
"""

# operations with data
data.dropna(inplace=True)
print(data.columns)
neighbourhood_group_num = LabelEncoder()
neighbourhood_num = LabelEncoder()
room_type_num = LabelEncoder()
neighbourhood_group_num.fit(data['neighbourhood_group'])
neighbourhood_num.fit(data['neighbourhood'])
room_type_num.fit(data['room_type'])
data['neighbourhood_group'] = neighbourhood_group_num.transform(data['neighbourhood_group'])
data['neighbourhood'] = neighbourhood_num.transform(data['neighbourhood'])
data['room_type'] = room_type_num.transform(data['room_type'])

drop_column = ['id','host_id', 'name', 'host_name', 'last_review']
data_d = data.drop(drop_column, axis=1)
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_d), columns=data_d.columns)

# Splitting the data
X = data_scaled.drop('price', axis=1)
y = data_scaled['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# ------- create the ElasticNet model
param_grid = {'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 1, 5, 10, 50, 100]}
cv = KFold(n_splits=5, shuffle=True, random_state=1)
enet = ElasticNet()
grid_enet = GridSearchCV(enet, param_grid, cv=cv, scoring='neg_mean_squared_error')
grid_enet.fit(X_train, y_train)
# print the best parameters and score
print('\nBest ElasticNet parameters:', grid_enet.best_params_)
print('Best ElasticNet score:', -grid_enet.best_score_)
# Evaluate ElasticNet on test set
enet_pred = grid_enet.predict(X_test)
enet_mse = mean_squared_error(y_test, enet_pred)  # mean squared error
enet_r2 = r2_score(y_test, enet_pred)
print("ElasticNet RMSE on test set: %.4f" % np.sqrt(enet_mse))  # root from MSE
print("ElasticNet MSE on test set: %.4f" % enet_mse)
print("ElasticNet R2 on test set: %.4f" % enet_r2)

# ------- create the Ridge model
ridge = Ridge()
ridge_param_grid = {
    "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    "max_iter": [1000, 5000, 10000]
}
grid_ridge = GridSearchCV(ridge, ridge_param_grid, cv=cv, scoring='neg_mean_squared_error')
grid_ridge.fit(X_train, y_train)
print('\nBest Ridge parameters:', grid_ridge.best_params_)
print('Best Ridge score:', -grid_ridge.best_score_)
ridge_pred = grid_ridge.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
print("Ridge RMSE on test set: %.4f" % np.sqrt(ridge_mse))
print("Ridge MSE on test set: %.4f" % ridge_mse)
print("Ridge R2 on test set: %.4f" % ridge_r2)

# ------- create the Lasso model
lasso = Lasso()
lasso_param_grid = {
              'alpha': [0.001, 0.01, 0.1, 1, 10],
              'max_iter': [100, 500, 1000, 5000]
}
grid_lasso = GridSearchCV(lasso, lasso_param_grid, cv=cv, scoring='neg_mean_squared_error')
grid_lasso.fit(X_train, y_train)
print('\nBest Lasso parameters:', grid_lasso.best_params_)
print('Best Lasso score:', -grid_lasso.best_score_)
lasso_pred = grid_lasso.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)
print("Lasso RMSE on test set: %.4f" % np.sqrt(lasso_mse))
print("Lasso MSE on test set: %.4f" % lasso_mse)
print("Lasso R2 on test set: %.4f" % lasso_r2)


# CSV report
models = {
    'ElasticNet': [np.sqrt(enet_mse), enet_mse, enet_r2],
    'Ridge': [np.sqrt(ridge_mse), ridge_mse, ridge_r2],
    'Lasso': [np.sqrt(lasso_mse), lasso_mse, lasso_r2]
}
with open('performance_metrics.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Model', 'RMSE', 'MSE', 'R2'])
    for model, metrics in models.items():
        writer.writerow([model, metrics[0], metrics[1], metrics[2]])

# importance analyse
enet_coef = grid_enet.best_estimator_.coef_
feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': enet_coef})
feature_importance = feature_importance.reindex(feature_importance['importance'].abs().sort_values(ascending=False).index)
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.show()


# Compute correlation matrix
corr = data_scaled.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


"""
sns.pairplot(num_data, height=2.5)
plt.savefig('myplot.png')
plt.show()
"""

