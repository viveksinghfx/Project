import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('Data/Delhi_v2.csv', sep=',')
if 'Unnamed: 0' in data.columns:
    data.drop(columns=['Unnamed: 0'], inplace=True)
data.dropna(inplace=True)

categorical_cols = [
    "type_of_building",
    "Furnished_status",
    "Status",
    "neworold"
]

data = data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data['Bedrooms'] = np.log(data['Bedrooms'] + 1)
data['Bathrooms'] = np.log(data['Bathrooms'] + 1)
data['Balcony'] = np.log(data['Balcony']+1)
data['parking'] = np.log(data['parking']+1)
data['Lift'] = np.log(data['Lift']+1)

X = data.drop(['price', 'Address', 'desc', 'Landmarks'], axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
#X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


#train_data = X_train.join(y_train)


from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train_scaled,y_train)
#print(reg.score(X_test_scaled,y_test))
#pred = reg.predict(X_test,y_test)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


forest = RandomForestRegressor(
    n_estimators=50,
    max_features=6,
    min_samples_split=2,
    random_state=42
)

forest.fit(X_train_scaled, y_train)

y_pred_rf = forest.predict(X_test_scaled)

print("Random Forest R²:", r2_score(y_test, y_pred_rf))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))


from sklearn.model_selection import GridSearchCV
#param_grid = {
   # "n_estimators": [50, 100, 200],
   # "max_features": [2, 4, 6],
   # "min_samples_split": [2, 4, 6]
#}

#grid_search = GridSearchCV(
  #  RandomForestRegressor(random_state=42),
 #   param_grid=param_grid,
  #  cv=5,
  #  scoring="neg_root_mean_squared_error",
  #  n_jobs=-1
#)

#grid_search.fit(X_train_scaled, y_train)

#print("Best params:", grid_search.best_params_)

from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("RMSE:", rmse)
print(y.describe())

residuals = y_test - y_pred_rf
plt.hist(residuals, bins=30)
plt.title("Residual Distribution")
plt.show()