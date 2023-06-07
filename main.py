import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('C:/Users/luana/OneDrive/Documentos/amq/ml-2023-1-trabalho-3/ds_salaries.csv')

mapping = {
    'SE': 1,
    'MI': 2,
    'EN': 3,
    'EX': 4
}

mapping_employment = {
    'FT': 4,
    'CT': 3,
    'FL': 2,
    'PT': 1
}

data['employment_type'] = data['employment_type'].map(mapping_employment)
data['experience_level'] = data['experience_level'].map(mapping)

encoded_job_title = pd.get_dummies(data['job_title'], prefix='job_title')
data_encoded = pd.concat([data, encoded_job_title], axis=1)
data_encoded.drop('job_title', axis=1, inplace=True) 

X = data_encoded[['work_year', 'experience_level', 'employment_type'] + list(encoded_job_title.columns)]

y = data_encoded['salary_in_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPRegressor(hidden_layer_sizes=(200, 200), activation='relu', solver='adam', max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
ymodel_one_train = model.predict(X_train)
ymodel_one_test = model.predict(X_test)

print("Model three training data:")
print("Mean Squared Error: ", mean_squared_error(y_train, ymodel_one_train))
print("Mean Absolute Error: ", mean_absolute_error(y_train, ymodel_one_train))
print("R2 Score: ", r2_score(y_train, ymodel_one_train), '\n')

print("Model three test data:")
print("Mean Squared Error: ", mean_squared_error(y_test, ymodel_one_test))
print("Mean Absolute Error: ", mean_absolute_error(y_test, ymodel_one_test))
print("R2 Score: ", r2_score(y_test, ymodel_one_test), '\n')

perfil_profissional = {
    'work_year': 2023,
    'experience_level': 'SE',
    'employment_type': 'FT',
    'job_title': 'Engineer'
}

df_perfil = pd.DataFrame([perfil_profissional])
df_perfil['employment_type'] = df_perfil['employment_type'].map(mapping_employment)
df_perfil['experience_level'] = df_perfil['experience_level'].map(mapping)

encoded_job_title = pd.get_dummies(df_perfil['job_title'], prefix='job_title')
df_perfil_encoded = pd.concat([df_perfil, encoded_job_title], axis=1)

df_perfil_encoded = df_perfil_encoded.reindex(columns=X.columns, fill_value=0)

if df_perfil_encoded.empty:
    salario_predito = 0
else:
    X_perfil = scaler.transform(df_perfil_encoded)

    salario_predito = model.predict(X_perfil)

print("Predicted Salary in USD:", salario_predito)
# y_pred_test = model.predict(X_test)
# acs = accuracy_score(y_test, y_pred_test)
# print("Acurácia: ", acs)