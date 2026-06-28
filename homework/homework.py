import pandas as pd
from pathlib import Path
import zipfile
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
import pickle
import gzip
import json

def calcular_matriz_confusion(X, y_verdadero, ruta, tipo_conjunto):

    modelo_cargado = None

    with gzip.open(ruta, "rb") as f:
        modelo_cargado = pickle.load(f)
    
    y_predicho = modelo_cargado.predict(X)

    matriz = confusion_matrix(y_verdadero, y_predicho)

    return {
        "type": "cm_matrix",
        "dataset": tipo_conjunto,
        "true_0": {
            "predicted_0": matriz.tolist()[0][0],
            "predicted_1": matriz.tolist()[0][1]
            },
        "true_1": {
            "predicted_0": matriz.tolist()[1][0],
            "predicted_1": matriz.tolist()[1][1]
            }
    }

def calcular_metricas(X, y_verdadero, ruta, tipo_conjunto):

    
    modelo_cargado = None

    with gzip.open(ruta, "rb") as f:
        modelo_cargado = pickle.load(f)
    
    y_predicho = modelo_cargado.predict(X)

    return {
        "type": "metrics",
        "dataset": tipo_conjunto,
        "precision": float(accuracy_score(y_verdadero, y_predicho)),
        "balanced_accuracy": float(balanced_accuracy_score(y_verdadero, y_predicho)),
        "recall": float(recall_score(y_verdadero, y_predicho, average='macro')), # 'macro' calcula la media para clases desbalanceadas
        "f1_score": float(f1_score(y_verdadero, y_predicho, average='macro'))
    }

# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#

ruta_zips = Path("files/input")
for zip in ruta_zips.iterdir():
    with zipfile.ZipFile(zip, 'r', metadata_encoding = "utf-8") as zip_ref:
        Path("files/input/csv").mkdir(parents=True, exist_ok=True)
        zip_ref.extractall("files/input/csv")

df_test = pd.read_csv("files/input/csv/test_default_of_credit_card_clients.csv")
df_train = pd.read_csv("files/input/csv/train_default_of_credit_card_clients.csv")


df_test.rename(columns = {"default payment next month": "default"}, inplace=True)
df_test = df_test.drop("ID", axis=1)
df_test = df_test.dropna()
df_test["EDUCATION"] = df_test["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
#df_test = df_test[(df_test["EDUCATION"] != 0) & (df_test["MARRIAGE"] != 0)]
#df_test["others"] = df_test["EDUCATION"].apply(lambda x: 1 if x>4 else 0)

df_train.rename(columns = {"default payment next month": "default"}, inplace=True)
df_train = df_train.drop("ID", axis=1)
df_train = df_train.dropna()
df_train["EDUCATION"] = df_train["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#

y_train = df_train["default"]
x_train = df_train.drop("default", axis=1)

y_test = df_test["default"]
x_test = df_test.drop("default", axis=1)

#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#

columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
columnas_numericas = [x for x in df_train.columns.to_list() if x not in columnas_categoricas]
columnas_numericas.remove("default")

preprocesador = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), columnas_categoricas),
        ('num', StandardScaler(with_mean=False), columnas_numericas),
    ],
    remainder='passthrough' # Mantiene las variables numéricas intactas (como 'edad')
)

pipeline = Pipeline([
    ('preprocessor', preprocesador),
    ('feature_selection', SelectKBest(score_func=f_classif, k=10)),
    ('pca', PCA(n_components=None)),
    ('classifier', MLPClassifier(
    random_state=42,
    max_iter=1000
    ))
])

#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#

param_grid = {
    'feature_selection__k': [12],

    'classifier__hidden_layer_sizes': [
        (50,)
    ],

    'classifier__activation': [
        'tanh'
    ],

    'classifier__solver': [
        'adam'
    ],

    'classifier__alpha': [
        0.001
    ],

    'classifier__learning_rate': [
        'adaptive'
    ]
}

# Búsqueda por validación cruzada
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring='balanced_accuracy',
    n_jobs=-1,
    refit=True
)

grid_search.fit(x_train, y_train)
print(grid_search.score(x_train, y_train))
print(grid_search.score(x_test, y_test))

#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#

Path("files/models").mkdir(parents=True, exist_ok=True)
archivo_modelo = 'files/models/model.pkl.gz'

# Guardamos el mejor pipeline encontrado por GridSearchCV de forma comprimida
with gzip.open(archivo_modelo, 'a+') as f:
    pickle.dump(grid_search, f)

#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#

Path("files/output").mkdir(parents=True, exist_ok=True)

metricas_train = calcular_metricas(x_train, y_train, "files/models/model.pkl.gz", "train")
metricas_test = calcular_metricas(x_test, y_test, "files/models/model.pkl.gz", "test")

with open("files/output/metrics.json", "w", encoding="utf-8") as file:
    file.write(json.dumps(metricas_train)+"\n")
    file.write(json.dumps(metricas_test)+"\n")

#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

matriz_train = calcular_matriz_confusion(x_train, y_train, "files/models/model.pkl.gz", "train")
matriz_test = calcular_matriz_confusion(x_test, y_test, "files/models/model.pkl.gz", "test")

with open("files/output/metrics.json", "a", encoding="utf-8") as file:
    file.write(json.dumps(matriz_train)+"\n")
    file.write(json.dumps(matriz_test)+"\n")