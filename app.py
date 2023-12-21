import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, dependencies
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

import pandas as pd                 # Para la manipulación de los datasets
import numpy as np                  # Para operaciones matematicas
import sklearn.datasets             # Para los datasets

import plotly.express as px         # Plotly
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pickle                       # Esto sirve para el K-Means Clustering y poder guardar el modelo en el formator correcto. 
from plotly.subplots import make_subplots

# * Librerias de Dash
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State

# * Librerías Machine Learning
# * Modelos Existentes (Regresión Lineal, Logistica, Random Forest, K-Means Clustering, SVM)

from sklearn.model_selection import train_test_split 
# ? - Regresión Lineal + Logisticas
from sklearn.linear_model import LinearRegression, SGDClassifier, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
# T-test
from scipy.stats import ttest_ind

# ? - Random Forest (+ Escalamiento necesario)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ? - Metricas de Precisión
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    classification_report,
    confusion_matrix,
    silhouette_score
)

from sklearn.pipeline import Pipeline # ? - Para generar modelos de machine learning con escalamiento automatico


# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script directory
os.chdir(script_dir)

app = dash.Dash()

server = app.server

app.title = 'AppDev'

# FIGURE 1

df = pd.read_csv("bank-full.csv", sep = ";")
df['job'] = df['job'].astype('category')
df['marital'] = df['marital'].astype('category')
df['education'] = df['education'].astype('category')
df['contact'] = df['contact'].astype('category')
df['poutcome'] = df['poutcome'].astype('category')
df['month'] = df['month'].astype('category')
df['day'] = df['day'].astype('category')

df['default'] = df['default'].map({'no': 0, 'yes': 1})
df['housing'] = df['housing'].map({'no': 0, 'yes': 1})
df['loan'] = df['loan'].map({'no': 0, 'yes': 1})
df['y'] = df['y'].map({'no': 0, 'yes': 1})

df_2 = df.copy()

cat_vars=['job','marital','education','contact','poutcome','month', 'day']

for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(df_2[var], prefix=var)
    data1=df_2.join(cat_list)
    df_2=data1

data_vars=df_2.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

df_2=df_2[to_keep]

df_2.rename(columns={'y': 'target'}, inplace=True)

X = df_2.drop("target", axis=1) #? - Quitamos la columna que queremos predecir. axis=1 para eliminar una columna, axis=0 para eliminar una fila.
y = df_2["target"]              #? - Nos quedamos con la columna que queremos predecir.

# ? - Separar los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 123) #? - test_size=0.2 para que el 20% de los datos sean de test

# ? - Crear el modelo
clf_log = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ("glm", LogisticRegression(max_iter=10000, tol=0.1)),
])

# clf_log = LogisticRegression(max_iter=10000, tol=0.1)
clf_log.fit(X_train, y_train)

predictions = clf_log.predict(X_test)

lmc = clf_log.named_steps['glm'].coef_

# Get the absolute values of the coefficients
abs_coefficients = abs(lmc[0])

# Create a dictionary with variable names as keys and absolute coefficients as values
coefficients_dict = dict(zip(X.columns, abs_coefficients))

# Sort the dictionary by absolute coefficient values in descending order
sorted_coefficients = sorted(coefficients_dict.items(), key=lambda x: x[1], reverse=True)

# Extract the variable names and absolute coefficients from the sorted list
sorted_variables = [item[0] for item in sorted_coefficients]
sorted_abs_coefficients = [item[1] for item in sorted_coefficients]

# Plot the sorted coefficients
fig_1 = go.Figure()

# Add the bar trace
fig_1.add_trace(
    go.Bar(
        x=sorted_abs_coefficients,
        y=sorted_variables,
        name="Absolute Coefficients",
        orientation="h"
    )
)

# Update the layout
fig_1.update_layout(
    title="Sorted GLM Coefficients Importance",
    xaxis_title="Absolute Coefficients",
    yaxis_title="Variables"
)


# FIGURE 2

df['y'] = df['y'].astype('category')

fig_2 = px.histogram(df, x="y", color="contact", barmode="stack", title="Counts of Contact Types by y")
fig_2.update_layout(bargap=0.2) 


# FIGURE 3 AND FIGURE 4

grouped_data = df.groupby(['y', 'poutcome']).size().reset_index(name='count')

fig_3 = px.pie(grouped_data[grouped_data['y'] == 1], values='count', names='poutcome', title='')
fig_4 = px.pie(grouped_data[grouped_data['y'] == 0], values='count', names='poutcome', title='')


app.layout = html.Div(
    children = [
        html.H1( # Título de la página
            children = [
                "Dashboard"
            ],
            id = "dahsboard_title",
            style = {
                "text-align": "Left",
                "font-family": "Arial",
            }
        ),
        html.Div(  # Container for the form
            children=[
                dcc.Dropdown(
                    options= [{'label': 'Tiene hipoteca', 'value': 'yes'}, {'label': 'No tiene hipoteca', 'value': 'no'}],
                    placeholder="Elija el tipo de hipoteca",
                    id="dropdown",
                    style={
                        "display": "inline-block",
                        "width": "175px",  # Adjusted width
                        "margin-right": "30px",  # Adjusted margin
                    }
                ),
            ],
            style={
                "font-family": "Arial",
                "display": "flex",  # Use flexbox layout
                "justify-content": "center",  # Center items horizontally
                "align-items": "center",  # Center items vertically
                "height": "100px",  # Adjust height as needed
                "margin-bottom": "10px",  # Margin for the entire form container
            }
        ),
        html.Div(  # Parent container for both rows
            children=[
                # First row: Clean Sheets and Pie Chart
                html.Div(
                    children=[
                        html.Div(
                            dcc.Graph(id="figure_1",
                                      figure=fig_1,
                                    style={"display": "inline_bloc"}),
                            style={"flex": "1"}  # Adjust flex value as needed
                        ),
                        html.Div(
                            dcc.Graph(id="figure_2",
                                        figure=fig_2,
                                    style={"display": "innline_bloc"}),
                            style={"flex": "1"}  # Adjust flex value as needed
                        )
                    ],
                    style={"display": "flex", "justify-content": "center"}
                ),    
                # Second row: Heatmap and Goals
                html.Div(
                    children=[
                        html.Div(
                            dcc.Graph(id="figure_3",
                                    figure=fig_3,
                                    style={"display": "inline_block"}),
                            style={"flex": "1"}  # Adjust flex value as needed
                        ),
                        html.Div(
                            dcc.Graph(id="figure_4",
                                      figure = fig_4,
                                    style={"display": "inline_block"}),
                            style={"flex": "1"}  # Adjust flex value as needed
                        )
                    ],
                    style={"display": "flex", "justify-content": "center"}
                ),
            ],
            style={"text-align": "center"}
        ),
    ],
    id = "dashboard_page",
    style = {
        "margin-right": "125px",
        "margin-left": "125px",
        "margin-top": "50px",
    } 
)

@app.callback(
    Output("figure_2", "figure"),
    Input("dropdown", "value")
)
def actualizar(sel):
    df = pd.read_csv("bank-full.csv", sep = ";")
    print(df.head())
    if sel == 'yes' or sel == 'no':
        df = df[df['housing'] == sel]    
    fig_2 = px.histogram(df, x="y", color="contact", barmode="stack", title="Counts of Contact Types by y")
    fig_2.update_layout(bargap=0.2) 
    return fig_2


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)