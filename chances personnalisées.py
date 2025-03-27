import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from dash import Dash, dcc, html, Input, Output


url = "https://raw.githubusercontent.com/Projet-Filiere-ADAD/Projet_ADAD/main/Donn%C3%A9es/fr-esr-parcoursup_2018-2024.csv"
df = pd.read_csv(url, sep=';')

# Regrouper les filières PASS, PACES et LAS
df['Filière de formation très agrégée'] = df['Filière de formation très agrégée'].replace('PASS', 'PACES/PASS/LAS')
df['Filière de formation très agrégée'] = df['Filière de formation très agrégée'].replace('PACES', 'PACES/PASS/LAS')
df['Filière de formation très agrégée'] = df['Filière de formation très agrégée'].replace('Licence_Las', 'PACES/PASS/LAS')

# Regrouper les écoles de commerce et d'ingénieur
df['Filière de formation très agrégée'] = df['Filière de formation très agrégée'].replace('Ecole de Commerce', 'Grandes écoles')
df['Filière de formation très agrégée'] = df['Filière de formation très agrégée'].replace("Ecole d'Ingénieur", 'Grandes écoles')

# Changement de nom
df['Filière de formation très agrégée'] = df['Filière de formation très agrégée'].replace('DUT', 'BUT')

df["Néo bacheliers en phase principale"] = df['Effectif des candidats néo bacheliers généraux en phase principale']+df['Effectif des candidats néo bacheliers technologiques en phase principale']+df['Effectif des candidats néo bacheliers professionnels en phase principale']
df["Néo bacheliers boursiers en phase principale"] = df['Dont effectif des candidats boursiers néo bacheliers généraux en phase principale']+df['Dont effectif des candidats boursiers néo bacheliers technologiques en phase principale']+df['Dont effectif des candidats boursiers néo bacheliers professionnels en phase principale']
df["% boursiers en phase principale"] = df["Néo bacheliers boursiers en phase principale"]/df["Néo bacheliers en phase principale"]*100

df["Néo bacheliers classés"] = df['Effectif des candidats néo bacheliers généraux classés par l’établissement']+df['Effectif des candidats néo bacheliers technologiques classés par l’établissement']+df['Effectif des candidats néo bacheliers professionnels classés par l’établissement']
df["Néo bacheliers boursiers classés"] = df['Dont effectif des candidats boursiers néo bacheliers généraux classés par l’établissement']+df['Dont effectif des candidats boursiers néo bacheliers technologiques classés par l’établissement']+df['Dont effectif des candidats boursiers néo bacheliers professionnels classés par l’établissement']
df["% boursiers classés"] = df["Néo bacheliers boursiers classés"]/df["Néo bacheliers classés"]*100

df["Néo bacheliers proposition"] = df['Effectif des candidats en terminale générale ayant reçu une proposition d’admission de la part de l’établissement']+df['Effectif des candidats en terminale technologique ayant reçu une proposition d’admission de la part de l’établissement']+df['Effectif des candidats en terminale professionnelle ayant reçu une proposition d’admission de la part de l’établissement']
df["Néo bacheliers boursiers proposition"] = df['Dont effectif des candidats boursiers en terminale générale ayant reçu une proposition d’admission de la part de l’établissement']+df['Dont effectif des candidats boursiers en terminale technologique ayant reçu une proposition d’admission de la part de l’établissement']+df['Dont effectif des candidats boursiers en terminale générale professionnelle ayant reçu une proposition d’admission de la part de l’établissement']
df["% boursiers proposition"] = df["Néo bacheliers boursiers proposition"]/df["Néo bacheliers proposition"]*100

df["Néo bacheliers admis"] = df["Effectif des admis néo bacheliers"]
df["Néo bacheliers boursiers admis"] = df["Dont effectif des admis boursiers néo bacheliers"]
df["% boursiers admis"] = df["Néo bacheliers boursiers admis"]/df["Néo bacheliers admis"] * 100

df["taux sélection boursiers admis / candidats"] = df["Néo bacheliers boursiers admis"]/df["Néo bacheliers boursiers en phase principale"]*100
df["taux sélection boursiers admis / classés"] = df["Néo bacheliers boursiers admis"]/df["Néo bacheliers boursiers classés"]*100
df["taux sélection boursiers admis / proposition"] = df["Néo bacheliers boursiers admis"]/df["Néo bacheliers boursiers proposition"]*100
df["taux sélection boursiers proposition / candidats"] = df["Néo bacheliers boursiers proposition"]/df["Néo bacheliers boursiers en phase principale"]*100

df["taux sélection admis / candidats"] = df["Néo bacheliers admis"] / df["Néo bacheliers en phase principale"] * 100
df["taux sélection admis / classés"] = df["Néo bacheliers admis"] / df["Néo bacheliers classés"] * 100
df["taux sélection admis / proposition"] = df["Néo bacheliers admis"] / df["Néo bacheliers proposition"] * 100
df["taux sélection proposition / candidats"] = df["Néo bacheliers proposition"] / df["Néo bacheliers en phase principale"] * 100
df["taux sélection classés / candidats"] = df["Néo bacheliers classés"] / df["Néo bacheliers en phase principale"] * 100

df["taux sélection non boursiers admis / candidats"] = (df["Néo bacheliers admis"] - df["Néo bacheliers boursiers admis"]) / (df["Néo bacheliers en phase principale"] - df["Néo bacheliers boursiers en phase principale"]) * 100

# Remplacer les valeurs infinies par des NaN
df = df.replace([np.inf, -np.inf], np.nan)
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go

# Create a Dash application
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Affichage de tes chances d'admission", style={'text-align': 'center', 'color': '#333'}),

    html.Div([
        html.Label('Établissement:', style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='school-selector',
            options=[{'label': school, 'value': school} for school in list(df['Établissement'].unique()) + ["Tous"]],
            value='Tous',
            placeholder="Sélectionnez un établissement",
            style={'width': '100%', 'margin-bottom': '10px'}
        )
    ], style={'width': '50%', 'margin': 'auto', 'margin-top': '20px'}),

    html.Div([
        html.Label('Filière:', style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='filiere-selector',
            options=[{'label': filiere, 'value': filiere} for filiere in list(df['Filière de formation très agrégée'].unique()) + ["Tous"]],
            value='Tous',
            placeholder="Sélectionnez une filière",
            style={'width': '100%', 'margin-bottom': '10px'}
        )
    ], style={'width': '50%', 'margin': 'auto'}),

    html.Div([
        html.Label('Filière détaillée:', style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='detailed-filiere-selector',
            options=[{'label': detailed, 'value': detailed} for detailed in list(df['Filière de formation détaillée'].unique()) + ["Tous"]],
            value='Tous',
            placeholder="Sélectionnez une filière détaillée",
            style={'width': '100%', 'margin-bottom': '10px'}
        )
    ], style={'width': '50%', 'margin': 'auto'}),

    html.Div([
        html.Label('Statut boursier:', style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='boursier-selector',
            options=[
                {'label': 'Oui', 'value': 'Oui'},
                {'label': 'Non', 'value': 'Non'},
                {'label': 'Tous', 'value': 'Tous'}
            ],
            value='Tous',
            placeholder="Sélectionnez un statut boursier",
            style={'width': '100%', 'margin-bottom': '10px'}
        )
    ], style={'width': '50%', 'margin': 'auto'}),

    # Display the result
    dcc.Graph(id='result-graph', style={'margin-top': '20px'})
], style={'font-family': 'Arial, sans-serif', 'padding': '20px'})

# Callback to update the result based on selections
@app.callback(
    Output('result-graph', 'figure'),
    [Input('school-selector', 'value'),
     Input('filiere-selector', 'value'),
     Input('detailed-filiere-selector', 'value'),
     Input('boursier-selector', 'value'),
    ]
)
def update_result(selected_school, selected_filiere, selected_detailed_filiere, selected_boursier):
    # Filter the DataFrame based on selections
    filtered_df = df
    if selected_school != 'Tous':
        filtered_df = filtered_df.loc[filtered_df['Établissement'] == selected_school]
    if selected_filiere != 'Tous':
        filtered_df = filtered_df.loc[filtered_df['Filière de formation très agrégée'] == selected_filiere]
    if selected_detailed_filiere != 'Tous':
        filtered_df = filtered_df.loc[filtered_df['Filière de formation détaillée'] == selected_detailed_filiere]

    # Select the appropriate column based on "boursier" selection
    if selected_boursier == 'Oui':
        taux_column = 'taux sélection boursiers admis / candidats'
    elif selected_boursier == 'Non':
        taux_column = 'taux sélection non boursiers admis / candidats'
    else:
        taux_column = 'taux sélection admis / candidats'

    # Calculate the average taux for the filtered data
    average_taux = filtered_df[taux_column].mean()

    # Create a gauge chart
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=average_taux,
        title={'text': "Taux de sélection moyen"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 30], 'color': "#d62728"},
                {'range': [30, 70], 'color': "#ff7f0e"},
                {'range': [70, 100], 'color': "#2ca02c"}
            ],
        }
    ))

    return figure

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False, port=8051)
