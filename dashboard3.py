import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.QUARTZ]

# Load data
raw_data = pd.read_csv('data_disp.csv')
raw_data['date'] = pd.to_datetime(raw_data['date'])

pred_data = pd.read_csv('pred.csv')
pred_data['date'] = pd.to_datetime(pred_data['date'])
date_vet = pred_data['date']

# Load features data
features_data = pd.read_csv('feats.csv')

# Define feature dictionary
features_dictionary = {
    'biomass': 'Biomass',
    'fossil_gas_comp': 'Fossil Gas Consumption',
    'hyd_p_stg': 'Hydro Pumped Storage',
    'hyd_river': 'Hydro River and Poundage',
    'hyd_reser': 'Hydro Water Reservoir',
    'other': 'Other',
    'solar': 'Solar',
    'wind_off': 'Wind Offshore',
    'wind_on': 'Wind Onshore'
}

other_features = {'time_of_day': 'Time of Day',
                  'season': 'Season'}

other_feature_values = {
    'time_of_day': {0: 'Night',
                    0.5: 'Day',
                    1: 'Afternoon and Evening'},
    'season': {1: 'Winter',
               0.25: 'Spring',
               0.0: 'Summer',
               0.75: 'Autumn'}}

ML_models = {'XGB':'all_stuff_XGB.csv',
             'NN':'all_stuff_NN.csv'}

# Dictionary for month selection
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

mwh_to_co2_pt = 185.0             #kg_co2/MWh

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define layout
app.layout = html.Div([
    html.H1('Energy Dashboard Project 3 - Vasco Morais', style={'textAlign': 'center'}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Project Overview', value='tab-1', style={'color': 'orange'}),
        dcc.Tab(label='Raw Data Display', value='tab-2', style={'color': 'orange'}),
        dcc.Tab(label='Exploratory Data Analysis', value='tab-3', style={'color': 'orange'}),
        dcc.Tab(label='Forecasting', value='tab-4', style={'color': 'orange'}),
        dcc.Tab(label='Energy and CO2 Budgets', value='tab-5', style={'color': 'orange'})
    ]),
    html.Div(id='tabs-content')
])


# Callback to render content for the selected tab
@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.Div([
                html.H2('Description', style={'color': 'white'}),
                html.P('In this project, a machine learning model was developed to forecast CO\u2082 emissions '
                       'resulting from the usage of natural gas in Portugal for the year 2024.', style={'color': 'black'}),
                html.P(
                    'The model was trained using features such as the time of day (morning, day, night), season of the year, '
                    'the usage of gas from the past year, and the difference of renewable energy generation from its mean value.', style={'color': 'black'}),
                html.P(
                    'To provide more insights, energy budgets were calculated, taking into account Portugal\'s grid load. '
                    'This allows one to observe the impact on CO\u2082 emissions, such as an increase or decrease in emissions '
                    'due to importations or exportations of energy from/to Spain.', style={'color': 'black'})
            ]),
            html.Div([
                html.H2('Navigation Instructions', style={'color': 'white'}),
                html.Ul([
                    html.Li('Raw Data Display: Explore sets of data from the raw data used against each other. '
                            'Filter them by month and year to see and compare their time evolution.', style={'color': 'black'}),
                    html.Li(
                        'Exploratory Data Analysis: Explore relations between features through scatter plots and histograms.', style={'color': 'black'}),
                    html.Li('Forecasting: View machine learning models\' predictions of energy and CO\u2082 emissions '
                            'on a monthly basis.', style={'color': 'black'}),
                    html.Li(
                        'Energy and CO\u2082 Budgets: Observe the influence of surplus/deficit of energy generation '
                        'on CO\u2082 emissions on a monthly basis.', style={'color': 'black'})
                ])
            ]),
            html.Div([
                html.H2('Data Sources', style={'color': 'white'}),
                html.Ul([
                    html.Li('ENTSOE API: Gather energy generation data of Portugal and its grid load.', style={'color': 'black'}),
                    html.Li('MWh to kg CO\u2082 conversion factor: '
                            'https://www.eeagrants.gov.pt/media/2776/conversion-guidelines.pdf', style={'color': 'black'}),
                    html.Li('Share of fossil fuel in total energy imported from Spain: '
                            'https://www.statista.com/statistics/1230294/share-of-primary-energy-consumption-by-source-in-spain/', style={'color': 'black'})
                ])
            ]),
            html.Div([
                html.H2('Additional Remarks', style={'color': 'white'}),
                html.Ul([
                    html.Li(
                        'The renewable energy generation of 2024 was assumed to be equal to 2023 to generate a feature.', style={'color': 'black'}),
                    html.Li('The grid load of 2024 was assumed to be equal to 2023 to create the budgets.', style={'color': 'black'}),
                html.P(),
                html.P(
                    'These simplifications were made because a different forecasting procedure would be required, '
                    'reaching a level similar to a thesis, which is beyond the scope of this project. '
                    'However, useful and interesting information is still provided using these assumptions.', style={'color': 'black'}),
                html.Li(
                    'I made the choice not to use a Geographic Information System (GIS). Typically, GIS is beneficial '
                    'when dealing with differences between geographical areas like countries, districts, or municipalities. '
                    'However, since this project involves forecasting for all of Portugal in a holistic manner, '
                    'I deemed it unnecessary to present the results in a GIS format.', style={'color': 'black'})
                ])
            ]),
            html.Div([
                html.H2('Contacts', style={'color': 'white'}),
                html.P('Email: vasco.a.morais@tecnico.ulisboa.pt', style={'color': 'black'}),
                html.P('Phone: +351 966147438', style={'color': 'black'})
            ])
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.Div([
                html.H5('Raw Data Display', style={'color': 'white'}),
                html.P('In this tab, you can explore sets of data from the raw data that was used against each other. '
                       'Filter them by month and year to see and compare their time evolution.',
                       style={'color': 'black'}),
                html.Div([
                    html.Label('Select first feature:', style={'margin-right': '10px'}),
                    dcc.Dropdown(
                        id='feature-dropdown-1',
                        options=[{'label': value, 'value': key} for key, value in features_dictionary.items()],
                        value='fossil_gas_comp',
                        style={'width': '250px', 'fontSize': 'large', 'color': 'rgb(0,0,255)'}
                    ),
                ], style={'display': 'inline-block', 'margin-right': '20px'}),
                html.Div([
                    html.Label('Select second feature:', style={'margin-right': '10px'}),
                    dcc.Dropdown(
                        id='feature-dropdown-2',
                        options=[{'label': value, 'value': key} for key, value in features_dictionary.items()],
                        value='wind_on',
                        style={'width': '250px', 'fontSize': 'large', 'color': 'rgb(0,0,255)'}
                    ),
                ], style={'display': 'inline-block'})
            ]),
            html.Div([
                html.Label('Select year:'),
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[{'label': year, 'value': year} for year in sorted(raw_data['date'].dt.year.unique())],
                    value=2022,
                    style={'width': '50%', 'fontSize': 'large', 'color': 'rgb(0,0,255)'}
                ),
                html.Label('Select month:'),
                dcc.Dropdown(
                    id='month-dropdown',
                    options=[{'label': month_name, 'value': month_num} for month_name, month_num in
                             month_mapping.items()],
                    value=1,
                    style={'width': '50%', 'fontSize': 'large', 'color': 'rgb(0,0,255)'}
                )
            ]),
            html.Div(id='plot-output')
        ])

    elif tab == 'tab-3':
        return html.Div([
            html.Div([
                html.H5('Exploratory Data Analysis', style={'color': 'white'}),
                html.P('Explore relations between features through scatter plots and histograms.',
                       style={'color': 'black'}),
                html.Div([
                    html.Label('Select plot type:', style={'margin-right': '10px'}),
                    dcc.Dropdown(
                        id='plot-type-dropdown',
                        options=[
                            {'label': 'Scatter', 'value': 'scatter'},
                            {'label': 'Histogram', 'value': 'histogram'}
                        ],
                        value='scatter',
                        style={'width': '250px', 'fontSize': 'large', 'color': 'rgb(0,0,255)'}
                    ),
                ], style={'display': 'inline-block', 'margin-right': '20px'}),
                html.Div([
                    html.Label('Select first feature:', style={'margin-right': '10px'}),
                    dcc.Dropdown(
                        id='eda-feature-dropdown-1',
                        options=[{'label': value, 'value': key} for key, value in features_dictionary.items()],
                        value='fossil_gas_comp',
                        style={'width': '250px', 'fontSize': 'large', 'color': 'rgb(0,0,255)'}
                    ),
                ], style={'display': 'inline-block', 'margin-right': '20px'}),
                html.Div([
                    html.Label('Select second feature:', style={'margin-right': '10px'}),
                    dcc.Dropdown(
                        id='eda-feature-dropdown-2',
                        options=[{'label': value, 'value': key} for key, value in features_dictionary.items()],
                        value='wind_on',
                        style={'width': '250px', 'fontSize': 'large', 'color': 'rgb(0,0,255)'}
                    ),
                ], style={'display': 'inline-block'})
            ]),
            html.Div([
                html.Label('Select year:'),
                dcc.Dropdown(
                    id='eda-year-dropdown',
                    options=[{'label': year, 'value': year} for year in sorted(raw_data['date'].dt.year.unique())],
                    value=2022,
                    style={'width': '50%', 'fontSize': 'large', 'color': 'rgb(0,0,255)'}
                ),
                html.Label('Select month:'),
                dcc.Dropdown(
                    id='eda-month-dropdown',
                    options=[{'label': month_name, 'value': month_num} for month_name, month_num in
                             month_mapping.items()],
                    value=1,
                    style={'width': '50%', 'fontSize': 'large', 'color': 'rgb(0,0,255)'}
                )
            ]),
            html.Div(id='eda-plot-output')
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.Div([
                html.H5('Forecasting', style={'color': 'white'}),
                html.P('View machine learning models\' predictions of energy and CO\u2082 emissions '
                            'on a monthly basis.', style={'color': 'black'}),
                html.P(
                    'In order to get the CO\u2082 emissions, a conversion of 185 kgCO\u2082/MWh was adopted from the literature found in Project Overview.',
                    style={'color': 'black'}),
                html.Label('Select Machine Learning Model:'),
                dcc.Dropdown(
                    id='ml-model-dropdown',
                    options=[{'label': key, 'value': key} for key in ML_models.keys()],  # Use keys instead of values
                    value='XGB',  # Set default value to the first ML model key
                    style={'width': '50%', 'fontSize': 'large', 'color': 'rgb(0,0,255)'}
                )
            ]),
            html.Div([
                html.Label('Select Data to Display:'),
                dcc.RadioItems(
                    id='data-radio',
                    options=[
                        {'label': 'Energy', 'value': 'energy'},
                        {'label': 'CO2 Emitted', 'value': 'co2'}
                    ],
                    value='energy',  # Set default value to 'Energy'
                    labelStyle={'display': 'block'}
                )
            ]),
            html.Div(id='forecast-output'),
            html.Div([
                html.P('**** Check Additional Remarks on Project Overview.', style={'color': 'black'})
            ])
        ])
    elif tab == 'tab-5':
        return html.Div([
            html.Div([
                html.H5('Energy and CO\u2082 Budgets', style={'color': 'white'}),
                html.P('Observe the influence of surplus/deficit of energy generation '
                        'on CO\u2082 emissions on a monthly basis. Using the best model which was XGBoosting.', style={'color': 'black'}),
                html.P(
                    'The energy budget was calculated from the difference between the sum of all generation methods (including the forecasted natural gas usage) and the load. Energy Budget = Forecasted Generation - Load',
                    style={'color': 'black'}),
                html.P(
                    'The CO\u2082 budget was calculated from the CO\u2082 predicted and from whether the energy budget is positive or negative. If it\'s positive, then we\'re exporting energy; if it\'s negative, we\'re importing energy.',
                    style={'color': 'black'}),
                html.P('CO\u2082 Budget = CO\u2082 Predicted - Energy Budget * MWh_to_CO2 * SHARE_gas. SHARE_gas is 0.35 if exporting and 0.5 if importing.', style={'color': 'black'}),
                html.Label('Select Budget Type:'),
                dcc.RadioItems(
                    id='budget-radio',
                    options=[
                        {'label': 'Energy Budget', 'value': 'energy'},
                        {'label': 'CO2 Budget', 'value': 'co2'}
                    ],
                    value='energy',  # Set default value to 'CO2 Budget'
                    labelStyle={'display': 'block'}
                )
            ]),
            html.Div(id='budget-plot-output'),
            html.Div([
                html.P('**** Check Additional Remarks on Project Overview.', style={'color': 'black'})
            ])
        ])

@app.callback(Output('plot-output', 'children'),
              [Input('feature-dropdown-1', 'value'),
               Input('feature-dropdown-2', 'value'),
               Input('year-dropdown', 'value'),
               Input('month-dropdown', 'value')])

def generate_line_plot(feature1, feature2, year, month):
    selected_data = raw_data[(raw_data['date'].dt.year == year) & (raw_data['date'].dt.month == month)]
    selected_data = selected_data.set_index('date')  # Set the 'date' column as index
    plot_data = {
        'data': [
            {'x': selected_data.index, 'y': selected_data[feature1], 'type': 'line', 'name': features_dictionary[feature1]},
            {'x': selected_data.index, 'y': selected_data[feature2], 'type': 'line', 'name': features_dictionary[feature2]}
        ],
        'layout': {
            'title': f'{features_dictionary[feature1]} vs {features_dictionary[feature2]}',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Energy Produced [MWh]'}
        }
    }
    return dcc.Graph(figure=plot_data)

# Callback to generate and display the EDA plot
@app.callback(Output('eda-plot-output', 'children'),
              [Input('plot-type-dropdown', 'value'),
               Input('eda-feature-dropdown-1', 'value'),
               Input('eda-feature-dropdown-2', 'value'),
               Input('eda-year-dropdown', 'value'),
               Input('eda-month-dropdown', 'value')])

def plot_eda(plot_type, feature1, feature2, year, month):
    selected_data = raw_data[(raw_data['date'].dt.year == year) & (raw_data['date'].dt.month == month)]
    if plot_type == 'scatter':
        plot_data = {
            'data': [
                {'x': selected_data[feature1], 'y': selected_data[feature2], 'mode': 'markers', 'type': 'scatter'}
            ],
            'layout': {
                'title': f'{features_dictionary[feature1]} vs {features_dictionary[feature2]}',
                'xaxis': {'title': features_dictionary[feature1]},
                'yaxis': {'title': features_dictionary[feature2]}
            }
        }
    elif plot_type == 'histogram':
        bins = 150  # You can adjust the number of bins as needed
        plot_data = {
            'data': [
                {'x': selected_data[feature1], 'type': 'histogram', 'name': features_dictionary[feature1],
                 'opacity': 0.5, 'nbinsx': bins},
                {'x': selected_data[feature2], 'type': 'histogram', 'name': features_dictionary[feature2],
                 'opacity': 0.5, 'nbinsx': bins}
            ],
            'layout': {
                'title': f'{features_dictionary[feature1]} and {features_dictionary[feature2]} Histogram',
                'barmode': 'overlay',
                'xaxis': {'title': 'Value'},
                'yaxis': {'title': 'Frequency'}
            }
        }
    else:
        return html.Div("Invalid plot type selected")

    return dcc.Graph(figure=plot_data)


@app.callback(
    Output('forecast-output', 'children'),
    [Input('ml-model-dropdown', 'value'),
     Input('data-radio', 'value')]
)
def generate_bar_plot(ml_model, data_option):
    # Load the data corresponding to the selected ML model
    data_file = ML_models[ml_model]
    ml_data = pd.read_csv(data_file)
    ml_data['date'] = pd.to_datetime(ml_data['date'])

    # Select the appropriate column based on the data option
    if data_option == 'energy':
        column = 'prediction'
        y_axis_title = 'Energy Predicted by Gas Usage [MWh]'
        bar_color = 'grey'  # Grey color for Energy bars
    elif data_option == 'co2':
        column = 'co2_predicted'
        y_axis_title = 'CO2 emitted by Gas Usage [tons]'
        bar_color = 'lightcoral'  # Light red color for CO2 bars
    else:
        return html.Div("Invalid data option selected")

    # Group data by month and sum the values
    ml_data['month'] = ml_data['date'].dt.month
    grouped_data = ml_data.groupby('month')[column].sum()

    # Convert month numbers to month names
    grouped_data.index = grouped_data.index.map(lambda x: list(month_mapping.keys())[list(month_mapping.values()).index(x)])

    # Create the bar plot with horizontally oriented bars
    bar_plot = {
        'data': [
            {'y': grouped_data.index, 'x': grouped_data.values, 'type': 'bar', 'name': 'Energy Prediction',
             'orientation': 'h', 'marker': {'color': bar_color}}  # Set bar color
        ],
        'layout': {
            'title': f'Forecasting of Gas Usage in Portugal by Month ({ml_model})',
            'xaxis': {'title': y_axis_title},  # Update x-axis title based on data_option
            'yaxis': {'title': 'Month'},  # Set y-axis title as Month
            'annotations': [{'x': value, 'y': key, 'text': f'{round(value):.0f}', 'xanchor': 'left', 'showarrow': False} for key, value in enumerate(grouped_data.values)]
        }
    }

    return dcc.Graph(figure=bar_plot)

@app.callback(
    Output('budget-plot-output', 'children'),
    [Input('budget-radio', 'value')]
)
def generate_budget_bar_plot(budget_option):
    if budget_option == 'co2':
        # Load the XGB data
        data_file = ML_models['XGB']
        ml_data = pd.read_csv(data_file)
        ml_data['date'] = pd.to_datetime(ml_data['date'])

        # Group data by month and sum the values
        ml_data['month'] = ml_data['date'].dt.month
        grouped_data = ml_data.groupby('month')[['co2_budget', 'co2_predicted']].sum()

        # Convert month numbers to month names
        grouped_data.index = grouped_data.index.map(lambda x: list(month_mapping.keys())[list(month_mapping.values()).index(x)])

        # Create the stacked bar plot
        bar_plot = {
            'data': [
                {'x': grouped_data['co2_predicted'], 'y': grouped_data.index, 'type': 'bar', 'name': 'CO2 Predicted',
                 'orientation': 'h', 'marker': {'color': 'blue', 'opacity': 0.5}},
                {'x': grouped_data['co2_budget'], 'y': grouped_data.index, 'type': 'bar', 'name': 'CO2 Budget',
                 'orientation': 'h', 'marker': {'color': 'lightcoral', 'opacity': 0.8}}
            ],
            'layout': {
                'title': 'CO2 Budget and Predicted CO2 by Month',
                'xaxis': {'title': 'CO2 [tons]'},
                'yaxis': {'title': 'Month'},
                'barmode': 'overlay'
            }
        }

        return dcc.Graph(figure=bar_plot)
    elif budget_option == 'energy':
        # Load the XGB data for energy budget
        data_file = ML_models['XGB']
        ml_data = pd.read_csv(data_file)
        ml_data['date'] = pd.to_datetime(ml_data['date'])

        # Group data by month and sum the values
        ml_data['month'] = ml_data['date'].dt.month
        grouped_data = ml_data.groupby('month')['budget'].sum()

        # Convert month numbers to month names
        grouped_data.index = grouped_data.index.map(lambda x: list(month_mapping.keys())[list(month_mapping.values()).index(x)])

        # Determine the color for negative and positive values
        colors = ['red' if val < 0 else 'green' for val in grouped_data.values]

        # Create the bar plot for energy budget
        bar_plot = {
            'data': [
                {'x': grouped_data.values, 'y': grouped_data.index, 'type': 'bar', 'name': 'Energy Budget',
                 'marker': {'color': colors}, 'orientation': 'h'}
            ],
            'layout': {
                'title': 'Energy Budget by Month',
                'xaxis': {'title': 'Energy [MWh]'},
                'yaxis': {'title': 'Month'},
                'annotations': [
                    {'x': value, 'y': key, 'text': f'{round(value):.0f}', 'xanchor': 'left', 'showarrow': False} for
                    key, value in enumerate(grouped_data.values)]
            }
        }

        return dcc.Graph(figure=bar_plot)
    else:
        return html.Div()  # Return an empty Div if an invalid option is selected


#if __name__ == '__main__':
#    app.run_server(debug=False)