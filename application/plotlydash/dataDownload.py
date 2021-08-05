import dash
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import base64
import pandas as pd
import io
import dash_table
import os
import ast

from seaborn.matrix import heatmap
from .layouts.layout import html_layout
from dash.exceptions import PreventUpdate
import requests
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import base64

from flask_sqlalchemy import SQLAlchemy
import psycopg2

from .layouts.page2_layout import server_layout
from .layouts.page3_layout import layout3
from .modelHelperFunctions import *

from pandas_profiling import ProfileReport
# from .layouts.EDA_layout import EDA_layout
# from .EDA_callbacks import EDA_callback

## size convert
import enum
# Enum for size units
df = pd.DataFrame([])

file = open('data/var_info.txt', 'r')
contents = file.read()
dictionary = ast.literal_eval(contents)
file. close()

file1 = open('data/section_name.txt', 'r')
contents1 = file1.read()
dictionary_name = ast.literal_eval(contents1)
file1.close()

file2 = open('data/categorized_type.txt', 'r')
contents2 = file2.read()
categories = ast.literal_eval(contents2)
file2.close()

def load_info_dict(file):
    f = open(file, 'r')
    cont = f.read()
    f.close()
    return ast.literal_eval(cont)

VAR_PATH = 'data/var_info.txt'
STATE_PATH = 'data/state_info.txt'
SECTION_PATH = 'data/section_name.txt'
REGRESSON_LIST = ["Linear", "Lasso", "Ridge",
                  "LassoLars", "Bayesian Ridge", "Elastic Net"]
REG_CRITERION = ['Index', 'Label', 'Model', 'Penalty', 'MAE', 'MSE']
CLASSIFICATION_LIST = ["Logistic", "LDA"]
#CLF_CRITERION = ["Index", "Label", "Model", "Penalty", "Accuracy", "ROC_AUC score", "Precision", "Recall", "F1-Score"]
CLF_CRITERION = ["Index", "Label", "Model", "Penalty",
                 "Accuracy"]

var_info = load_info_dict(VAR_PATH)
section_info = load_info_dict(SECTION_PATH)
state_info = load_info_dict(STATE_PATH)

SECTION = list(section_info.keys())
STATE = list(state_info.keys())
class SIZE_UNIT(enum.Enum):
   BYTES = 1
   KB = 2
   MB = 3
   GB = 4
def convert_unit(size_in_bytes, unit):
   """ Convert the size from bytes to other units like KB, MB or GB"""
   if unit == SIZE_UNIT.KB:
       return size_in_bytes/1024
   elif unit == SIZE_UNIT.MB:
       return size_in_bytes/(1024*1024)
   elif unit == SIZE_UNIT.GB:
       return size_in_bytes/(1024*1024*1024)
   else:
       return size_in_bytes

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def dataDownload(server):
    global df
    app = dash.Dash(server=server,
                         routes_pathname_prefix='/dashapp/',
                         external_stylesheets=[
                             'https://codepen.io/dapraxis/pen/gOPGzPj.css',
                             '/static/dist/css/styles.css',
                             'https://fonts.googleapis.com/css?family=Lato',
                             'https://codepen.io/chriddyp/pen/bWLwgP.css'
                             ])

    allData = {'BRFSS':'1tNWPT9xW1jc3Qta_h4CGHp9lRbHM1540'}
    
    app.index_string = html_layout
    
    db = SQLAlchemy(server)

    app.scripts.config.serve_locally = True  # Uploaded to npm, this can work online now too.
    
    df = pd.DataFrame([])
    # df.to_csv(os.stat(r`str(os.getcwd())+'\\uploads\\'+str('download.csv')`))
    
    all_performance_layouts = []

    org_layout = html.Div([
        html.Div([], id='hidden-div', style={'display': 'none'}),
        dcc.Dropdown(
            id='demo-dropdown',
            options=[
                {'label': 'BRFSS', 'value': 'BRFSS'}
            ],
            searchable=False,
            placeholder="Select A Medical Dataset",
            style = {
                'width':'50%',
                'margin':'2% 0%'
            }
        ),
        html.Div(id='dd-output-container'),
        # html.Div(id='output')
        dcc.Loading(
            children=[
                html.Div(id='output')
                ], type='cube')
        ],style={
            'width': '70%',
            'margin': '5% 15%',
            # 'text-align': 'center',
        })

    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content')
    ])
    
    # @app.callback(Input('test', 'n_clicks'))
    
    @app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
    def display_page(pathname):
        if pathname == '/dashapp/':
            return org_layout
        elif pathname == '/dashapp/EDA/':
            # create_EDA(app, df)
            return server_layout
        elif pathname == '/dashapp/RFA/':
            return layout3
        
            # return '404'

    @app.callback(
        dash.dependencies.Output('dd-output-container', 'children'),
        [dash.dependencies.Input('demo-dropdown', 'value')])
    def update_output(value):
        if(not value):
            raise PreventUpdate
        return 'You have selected "{}"'.format(value)

    @app.callback(Output('output', 'children'),
                [Input('demo-dropdown', 'value')])
    def update_output(value):
        global df
        if value is not None:
            # file_id = allData[value]
            # destination = 'download.csv'
            # download_file_from_google_drive(file_id, destination)
            # connection = False
            if value =='BRFSS':
                df = pd.read_sql_query(sql = 'SELECT * FROM brfss2 LIMIT 100', con=db.engine, index_col='object_id')
                df.columns = df.columns.str.upper()
                return parse_contents(value)
        
    def parse_contents(filename):
        global df
        '''
        profile = ProfileReport(df, title = "Pandas Profiling Report")
        profile.to_file("report.html") #change it so that it automatically saves it in the folder where this file is
        These lines create the report using pandas profiling, however it takes quite long as the report is 11 MB. For now,
        only one dataset is used so these lines don't neeed to run each time
        '''
        new_df = df.iloc[:5, :5]
        return html.Div([
            # html.H5("Upload File: {}".format(filename)),
            html.Hr(),
            html.A(html.Button('Next', id='btn'), href='/dashapp/EDA/'),
            html.Hr(),
            dcc.Loading(children=[
                dash_table.DataTable(
                id='database-table',
                columns=[{'name': i, 'id': i} for i in new_df.columns],
                data=new_df.to_dict('records'),
                sort_action="native",
                sort_mode='native',
                page_size=300,
                fixed_rows = 100,
                style_table={
                    'maxHeight': '80ex',
                    #'overflowY': 'scroll',
                    #'overflowX': 'scroll',
                    'width': '100%',
                    'border' : '1px solid blue'
                },
                style_cell={
                    'color': 'blue'
                },
            ),

            html.Hr(),  # horizontal line
            #html.A(html.Button('Next', id='btn'), href='/dashapp/EDA/')
            ], type='cube')
        ])

    ## EDA STARTS
    @app.callback(dash.dependencies.Output('dropdown_content', 'children'),
                       [dash.dependencies.Input('dropdown_section_name', 'value')])
    def render_tab_preparation_multiple_dropdown(value):
        #print (dictionary_name)
        all_vals = []
        for i in dictionary_name.values():
            for j in i:
                all_vals.append(j)
        if value:
            for key in dictionary_name:
                if key == value:
                    return_div = html.Div([
                        html.Br(),
                        dcc.Dropdown(
                            id='dropdown',
                            options=[
                                {'label': i, 'value': i} for i in dictionary_name[key]
                            ],
                            placeholder="Select First Feature",
                            # value='features'
                        ),
                        #dcc.Dropdown(
                        #    id = 'dropdown2',
                        #    options = [
                        #        {'label': i, 'value' : i} for i in all_vals
                        #    ],
                        #    placeholder="Select Second Feature",
                        #),
                        html.Div(id='single_commands'),
                    ])
                    return return_div
        else:
            raise PreventUpdate


    '''
    @app.callback(dash.dependencies.Output('dropdown_content', 'children'),
                       [dash.dependencies.Input('dropdown_section_name', 'value2')])
    def render_tab_preparation_multiple_dropdown2(value2):
        all_vals = []
        for i in dictionary_name.values():
            for j in i:
                all_vals.append(j)
        return_div = html.Div([
            html.Br(),
            dcc.Dropdown(
                id = 'dropdown2',
                options = [
                    {'label': i, 'value' : i} for i in all_vals
                ],
                placeholder="Select Second Feature",
            ),
            html.Div(id='single_commands2'),
        ])
        return return_div
    '''


    @app.callback(
        dash.dependencies.Output('dd-notice', 'children'),
        [dash.dependencies.Input('dropdown', 'value'),])
    def update_selected_feature_div(value):
        if value:
            result = []
            for key, values in dictionary[value].items():
                result.append('{}:{}'.format(key, values))
            div = html.Div([
                html.Div([
                    html.H3('First Feature Informatation')
                ]),
                html.Div([
                    html.Ul([html.Li(x) for x in result])
                ]),
            ])

            return div
        else:
            raise PreventUpdate

    @app.callback(
        [dash.dependencies.Output('dd-output-container2', 'children'),
         dash.dependencies.Output('graph_plot', 'children')],
        [dash.dependencies.Input('dropdown', 'value')])
    def preparation_tab_information_report(value):
        global df
        if value:
            str_value = str(value)
            df = pd.read_sql_query(sql = 'SELECT {}, object_id FROM brfss2 LIMIT 100'.format(str_value.lower()), con=db.engine, index_col='object_id')
            R_dict = df.describe().to_dict()
            result = []
            for key in R_dict[str_value.lower()]:
                result.append('{}: {}'.format(key, R_dict[str_value.lower()][key]))

            div = html.Div([
                html.Div([
                    html.H3('First Feature Statistics')
                ]),
                html.Div([
                    html.Ul([html.Li(x) for x in result])
                ]),
            ])
            
            g = dcc.Loading(id='graph_loading', children=[
                                dcc.Graph(
                                figure={"layout": {
                                    "xaxis": {"visible": False},
                                    "yaxis": {"visible": False},
                                    "annotations": [{
                                        "text": "Please Select the Feature you would like to Visualize",
                                        "xref": "paper",
                                        "yref": "paper",
                                        "showarrow": False,
                                        "font": {"size": 28}
                                    }]
                                }}, id='dd-figure'),
                            ])
            return [div, g]
        else:
            raise PreventUpdate

        # Define a function for drawing box plot for selected feature

    @app.callback(
        dash.dependencies.Output('dd-figure', 'figure'),
        # [dash.dependencies.Input('dropdown', 'value'),dash.dependencies.Input("hidden-div", 'children')])
        [dash.dependencies.Input('dropdown', 'value')])
    def preparation_tab_visualize_features(value):
        global df
        if value:
            str_value = str(value).lower()
            df = pd.read_sql_query(sql = 'SELECT {}, object_id FROM brfss2 LIMIT 100'.format(str_value.lower()), con=db.engine, index_col='object_id')
            integers = categories[0]
            floats = categories[1]
            if str_value in integers:
                fig = px.histogram(df[str_value], y=str_value)
            elif str_value in floats:
                fig = px.box(df[str_value], y=str_value)
            else:
                fig = px.histogram(df[str_value], y=str_value)
            return fig
        else:
            raise PreventUpdate
        
    @app.callback(Output(component_id='feature_dropdown', component_property='options'),
                       [Input(component_id='section_dropdown', component_property='value')])
    def update_feature_dropdown(section):
        if section == None:
            return dash.no_update
        lst = section_info.get(section)
        return [{'label': '{}: {}'.format(i, var_info.get(i).get('Label')), 'value': i} for i in lst]

    @app.callback(Output(component_id='model_dropdown', component_property='options'),
                       [Input(component_id='type_dropdown', component_property='value')])
    def update_model_dropdown(task):
        if task == "Regression":
            return [{'label': i, 'value': i} for i in REGRESSON_LIST]
        elif task == "Classification":
            return [{'label': i, 'value': i} for i in CLASSIFICATION_LIST]
        else:
            return dash.no_update

    @app.callback(Output('slider-output-container', 'children'), [Input('num-of-factors', 'value')])
    def update_output(value):
        return 'You have selected {} top risk factors'.format(value)

    @app.callback(Output('penalty-output-container', 'children'), [Input('penalty', 'value')])
    def update_output(value):
        return 'You have selected {} as penalty multiplier'.format(value)

    @app.callback([Output('RFA_output', 'children'),
                        Output('reg_rec', 'data'),
                        Output('clf_rec', 'data')],
                       [Input('run_button', 'n_clicks')],
                       [State('state_dropdown', 'value'),
                        State('feature_dropdown', 'value'),
                        State('type_dropdown', 'value'),
                        State('model_dropdown', 'value'),
                        State('penalty', 'value'),
                        State('num-of-factors', 'value'),
                        State('reg_rec', 'data'),
                        State('clf_rec', 'data')])
    def perform_risk_factor_analysis(n_clicks, state, label, task_type, model_type, penalty, num_of_factor, reg_data, clf_data):
        global df

        if n_clicks > 0:

            if((label == None) or (task_type == None) or (model_type == None)):
                # return [], reg_data, clf_data, True
                return dash.no_update, dash.no_update, dash.no_update

            state_df = pd.read_sql_query(sql = 'SELECT * FROM brfss2 WHERE _state = {}'.format(int(state)), con=db.engine, index_col='object_id')
            state_df.columns = state_df.columns.str.upper()
            y = state_df[label]
            X = state_df.drop([label], axis=1)
            col_names = X.columns

            if task_type == "Regression":
                model_res = regression_models(
                    X, y, model_type, True, alpha=penalty)
                model = model_res[0]
                res = reg_risk_factor_analysis(model, col_names, num_of_factor)

                performance_layout = html.Div(
                    html.Div(
                        dash_table.DataTable(
                            id="reg_table",
                            columns=[{'name': val, 'id': val}
                                     for val in REG_CRITERION[1:]],
                            data=[{"Label": label, 'Model': model_type, 'Penalty': penalty, 'MAE': round(model_res[1], 5), 'MSE':round(model_res[2], 5),
                                   'R2':round(model_res[3], 5)}],
                            style_cell={
                                'height': 'auto',
                                'textAlign': 'right'
                                # all three widths are needed
                                # 'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                # 'whiteSpace': 'normal'
                            },
                        )
                    ),
                )
                info = "Perform Risk Factor Analysis with normalized data based on {} regression".format(
                    model_type)
            elif task_type == "Classification":
                model_res = classification_models(
                    X, y, model_type, True, C=penalty)
                model = model_res[0]
                cfn_matrix = model_res[5]
                #heatmap_filename = model_res[-1]
                #encoded_image = base64.b64encode(open(heatmap_filename, 'rb').read())
                res = clf_risk_factor_analysis(model, col_names, num_of_factor, label)[0]
                table_columns = []
                categories = clf_risk_factor_analysis(model, col_names, num_of_factor, label)[1]
                table_columns.append(categories) #categories (yes, no, etc.)
                table_columns.append(model_res[2]) #precision
                table_columns.append(model_res[3]) #recall
                table_columns.append(model_res[4]) #f1
                accuracy_column = [model_res[1]] * len(table_columns[1])
                table_columns.append(accuracy_column)
                #print (accuracy_column)
                CLF_CRITERION2 = ["Category", "Precision", "Recall", "F1", "Accuracy"]
                #print (table_columns)
                if len(table_columns[0]) == len(table_columns[1]) + 1:
                    table_columns[0] = table_columns[0][:-1]
                elif len(table_columns[0]) == len(table_columns[1]) + 2:
                    table_columns[0] = table_columns[0][:-2]
                elif len(table_columns[0]) == len(table_columns[1]) - 1:
                    i = 1
                    while (i <= 4):
                        table_columns[i] = table_columns[i][:-1]
                        i += 1
                table_dict = {"Category" : table_columns[0], "Precision" : table_columns[1], "Recall" : table_columns[2], "F1": table_columns[3], "Accuracy" : table_columns[4]}
                table_df = pd.DataFrame(table_dict)
                #print (table_df)
                performance_layout = html.Div(
                    #html.Div(
                    #    dash_table.DataTable(
                    #        id="clf_table",
                    #        columns=[{'name': val, 'id': val}
                    #                 for val in CLF_CRITERION[1:]],
                    #        #data=[{"Label": label, 'Model': model_type, "Penalty": penalty, "Accuracy": round(model_res[1], 5),
                    #        #       "Precision":round(model_res[2], 5), "Recall":round(model_res[3], 5), "F1-Score":round(model_res[4], 5)}],
                    #        data=[{"Label": label, 'Model': model_type, "Penalty": penalty, "Accuracy": model_res[1]}],
                    #        style_cell={
                    #            'height': 'auto',
                    #            'textAlign': 'right'
                    #            # all three widths are needed
                    #            # 'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                    #            # 'whiteSpace': 'normal'
                    #        }
                    #    )
                    #),
                    html.Div(
                        dash_table.DataTable(
                            id="clf_table",
                            columns = [{'name' : val, 'id' : val}
                                    for val in CLF_CRITERION2],
                            data = table_df.to_dict('records'),
                            style_cell = {
                                'height' : 'auto',
                                'textAllign' : 'right',
                            }
                        )
                    )
                )
                all_performance_layouts.append(performance_layout)
                info = "Perform Risk Factor Analysis with normalized data based on {} model".format(
                    model_type)
            else:
                return [], reg_data, clf_data

            res_tab_col = ["Rank", "Factor", "Absolute Weight", "Sign"]
            #res = reg_risk_factor_analysis(model, col_names, num_of_factor)

            if task_type == "Classification":
                history_html = html.Div(children=[])
                for i in range(len(all_performance_layouts)):
                    text = "Performance table for Index " + str(i+1)
                    temp_html = html.Details([
                        html.Summary(text),
                        all_performance_layouts[i]])
                    history_html.children.append(temp_html)
                ######## so history_html is a list with all the temp_htmls, and they need to be combined via comments like
                ''' 
                html.Details([
                    html.Summray(text),
                    all_performance_layouts[0])],
                html.Details([
                    html.Summary(text), 
                    all_performance_layouts[1])],
                and so on
                '''
                layout = html.Div(children=[
                    html.P(
                        html.Label(info)
                    ),
                    html.Div(
                        dash_table.DataTable(
                            id="RFA_table",
                            columns=[
                                {'name': i, 'id': i} for i in res_tab_col
                            ],
                            data=res,
                            style_cell={
                                'height': 'auto',
                                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                'whiteSpace': 'normal',
                                'textAlign': 'right'
                            },
                            style_header={
                                'backgroundColor': 'white',
                                'fontWeight': 'bold'
                            },
                            style_data_conditional=[
                                {
                                    'if': {
                                        'column_id': 'Sign',
                                        'filter_query': '{Sign} = "-"'
                                    },
                                    'backgroundColor': 'dodgerblue',
                                    'color': 'white'
                                },
                                {
                                    'if': {
                                        'column_id': 'Sign',
                                        'filter_query': '{Sign} = "+"'
                                    },
                                    'backgroundColor': '#85144b',
                                    'color': 'white'
                                },
                            ],
                        )
                    ),

                    html.P(
                        html.Label("{} model performance: ".format(model_type))
                    ),
                    performance_layout,
                    #html.Div([
                    #    html.Label("Heatmap for the Confusion Matrix:"),
                    #    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))   #encoded_image does not exist for continous variables
                    #]),
                    html.Div([
                        html.Details([
                            html.Summary("Performance of History Models"),
                            html.Details([ 
                                html.Summary("Performance Records for Classification Model"),
                                html.Div(
                                    dash_table.DataTable(
                                        id="clf_rec",
                                        columns=[{'name': val, 'id': val}
                                                 for val in CLF_CRITERION],
                                        data=[],
                                        style_cell={
                                            'height': 'auto',
                                            'textAlign': 'right'
                                            # all three widths are needed
                                            # 'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                            # 'minWidth': '100px', 'width': '120px', 'maxWidth': '240px',
                                            # 'whiteSpace': 'normal'
                                        }
                                    )
                                ),
                                #html.Details([
                                    #html.Summary("Performance Table"),
                                    #performance_layout])
                                history_html
                            ])
                        ])
                    ])



                ])
            elif task_type == "Regression":
                layout = html.Div(children=[
                    html.P(
                        html.Label(info)
                    ),
                    html.Div(
                        dash_table.DataTable(
                            id="RFA_table",
                            columns=[
                                {'name': i, 'id': i} for i in res_tab_col
                            ],
                            data=res,
                            style_cell={
                                'height': 'auto',
                                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                'whiteSpace': 'normal',
                                'textAlign': 'right'
                            },
                            style_header={
                                'backgroundColor': 'white',
                                'fontWeight': 'bold'
                            },
                            style_data_conditional=[
                                {
                                    'if': {
                                        'column_id': 'Sign',
                                        'filter_query': '{Sign} = "-"'
                                    },
                                    'backgroundColor': 'dodgerblue',
                                    'color': 'white'
                                },
                                {
                                    'if': {
                                        'column_id': 'Sign',
                                        'filter_query': '{Sign} = "+"'
                                    },
                                    'backgroundColor': '#85144b',
                                    'color': 'white'
                                },
                            ],
                        )
                    ),

                    html.P(
                        html.Label("{} model performance: ".format(model_type))
                    ),
                    performance_layout,
                ])


            if task_type == "Regression":
                return layout, reg_data + [{"Index": len(reg_data)+1, "Label": label, 'Model': model_type, 'Penalty': penalty, 'MAE': round(model_res[1], 5), 'MSE':round(model_res[2], 5),
                                            }], clf_data
            elif task_type == "Classification":
                return layout, reg_data, clf_data + [{"Index": len(clf_data)+1, "Label": label, 'Model': model_type, "Penalty": penalty, "Accuracy": model_res[1]}]
            else:
                return [], reg_data, clf_data
        else:
            # return [], reg_data, clf_data, False
            return dash.no_update, dash.no_update, dash.no_update


