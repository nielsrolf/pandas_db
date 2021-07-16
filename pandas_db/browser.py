from pandas_db.b import apply_filters
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_table
import dash_html_components as html
import pandas as pd
from pandas_db.pandas_db import pandas_db, maybe_float, DEFAULT_PANDAS_DB_PATH
import numpy as np
from dash_extensions import Keyboard
import os
import base64
import json
import click


@click.command()
@click.argument("view_name")
def main(view_name):
    """entry point to start the app"""
    with open(os.path.join(os.environ['PANDAS_DB_PATH'], ".pandas_db_views.json")) as json_file:
        views = json.load(json_file)
    view = views[view_name]
    app = init_app(keys=view['keys'], columns=view.get('columns'), file_id=view.get('file_id'))
    app.run_server(debug=True)


def init_app(keys, columns, file_id):
    """Initialize the html structure of the app, such that later the
    content can be filled via callback"""
    state = State(keys, columns, file_id)
    app = dash.Dash(__name__, title="Pandas DB")
    search = dcc.Input('global-search', type='text', style={"width": "100%", "height": "30px", "position": "fixed", "top": "0px", "z-index": "10"}, placeholder="Keyword search: enter any number of words you'd like to search")
    
    table_view = dcc.Loading(html.Div(get_main_table(state), id='table-view',
                            style={"padding-bottom": "500px"}))
    detail_view = html.Div(dcc.Loading(html.Div(id='medias')), style={
                "width": "100%",
                "background-color": "#2cb2cb",
                "max-height": "500px",
                "bottom": "30px",
                "position": "fixed",
                "overflow": "scroll"})
    file_filter = get_file_filter(state)
    app.layout = html.Div([
        search,
        table_view,
        detail_view,
        file_filter,
        html.Div(id='hidden'),
        Keyboard(id="keyboard")
    ])

    @app.callback(
        Output('table-filtering', 'data'),
        Input('global-search', 'value'),
        Input('table-filtering', "filter_query"))
    def update_table_data_clb(search_str, table_filters):
        return update_table_data(state, search_str, table_filters)
    
    @app.callback(
        Output('medias', 'children'),
        Input('global-search', 'value'),
        Input('table-filtering', "filter_query"),
        Input('file-filtering', "filter_query"),
        Input('table-filtering', 'active_cell'),
        Input('table-filtering', 'data'))
    def update_medias_clb(search_str, table_filters, file_filters, active_cell, table_data):
        return update_medias(state, search_str, table_filters, file_filters, active_cell, table_data)
    
    @app.callback(
        Output('hidden', 'children'),
        Input('table-filtering', 'data'),
        Input("keyboard", "keydown"))
    def save_clb(rows, key_event):
        save(state, rows, key_event)
    
    return app

    
def save(state, rows, key_event):
    if key_event is None:
        return
    if not (key_event['key'] == "s" and key_event["ctrlKey"] == True):
        return None
    edited = pd.DataFrame(rows, columns=state.keys+state.columns)
    cols = [c for c in edited.columns if c != "pandas_db.created"]
    edited = cols_maybe_float(edited)[cols]
    original = pandas_db.latest(keys=state.keys, df=state.get_transactions()).reset_index()
    original = edited[state.keys].merge(original, on=state.keys)
    original = original[edited.columns]
    row_idx, col_idx = np.where(original != edited)
    for r, c in zip(row_idx, col_idx):
        row = edited.iloc[r]
        previous = original.iloc[r]
        data = {k: row[k] for k in state.keys if row[k] != "-"}
        if row[edited.columns[c]] not in ["-", "?"]:
            data[edited.columns[c]] = row[edited.columns[c]]
        pandas_db.save(**data)
    state.fetch()
    return None

def cols_maybe_float(df):
    df = df.copy()
    for c in df.columns:
        df[c] = df[c].apply(maybe_float)
    return df


class State():
    """Class that caches the state of pandas_db and imlements transaction search"""
    def __init__(self, keys, columns, file_id):
        self.keys = keys
        self.columns = [c for c in columns if not c in self.keys]
        self.file_id = file_id
        self._transactions = None
        self.file_info = None
        self.fetch()
    
    def get_transactions(self, search_str=""):
        if self._transactions is None:
            return None
        dff = self._transactions
        if search_str is not None and search_str != "":
            search_index = self.search
            for i in search_str.replace(" ", ",").split(","):
                dff = dff.loc[search_index.str.contains(i)]
                search_index = search_index[search_index.str.contains(i)]
        return dff

    def fetch(self):
        self._transactions = pandas_db.get_df()
        self.search = self._transactions.fillna("").apply(concat_as_str, axis=1)
        self.file_info = pandas_db.latest(keys=["file"], df=self._transactions).reset_index()


def concat_as_str(values):
    return "".join([str(i) for i in values])


def update_table_data(state, search_str, table_filters):
    """Given all relevant input fields, rebuild the app but now filled with data"""
    transactions = state.get_transactions(search_str)
    df_main = apply_filters(transactions, table_filters)
    df_main = pandas_db.latest(keys=state.keys, df=df_main).reset_index()
    return df_main.to_dict('records')


def update_medias(state, search_str, table_filters, file_filters, active_cell, table_data):
    if active_cell is None:
        return

    transactions = state.get_transactions(search_str)
    df_main = apply_filters(transactions, table_filters)
    df_table = pd.DataFrame(table_data)
    df_files = apply_filters(df_main, file_filters)
    df_files = pandas_db.latest(keys=state.file_id, df=df_files).reset_index()
    row = df_table.iloc[active_cell['row']]
    sep = "!!!"
    files = df_files.groupby(by=state.keys).aggregate({"file": lambda i: sep.join(i.unique())}).reset_index()
    selected = pd.merge(
        pd.DataFrame(row).T[state.keys],
        files, how='left', on=state.keys).fillna("")
    selected = selected.sort_values(state.keys)
    if len(selected) == 0:
        return
    selected = selected['file'].iloc[0].split(sep)
    medias = []
    for rel_path in selected[:10]:
        try:
            file_info = state.file_info.loc[state.file_info['file']==rel_path].iloc[0]
            filepath = os.path.join(DEFAULT_PANDAS_DB_PATH, ".pandas_db_files", rel_path)
            medias += [show_media(state, filepath, file_info)]
        except (KeyError, FileNotFoundError, IndexError) as e:
            print(e)
            pass
    return medias


def show_media(state, media_file, file_info):
    data = str(base64.b64encode(open(media_file, 'rb').read()))[2:-1]
    media = None
    if media_file.endswith(".png"):
        media = html.Img(src='data:image/png;base64,{}'.format(data), style={"height": "300px", "width": "auto"})
    if media_file.endswith(".wav"):
        media = html.Audio(src='data:audio/wav;base64,{}'.format(data), controls=True)
    file_info = pd.DataFrame(file_info).T[state.file_id].T.reset_index()
    rename_cols = dict(zip(file_info.columns, ["key", "value"]))
    file_info = file_info.rename(columns=rename_cols)
    info = dash_table.DataTable(
        columns = [{"name": "", "id": "key"}, {"name": "", "id": "value"}],
        data=file_info.to_dict('records'),
        style_cell={'padding': '5px', 'width': "100px"},
        style_header = {'display': 'none'}
    )
    return html.Div([html.Div(info, style={"flex": "50%"}),
                        html.Div(media, style={
                            "flex": "50%",
                            "height": "300px",
                            "display": "flex",
                            "align-items": "center",
                            "padding": "5px"})],
                        style={
                            "display": "flex", 
                            "background-color": "rgba(0.9, 0.9, 0.9, 0.5)", 
                            "margin-bottom": "5px", "margin-top": "5px"})


def get_main_table(state, df=None):
    columns = state.keys + state.columns
    table = dash_table.DataTable(
        id='table-filtering',
        columns=[
            {"name": i, "id": i, "hideable": True, "editable": i not in state.keys} for i in columns
        ],
        style_cell={'padding': '5px', 'min-width': "50px"},
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold'
        },
        style_as_list_view=True,
        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'left'
            } for c in state.keys
        ],
        # fixed_rows={'headers': True},
        style_table={"height": "100%"},
        # page_current=0,
        # page_size=PAGE_SIZE,
        page_action='custom',
        filter_action='custom',
        filter_query=''
    )
    return table


def get_file_filter(STATE):
    file_filter = (STATE.file_id if STATE.file_id is not None else STATE.columns)
    file_filter = [i for i in file_filter if not i in STATE.keys]
    file_filter = dash_table.DataTable(
        id='file-filtering',
        columns=[
            {"name": i, "id": i} for i in file_filter
        ],
        style_cell={'padding': '5px', 'min-width': "50px"},
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold'
        },
        style_as_list_view=True,
        filter_action='custom',
        filter_query=''
    )
    return html.Div(file_filter, style={"position": "fixed", "bottom": "0", "height": "30", "width": "100%"})


operators = [['ge ', '>='],
            ['le ', '<='],
            ['lt ', '<'],
            ['gt ', '>'],
            ['ne ', '!='],
            ['eq ', '='],
            ['contains '],
            ['datestartswith ']]


def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


def apply_filters(dff, filter):
    filtering_expressions = filter.split(' && ')
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)
        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[(dff[col_name].apply(lambda i: isinstance(i, float))) & 
                            (getattr(dff[col_name].apply(float_or_zero), operator))(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].astype(str).str.contains(str(filter_value))]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].astype(str).str.startswith(filter_value)]
    return dff


if __name__ == '__main__':
    main()
