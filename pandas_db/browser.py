from numpy.lib.type_check import imag
import dash
from dash.dependencies import Input, Output
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


PAGE_SIZE = 30




def drop_constant_columns(df):
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df = df.drop(col, axis=1)
    return df


def cols_maybe_float(df):
    df = df.copy()
    for c in df.columns:
        df[c] = df[c].apply(maybe_float)
    return df


def float_or_zero(v):
    if isinstance(v, float):
        return v
    else:
        return 0.


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


def get_selection_view(STATE):
    pass


def get_main_table(STATE):
    return dash_table.DataTable(
        id='table-filtering',
        columns=[
            {"name": i, "id": i, "hideable": True, "editable": i not in STATE.keys} for i in STATE.df.columns
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
            } for c in STATE.keys
        ],
        # fixed_rows={'headers': True},
        style_table={"height": "100%"},
        page_current=0,
        page_size=PAGE_SIZE,
        page_action='custom',
        filter_action='custom',
        filter_query=''
    )


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
        page_current=0,
        page_size=1,
        page_action='custom',
        filter_action='custom',
        filter_query=''
    )
    return html.Div([file_filter, html.Div(style={"content": '""', "clear": "both", "display": "table"})])

class State():
    def __init__(self, keys, columns=None, file_id=None):
        self.keys = keys
        self.columns = columns
        self.file_id = file_id
        self.df = None
        self.fetch()
    
    def fetch(self):
        print("fetch")
        df = pandas_db.latest(keys=self.keys).reset_index()
        if self.columns is not None:
            columns = self.columns
        else:
            columns = df.columns
        columns = [c for c in columns if not c in self.keys]
        df = df[self.keys + columns]
        df = drop_constant_columns(df)
        self.df = cols_maybe_float(df)
        print("fetched", len(self.df))


def get_app(keys, columns=None, file_id=None):
    print("file_id", file_id)

    STATE = State(keys, columns, file_id)
    external_stylesheets = ['https://raw.githubusercontent.com/plotly/dash-app-stylesheets/master/dash-diamonds-explorer.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets, title="Pandas DB")
    
    selection_view = get_selection_view(STATE)
    main_table = get_main_table(STATE)
    file_filter = get_file_filter(STATE)
    
    

    @app.callback(
        Output('detail-view', 'children'),
        Input('table-filtering', 'active_cell'),
        Input('table-filtering', "page_current"),
        Input('table-filtering', "page_size"),
        Input('table-filtering', "filter_query"),
        Input('file-filtering', "filter_query"))
    def show_detail_view(active_cell, page_current, page_size, filter, file_filter):
        if STATE.df is None or active_cell is None:
            return ""
        df = get_current_selection(page_current, page_size, filter)
        row = df.iloc[active_cell['row']]
        # Find all media files connected to the key of the row
        df = pandas_db.get_df().fillna("?")
        # df = get_current_selection(page_current=0, page_size=10000, filter=filter, dff=df)
        df = apply_filters(df, file_filter)
        df = apply_filters(df, filter)
        if len(df) == 0:
            return []
        def latest_entry(values):
            values = values.dropna()
            if len(values) == 0:
                return None
            return values[-1]
        file_id = None
        if STATE.file_id:
            file_id = STATE.file_id
            df = df.sort_values("pandas_db.created")\
                    .groupby(by=file_id)\
                    .aggregate(latest_entry)\
                    .reset_index()
        sep = "!!!"
        files = df.groupby(by=STATE.keys).aggregate({"file": lambda i: sep.join(i.unique())}).reset_index()
        selected = pd.merge(
            pd.DataFrame(row).T[STATE.keys],
            files, how='left', on=STATE.keys).fillna("")
        selected = selected.sort_values(STATE.keys)
        if file_id is None:
            file_id = selected.columns
        if len(selected) == 0:
            return []
        selected = selected['file'].iloc[0].split(sep)
        print(selected)
        medias = []
        for rel_path in selected[:5]:
            try:
                file_info = df.loc[df['file']==rel_path].iloc[0]
                filepath = os.path.join(DEFAULT_PANDAS_DB_PATH, ".pandas_db_files", rel_path)
                medias += [show_media(filepath, file_info)]
            except (KeyError, FileNotFoundError, IndexError):
                pass
        return medias


    def show_media(media_file, file_info):
        data = str(base64.b64encode(open(media_file, 'rb').read()))[2:-1]
        media = None
        if media_file.endswith(".png"):
            media = html.Img(src='data:image/png;base64,{}'.format(data), style={"height": "300px", "width": "auto"})
        if media_file.endswith(".wav"):
            media = html.Audio(src='data:audio/wav;base64,{}'.format(data), controls=True)
        file_id = STATE.file_id if STATE.file_id is not None else file_info.keys()
        file_info = pd.DataFrame(file_info).T[file_id].T.reset_index()
        rename_cols = dict(zip(file_info.columns, ["key", "value"]))
        file_info = file_info.rename(columns=rename_cols)
        info = dash_table.DataTable(
            columns = [{"name": "", "id": "key"}, {"name": "", "id": "value"}],
            data=file_info.to_dict('records'),
            style_cell={'padding': '5px', 'width': "100px"},
            style_header = {'display': 'none'}
        )
        # return media
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
                             "margin": "5px"})


    @app.callback(
        Output('hidden', 'children'),
        Input('table-filtering', 'data'),
        Input("keyboard", "keydown"))
    def save(rows, key_event):
        if key_event is None:
            return
        if not (key_event['key'] == "s" and key_event["ctrlKey"] == True):
            return None
        STATE.fetch()
        edited = pd.DataFrame(rows, columns=STATE.df.columns)
        cols = [c for c in edited.columns if c != "pandas_db.created"]
        edited = cols_maybe_float(edited)[cols]
        original = STATE.df[cols]

        original = edited[STATE.keys].merge(original, on=STATE.keys)
        original = original[edited.columns]
        row_idx, col_idx = np.where(original != edited)
        for r, c in zip(row_idx, col_idx):
            row = edited.iloc[r]
            previous = original.iloc[r]
            print(edited.columns[c], f":{previous[c]} -> {row[c]}")
            data = {k: row[k] for k in STATE.keys}
            data[edited.columns[c]] = row[edited.columns[c]]
            pandas_db.save(**data)
        STATE.fetch()
        return None

    @app.callback(
        Output('table-filtering', 'data'),
        Input('table-filtering', "page_current"),
        Input('table-filtering', "page_size"),
        Input('table-filtering', "filter_query"))
    def update_table(page_current,page_size, filter):
        return get_current_selection(page_current,page_size, filter).to_dict('records')
    
    def get_current_selection(page_current, page_size, filter, dff=None):
        print(filter)
        dff = dff if dff is not None else STATE.df
        dff_original = dff
        dff = apply_filters(dff, filter)
        if len(dff) == 0:
            dff = dff_original
        dff = dff.iloc[
            page_current*page_size:(page_current+ 1)*page_size
        ]
        print(len(dff))
        return dff


    table_view = html.Div(main_table, id='table-view', style={"position": "absolute", "padding-bottom": "500px"})
    detail_view = html.Div([file_filter, html.Div(id='detail-view')], style={
                "position":
                "fixed", "bottom": "0",
                "width": "100%",
                "background-color": "#2cb2cb",
                "max-height": "500px",
                "padding": "10px",
                "overflow": "scroll"})
    hidden_view = html.Div(id='hidden')

    app.layout = html.Div([
        table_view,
        detail_view,
        hidden_view,
        Keyboard(id="keyboard")
    ])
    return app


@click.command()
@click.argument("view_name")
def main(view_name):
    with open(os.path.join(os.environ['PANDAS_DB_PATH'], ".pandas_db_views.json")) as json_file:
        views = json.load(json_file)
    view = views[view_name]
    app = get_app(keys=view['keys'], columns=view.get('columns'), file_id=view.get('file_id'))
    app.run_server(debug=True)


if __name__ == '__main__':
    main()
