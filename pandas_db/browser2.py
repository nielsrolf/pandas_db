import pandas_db
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_table
import dash_html_components as html
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pandas_db.pandas_db import PandasDB, maybe_float, DEFAULT_PANDAS_DB_PATH
import numpy as np
from dash_extensions import Keyboard
import os
import base64
import json
import click
import functools


if os.path.exists(os.path.join(DEFAULT_PANDAS_DB_PATH, "migrated.csv")):
    pandas_db = PandasDB(csv_path=os.path.join(DEFAULT_PANDAS_DB_PATH, "migrated.csv"))
else:
    pandas_db = PandasDB()


@click.command()
@click.argument("view_name")
def main(view_name):
    """entry point to start the app"""
    with open(os.path.join(DEFAULT_PANDAS_DB_PATH, ".pandas_db_views.json")) as json_file:
        views = json.load(json_file)
    view = views[view_name]
    app = init_app(keys=view['keys'], columns=view.get('columns'), file_id=view.get('file_id'))
    app.run_server(host='0.0.0.0', port=8050)


def jupyter(keys, columns, file_id, **server_args):
    """Run the browser in a jupyter browser
    
    Arguments:
        keys (List[str]): columns that you want to group information by
        columns (List[str]): columns that hold metrics or other data,
            where you want to see the latest info per group
        file_id (List[str]): columns that together uniquely identify a file.
            If multiple files exist per group, the latest is displayed
    """
    app = init_app(keys=keys, columns=columns, file_id=file_id, jupyter=True)
    app.run_server(**server_args)


def init_app(keys, columns, file_id, jupyter=False):
    """Initialize the html structure of the app, such that later the
    content can be filled via callback"""
    state = State(keys, columns, file_id)
    if not jupyter:
        app = dash.Dash(__name__)
    else:
        app = JupyterDash(__name__)
    
    df = pandas_db.get_df()
    metrics_df = pandas_db.latest(keys=keys, metrics=state.columns, df=df)

    dropdown_fields = state.keys + [i for i in state.file_id if i not in state.keys]
    
    dropdowns =  html.Div(dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id=f"dropdown-{key}",
                            options=[{'label': i, 'value': i} for i in list(df[key].dropna().unique()) if i != ""],
                            multi=True,
                            style={"font-size": "13px"},
                            placeholder=key)
                    ], md=1) for key in dropdown_fields], align="center", no_gutters=True))
    
    metrics_plots = html.Div(id='metrics-view')

    @functools.lru_cache(maxsize=20)
    def filter_df(*dropdown_values):
        dropdown_values = unsmask_dropdown_values(dropdown_values)
        groupby_keys = []
        filtered_df = metrics_df.reset_index()
        for key, value in zip(dropdown_fields, dropdown_values):
            if value is not None and len(value) > 0:
                filtered_df = filtered_df[filtered_df[key].isin(value)]
                if len(value) > 1:
                    groupby_keys.append(key)
        filtered_df["key"] = filtered_df[groupby_keys].apply(lambda x: "\n".join([str(i) for i in x.to_dict().values()]), axis=1)
        return filtered_df, groupby_keys

    @app.callback(
        Output('metrics-view', 'children'),
        [Input('dropdown-{}'.format(key), 'value') for key in dropdown_fields])
    def get_metrics_view(*dropdown_values):
        dropdown_values = mask_dropdown_values(dropdown_values)
        filtered_df, groupby_keys = filter_df(*dropdown_values)
        return get_metric_plot(filtered_df, state.columns, groupby_keys)
    
    search = dcc.Input('global-search', type='text', style={"width": "100%", "height": "30px", "z-index": "10", "border": "2px solid #2cb2cb", "position": "fixed", "bottom": "1px"}, placeholder="Keyword search: enter any number of words you'd like to search")

    detail_view = html.Div([
        search,
        dcc.Loading(html.Div(id='medias'))],
        style={
                "width": "100%",
                "background-color": "#2cb2cb",
                "bottom": "30px"})
    @functools.lru_cache(maxsize=20)
    def global_search(*dropdown_values, search_str=""):
        dropdown_values = unsmask_dropdown_values(dropdown_values)
        groupby_keys = []
        filtered_df = state.file_info.reset_index()
        search_index = state.search
        for key, value in zip(dropdown_fields, dropdown_values):
            if value is not None and len(value) > 0:
                selection = filtered_df[key].isin(value)
                search_index = search_index.loc[selection]
                filtered_df = filtered_df.loc[selection]
                if len(value) > 1:
                    groupby_keys.append(key)
        filtered_df["key"] = filtered_df[groupby_keys].apply(lambda x: "\n".join([str(i) for i in x.to_dict().values()]), axis=1)
        if filtered_df is None:
            return None
        if search_str is not None and search_str != "":
            for i in search_str.replace(" ", ",").split(","):
                selection = search_index['search_index'].str.contains(i)
                filtered_df = filtered_df.loc[selection]
                search_index = search_index.loc[selection]
        return filtered_df, groupby_keys

    @app.callback(
        Output('medias', 'children'),
        [Input('global-search', 'value')] + \
        [Input('dropdown-{}'.format(key), 'value') for key in dropdown_fields])
    def update_medias_clb(search_str, *dropdown_values):
        dropdown_values = mask_dropdown_values(dropdown_values)
        df_files , groupby_keys = global_search(*dropdown_values, search_str=search_str)
        if len(df_files) == 0:
            return
        keys = [key for key in state.file_id if key not in groupby_keys] + groupby_keys

        df_files = df_files.fillna("")
        df_files = pandas_db.latest(keys=keys, df=df_files).reset_index()
        df_files = df_files.loc[(df_files.file!="") & (df_files.file != "?")]
        return update_medias(state, df_files)

    app.layout = dbc.Container([
        dropdowns,
        metrics_plots,
        detail_view
    ])
    return app


def mask_dropdown_values(dropdown_values):
    masked_values = ["||".join(i) if isinstance(i, list) else i for i in dropdown_values]
    return [i if i != "" else None for i in masked_values]


def unsmask_dropdown_values(dropdown_values):
    return [i.split("||") if isinstance(i, str) else i for i in dropdown_values]


def get_metric_plot(df, metrics, groupby_keys):
    metric_plots = []
    vals = df.reset_index()

    if len(vals) == 0:
        return None

    mean_metrics = vals.groupby("key").aggregate(np.nanmean)
    for metric in metrics:
        mean_metric = mean_metrics.sort_values(metric)
        ordered_keys = mean_metric.index.values
        category_order = {"key": ordered_keys}
        fig = go.Figure()
        for key in ordered_keys:
            res = vals.loc[vals.key==key][metric].dropna().values
            fig.add_trace(go.Histogram(x=res, name=key, histnorm='probability'))
        fig.update_layout(barmode='overlay')
        fig.update_traces(opacity=min(1, 0.25 + 1/len(vals)))
        fig.update_layout(
            title=metric,
            xaxis_title=metric,
            yaxis_title="Density",
        )
        histograms = dcc.Graph(id=f"scatter-{metric}", figure=fig)
        fig = px.box(vals, x="key", y=metric, category_orders=category_order, color="key")
        boxplots = dcc.Graph(id=f"boxplot-{metric}", figure=fig)
        metric_plots += [html.Div([histograms, boxplots])]
    return metric_plots


class State():
    """Class that caches the state of pandas_db and imlements transaction search"""
    def __init__(self, keys, columns, file_id):
        self.keys = keys
        self.columns = [c for c in columns if not c in self.keys]
        self.file_id = file_id
        self._transactions = None
        self.file_info = None
        self.fetch()

    def fetch(self):
        self._transactions = pandas_db.get_df()
        self.file_info = pandas_db.latest(keys=["file"], df=self._transactions)
        self.search = self.file_info.fillna("").reset_index()
        self.search['search_index'] = self.search.apply(concat_as_str, axis=1)


def concat_as_str(values):
    return "".join([str(i) for i in values])


def update_medias(state, df_files):
    medias = []
    for rel_path in df_files["file"].values[:5]:
        try:
            file_info = state.file_info.loc[rel_path]
            medias += [show_media(state, rel_path, file_info)]
        except (KeyError, FileNotFoundError, IndexError) as e:
            print(e)
            pass
    return medias


def show_media(state, media_file, file_info):
    if os.environ.get("PANDAS_DB_S3_PREFIX") is not None:
        if media_file.endswith(".png"):
            media = html.Img(src=f"{os.environ.get('PANDAS_DB_S3_PREFIX')}.pandas_db_files/{media_file}", style={"height": "300px", "width": "auto"})
        elif media_file.endswith(".wav"):
            media = html.Audio(src=f"{os.environ.get('PANDAS_DB_S3_PREFIX')}.pandas_db_files/{media_file}", controls=True)
        else:
            media = "Not found"
    else:
        media_file = os.path.join(DEFAULT_PANDAS_DB_PATH, ".pandas_db_files", media_file)
        data = str(base64.b64encode(open(media_file, 'rb').read()))[2:-1]
        media = None
        if media_file.endswith(".png"):
            media = html.Img(src='data:image/png;base64,{}'.format(data), style={"height": "300px", "width": "auto"})
        if media_file.endswith(".wav"):
            media = html.Audio(src='data:audio/wav;base64,{}'.format(data), controls=True)
    file_info = file_info.dropna().reset_index()
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





if __name__ == '__main__':
    main()
