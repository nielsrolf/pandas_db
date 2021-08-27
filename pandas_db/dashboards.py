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



def get_dashboard(view: dict, df: pd.DataFrame, app: dash.Dash):
    """Create a dashboard for a given (prefiltered) df and a view
    
    Arguments:
        view (dict): defines which columns contain model params, metrics or define a file
                     example is given in views.json
        df (pd.DataFrame): (filtered) output of pd.get_df()
    """
    state = State(view['keys'], view['columns'], view['file_id'], view.get('file_references', {}), view['prefix'], df)
    metrics_df = pandas_db.latest(keys=state.model_id, metrics=state.metrics, df=df)
    full_df = pandas_db.latest(keys=list(set(state.model_id + state.file_id)), df=df)

    dropdown_fields_top = state.model_id
    dropdown_fields_files = [i for i in state.file_id if i not in state.model_id]
    dropdown_fields = dropdown_fields_top + dropdown_fields_files
    
    dropdowns_top =  html.Div(dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id=f"{state.prefix}-dropdown-{key}",
                            options=[{'label': i, 'value': i} for i in list(df[key].dropna().unique()) if i != ""],
                            multi=True,
                            style={"font-size": "13px"},
                            value=view['default_selection'].get(key),
                            placeholder=key)
                    ], md=1) for key in dropdown_fields_top], align="center", no_gutters=True))
    dropdowns_files =  html.Div(dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id=f"{state.prefix}-dropdown-{key}",
                            options=[{'label': i, 'value': i} for i in list(df[key].dropna().unique()) if i != ""],
                            multi=True,
                            style={"font-size": "13px"},
                            value=view['default_selection'].get(key),
                            placeholder=key)
                    ], md=1) for key in dropdown_fields_files], align="center", no_gutters=True))

    
    metrics_plots = dcc.Loading(id=f"{state.prefix}-metrics-view", style={'min-height': '100px'})

    @functools.lru_cache(maxsize=20)
    def filter_df(*dropdown_values, use_full_df=False):
        dropdown_values = unsmask_dropdown_values(dropdown_values)
        groupby_keys = []
        if use_full_df:
            filtered_df = full_df.reset_index()
        else:
            filtered_df = metrics_df.reset_index()
        for key, value in zip(dropdown_fields[:len(dropdown_values)], dropdown_values):
            if value is not None and len(value) > 0:
                try:
                    filtered_df = filtered_df[filtered_df[key].isin(value)]
                except:
                    breakpoint()
                if len(value) > 1:
                    groupby_keys.append(key)
        filtered_df["key"] = filtered_df[groupby_keys].apply(lambda x: "\n".join([str(i) for i in x.to_dict().values()]), axis=1)
        return filtered_df, groupby_keys

    @app.callback(
        Output(f"{state.prefix}-metrics-view", 'children'),
        [Input(f"{state.prefix}-dropdown-{key}", 'value') for key in dropdown_fields_top])
    def get_metrics_view(*dropdown_values):
        dropdown_values = mask_dropdown_values(dropdown_values)
        filtered_df, _ = filter_df(*dropdown_values)
        return get_metric_plot(filtered_df, state)
    
    search = dcc.Input(f"{state.prefix}-global-search", type='text', style={
        "width": "100%", "height": "30px", "z-index": "10",
        "border": "2px solid #2cb2cb", "position": "fixed", "bottom": "1px"},
        placeholder="Keyword search: enter any number of words you'd like to search")
    
    cheap_pagination_1 = dcc.Input(f"{state.prefix}-file_id_first", type='number', style={"display": "inline"}, placeholder="0")
    cheap_pagination_2 = dcc.Input(f"{state.prefix}-file_id_last", type='number', style={"display": "inline"}, placeholder="5")
    pagination = html.Div([cheap_pagination_1, " - ", cheap_pagination_2], style={"margin": "auto", "width": "400px"})

    detail_view = html.Div([
        dropdowns_files,
        pagination,
        search,
        dcc.Loading(html.Div(id=f"{state.prefix}-medias", style={"min-height": "100px"}))],
        style={"width": "100%", "bottom": "30px"})
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
        Output(f"{state.prefix}-medias", 'children'),
        [Input(f"{state.prefix}-global-search", 'value'),
        Input(f"{state.prefix}-file_id_first", 'value'),
        Input(f"{state.prefix}-file_id_last", 'value')] + \
        [Input(f"{state.prefix}-dropdown-{key}", 'value') for key in dropdown_fields])
    def update_medias_clb(search_str, file_id_first, file_id_last, *dropdown_values):
        dropdown_values = mask_dropdown_values(dropdown_values)
        df_files , groupby_keys = global_search(*dropdown_values, search_str=search_str)
        if len(df_files) == 0:
            return None
        keys = [key for key in state.file_id if key not in groupby_keys] + groupby_keys

        df_files = df_files.fillna("")
        df_files = pandas_db.latest(keys=keys, df=df_files).reset_index()
        df_files = df_files.loc[(df_files.file!="") & (df_files.file != "?")]
        rel_paths = df_files["file"].values[(file_id_first or 0):(file_id_last or 5)]
        return update_medias(state, rel_paths)
    
    
    for field in dropdown_fields:
        # the default value is evaluated at definition time, otherwise
        # field will always be set to the last value of the loop at execution time
        # therefore it is fixed as kwarg
        def hide_field(*dropdown_values, field=field): 
            value = dict(zip(dropdown_fields, dropdown_values))[field]
            if value is not None and value != "":
                return {"font-size": "13px", "visibility": "visible"}
            dropdown_values = mask_dropdown_values(dropdown_values)
            df, _ = filter_df(*dropdown_values, use_full_df=True)
            if len(df[field].unique()) < 2:
                return {"font-size": "13px", "visibility": "hidden"}
            else:
                return {"font-size": "13px", "visibility": "visible"}
        app.callback(
            Output(f"{state.prefix}-dropdown-{field}", 'style'),
            [Input(f"{state.prefix}-dropdown-{key}", 'value') for key in dropdown_fields]
                )(hide_field)


    return dbc.Container([
        dropdowns_top,
        metrics_plots,
        detail_view
    ])
    



@click.command()
@click.argument("view_name")
def main(view_name):
    """entry point to start the app"""
    with open(os.path.join(DEFAULT_PANDAS_DB_PATH, ".pandas_db_views.json")) as json_file:
        views = json.load(json_file)
    view = views[view_name]
    app = dash.Dash(__name__)
    df = pandas_db.get_df()
    app.layout = get_dashboard(view, df, app)
    app.run_server(host='0.0.0.0', port=8050, debug=("nielswarncke" in os.getcwd()))


def jupyter(view, **server_args):
    """Run the browser in a jupyter browser
    
    Arguments:
        keys (List[str]): columns that you want to group information by
        columns (List[str]): columns that hold metrics or other data,
            where you want to see the latest info per group
        file_id (List[str]): columns that together uniquely identify a file.
            If multiple files exist per group, the latest is displayed
    """
    app = JupyterDash(__name__)
    df = pandas_db.get_df()
    app.layout = get_dashboard(view, df, app)
    app.run_server(**server_args)


class State():
    """Class that caches the state of pandas_db and imlements transaction search"""
    def __init__(self, model_id, metrics, file_id, file_references, prefix, df):
        self.prefix = prefix
        self.model_id = model_id
        self.metrics = [c for c in metrics if not c in self.model_id]
        self.file_id = file_id
        self.file_info = pandas_db.latest(keys=["file"], df=df)
        self.search = self.file_info.fillna("").reset_index()
        self.search['search_index'] = self.search.apply(concat_as_str, axis=1)
        self.file_references = file_references


def concat_as_str(values):
    return "".join([str(i) for i in values])


def mask_dropdown_values(dropdown_values):
    masked_values = ["||".join(i) if isinstance(i, list) else i for i in dropdown_values]
    return [i if i != "" else None for i in masked_values]


def unsmask_dropdown_values(dropdown_values):
    return [i.split("||") if isinstance(i, str) else i for i in dropdown_values]


def get_metric_plot(df, state):
    metric_plots = []
    vals = df.reset_index()

    if len(vals) == 0:
        return None

    mean_metrics = vals.groupby("key").aggregate(np.nanmean)
    for metric in state.metrics:
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
        histograms = dcc.Graph(id=f"{state.prefix}-scatter-{metric}", figure=fig)
        fig = px.box(vals, x="key", y=metric, category_orders=category_order, color="key")
        boxplots = dcc.Graph(id=f"{state.prefix}-boxplot-{metric}", figure=fig)
        metric_plots += [html.Div([histograms, boxplots])]
    return metric_plots


def update_medias(state, rel_paths):
    medias = []
    for rel_path in rel_paths:
        try:
            file_info = state.file_info.loc[rel_path]
            medias += [show_media(state, rel_path, file_info)]
        except (KeyError, FileNotFoundError, IndexError) as e:
            print(e)
            pass
    return medias


def show_media(state, media_file, file_info):
    media = resolve_and_render(media_file)
    table_rows = []
    for key, value in file_info.to_dict().items():
        if value is None or str(value)=="nan":
            continue
        if key in state.file_references.keys() and (str(value).endswith(".png") 
                                                or str(value).endswith(".wav")
                                                 or str(value).endswith(".mp3")):
            value_path = f"{state.file_references[key]}/{value}"
            value_rendered = render_remote_or_local_path(value_path)
            value = html.Div([value_rendered, value])
        row = html.Tr([html.Td(key), html.Td(value)])
        table_rows.append(row)
    
    data_table = html.Table(table_rows)

    return html.Div([html.Div(data_table, style={"flex": "50%"}),
                        html.Div(media, style={
                            "flex": "50%",
                            "height": "300px",
                            "display": "flex",
                            "align-items": "center",
                            "padding": "5px"})],
                        style={
                            "display": "flex", 
                            "border": "3px solid #2cb2cb", 
                            "border-radius": "3px",
                            "margin-bottom": "5px", "margin-top": "5px"})


def resolve_and_render(media_file):
    if os.environ.get("PANDAS_DB_S3_PREFIX") is not None:
        return render_s3(f"{os.environ.get('PANDAS_DB_S3_PREFIX')}.pandas_db_files/{media_file}")
    else:
        return render_local(os.path.join(DEFAULT_PANDAS_DB_PATH, ".pandas_db_files", media_file))


def render_s3(src):
    if src.endswith(".png"):
        media = html.Img(src=src, style={"height": "300px", "width": "auto"})
    elif src.endswith(".wav") or src.endswith(".mp3"):
        media = html.Audio(src=src, controls=True)
    else:
        media = "Not found"
    return media


def render_local(media_file):
    data = str(base64.b64encode(open(media_file, 'rb').read()))[2:-1]
    media = None
    if media_file.endswith(".png"):
        media = html.Img(src='data:image/png;base64,{}'.format(data), style={"height": "300px", "width": "auto"})
    if media_file.endswith(".wav"):
        media = html.Audio(src='data:audio/wav;base64,{}'.format(data), controls=True)
    return media


def render_remote_or_local_path(media_file):
    if os.path.exists(media_file):
        return render_local(media_file)
    else:
        return render_s3(media_file)


if __name__ == '__main__':
    main()
