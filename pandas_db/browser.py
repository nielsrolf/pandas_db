import dash
from dash.dependencies import Input, Output
import dash_table
import dash_html_components as html
import pandas as pd
from pandas_db import pandas_db, maybe_float
import numpy as np



app = dash.Dash(__name__)


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


class State():
    def __init__(self, keys):
        self.keys = keys
        self.df = None
        self.fetch()
    
    def fetch(self):
        df = pandas_db.latest(keys=self.keys).reset_index()
        df = drop_constant_columns(df)
        self.df = cols_maybe_float(df)


STATE = State(['model', 'file'])



PAGE_SIZE = 30


main_table = dash_table.DataTable(
    id='table-filtering',
    columns=[
        {"name": i, "id": i, "hideable": True, "editable": i not in STATE.keys} for i in STATE.df.columns
    ],
    page_current=0,
    page_size=PAGE_SIZE,
    page_action='custom',

    filter_action='custom',
    filter_query=''
)


@app.callback(
    Output('hidden', 'children'),
    Input('table-filtering', 'data'))
def update_db(rows):
    STATE.fetch()
    edited = pd.DataFrame(rows, columns=STATE.df.columns)
    edited = cols_maybe_float(edited)
    original = cols_maybe_float(STATE.df.copy())
    edited['tmp_key'] = ""
    
    original['tmp_key'] = ""
    for key in STATE.keys:
        edited['tmp_key'] = edited['tmp_key'].str.cat(edited[key])
        original['tmp_key'] = original['tmp_key'].str.cat(original[key])

    original = edited[['tmp_key']].merge(original, on='tmp_key')
    original = original[edited.columns]
    row_idx, col_idx = np.where(original != edited)
    for r, c in zip(row_idx, col_idx):
        row = edited.iloc[r]
        data = {k: row[k] for k in STATE.keys}
        data[edited.columns[c]] = row[edited.columns[c]]
        print("pandas_db.save: ", data)
        pandas_db.save(**data)
    STATE.fetch()
    return None


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


@app.callback(
    Output('table-filtering', 'data'),
    Input('table-filtering', "page_current"),
    Input('table-filtering', "page_size"),
    Input('table-filtering', "filter_query"))
def update_table(page_current,page_size, filter):
    print(filter)
    filtering_expressions = filter.split(' && ')
    dff = STATE.df
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]
    return dff.iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')


detail_view = html.Div(id='detail-view')
hidden_view = html.Div(id='hidden')

app.layout = html.Div([
    detail_view,
    main_table,
    hidden_view
])


if __name__ == '__main__':
    app.run_server(debug=True)

