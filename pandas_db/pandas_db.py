import pandas as pd
import os
from uuid import uuid4
import datetime as dt
from contextlib import contextmanager


DEFAULT_PANDAS_DB_PATH = os.environ.get("PANDAS_DB_PATH")


def maybe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


class PandasDB():
    def __init__(self, path=None, context=None):
        self.path = path or DEFAULT_PANDAS_DB_PATH
        self.context = context or {}
    
    def get_df(self):
        try:
            df = pd.read_csv(os.path.join(self.path, ".local_db.csv"), index_col=0)
            return df
        except FileNotFoundError:
            return pd.DataFrame()
    
    def latest(self, keys=None, metrics=None):
        assert not (keys is None and metrics is None), "Specify either keys or metrics"
        df = self.get_df()
        cols = df.columns
        if keys is None:
            keys = [c for c in cols if not c in metrics and c != "entry_created"]
        def latest_entry(values):
            values = values.dropna()
            if len(values) == 0:
                return None
            return values[-1]
        df = df.sort_values("entry_created")\
                 .groupby(keys)\
                 .aggregate(latest_entry)
        if metrics is None:
            return df
        return df[metrics]

    def save(self, **data):
        df = self.get_df()
        data['entry_created'] = dt.datetime.now()
        data.update(self.context)
        data = {k: [maybe_float(v)] for k, v in data.items()}
        df = pd.concat([df, pd.DataFrame(data, index=[uuid4()])], axis=0)
        df.to_csv(os.path.join(self.path, ".local_db.csv"))
    
    @contextmanager
    def set_context(self, **data):
        original_context = self.context.copy()
        try:
            self.context.update(data)
            yield self
        finally:
            self.context = original_context



pandas_db = PandasDB()
