import pandas as pd
import os
from uuid import uuid4
import datetime as dt
from contextlib import contextmanager
import shutil
import pathlib
from glob import glob


DEFAULT_PANDAS_DB_PATH = os.environ.get("PANDAS_DB_PATH")


def maybe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def modification_time(filepath):
    try:
        fname = pathlib.Path(filepath)
        return dt.datetime.fromtimestamp(fname.stat().st_mtime)
    except FileNotFoundError:
        return None


class PandasDB():
    def __init__(self, path=None, context=None):
        self.path = path or DEFAULT_PANDAS_DB_PATH
        self.context = context or {}
        self._df = None
        self._loaded = None
    
    def get_df(self):
        csv_path = os.path.join(self.path, ".local_db.csv")
        if self._df is not None:
            if self._loaded == modification_time(csv_path):
                return self._df
        try:
            self._df = pd.read_csv(csv_path, index_col=0)
            self._loaded = modification_time(csv_path)
            return self._df
        except FileNotFoundError:
            return pd.DataFrame(columns=['pandas_db.created'])
    
    def latest(self, keys=None, metrics=None, df=None):
        assert not (keys is None and metrics is None), "Specify either keys or metrics"
        na_placeholder = "-"
        df = df if df is not None else self.get_df()
        df = df.fillna(na_placeholder)
        cols = df.columns
        if keys is None:
            keys = [c for c in cols if not c in metrics and c != "pandas_db.created"]
        def latest_entry(values):
            values = values[values!=na_placeholder]
            if len(values) == 0:
                return None
            return values[-1]
        df = df.sort_values("pandas_db.created")\
                .groupby(keys)\
                .aggregate(latest_entry)\
                .fillna("?")
        if metrics is None:
            return df
        return df[metrics]

    def save(self, **data):
        df = self.get_df()
        data['pandas_db.created'] = dt.datetime.now()
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
    
    def save_artifact(self, filepath, **data):
        file_id = str(uuid4().hex[:6])
        filename = filepath.split("/")[-1]
        dest_dir = os.path.join(self.path,
                            ".pandas_db_files",
                            file_id)
        os.makedirs(dest_dir)
        dest = os.path.join(dest_dir, filename)
        shutil.copy(filepath, dest)
        self.save(file=f"{file_id}/{filename}", filename=filename, **data)
    
    def mount_dir(self, dir_path, **data):
        pass

    def save_dir(self, dir_path, **data):
        for file in glob(f"{dir_path}/*"):
            if os.path.isdir(file):
                self.save_dir(self, added_from_dir=dir_path, **data)
            else:
                self.save_artifact(file, **data)


pandas_db = PandasDB()
