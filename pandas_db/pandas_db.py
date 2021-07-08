import pandas as pd
import os
from uuid import uuid4
import datetime as dt


DEFAULT_PANDAS_DB_PATH = os.environ.get("PANDAS_DB_PATH")

class PandasDB():
    def __init__(self, path=None):
        self.path = path or DEFAULT_PANDAS_DB_PATH
    
    def get_df(self):
        try:
            df = pd.read_csv(os.path.join(self.path, ".local_db.csv"), index_col=0)
            return df
        except FileNotFoundError:
            return pd.DataFrame()

    def save(self, **data):
        df = self.get_df()
        data['entry_created'] = dt.datetime.now()
        data = {k: [v] for k, v in data.items()}
        df = pd.concat([df, pd.DataFrame(data, index=[uuid4()])], axis=0)
        df.to_csv(os.path.join(self.path, ".local_db.csv"))
