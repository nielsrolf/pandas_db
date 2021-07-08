# pandas-db

A simpler and minified version of MLFlow model tracking.

```python
from pandas_db import PandasDB


db = PandasDB("some/path")
db.save(foo="foo", bar="bar")
db.save(foo="FOO", another_column=42)
df = db.get_df()
```
|                                      | foo   | bar   | entry_created              |   another_column |
|:-------------------------------------|:------|:------|:---------------------------|-----------------:|
| e2bfa08f-b055-4526-b6a5-e965282e62dc | foo   | bar   | 2021-07-08 17:53:34.087882 |              nan |
| 8e99fc43-576e-4af6-8f4d-5b6ef33ee029 | FOO   | nan   | 2021-07-08 17:53:34.099407 |               42 |

Or use context managers to set default values for rows that are created within the context:
```python
from pandas_db import pandas_db

with pandas_db.set_context(foo="foo"):
    pandas_db.save(bar="bar")

# Equivalent to:
# pandas_db.save(foo="foo", bar="bar")
```

For this, you have to put `export PANDAS_DB_PATH="/some/path" ` in your bashrc.

## Setup
```sh
# Install dependencies
pip install git+git://github.com/nielsrolf/pandas_db
```
