# pandas-db

A simpler and minified version of MLFlow model tracking.

```python
form pandas_db import PandasDB


db = PandasDB("some/path")
db.save(foo="foo", bar="bar")
db.save(foo="FOO", another_column=42)
df = db.get_df()
```
|                                      |   a |   b | entry_created              | c     |
|:-------------------------------------|----:|----:|:---------------------------|:------|
| e41fb6f1-6311-49a4-b8a6-7bd4590a1b1e |   1 |   1 | 2021-07-08 16:49:44.882154 | nan   |
| 41db0cd0-0baf-474d-977f-6caf70309990 |   2 | nan | 2021-07-08 16:49:44.889189 | hello |

You can also put `export PANDAS_DB_PATH="/some/path" ` in your bashrc and init via `db = PandasDB()`.

## Setup
```sh
# Install dependencies
pip install git+git://github.com/nielsrolf/pandas_db
```
