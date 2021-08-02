# pandas-db

A minimalistic local (meta) data store.
- save any tabular data, like model metrics. Instead of a metrics UI, you get to use jupyter to plot anything
- save artifacts with metadata, like model artifacts or training data
- browse artifacts in the UI

# Setup
```sh
# Install dependencies
pip install git+git://github.com/nielsrolf/pandas_db
```
Add `export PANDAS_DB_PATH=some/path` to your `.bashrc`.
To use the UI, define a `views.json` (for an example, see `pandas_db/views.json`) and save this file to `$PANDAS_DB_PATH/.pandas_db_views.json`.

# Saving Meta Data

```python
from pandas_db import PandasDB

pandas_db = PandasDB("~/Downloads") # or: from pandas_db import pandas_db for using the defaukt path

pandas_db.save(hello="world", example=1, comment="create a new entry")
pandas_db.save(hello="world", example=2, author="me")
pandas_db.save(hello="cutie", author="me")
pandas_db.latest(keys=['hello'])
```
| hello   | pandas_db.created          | example   | comment            | author   |
|:--------|:---------------------------|:----------|:-------------------|:---------|
| cutie   | 2021-07-16 22:38:28.454400 | ?         | ?                  | me       |
| world   | 2021-07-16 22:37:45.990464 | 2.0       | create a new entry | me       |

Or use context managers to set default values for rows that are created within the context:
```python
from pandas_db import pandas_db

with pandas_db.set_context(foo="foo"):
    pandas_db.save(bar="bar")

# Equivalent to:
# pandas_db.save(foo="foo", bar="bar")
```

# Saving artifacts
```python
from pandas_db import pandas_db

with pandas_db.set_context(foo="foo"):
    pandas_db.save_artifact("my_file.png") 
    #  file is saved, and meta data from context is also saved like any tabular data

```

# Using the UI
If you defined your `views.json` as in the example, you can start the UI via `pandasdb {files/metrics}`.
![pandasdb ui](img/ui.png)
You can edit the data in the top table and save via `ctrl+s`. The filters in the top (search bar + table filters) apply to all displayed data, the filters at the bottom apply to media files only.

# Using the UI in colab
You can upload all data created by pandas db to some public s3 bucket or whatever, and then show your data to other people via colab:

```
#@title Start Browser (takes a few minutes)
print("Install dependencies")
!pip install --upgrade git+git://github.com/nielsrolf/pandas_db &> /dev/null
print("Download data")
!wget https://pandasdb-ddsp-demo.s3.eu-central-1.amazonaws.com/pandasdb.zip
!unzip pandasdb.zip &> /dev/null
print("Start browser")
import os
os.environ['PANDAS_DB_PATH'] = 'pandasdb'
from pandas_db import browser

browser.jupyter(
    keys=["model", "dataset", "audio_file"],
    columns=["unskewed_spectral_loss", "spectral_loss", "reconstruction_loss", "cycle_reconstruction_loss"],
    file_id=["model", "audio_file", "dataset", "sample_idx", "audio_type", "plot_type", "timbre", "melody"]
)
```
