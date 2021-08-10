import pandas_db
import dash
import dash_html_components as html
import pandas as pd
from pandas_db.pandas_db import PandasDB, DEFAULT_PANDAS_DB_PATH
import numpy as np
import os
from pandas_db.dashboards import get_dashboard


"""
Experiment 0: Baseline
- Reconstruction works fine
- Timbre transfer possible if the model was trained on a single model
- Timbre transfer sounds poor if the model was trained on multiple instruments

Experiment 1: Enforcing z to be constant over time improves timbre transfer without hurting reconstruction
- Now consider multi instrument models
- We enforce z to be constant over time by meaning it before passing it to the decoder
- This enforces the model to use z as 'timbre vector'
- Timbre transfer can now happen by extracting z from one sound and using it together with (f0, loudness) from other sound
- Does not sound as natural as we would like

Experiment 2: Loudness bug creates unpleasant squeaky noises
- Show: reconstruction old/new method of a good sample
- Show loudness curves extracted with different methods

Experimt 3: Spectral loss does not highlight the perceived errors as well as unskewed loss
- Show logmag vs s=100 error heatmap
- sound examples unskewed vs old loss

Experiment 4: Giving the encoder more capabilities gives slight improvements
- Taking the mean of the z-vector has the disadvantage that uninformative silent parts introduce noise on the mean
- Idea: let it predict a confidence score and use weighted mean
- Inspired by multi headed attention and resnext: use z-groups

Experiment 5: Fine-tuning on each inference sample improves reconstruction
"""

if os.path.exists(os.path.join(DEFAULT_PANDAS_DB_PATH, "migrated.csv")):
    pandas_db = PandasDB(csv_path=os.path.join(DEFAULT_PANDAS_DB_PATH, "migrated.csv"))
else:
    pandas_db = PandasDB()


def main():
    """entry point to start the app"""
    app = dash.Dash(__name__)
    app.layout = html.Div([
        get_experiment_1(app),
        # get_experiment_2(app)
    ])
    app.run_server(host='0.0.0.0', port=8050, debug=("nielswarncke" in os.getcwd()))


def get_experiment_1(app):
    title = "Experiment 1: Enforcing z to be constant over time improves timbre transfer without hurting reconstruction"
    description = ("The baseline architecture allows the model to use a time distributed latent variable z."
                   "Restricting z to be constant over time by meaning it before passing it to the decoder"
                   "enforces the model to describe the timbre via the z-vector, if the model is trained on multiple instruments")
    df = pandas_db.get_df()
    df = df.loc[(df['z_aggregation'].isin(['none', 'mean'])) 
                & (df['loudness_algorithm'] == 'old_loudness')
                & (df['loss_function'] == "spectral_loss")
                & (df['train_data'] == "urmp_train")
                & (df['crepe_f0'] == "CREPE used")
                & (df['s'].fillna("100.0") == "100.0")]
    
    view = {
        "prefix": "experiment_1",
        "keys": ["z_aggregation", "test_data", "audio_file"],
        "file_id": ["audio_file", "test_data", "sample_idx", "audio_type", "plot_type", "s", "melody", "timbre", "model"],
        "columns": ["unskewed_spectral_loss", "spectral_loss", "reconstruction_loss", "cycle_reconstruction_loss"],
    }

    dashboard = get_dashboard(view, df, app)
    return html.Div([
        html.H1(title),
        html.P(description),
        dashboard])


if __name__ == "__main__":
    main()
