import pandas_db
import dash
import dash_html_components as html
import pandas as pd
from pandas_db.pandas_db import PandasDB, DEFAULT_PANDAS_DB_PATH
import numpy as np
import os
from pandas_db.dashboards import get_dashboard


"""

Experiment 4: Before vs after
- Show reconstruction, cycle reconstruction and timbre transfer for baseline, improved baseline and generalist
- Dashboard with all baseline + improved baseline models

Experiment 5: Fine-tuning on each inference sample improves reconstruction
"""

if os.path.exists(os.path.join(DEFAULT_PANDAS_DB_PATH, "migrated.csv")):
    pandas_db = PandasDB(csv_path=os.path.join(DEFAULT_PANDAS_DB_PATH, "migrated.csv"))
else:
    pandas_db = PandasDB()


samples_path = "https://pandasdb-ddsp-demo.s3.eu-central-1.amazonaws.com/.pandas_db_files"


def main():
    """entry point to start the app"""
    app = dash.Dash(__name__)
    app.layout = html.Div([
        get_experiment_1(app),
        get_experiment_2(app),
        get_experiment_3(app)
    ])
    app.run_server(host='0.0.0.0', port=8050, debug=("nielswarncke" in os.getcwd()))


def get_experiment_1(app):
    title = "Experiment 1: Enforcing z to be constant over time improves timbre transfer without hurting reconstruction"
    description = ("The baseline architecture allows the model to use a time distributed latent variable z."
                   "Restricting z to be constant over time by meaning it before passing it to the decoder"
                   "enforces the model to describe the timbre via the z-vector, if the model is trained on multiple instruments."
                   "To listen to reconstructions, select audio_type=reconstruction and audio_file=<something that matches the test_data_group>."
                   "To listen to a cycle reconstruction, do the same but select audio_type=cycle."
                   "To listen to a timbre transfer, selectio audio_type=transfer, leave audio_file and test_data_group blank and select the fields timbre and melody."
                   "Additionally, you can always select the model used to generate the reconstruction/cycle reconstruction/timbre transfer.")
    df = pandas_db.get_df()
    df = df.loc[(df['model'].isin(['baseline_ae_urmp_train', 'time_average_mfcc', 'improved_baseline_ae_urmp_train'])) 
                & (df['train_data'] == "urmp_train")
                & (df['crepe_f0'] == "CREPE used")
                & (df['fine_tune'] == "fine-tuning not used")
                & ((df['test_data_group'] == 'urmp_test') | (df['timbre'].str.contains("urmp_test")))
                & ( (df['s']=="100.0") | (df['plot_type'].isnull()) )]
    view = {
        "default_selection": {
            "z_aggregation": ["none", "mean", "groupwise"],
            "audio_type": ["cycled"],
            "audio_file": ["samples/urmp_test/AuSep_2_fl_14_Waltz.wav"]
        },
        "file_references": {
            "audio_file": samples_path,
            "timbre": samples_path,
            "melody": samples_path,
            "intermediate": samples_path
        },
        "prefix": "experiment_1",
        "keys": ["z_aggregation"],
        "file_id": ["test_data_src", "audio_type", "timbre", "melody", "s", "intermediate", "model"],
        "columns": ["unskewed_spectral_loss",  "cycle_reconstruction_loss"],
    }

    dashboard = get_dashboard(view, df, app)
    return html.Div([
        html.H1(title),
        html.P(description),
        dashboard])


def get_experiment_2(app):
    title = "Section 3.2.2 & 3.2.3: Fixing the loudness bug and switching to unskewed MSS gets us rid of unpleasant squeaky noises"
    description = ("Consider the reconstruction of samples/urmp_test/AuSep_2_fl_14_Waltz.wav"
                   "With the baseline settings for loss and loudness, there is an unpleasant squeaky noise in the beginning of the reconstruction."
                   "The model that used the new loudness gets rid of this problem, the reconstruction sounds much better."
                   "The error heatmap for the unskewed loss with s=100 shows this squeaky noise in red, while this mistake is almost invisble in the logmag spectrogram."
                   "However, the total unskewed loss does not differ as much, indicating that the loss function could be improved further.")
    """
    Preset:
    test_data_group='urmp_test'
    loudness=['new_loudness', 'old_loudness']
    audio_file='samples/urmp_test/AuSep_2_fl_14_Waltz.wav'
    audio_type='reconstruction'
    model=['loudness_and_loss_autoencoder_old_loudness_spectral_loss_urmp_train', 'loudness_and_loss_autoencoder_new_loudness_unskewed_loss_urmp_train']
    """
    df = pandas_db.get_df()
    df = df.loc[(df['model'].str.contains('loudness_and_loss')) 
                & (df['train_data'] == "urmp_train")
                & (df['crepe_f0'] == "CREPE used")
                & ((df['test_data_group'] == 'urmp_test') | (df['timbre'].str.contains("urmp_test")))
                & (df['fine_tune'] == "fine-tuning not used")
                & ( (df['s']=="100.0") | (df['plot_type'].isnull()) )]
    view = {
        "default_selection": {
            "loudness_algorithm": ['new_loudness', 'old_loudness'],
            "loss_function": ["unskewed_loss", "spectral_loss"],
            "audio_type": ["reconstruction"],
            "audio_file": ["samples/urmp_test/AuSep_2_fl_14_Waltz.wav"],
            "model": ['loudness_and_loss_autoencoder_old_loudness_spectral_loss_urmp_train', 'loudness_and_loss_autoencoder_new_loudness_unskewed_loss_urmp_train']
        },
        "file_references": {
            "audio_file": samples_path,
            "timbre": samples_path,
            "melody": samples_path,
            "intermediate": samples_path
        },
        "prefix": "experiment_2",
        "keys": ["loudness_algorithm", "loss_function"],
        "file_id": ["audio_file", "audio_type", "plot_type", "s", "melody", "timbre", "intermediate", "model"],
        "columns": ["unskewed_spectral_loss",  "cycle_reconstruction_loss"],
    }

    dashboard = get_dashboard(view, df, app)
    return html.Div([
        html.H1(title),
        html.P(description),
        dashboard])


def get_experiment_3(app):
    title = "Section 3.2.4: Baseline vs improved baseline on all datasets"
    description = ("here, you can check out how the baseline vs the improved baseline sounds."
                    "Models have been trained on different datasets, note how a model trained"
                    "on guitar makes everything sound like a guitar. The baseline works well"
                    " for timbre transfer in these cases, if the target timbre is also a guitar."
                    "An interesting weakness of the loss function can be seen when you look into the cycle reconstruction error of sh101 sounds."
                    "Clearly, the improved model sounds much better than the baseline model."
                    "However, the loss of the baseline model is smaller than that of the improved model!"
                    "This is, why manual eevaluation remains important for tasks that target human perception.")
    df = pandas_db.get_df()
    df = df.loc[(df['model'].str.contains('baseline')) 
                & (df['crepe_f0'] == "CREPE used")
                & (df['fine_tune'] == "fine-tuning not used")
                & ( (df['s']=="100.0") | (df['plot_type'].isnull()) )]
    view = {
        "prefix": "experiment_3",
        "file_references": {
            "audio_file": samples_path,
            "timbre": samples_path,
            "melody": samples_path,
            "intermediate": samples_path
        },
        "default_selection": {
            "audio_type": ["cyled"],
            "train_data": ["sh101_train", "combined_train"],
            "test_data_group": ["sh101_test"],
            "audio_file": ["samples/sh101/sh101_arps_dataset_v3_0113.wav"],
            "baseline_or_improved": ['baseline', 'improved'],
            "intermediate": ['samples/guitar/AR_Lick4_FN.wav'],
            "audio_type": ["cycled"]
        },
        "keys": ["baseline_or_improved", "train_data", "test_data_group"],
        "file_id": ["audio_file", "audio_type", "plot_type", "s", "melody", "timbre", "intermediate", "model"],
        "columns": ["unskewed_spectral_loss",  "cycle_reconstruction_loss"],
    }

    dashboard = get_dashboard(view, df, app)
    return html.Div([
        html.H1(title),
        html.P(description),
        dashboard])


if __name__ == "__main__":
    main()
