import os
import pandas as pd
from pandas_db import pandas_db
import numpy as np

pandas_db_csv_path = os.path.join(pandas_db.path, ".local_db.csv")
df = pd.read_csv(pandas_db_csv_path)

# train_data_set
def get_train_dataset(model):
    if not isinstance(model, str):
        return None
    options = ["sh101_train", "urmp_train", "idmt_drum_train", "combined_train", "guitar_train", "bass_train"]
    for option in options:
        if option in model:
            return option
    return "urmp_train"

df['train_data'] = df['model'].apply(get_train_dataset)

# test_data
def get_test_dataset(row):
    if isinstance(row['dataset'], str) and row['dataset'] != "":
        return row["dataset"]
    
    options = ["sh101_test", "urmp_test", "idmt_drum_test", "combined_test", "guitar_test", "bass_test"]
    src = row['tag']
    if isinstance(src, str):
        for option in options:
            if option in src:
                return option
    src = row['audio_file']
    if isinstance(src, str):
        for option in options:
            if option in src:
                return option
    return None

df['test_data'] = df.apply(get_test_dataset, axis=1)

# loss_function
def get_loss_function(model):
    if not isinstance(model, str):
        return None
    options = ["unskewed_loss", "spectral_loss"]
    for option in options:
        if option in model:
            return option
    if "improved" in model:
        return "unskewed_loss"
    if "baseline_ae" in model:
        return "spectral_loss"
    if "time_aggregation"in model:
        return "unskewed_loss"
    elif "time_constant" in model:
        return "spectral_loss"
    if "no_crepe" in model:
        return "unskewed_loss"
    return None
df['loss_function'] = df['model'].apply(get_loss_function)

# loss_function
def get_loudness_algorithm(model):
    if not isinstance(model, str):
        return None
    options = ["old_loudness", "new_loudness"]
    for option in options:
        if option in model:
            return option
    if "improved" in model:
        return "new_loudness"
    if "baseline_ae" in model:
        return "old_loudness"
    if "time_aggregation"in model:
        return "new_loudness"
    elif "time_constant" in model:
        return "old_loudness"
    if "no_crepe" in model:
        return "new_loudness"
    return None

df['loudness_algorithm'] = df['model'].apply(get_loudness_algorithm)
# z_aggregation
def get_z_aggregation(model):
    if not isinstance(model, str):
        return None
    if "groupwise" in model or "improved" in model:
        return "groupwise"
    elif "confidence" in model:
        return "confidence"
    elif "time_average" in model or "loudness_and_loss" in model:
        return "mean"
    elif "baseline_ae" in model:
        return "none"
    if "no_crepe" in model:
        return "groupwise"
df['z_aggregation'] = df['model'].apply(get_z_aggregation)


# Use CREPE
def get_crepe_f0(model):
    if not isinstance(model, str):
        return None
    return "CREPE not used" if "no_crepe" in model else "CREPE used"
df['crepe_f0'] = df['model'].apply(get_crepe_f0)

df.to_csv(os.path.join(pandas_db.path, "migrated.csv"), index=False)
