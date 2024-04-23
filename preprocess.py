import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path.cwd().parent/'ADSE4001_Project_Data preprocessed'

Objects = ['Nothing','Pencil_Sharpener','Clip','Staple']

# Load data from csv files
df1 = pd.read_csv(ROOT/Objects[1]/'Processed_5.csv', header=0)

def window_slicing(df, sampling_length=128, overlap=0.75):
    step_size = int(sampling_length * (1 - overlap))  # 32

    windows = []
    num_windows = int((len(df) - sampling_length) / step_size) + 1

    # Extract windows
    for i in range(num_windows):
        start_index = i * step_size
        end_index = start_index + sampling_length
        window = df[start_index:end_index]
        if len(window) == sampling_length: # 最尾尾唔要
            window['window_label'] = i

            windows.append(window)

    return pd.concat(windows).reset_index(drop=True)

def jitter(df, std=0.015):
    jittered = df + np.random.normal(loc=0.0, scale=std, size=df.shape)
    return jittered

def shifting(df, shift_max_ratio=0.15):
    shift_amount = np.random.randint(-int(shift_max_ratio * len(df)), int(shift_max_ratio * len(df)))
    # Assuming window0 is a DataFrame with columns ['normalized_x', 'normalized_y', 'normalized_z']
    window_shifted = df.copy()  # Create a copy to avoid modifying the original DataFrame

    # Shift each column in the DataFrame
    for column in window_shifted.columns:
        window_shifted[column] = np.roll(window_shifted[column], shift_amount)
        # Zero-pad the start of the column after the shift
        #window0_shifted[column].iloc[:shift_amount] = 0
    return window_shifted

def scaling(df, scaling_factor_range=(0.8, 1.2)):
    # Randomly choose a scaling factor
    scaling_factor = np.random.uniform(*scaling_factor_range)
    return df * scaling_factor

# Function for data augmentation
def augment_data(sliced_df):
    augmented_data = pd.DataFrame(columns=["normalized_x", 'normalized_y', 'normalized_z', 'window_label', 'Augmentation'])
    for i in sliced_df.window_label.unique():
        window_df = sliced_df[sliced_df.window_label == i].drop('window_label',axis=1)

        window_df_jitter = jitter(window_df)
        window_df_jitter['window_label'] = 4*i + 2
        window_df_jitter['Augmentation'] = "jitter"

        window_df_shifting = shifting(window_df)
        window_df_shifting['window_label'] = 4*i + 3
        window_df_shifting['Augmentation'] = "shifting"

        window_df_scaling = scaling(window_df)
        window_df_scaling['window_label'] = 4*i + 4
        window_df_scaling['Augmentation'] = "scaling"

        window_df['window_label'] = 4*i + 1
        window_df['Augmentation'] = "original"

        augmented_data = pd.concat([augmented_data,window_df, window_df_jitter, window_df_shifting, window_df_scaling]) 
    return augmented_data

for object in Objects:
    df = pd.read_csv(ROOT/object/'Processed_5.csv', header=0)
    df.drop(['row_id', 'window_id'],axis=1, inplace=True)
    df2 = window_slicing(df)
    final_df = augment_data(df2)
    final_df['Object'] = object
    final_df.to_csv(object+'_Augmented_5.csv', index=False)