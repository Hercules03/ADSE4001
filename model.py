import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# Define a list of filenames for each object (replace with actual filenames)
pencil_sharpener_files = ["Pencil_Sharpener_Augmented_"+str(i+1)+".csv" for i in range(5)]
nothing_files = ["Nothing_Augmented_"+str(i+1)+".csv" for i in range(5)]
clip_files = ["Clip_Augmented_"+str(i+1)+".csv" for i in range(5)]
staple_files = ["Staple_Augmented_"+str(i+1)+".csv" for i in range(5)]

# Initialize empty lists to store data
X_df_pencil_sharpener = []
X_df_nothing = []
X_df_clip = []
X_df_staple = []

# Loop through filenames for each object
for filename in pencil_sharpener_files:
  df_temp = pd.read_csv(filename)
  df_temp = df_temp.drop(columns=['Augmentation','Object','window_label']).values  # drop unnecessary columns
  X_df_pencil_sharpener.append(df_temp)  # append processed data to list
  
for filename in nothing_files:
  df_temp = pd.read_csv(filename)
  df_temp = df_temp.drop(columns=['Augmentation','Object','window_label']).values  # drop unnecessary columns
  X_df_nothing.append(df_temp)  # append processed data to list
  
for filename in clip_files:
  df_temp = pd.read_csv(filename)
  df_temp = df_temp.drop(columns=['Augmentation','Object','window_label']).values  # drop unnecessary columns
  X_df_clip.append(df_temp)  # append processed data to list
  
for filename in staple_files:
  df_temp = pd.read_csv(filename)
  df_temp = df_temp.drop(columns=['Augmentation','Object','window_label']).values  # drop unnecessary columns
  X_df_staple.append(df_temp)  # append processed data to list

# Stacking arrays along columns
X_df_pencil_sharpener = np.vstack(X_df_pencil_sharpener)
X_df_nothing = np.vstack(X_df_nothing)
X_df_clip = np.vstack(X_df_clip)
X_df_staple = np.vstack(X_df_staple)

X = np.concatenate([
    X_df_clip, 
    X_df_pencil_sharpener, 
    X_df_nothing, 
    X_df_staple
], axis=0)

y = np.concatenate([
    np.full(X_df_clip.shape[0], 0),  # Clip
    np.full(X_df_pencil_sharpener.shape[0], 1),  # Pencil Sharpener
    np.full(X_df_nothing.shape[0], 2),  # Nothing
    np.full(X_df_staple.shape[0], 3)   # Staple
])