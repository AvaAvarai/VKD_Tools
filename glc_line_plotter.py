
"""
GLC-L Type Graph Plotter

Mathematical Algorithm:
-----------------------
Given a dataset with 'n' attributes and a label column named 'class':
1. Normalize the attributes to the range [0, 1].
2. For each data point in the dataset:
    a. Initialize x_prev, y_prev = 0, 0 (starting at the origin).
    b. For each attribute a_i:
        i. Calculate theta_i = arccos(abs(a_i)).
        ii. Update x_i = x_prev + a_i * cos(theta_i) and y_i = y_prev + a_i * sin(theta_i).
        iii. Plot a line segment from (x_prev, y_prev) to (x_i, y_i).
        iv. Update x_prev, y_prev to x_i, y_i for the next iteration.
3. Color code each glyph according to its class label.

"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def normalize_and_clip(df, feature_columns):
    """
    Normalize the feature columns to the range [0, 1] using Min-Max scaling and clip any out-of-range values.
    
    Parameters:
    - df: DataFrame containing the data.
    - feature_columns: List of feature column names to be normalized.
    
    Returns:
    - DataFrame with normalized and clipped feature columns.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    df[feature_columns] = np.clip(df[feature_columns], 0, 1)
    return df

def plot_glyphs(df, dataset_name):
    """
    Plot the GLC-L type graph for the given dataset.
    
    Parameters:
    - df: DataFrame containing the data.
    - dataset_name: Name of the dataset to be used in the plot title.
    """
    feature_columns = [col for col in df.columns if col != 'class']
    label_column = 'class'
    df = normalize_and_clip(df, feature_columns)
    
    unique_labels = df[label_column].unique()
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))
    
    plt.figure(figsize=(10, 20))
    
    max_x_value = 0
    for index, row in df.iterrows():
        x_prev = 0
        for feature in feature_columns:
            a_i = row[feature]
            theta_i = np.abs(a_i)
            x_i = x_prev + a_i * np.cos(theta_i)
            x_prev = x_i
        max_x_value = max(max_x_value, x_i)
    
    plt.subplot(2, 1, 1)
    first_class = unique_labels[0]
    for index, row in df[df[label_column] == first_class].iterrows():
        x_prev, y_prev = 0, 0
        for feature in feature_columns:
            a_i = row[feature]
            theta_i = np.abs(a_i)
            x_i = x_prev + a_i * np.cos(theta_i)
            y_i = y_prev + a_i * np.sin(theta_i)
            plt.plot([x_prev, x_i], [y_prev, y_i], color=label_to_color[row[label_column]], alpha=0.33)
            x_prev, y_prev = x_i, y_i
        plt.scatter(x_i, y_i, color=label_to_color[row[label_column]], s=30)
        plt.scatter(x_i, 0, marker='|', color=label_to_color[row[label_column]], s=100)
        
    plt.xlim(0, max_x_value + 0.1)
    plt.title(f'GLC-L Graph of {dataset_name} - Class: {first_class}')
    
    plt.subplot(2, 1, 2)
    for index, row in df[df[label_column] != first_class].iterrows():
        x_prev, y_prev = 0, 0
        for feature in feature_columns:
            a_i = row[feature]
            theta_i = np.abs(a_i)
            x_i = x_prev + a_i * np.cos(theta_i)
            y_i = y_prev + a_i * np.sin(theta_i)
            plt.plot([x_prev, x_i], [y_prev, y_i], color=label_to_color[row[label_column]], alpha=0.5)  # Set alpha to 0.5
            x_prev, y_prev = x_i, y_i
        plt.scatter(x_i, y_i, color=label_to_color[row[label_column]], s=30)
        plt.scatter(x_i, 0, marker='|', color=label_to_color[row[label_column]], s=100)
        
    plt.xlim(0, max_x_value + 0.1)
    plt.gca().invert_yaxis()
    plt.title(f'GLC-L Graph of {dataset_name} - Additional Classes')
    
    custom_lines = [plt.Line2D([0], [0], color=color, lw=4) for color in label_to_color.values()]
    plt.legend(custom_lines, unique_labels, title='Class')
    
    def close(event):
        if event.key == 'escape' or (event.key == 'w' and event.inaxes == None):
            plt.close(event.canvas.figure)
            
    plt.connect('key_press_event', close)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot GLC-L Type Graph.')
    parser.add_argument('--file_path', type=str, help='Path to the dataset CSV file.')
    args = parser.parse_args()
    
    df = pd.read_csv(args.file_path)
    dataset_name = args.file_path.split('/')[-1].split('.')[0]
    
    plot_glyphs(df, dataset_name)
