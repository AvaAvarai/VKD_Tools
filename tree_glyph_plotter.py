import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations
from sklearn.preprocessing import MinMaxScaler
import argparse

# Initialize global index for permutation cycling
current_perm_index = 0
permutation_list = []

def main():
    global permutation_list
    parser = argparse.ArgumentParser(description='Plot tree glyphs based on the input dataset.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the dataset CSV file.')
    
    args = parser.parse_args()
    
    # Load the dataset
    file_path = args.file_path
    class_column = 'class'
    
    df = pd.read_csv(file_path)
    features = df.drop(columns=[class_column])
    labels = df[class_column]
    
    scaler = MinMaxScaler()
    features_normalized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    dataset_name = os.path.basename(file_path).split(".")[0]
    
    # Generate the list of permutations
    num_features = features_normalized.shape[1]
    permutation_list = list(permutations(range(num_features), num_features))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initial plot
    plot_current_permutation(features_normalized, labels, dataset_name, ax)

    def on_scroll(event):
        global current_perm_index
        if event.button == 'up':
            current_perm_index = (current_perm_index + 1) % len(permutation_list)
        else:
            current_perm_index = (current_perm_index - 1) % len(permutation_list)
        
        plot_current_permutation(features_normalized, labels, dataset_name, ax)
        plt.draw()
    
    def on_key(event):
        if event.key == 'escape':
            plt.close()
        elif event.key == 'ctrl+w':
            plt.close()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

def plot_current_permutation(data, labels, dataset_name, ax=None):
    global current_perm_index
    global permutation_list
    
    if ax is None:
        plt.close('all')
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax.clear()
        
    cols = permutation_list[current_perm_index]
    ax.set_title(f"{dataset_name} with Perm: {cols}")

    plot_trees(ax, data.iloc[:, list(cols)], labels, stretch_factor=5.0)

def draw_tree_colored_stretched(ax, x, y, z, trunk_length, branches, color, stretch_factor=1.0):
    x *= stretch_factor
    y *= stretch_factor
    z *= stretch_factor
    ax.plot([x, x], [y, y], [z, z + trunk_length], c=color)
    z_current = z + trunk_length
    for length, angle in branches:
        angle_scaled = angle * 45
        dx = length * math.sin(math.radians(angle_scaled))
        dy = length * math.cos(math.radians(angle_scaled))
        ax.plot([x, x + dx], [y, y + dy], [z_current, z_current], c=color)

def plot_trees(ax, data, labels, stretch_factor):
    unique_labels = labels.unique()
    colormap = plt.get_cmap('tab10')
    legend_handles = []
    
    for label in unique_labels:
        color_index = np.where(unique_labels == label)[0][0]
        color = colormap(color_index / len(unique_labels))
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label))
        
    for i, (index, row) in enumerate(data.iterrows()):
        label = labels.iloc[i]
        color_index = np.where(unique_labels == label)[0][0]
        color = colormap(color_index / len(unique_labels))
        draw_tree_colored_stretched(ax, row[0], row[1], row[2], row[3], zip(row[4:], row[4:]), color, stretch_factor)

    ax.legend(handles=legend_handles, title='Classes')

if __name__ == '__main__':
    main()
