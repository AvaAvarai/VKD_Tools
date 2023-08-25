import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations
from sklearn.preprocessing import MinMaxScaler
import argparse

def main():
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
    plot_combinatorial_trees(features_normalized, labels, stretch_factor=5.0, dataset_name=dataset_name)

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
    for i, (index, row) in enumerate(data.iterrows()):
        label = labels.iloc[i]
        color_index = np.where(unique_labels == label)[0][0]
        color = colormap(color_index / len(unique_labels))
        draw_tree_colored_stretched(ax, row[0], row[1], row[2], row[3], zip(row[4:], row[4:]), color, stretch_factor)

def plot_combinatorial_trees(data, labels, num_cols=4, stretch_factor=1.0, dataset_name="output"):
    output_folder = f"{dataset_name}_output_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    num_features = data.shape[1]
    fig_count = 0
    ax_count = 0
    num_rows_per_fig = 4
    num_subplots_per_fig = num_rows_per_fig * num_cols

    for i, cols in enumerate(permutations(range(num_features), num_features)):  # Using all features
        if ax_count == 0:
            plt.close('all')  # Close the previous plot
            fig = plt.figure(figsize=(24, 24))
            fig_count += 1
            print(f"Processing plot number {fig_count}")

        ax = fig.add_subplot(num_rows_per_fig, num_cols, ax_count + 1, projection='3d')
        ax.set_title(f"Perm: {cols}")
        plot_trees(ax, data.iloc[:, list(cols)], labels, stretch_factor)

        ax_count += 1

        if ax_count >= num_subplots_per_fig:
            ax_count = 0
            unique_labels = labels.unique()
            colormap = plt.get_cmap('tab10')
            legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                         markersize=10, markerfacecolor=colormap(i / len(unique_labels)))
                              for i, label in enumerate(unique_labels)]
            fig.legend(handles=legend_handles, loc='upper right', title='Classes')
            save_path = os.path.join(output_folder, f"{dataset_name}_combinatorial_trees_{fig_count}.png")
            plt.tight_layout()
            plt.savefig(save_path)

    if ax_count > 0:
        unique_labels = labels.unique()
        colormap = plt.get_cmap('tab10')
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                     markersize=10, markerfacecolor=colormap(i / len(unique_labels)))
                          for i, label in enumerate(unique_labels)]
        fig.legend(handles=legend_handles, loc='upper right', title='Classes')
        save_path = os.path.join(output_folder, f"{dataset_name}_combinatorial_trees_{fig_count}.png")
        plt.tight_layout()
        plt.savefig(save_path)

if __name__ == '__main__':
    main()
