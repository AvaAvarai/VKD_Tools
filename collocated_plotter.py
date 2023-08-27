import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from itertools import permutations
from sklearn.preprocessing import MinMaxScaler
import argparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Global variable to store LDA coefficients
lda_coefficients = None

def sort_permutations_by_lda(perms, lda_coefficients):
    # Sort individual features by LDA coefficients
    sorted_features = np.argsort(lda_coefficients)[::-1]
    
    # Create the "most important" permutation based on sorted features
    most_important_perm = tuple(sorted_features)
    
    # Add this permutation at the beginning
    if most_important_perm not in perms:
        perms = [most_important_perm] + perms
    else:
        # Move the most important permutation to the beginning
        perms.remove(most_important_perm)
        perms = [most_important_perm] + perms
    
    return perms

def run_lda(features, labels):
    global lda_coefficients
    lda = LinearDiscriminantAnalysis()
    lda.fit(features, labels)
    lda_coefficients = np.abs(lda.coef_).mean(axis=0)  # Taking the absolute and averaging over classes if multi-class

# Initialize the current index for permutations
current_idx = 0

def exit_program(event):
    if event.key == 'escape' or event.key == 'ctrl+w':
        plt.close()

def update_plot(event):
    global current_idx
    if event.button == 'up':
        current_idx += 1
    elif event.button == 'down':
        current_idx -= 1
    plt.clf()
    draw_plot()

def draw_plot():
    global current_idx, features_normalized, labels, unique_labels, colors, perms
    
    # Create a new figure
    fig = plt.gcf()
    
    # Attach exit function to the 'key_press_event'
    fig.canvas.mpl_connect('key_press_event', exit_program)

    # Bound the current index
    current_idx %= len(perms)

    # Get the current permutation
    perm = perms[current_idx]

    # Calculate the number of subplots needed
    num_subplots = len(perm) // 2 + (len(perm) % 2)
    axes = [fig.add_subplot(1, num_subplots, i+1) for i in range(num_subplots)]

    # Create a separate axis for the legend at the top left, below the title
    legend_axis = fig.add_axes([0.05, 0.85, 0.2, 0.1])
    legend_axis.axis('off')  # Turn off axis lines and ticks

    # Create a legend for the colors
    legend_handles = [Line2D([0], [0], marker='x', color='w', markeredgecolor=colors[i], label=unique_labels[i], markersize=10) for i in range(len(unique_labels))]

    # Add the legend to the new axis
    legend_axis.legend(handles=legend_handles, loc='upper left')
    
    # Plot points in each subplot
    for ax, i, j in zip(axes, perm[::2], perm[1::2]):
        for l in range(len(features_normalized)):
            x, y = features_normalized[l, i], features_normalized[l, j]
            label = labels[l]
            color = colors[np.where(unique_labels == label)[0][0]]

            # Plot the point using 'x' marker
            ax.scatter(x, y, color=color, s=50, marker='x')

        # Move the axis labels to the top-left corner by setting them as the title
        ax.set_title(f"X{i + 1} on x / X{j + 1} on y")

    # Connect points for each feature vector sample across subplots
    for l in range(len(features_normalized)):
        color = colors[np.where(unique_labels == labels[l])[0][0]]
        xy_points = [(features_normalized[l, perm[i]], features_normalized[l, perm[i+1]]) for i in range(0, len(perm)-1, 2)]

        for ax1, ax2, point1, point2 in zip(axes[:-1], axes[1:], xy_points[:-1], xy_points[1:]):
            coord1 = ax1.transData.transform(point1)
            coord2 = ax2.transData.transform(point2)

            fig_coord1 = fig.transFigure.inverted().transform(coord1)
            fig_coord2 = fig.transFigure.inverted().transform(coord2)

            line = Line2D([fig_coord1[0], fig_coord2[0]], [fig_coord1[1], fig_coord2[1]], transform=fig.transFigure, color=color)
            fig.lines.append(line)
    plt.suptitle(f'{dataset_name} in Collocated Paired Coordinates')
    plt.draw()

def visualize_dataset(file_path):
    global features_normalized, labels, unique_labels, colors, perms, lda_coefficients, current_idx, dataset_name
    
    # Extract the dataset name from the file path
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # Check if the number of features is odd, if so duplicate the last column
    if (df.shape[1] - 1) % 2 != 0:
        last_col = df.columns[-2]
        df[last_col + '_dup'] = df[last_col]

    # Normalize the features
    scaler = MinMaxScaler()
    features = df.drop(columns=['class']).values
    features_normalized = scaler.fit_transform(features)

    # Get class labels
    labels = df['class'].values
    unique_labels = np.unique(labels)

    # Create a color map
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    # Run LDA and get coefficients
    run_lda(features_normalized, labels)

    # Generate all permutations of the feature vectors
    num_features = features_normalized.shape[1]
    perms = list(permutations(range(num_features)))

    # Sort permutations based on LDA coefficients
    perms = sort_permutations_by_lda(perms, lda_coefficients)

    # Reset current_idx to 0 so that the first permutation displayed is the most important one
    current_idx = 0  # Reset index
    
    fig = plt.figure(figsize=(15, 8))
    fig.canvas.mpl_connect('scroll_event', update_plot)
    fig.canvas.mpl_connect('key_press_event', exit_program)  # Connect exit function here
    
    draw_plot()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize dataset.')
    parser.add_argument('--file_path', type=str, help='Path to the CSV file containing the dataset')
    args = parser.parse_args()

    if args.file_path:
        visualize_dataset(args.file_path)
