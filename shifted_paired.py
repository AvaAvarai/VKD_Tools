import math
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from itertools import islice, cycle, permutations

# Initialize global variables
current_perm_index = 0
current_permutation = None
perms = None  # This will hold the cycle object

def perm_generator(n):
    if n == 1:
        yield (0,)
    else:
        for perm in perm_generator(n - 1):
            for i in range(n):
                yield perm[:i] + (n - 1,) + perm[i:]

# This function regenerates the generator up to a given index
def regenerate_perms_to_index(n, index):
    gen = perm_generator(n)
    for _ in range(index + 1):
        perm = next(gen)
    return gen, perm

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
    global current_permutation, perms, current_perm_index  # Use global instead of nonlocal
    num_features = features_normalized.shape[1]
    if event.button == 'up':
        current_permutation = next(perms)
        current_perm_index += 1
    else:
        # To go to the previous permutation, regenerate the cycle up to the current index - 1
        if current_perm_index > 0:
            current_perm_index -= 1
            perms = cycle(permutations(range(num_features), num_features))
            current_permutation = next(islice(perms, current_perm_index, current_perm_index + 1))

    plt.clf()
    draw_plot()

def draw_plot():
    global current_idx, features_normalized, labels, unique_labels, colors, current_permutation  # Added current_permutation here
    
    # Create a new figure
    fig = plt.gcf()
    
    # Attach exit function to the 'key_press_event'
    fig.canvas.mpl_connect('key_press_event', exit_program)

    # Calculate the total number of permutations (factorial of the number of features)
    total_perms = math.factorial(features_normalized.shape[1])
    
    # Modulo operation to ensure current_idx is within range
    current_idx %= total_perms

    # Get the current permutation
    perm = current_permutation  # Replaced current_perm with current_permutation

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
        ax.set_title(f"(X{i + 1}, X{j + 1})")

    # Connect points for each feature vector sample across subplots
    for l in range(len(features_normalized)):
        color = colors[np.where(unique_labels == labels[l])[0][0]]
        xy_points = [(features_normalized[l, perm[i]], features_normalized[l, perm[i+1]]) for i in range(0, len(perm)-1, 2)]

        for ax1, ax2, point1, point2 in zip(axes[:-1], axes[1:], xy_points[:-1], xy_points[1:]):
            coord1 = ax1.transData.transform(point1)
            coord2 = ax2.transData.transform(point2)

            fig_coord1 = fig.transFigure.inverted().transform(coord1)
            fig_coord2 = fig.transFigure.inverted().transform(coord2)

            line = Line2D([fig_coord1[0], fig_coord2[0]], [fig_coord1[1], fig_coord2[1]], transform=fig.transFigure, color=color, alpha=0.33)
            fig.lines.append(line)
    cols = current_permutation
    plt.suptitle(f'{dataset_name} in Shifted Paired Coordinates with Permutation: {perm}')
    plt.draw()

def visualize_dataset(file_path):
    global features_normalized, labels, unique_labels, colors, lda_coefficients, current_perm_index, current_permutation, perms, dataset_name
    
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

    # Initialize the generator and current index
    num_features = features_normalized.shape[1]
    perms = cycle(permutations(range(num_features), num_features))
    current_perm_index = 0  # Initialize to zero
    
    # Get the first permutation based on LDA coefficients
    current_permutation = sort_permutations_by_lda([next(perms)], lda_coefficients)[0]
    
    # Reset the generator and current index
    current_idx = 0
    perms, current_perm = regenerate_perms_to_index(num_features, current_idx)
    
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
