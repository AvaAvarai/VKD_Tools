import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import argparse

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-dimensional Glyph Plotter")
    parser.add_argument("--file_path", required=True, help="Path to the CSV file")
    return parser.parse_args()

# Function to draw a glyph
def draw_glyph(ax, x, y, angles, color):
    xs = [x + np.cos(angle) for angle in angles]
    ys = [y + np.sin(angle) for angle in angles]
    xs.append(xs[0])  # Close the shape by appending the first point at the end
    ys.append(ys[0])  # Same as above
    ax.fill(xs, ys, color=color, alpha=0.7)

def main():
    args = parse_args()
    df = pd.read_csv(args.file_path)

    features = [col for col in df.columns if col.lower() != 'class']
    feature_combinations = list(itertools.combinations(features, 2))

    # Calculate the number of rows needed for subplots
    num_rows = int(np.ceil(len(feature_combinations) / 2.0))

    # Initialize a figure to plot subplots
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 15))

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Remove extra subplots if any
    for i in range(len(axes), len(feature_combinations), -1):
        fig.delaxes(axes[i-1])

    for i, (x_feature, y_feature) in enumerate(feature_combinations):
        ax = axes[i]
        
        angle_features = [f for f in features if f not in [x_feature, y_feature]]
        
        for cls, color in zip(df['class'].unique(), ['blue', 'orange', 'green']):
            subset = df[df['class'] == cls]
            
            for _, row in subset.iterrows():
                x, y = row[x_feature], row[y_feature]
                angles = [2 * np.pi * (row[feature] - subset[feature].min()) / (subset[feature].max() - subset[feature].min()) for feature in angle_features]
                draw_glyph(ax, x, y, angles, color)
                
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.legend(handles=[plt.Rectangle((0,0),1,1, color=color, alpha=0.7, label=cls) for cls, color in zip(df['class'].unique(), ['blue', 'orange', 'green'])])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
