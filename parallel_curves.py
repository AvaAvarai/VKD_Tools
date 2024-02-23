import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def fourier_series_same_coeff(x, coefficients, T=2*np.pi):
    result = np.zeros_like(x)
    
    for n in range(len(coefficients)):
        result += coefficients[n] * (np.cos(2 * np.pi * n * x / T) + np.sin(2 * np.pi * n * x / T))
        
    return result

def main(file_path):
    # Load and normalize the dataset
    df = pd.read_csv(file_path)
    df_normalized = (df.drop('class', axis=1) - df.drop('class', axis=1).min()) / (df.drop('class', axis=1).max() - df.drop('class', axis=1).min())
    df_normalized['class'] = df['class']

    # Generate x values
    x_values = np.linspace(0, 2*np.pi, 500)

    # Generate unique colors for each class
    unique_classes = df_normalized['class'].unique()
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_classes)))
    class_color_map = dict(zip(unique_classes, colors))

    # Plot the first few data points in the dataset as Fourier series
    plt.figure(figsize=(14, 8))

    for class_name, color in class_color_map.items():
        df_class = df_normalized[df_normalized['class'] == class_name]
        for idx, row in df_class.iterrows():
            coefficients = row.drop('class').values
            y_values = fourier_series_same_coeff(x_values, coefficients)
            plt.plot(x_values, y_values, color=color, label=class_name if idx == df_class.index[0] else "")

    # Create a custom legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Classes")

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    plt.title(f"Andrew's Curves of {file_name} Dataset")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render parallel coordinates from a CSV file.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the CSV file')
    args = parser.parse_args()

    main(args.file_path)
