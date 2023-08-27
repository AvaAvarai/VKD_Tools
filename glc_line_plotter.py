import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex

def lighten_color(color, amount=0.2):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3, .55, .1), 0.5)
    """
    try:
        c = to_rgb(color)
    except ValueError:
        c = (1.0, 1.0, 1.0)
    c = [(1 - amount) * c[i] + amount for i in [0, 1, 2]]
    return to_hex(c)

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
            theta_i = a_i
            x_i = x_prev + a_i * np.cos(theta_i)
            x_prev = x_i
        max_x_value = max(max_x_value, x_i)
    
    plt.subplot(2, 1, 1)
    first_class = unique_labels[0]
    for index, row in df[df[label_column] == first_class].iterrows():
        x_prev, y_prev = 0, 0
        for feature in feature_columns:
            a_i = row[feature]
            theta_i = a_i
            x_i = x_prev + a_i * np.cos(theta_i)
            y_i = y_prev + a_i * np.sin(theta_i)
            plt.plot([x_prev, x_i], [y_prev, y_i], color=label_to_color[row[label_column]], alpha=0.33)
            x_prev, y_prev = x_i, y_i
        plt.scatter(x_i, 0, marker='|', color=label_to_color[row[label_column]], s=100)
        endpoint_color = lighten_color(label_to_color[row[label_column]], 0.3)  # Lighten the color
        plt.scatter(x_i, y_i, color=endpoint_color, s=30)
    
    plt.xlim(0, max_x_value + 0.1)
    plt.title(f'GLC-L Graph of {dataset_name} - Class: {first_class}')
    
    plt.subplot(2, 1, 2)
    for index, row in df[df[label_column] != first_class].iterrows():
        x_prev, y_prev = 0, 0
        for feature in feature_columns:
            a_i = row[feature]
            theta_i = a_i
            x_i = x_prev + a_i * np.cos(theta_i)
            y_i = y_prev + a_i * np.sin(theta_i)
            plt.plot([x_prev, x_i], [y_prev, y_i], color=label_to_color[row[label_column]], alpha=0.33)
            x_prev, y_prev = x_i, y_i
        plt.scatter(x_i, 0, marker='|', color=label_to_color[row[label_column]], s=100)
        endpoint_color = lighten_color(label_to_color[row[label_column]], 0.3)  # Lighten the color
        plt.scatter(x_i, y_i, color=endpoint_color, s=30)
        
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
