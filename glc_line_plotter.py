import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

def compute_angles(coefficients):
    c_max = np.max(coefficients)
    normalized_coefficients = coefficients / c_max
    transformed_coefficients = np.cos(np.arccos(normalized_coefficients))
    angles = np.arccos(np.abs(transformed_coefficients))
    return angles

def get_lda_coefficients(df, feature_columns, label_column):
    X = df[feature_columns].values
    y = df[label_column].values
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    return lda.coef_[0]

def plot_glyphs(df, dataset_name, coefficients=None):
    feature_columns = [col for col in df.columns if col != 'class']
    label_column = 'class'
    df = normalize_and_clip(df, feature_columns)
    
    if coefficients is None:
        coefficients = np.ones(len(feature_columns))
        
    angles = compute_angles(coefficients)
    
    unique_labels = df[label_column].unique()
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))
    
    plt.figure(figsize=(10, 20))
    
    max_x_value = 0
    for index, row in df.iterrows():
        x_prev = 0
        for i, feature in enumerate(feature_columns):
            a_i = row[feature]  # Length of line segment
            theta_i = angles[i]  # Angle based on coefficient
            x_i = x_prev + a_i * np.cos(theta_i)
            x_prev = x_i
        max_x_value = max(max_x_value, x_i)
    
    plt.subplot(2, 1, 1)
    first_class = unique_labels[0]
    for index, row in df[df[label_column] == first_class].iterrows():
        x_prev, y_prev = 0, 0
        for i, feature in enumerate(feature_columns):  # Added 'i' to enumerate
            a_i = row[feature]
            theta_i = angles[i]
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
        for i, feature in enumerate(feature_columns):  # Added 'i' to enumerate
            a_i = row[feature]
            theta_i = angles[i]
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
    parser.add_argument('--coefficients', type=str, help='Comma-separated coefficients for the linear function. Should match the number of features.')
    args = parser.parse_args()
    
    df = pd.read_csv(args.file_path)
    dataset_name = args.file_path.split('/')[-1].split('.')[0]
    
    feature_columns = [col for col in df.columns if col != 'class']
    label_column = 'class'
    
    if args.coefficients:
        coefficients = np.array([float(x) for x in args.coefficients.split(',')])
        if len(coefficients) != len(feature_columns):
            print("Error: The number of coefficients must match the number of features.")
            exit(1)
    else:
        coefficients = get_lda_coefficients(df, feature_columns, label_column)
    
    plot_glyphs(df, dataset_name, coefficients)
