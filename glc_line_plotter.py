import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def attribute_based_scaling(df, feature_columns):
    for col in feature_columns:
        col_max = df[col].max()
        df[col] = df[col] / col_max
    return df

def compute_angles(coefficients):
    c_max = np.max(coefficients)
    normalized_coefficients = coefficients / c_max
    # Clip the values to be in the range [-1, 1]
    normalized_coefficients = np.clip(normalized_coefficients, -1, 1)
    transformed_coefficients = np.cos(np.arccos(normalized_coefficients))
    angles = np.arccos(np.abs(transformed_coefficients))
    return angles

def get_lda_coefficients(df, feature_columns, label_column):
    X = df[feature_columns].values
    y = df[label_column].values
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    return lda.coef_[0]

# Add this function to calculate the final x-coordinate of the GLC-L glyph
def calculate_final_x(row, angles):
    x_prev = 0
    for i, feature in enumerate(row.index):
        a_i = row[i] / df[feature].max()  # Length of line segment
        theta_i = angles[i]  # Angle based on coefficient
        x_i = x_prev + a_i * np.cos(theta_i)
        x_prev = x_i
    return x_i

# Modify this function to use calculate_final_x
def find_lda_separation_line(df, lda_model, feature_columns, label_column, angles):
    # Get the actual and predicted labels
    X = df[feature_columns].values
    y = df[label_column].values
    y_pred = lda_model.predict(X)

    # Identify the misclassified points
    misclassified = (y != y_pred)
    misclassified_df = df[misclassified]

    # Get the x-projections of the misclassified points
    x_projections = misclassified_df[feature_columns].apply(lambda row: calculate_final_x(row, angles), axis=1)
    
    # Find the left-most and right-most misclassified points
    leftmost_x = x_projections.min()
    rightmost_x = x_projections.max()

    # Calculate the midpoint between the left-most and right-most misclassified points
    midpoint_x = (leftmost_x + rightmost_x) / 2
    return midpoint_x

def plot_lda_separation_line(midpoint_x):
    plt.axvline(x=midpoint_x, color='orange', linestyle='--', linewidth=1)

def plot_glyphs(df, dataset_name, coefficients=None):
    feature_columns = [col for col in df.columns if col != 'class']
    label_column = 'class'
    df = attribute_based_scaling(df, feature_columns)
    
    if coefficients is None:
        coefficients = np.ones(len(feature_columns))
        
    angles = compute_angles(coefficients)
    
    unique_labels = df[label_column].unique()
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))
    # Fit the LDA model
    lda_model = LinearDiscriminantAnalysis()
    X = df[feature_columns].values
    y = df[label_column].values
    lda_model.fit(X, y)
    lda_accuracy = lda_model.score(X, y)
    plt.figure(figsize=(8, 8))
    
    max_x_value = 0
    for unique_label in unique_labels:
        for index, row in df[df[label_column] == unique_label].iterrows():
            x_prev = 0
            for i, feature in enumerate(feature_columns):
                a_i = row[feature]
                theta_i = angles[i]
                x_i = x_prev + a_i * np.cos(theta_i)
                x_prev = x_i
            max_x_value = max(max_x_value, x_i)
    
    plt.subplot(2, 1, 1)
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)

    first_class = unique_labels[0]
    for index, row in df[df[label_column] == first_class].iterrows():
        x_prev, y_prev = 0, 0
        for i, feature in enumerate(feature_columns):
            a_i = row[feature]
            theta_i = angles[i]
            x_i = x_prev + a_i * np.cos(theta_i)
            y_i = y_prev + a_i * np.sin(theta_i)
            plt.plot([x_prev, x_i], [y_prev, y_i], color=label_to_color[row[label_column]], alpha=0.1, zorder=2)
            x_prev, y_prev = x_i, y_i
        plt.scatter(x_i, 0, marker='|', color=label_to_color[row[label_column]], s=100)
        plt.scatter(x_i, y_i, marker='s', color='white', s=12, zorder=3)
        plt.scatter(x_i, y_i, marker='s', color='black', s=10, zorder=3)
    # Find the midpoint for the LDA separation line and plot it
    midpoint_x = find_lda_separation_line(df, lda_model, feature_columns, label_column, angles)
    plot_lda_separation_line(midpoint_x)
    plt.xlim(0, max_x_value + 0.1)
    classes = ', '.join(unique_labels[1:])
    plt.title(f'GLC-L Graph of {dataset_name} - {first_class} vs {classes}  LDA Accuracy: {lda_accuracy:.2f}')
    
    plt.subplot(2, 1, 2)
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)

    for other_class in unique_labels[1:]:
        for index, row in df[df[label_column] == other_class].iterrows():
            x_prev, y_prev = 0, 0
            for i, feature in enumerate(feature_columns):
                a_i = row[feature]
                theta_i = angles[i]
                x_i = x_prev + a_i * np.cos(theta_i)
                y_i = y_prev + a_i * np.sin(theta_i)
                plt.plot([x_prev, x_i], [y_prev, y_i], color=label_to_color[row[label_column]], alpha=0.1, zorder=2)
                x_prev, y_prev = x_i, y_i
            plt.scatter(x_i, 0, marker='|', color=label_to_color[row[label_column]], s=100)
            plt.scatter(x_i, y_i, marker='s', color='white', s=12, zorder=3)
            plt.scatter(x_i, y_i, marker='s', color='black', s=10, zorder=3)
    # Find the midpoint for the LDA separation line and plot it
    midpoint_x = find_lda_separation_line(df, lda_model, feature_columns, label_column, angles)
    plot_lda_separation_line(midpoint_x)

    plt.xlim(0, max_x_value + 0.1)
    
    plt.gca().invert_yaxis()
    
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
    
    feature_columns = [col for col in df.columns if col != 'class']
    label_column = 'class'
    
    coefficients = get_lda_coefficients(df, feature_columns, label_column)
    
    plot_glyphs(df, dataset_name, coefficients)
