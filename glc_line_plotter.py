import argparse
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from concurrent.futures import ThreadPoolExecutor

def attribute_based_scaling(df, feature_columns):
    for col in feature_columns:
        col_max = df[col].max()
        df[col] = df[col] / col_max
    return df

def compute_angles(coefficients):
    c_max = np.max(coefficients)
    normalized_coefficients = coefficients / c_max
    normalized_coefficients = np.clip(normalized_coefficients, -1, 1)
    transformed_coefficients = np.cos(np.arccos(normalized_coefficients))
    angles = np.arccos(np.abs(transformed_coefficients))
    return angles

def evaluate_thread(coefficients, df, feature_columns, label_column, result):
    angles = compute_angles(coefficients)
    current_accuracy = evaluateCoefficients(df, feature_columns, label_column, angles)
    result.append((coefficients, current_accuracy))

def coefficients_search(df, feature_columns, label_column, epochs=10, n_threads=4):
    best_coefficients = []
    best_accuracy = 0
    results = []

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for n in range(epochs):
            coefficients = [random.uniform(-1, 1) for _ in range(len(feature_columns))]
            executor.submit(evaluate_thread, coefficients, df, feature_columns, label_column, results)
            
    for coefficients, current_accuracy in results:
        if current_accuracy > best_accuracy:
            best_coefficients = coefficients
            best_accuracy = current_accuracy
    
    return best_coefficients, best_accuracy

def evaluateCoefficients(df, feature_columns, label_column, angles):
    X = df[feature_columns].apply(lambda row: calculate_final_x(row, angles), axis=1).values.reshape(-1, 1)
    y = df[label_column].values
    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X, y)
    return lda_model.score(X, y)

# Add this function to calculate the final x-coordinate of the GLC-L glyph
def calculate_final_x(row, angles):
    x_prev = 0
    for i, feature in enumerate(row.index):
        a_i = row[i] / df[feature].max()  # Length of line segment
        theta_i = angles[i]  # Angle based on coefficient
        x_i = x_prev + a_i * np.cos(theta_i)
        x_prev = x_i
    return x_i

def calculate_final_y(row, angles):
    y_prev = 0
    for i, feature in enumerate(row.index):
        a_i = row[i] / df[feature].max()  # Length of line segment
        theta_i = angles[i]  # Angle based on coefficient
        y_i = y_prev + a_i * np.sin(theta_i)
        y_prev = y_i
    return y_i

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
    y_projections = misclassified_df[feature_columns].apply(lambda row: calculate_final_y(row, angles), axis=1)
    
    # Find the left-most and right-most misclassified points
    leftmost_x = x_projections.min()
    rightmost_x = x_projections.max()
    
    leftmost_y = y_projections.min()
    rightmost_y = y_projections.max()

    # Calculate the midpoint between the left-most and right-most misclassified points
    midpoint_x = (leftmost_x + rightmost_x) / 2
    midpoint_y = (leftmost_y + rightmost_y) / 2
    return midpoint_x, midpoint_y

def plot_lda_separation_line(midpoint_x, midpoint_y):
    plt.axvline(x=midpoint_x, color='orange', linestyle='--', linewidth=1)
    #plt.axhline(y=midpoint_y, color='orange', linestyle='--', linewidth=1)

def calculate_endpoint_percentages(df, lda_model, feature_columns, label_column, angles, midpoint_x):
    x_projections = df[feature_columns].apply(lambda row: calculate_final_x(row, angles), axis=1)
    class_labels = df[label_column].unique()
    percentages = {}

    for label in class_labels:
        label_filter = (df[label_column] == label)
        total_points = sum(label_filter)
        left_points = sum((x_projections[label_filter] < midpoint_x))
        right_points = total_points - left_points
        left_percentage = (left_points / total_points) * 100 if total_points else 0
        right_percentage = (right_points / total_points) * 100 if total_points else 0
        percentages[label] = (left_percentage, right_percentage)

    return percentages


def plot_glyphs(df, dataset_name, coefficients=None, accuracy=None):
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
    plt.figure(figsize=(8, 8), constrained_layout=True)
    
    max_x_value = 0
    max_y_value = 0
    for unique_label in unique_labels:
        for index, row in df[df[label_column] == unique_label].iterrows():
            x_prev = 0
            y_prev = 0
            for i, feature in enumerate(feature_columns):
                a_i = row[feature]
                theta_i = angles[i]
                x_i = x_prev + a_i * np.cos(theta_i)
                y_i = y_prev + a_i * np.sin(theta_i)
                x_prev = x_i
                y_prev = y_i
            max_x_value = max(max_x_value, x_i)
            max_y_value = max(max_y_value, y_i)
    
    max_max = max(max_x_value, max_y_value)
    
    midpoint_x, midpoint_y = find_lda_separation_line(df, lda_model, feature_columns, label_column, angles)
    percentages = calculate_endpoint_percentages(df, lda_model, feature_columns, label_column, angles, midpoint_x)
    half_y_value = max_y_value / 2
    plt.subplot(2, 1, 1)
    #plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    plt.gca().set_facecolor('lightgrey')
    custom_lines = [plt.Line2D([0], [0], color=color, lw=4) for color in label_to_color.values()]
    plt.legend(custom_lines, unique_labels, title='Class')
    first_class = unique_labels[0]
    for index, row in df[df[label_column] == first_class].iterrows():
        x_prev, y_prev = 0, 0
        for i, feature in enumerate(feature_columns):
            a_i = row[feature]
            theta_i = angles[i]
            x_i = x_prev + a_i * np.cos(theta_i)
            y_i = y_prev + a_i * np.sin(theta_i)
            # plot a line from (x_prev, y_prev) to (x_i, y_i)
            plt.plot([x_prev, x_i], [y_prev, y_i], color=label_to_color[row[label_column]], alpha=0.1, zorder=2)
            x_prev, y_prev = x_i, y_i
        plt.scatter(x_i, 0, marker='|', color=label_to_color[row[label_column]], s=100)
        plt.scatter(x_i, y_i, marker='s', color='white', s=12, zorder=3)
        plt.scatter(x_i, y_i, marker='s', color='black', s=8, zorder=3)

    first_class = unique_labels[0]
    left_percentage, right_percentage = percentages[first_class]
    plt.xticks([])  # Remove x-axis numbering
    plt.yticks([])  # Remove y-axis numbering
    plt.text(0.05, 0.52, f"{left_percentage:.2f}% of {first_class}", fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.95, 0.52, f"{right_percentage:.2f}% of {first_class}", fontsize=12, ha="right", transform=plt.gcf().transFigure)

    plot_lda_separation_line(midpoint_x, midpoint_y)
    plt.xlim(0, max_max + 0.1)
    plt.ylim(0, max_max + 0.1)
    classes = ', '.join(map(str, unique_labels[1:]))
    plt.title(f'GLC-L Graph of {dataset_name} - {first_class} vs {classes}  LDA Accuracy: {accuracy:.2f}')
    
    plt.subplot(2, 1, 2)
    #plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    plt.gca().set_facecolor('lightgrey')
    for idx, other_class in enumerate(unique_labels[1:]):
        y_increment = 0.025 * (max_y_value / (len(unique_labels) - 1))
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
            plt.scatter(x_i, y_i, marker='s', color='black', s=8, zorder=3)
        left_percentage, right_percentage = percentages[other_class]
        y_position = idx * y_increment
        plt.text(0.05, 0.025 + y_position, f"{left_percentage:.2f}% of {other_class}", fontsize=12, transform=plt.gcf().transFigure)
        plt.text(0.95, 0.025 + y_position, f"{right_percentage:.2f}% of {other_class}", fontsize=12, ha="right", transform=plt.gcf().transFigure)
    plt.xticks([])  # Remove x-axis numbering
    plt.yticks([])  # Remove y-axis numbering  
    # Find the midpoint for the LDA separation line and plot it
    midpoint_x, midpoint_y = find_lda_separation_line(df, lda_model, feature_columns, label_column, angles)
    plot_lda_separation_line(midpoint_x, midpoint_y)

    plt.xlim(0, max_max + 0.1)
    plt.ylim(0, max_max + 0.1)
    plt.gca().invert_yaxis()
    
    def close(event):
        if event.key == 'escape' or (event.key == 'w' and event.inaxes == None):
            plt.close(event.canvas.figure)
            
    plt.connect('key_press_event', close)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot GLC-L Type Graph.')
    parser.add_argument('--file_path', type=str, help='Path to the dataset CSV file.')
    args = parser.parse_args()
    
    df = pd.read_csv(args.file_path)
    dataset_name = args.file_path.split('/')[-1].split('.')[0]
    
    feature_columns = [col for col in df.columns if col != 'class']
    label_column = 'class'
    
    coefficients, accuracy = coefficients_search(df, feature_columns, label_column, epochs=100)

    plot_glyphs(df, dataset_name, coefficients, accuracy)
