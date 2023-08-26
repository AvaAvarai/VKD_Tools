
"""
GLC-L Type Graph Plotter

Mathematical Algorithm:
-----------------------
Given a dataset with 'n' attributes and a label column named 'class':
1. Normalize the attributes to the range [0, 1].
2. For each data point in the dataset:
    a. Initialize x_prev, y_prev = 0, 0 (starting at the origin).
    b. For each attribute a_i:
        i. Calculate theta_i = arccos(abs(a_i)).
        ii. Update x_i = x_prev + a_i * cos(theta_i) and y_i = y_prev + a_i * sin(theta_i).
        iii. Plot a line segment from (x_prev, y_prev) to (x_i, y_i).
        iv. Update x_prev, y_prev to x_i, y_i for the next iteration.
3. Color code each glyph according to its class label.

"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tkinter import Frame, Tk, Label, Button, StringVar, OptionMenu

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

def plot_glyphs(df, dataset_name, angle_function):
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
    
    plt.figure(figsize=(10, 10))
    furthest_endpoints = {label: 0 for label in unique_labels}
    
    for index, row in df.iterrows():
        x_prev, y_prev = 0, 0
        for feature in feature_columns:
            a_i = row[feature]
            theta_i = angle_function(np.abs(a_i))
            x_i = x_prev + a_i * np.cos(theta_i)
            y_i = y_prev + a_i * np.sin(theta_i)
            plt.plot([x_prev, x_i], [y_prev, y_i], color=label_to_color[row[label_column]])
            x_prev, y_prev = x_i, y_i
        furthest_endpoints[row[label_column]] = max(furthest_endpoints[row[label_column]], x_i)
            
    for label, x in furthest_endpoints.items():
        plt.plot([x, x], [0, 0.1], color=label_to_color[label], linestyle='--', linewidth=2)
    
    custom_lines = [plt.Line2D([0], [0], color=color, lw=4) for color in label_to_color.values()]
    plt.legend(custom_lines, unique_labels, title='Class')
    
    plt.title(f'GLC-L Type Graph of {dataset_name} with {angle_function.__name__}()')

    def close(event):
        """
        Close the plot window on pressing 'Escape' or 'Ctrl+W' keys.
        """
        if event.key == 'escape' or (event.key == 'w' and event.inaxes == None):
            plt.close(event.canvas.figure)
            
    plt.connect('key_press_event', close)
    plt.show()

class AngleFunctionSelector:

    def __init__(self, master):
        self.master = master
        master.title("Select Angle Function")
        
        # Center the window on the screen
        window_width = 250
        window_height = 150
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))
        master.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

        # Create a frame and pack it in the center
        self.frame = Frame(master)
        self.frame.pack(side="top", pady=20)
        
        # Add the label to the frame
        self.label = Label(self.frame, text="Select the angle function for plotting")
        self.label.pack(side="top")
        
        self.angle_function = StringVar(master)
        self.angle_function.set("arccos")  # default value
        self.angle_function_map = {
            'arccos': np.arccos,
            'arcsin': np.arcsin,
            'arctan': np.arctan,
        }
        
        # Add the dropdown to the frame
        self.dropdown = OptionMenu(self.frame, self.angle_function, *self.angle_function_map.keys())
        self.dropdown.pack(side="top")
        
        # Add the button to the frame
        self.plot_button = Button(self.frame, text="Plot", command=self.plot)
        self.plot_button.pack(side="top")

    def plot(self):
        selected_function = self.angle_function_map[self.angle_function.get()]
        plot_glyphs(df, dataset_name, selected_function)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot GLC-L Type Graph.')
    parser.add_argument('--file_path', type=str, help='Path to the dataset CSV file.')
    args = parser.parse_args()
    
    df = pd.read_csv(args.file_path)
    dataset_name = args.file_path.split('/')[-1].split('.')[0]
    
    root = Tk()
    app = AngleFunctionSelector(root)
    root.mainloop()