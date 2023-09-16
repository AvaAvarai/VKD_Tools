import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tkinter import Tk, filedialog
from typing import List

def polar_to_cartesian(angle, radius):
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return x, y

def plot_circular_coordinates(data: np.ndarray, target: np.ndarray, attribute_labels: List[str], class_labels: List[str]):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    unique_classes = np.unique(target)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}
    angles = np.linspace(0, 2 * np.pi, len(attribute_labels), endpoint=False).tolist()
    angles = [(np.pi/2 - angle) % (2 * np.pi) for angle in angles]
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    radii = np.linspace(1, len(np.unique(target)), len(np.unique(target)))
    ax.set_xlim(-radii[-1] - 1, radii[-1] + 1)
    ax.set_ylim(-radii[-1] - 1, radii[-1] + 1)
    for radius in radii:
        circle = plt.Circle((0, 0), radius, color='white', fill=False)
        ax.add_artist(circle)
    for i, (angle, label) in enumerate(zip(angles, attribute_labels)):
        x, y = polar_to_cartesian(angle, radii[-1] + 0.3)
        prefixed_label = f"X{{i}} - {{label}}"
        ax.text(x, y, prefixed_label.format(i=i+1, label=label), color='white', horizontalalignment='center', verticalalignment='center')
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[cls], markersize=10) for cls in unique_classes]
    ax.legend(legend_handles, class_labels, loc='upper right')
    angle_step = 2 * np.pi / len(attribute_labels)
    for radius, cls in zip(reversed(radii), reversed(unique_classes)):
        class_indices = np.where(target == cls)[0]
        class_data = scaled_data[class_indices, :]
        for i in range(len(class_data)):
            color = color_map[cls]
            scaled_row = class_data[i]
            x_values = []
            y_values = []
            for j, value in enumerate(scaled_row):
                start_angle = angles[j]
                end_angle = start_angle - angle_step
                point_angle = start_angle - (value * angle_step)
                x, y = polar_to_cartesian(point_angle, radius)
                x_values.append(x)
                y_values.append(y)
            ax.plot(x_values, y_values, color=color, marker='o', markersize=3)
    plt.show()

def select_dataset_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    return file_path

def plot_from_file(file_path: str):
    df = pd.read_csv(file_path)
    target_col = 'class'
    data = df.drop(target_col, axis=1).values
    target = df[target_col].values
    attribute_labels = list(df.drop(target_col, axis=1).columns)
    class_labels = np.unique(target)
    plot_circular_coordinates(data, target, attribute_labels, class_labels)

selected_file_path = select_dataset_file()
if selected_file_path:
    plot_from_file(selected_file_path)
