import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import tkinter as tk
from tkinter import filedialog
import seaborn as sns

class DraggableNDPoint:
    def __init__(self, scc_instance):
        self.scc_instance = scc_instance
        self.selected_nd_point_idx = None
        self.press = None
        self.initial_point = None

    def connect(self):
        self.cidpress = self.scc_instance.ax.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.scc_instance.ax.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.scc_instance.ax.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        for idx, (_, _, start_point, _, row_idx, _) in enumerate(self.scc_instance.lines):
            if self.scc_instance.lines[idx][0].contains(event)[0]:
                self.press = True
                self.selected_nd_point_idx = row_idx
                self.initial_point = (event.xdata, event.ydata)
                break

    def on_motion(self, event):
        if not self.press or self.selected_nd_point_idx is None:
            return

        # Calculate the new angle based on the dragged position
        angle = np.arctan2(event.ydata, event.xdata)
        
        # Compute the circle radius based on the attribute count
        r = len(self.scc_instance.data.columns) - 1
        r = r / (2 * np.pi)
        
        # Move the points of the selected nD point along the circular axis
        for _, _, start_point, end_point, row_idx, _ in self.scc_instance.lines:
            if row_idx == self.selected_nd_point_idx:
                # Update positions based on the new angle and the computed radius
                start_point[0] = r * np.cos(angle)
                start_point[1] = r * np.sin(angle)
                end_point[0] = r * np.cos(angle + (np.pi / len(self.scc_instance.data.columns)))
                end_point[1] = r * np.sin(angle + (np.pi / len(self.scc_instance.data.columns)))
        
        # Refresh the SCCWithChords plot
        self.scc_instance.update_plot()

    def on_release(self, event):
        self.press = None
        self.selected_nd_point_idx = None
        self.initial_point = None

class DraggableLine:
    def __init__(self, line):
        self.line = line
        self.press = None
        self.background = None

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.line.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.line.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.line.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.line.axes: return
        contains, attrd = self.line.contains(event)
        if not contains: return
        self.press = True

    def on_motion(self, event):
        'on motion we will move the line if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.line.axes: return

        # Only change the angle of rotation, not the position
        x0, y0 = 0, 0
        dx = event.xdata - x0
        dy = event.ydata - y0
        angle = np.arctan2(dy, dx)
        r = 2  # fixed distance
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        self.line.set_data([x0, x], [y0, y])
        self.update_predictions(angle)
        
        self.line.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.line.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.line.figure.canvas.mpl_disconnect(self.cidpress)
        self.line.figure.canvas.mpl_disconnect(self.cidrelease)
        self.line.figure.canvas.mpl_disconnect(self.cidmotion)

def lighten_color(color, factor=0.2):
    """
    Lightens the given color.
    
    Parameters:
    color : tuple of float
        Original color as an (R, G, B) tuple.
    factor : float
        Factor to lighten the color. 
        0 means no change, 1 means white color. 
        Default is 0.2.

    Returns:
    tuple of float
        Lightened color as an (R, G, B) tuple.
    """
    r = color[0] + (1 - color[0]) * factor
    g = color[1] + (1 - color[1]) * factor
    b = color[2] + (1 - color[2]) * factor
    return (r, g, b)

def adjusted_bezier_curve(p0, p1, class_order, radius, coef=100):
    """Calculate quadratic Bezier curve points with control points adjusted based on attribute count."""
    
    # Calculate the midpoint between p0 and p1
    x = (p0[0] + p1[0]) / 2
    y = (p0[1] + p1[1]) / 2

    # Calculate the distance between the center point (0,0) and the midpoint
    mDistance = np.sqrt(np.power(-x, 2) + np.power(-y, 2))

    if class_order == 0:  # Inner curve
        # Calculate the scaling factor for the inner curve
        factor = 0.1 * coef / 100
    else:  # Outer curve
        # Calculate the scaling factor for the outer curve
        factor = 4 * coef / 100
    
    mScale = factor * radius / mDistance
    
    # Calculate the control point
    control_point = np.array([x * mScale, y * mScale])

    # Invert the direction for the second class (or any subsequent class)
    if class_order > 0:
        direction = -control_point / np.linalg.norm(control_point)
        control_point = control_point + 2 * direction * mDistance

    # Calculate the Bezier curve points
    t_values = np.linspace(0, 1, 10)
    curve_points = []
    for t in t_values:
        point = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * control_point + t ** 2 * p1
        curve_points.append(point)
    
    return np.array(curve_points)

class SCCWithChords:
    def __init__(self, dataframe):
        self.data = dataframe
        self.positions = []
        self.colors = []
        self.transform_data()
        
    def transform_data(self):
        # The transformation code remains the same as the previous class
        attribute_count = self.data.shape[1] - 1
        scaler = MinMaxScaler((0, 1))
        
        class_column_index = self.data.columns.get_loc('class')
        numeric_data = self.data.drop(columns='class')
        class_data = self.data.iloc[:, class_column_index]
        numeric_data = scaler.fit_transform(numeric_data)
        self.data.iloc[:, [i for i in range(self.data.shape[1]) if i != class_column_index]] = numeric_data

        section_array = np.linspace(0, 1, attribute_count)
        classes = self.data['class'].unique()
        color_palette = sns.color_palette('husl', len(classes))
        self.color_map = dict(zip(classes, color_palette))

        self.all_positions = []
        self.all_colors = []

        for class_order, class_name in enumerate(classes):
            class_positions = []
            class_colors = []
            df_name = self.data[self.data['class'] == class_name]
            for index, row in df_name.iterrows():
                positions = []
                colors = []
                y_values = row.drop('class').values
                x_coord = np.linspace(0, 1, attribute_count)
                arc_length = 0
                for i, y in enumerate(y_values):
                    arc_length += y
                    radius = attribute_count / (2 * np.pi)
                    center_angle = arc_length * 360 / (2 * np.pi * radius)
                    center_angle = np.pi * center_angle / 180
                    x = radius * np.sin(center_angle)
                    y = radius * np.cos(center_angle)
                    positions.append([x, y])
                    colors.append(self.color_map[class_name])
                class_positions.extend(positions)
                class_colors.extend(colors)
            self.all_positions.append(class_positions)
            self.all_colors.append(class_colors)
    
    def on_press(self, event):
        """Called when the mouse button is pressed."""
        # Check if the cursor is on top of a point
        for idx, (positions, colors, class_name) in enumerate(zip(self.all_positions, self.all_colors, self.data['class'].unique())):
            distances = np.linalg.norm(np.array(positions) - [event.xdata, event.ydata], axis=1)
            if np.min(distances) < 0.1:  # 0.1 is a threshold to determine if a point is under the cursor
                self.selected_point_idx = np.argmin(distances)
                self.selected_class_idx = idx
                break

    def on_release(self, event):
        """Called when the mouse button is released."""
        self.selected_point_idx = None
        self.selected_class_idx = None

    def on_drag(self, event):
        """Called when the mouse is dragged."""
        if self.selected_point_idx is None or self.selected_class_idx is None:
            return

        # Compute the new position on the circle
        angle = np.arctan2(event.ydata, event.xdata)
        r = len(self.data.columns) - 1  # the radius
        x = r * np.cos(angle)
        y = r * np.sin(angle)

        # Update the point's position in the plot
        self.all_positions[self.selected_class_idx][self.selected_point_idx] = [x, y]
        self.update_plot()

        # Update the dataset (This step is a bit more involved because you need to compute the new attribute value based on the new position)

    def update_plot(self):
        # Recreate your plot based on the updated data points

        # For simplicity, let's just clear the existing plot and redraw. 
        # You can optimize this if needed.
        self.ax.clear()

        for class_order, (positions, colors) in enumerate(zip(self.all_positions, self.all_colors)):
            positions = np.array(positions)
            self.ax.scatter(positions[:, 0], positions[:, 1], color=colors, s=20, alpha=0.5)

            lightened_color = lighten_color(colors[0])
            for i in range(0, len(positions) - 1, self.attribute_count):  
                for j in range(i, i + self.attribute_count - 1):  # Connect each point to the next within the row
                    start_pos = positions[j]
                    end_pos = positions[j + 1]
                    curve_points = adjusted_bezier_curve(start_pos, end_pos, class_order, self.circle_radius)
                    line, = self.ax.plot(curve_points[:, 0], curve_points[:, 1], color=lightened_color, alpha=0.3)
                    self.lines.append((line, lightened_color, start_pos, end_pos, j // self.attribute_count, class_order))

        # Additional code to plot other elements like circles, labels, etc.

        self.ax.figure.canvas.draw()

    
    def point_side_of_line(self, point, angle):
        x, y = point
        coef = [np.cos(angle), np.sin(angle)]
        boundary = -coef[0] / coef[1]
        
        # y = mx is the line equation; if y > mx, then point is above the line (or to the right)
        return 'right' if y > boundary * x else 'left'

    def update_predictions(self, angle):
        X = self.line.axes.figure.scc_instance.data.drop(columns='class').values
        y = self.line.axes.figure.scc_instance.data['class'].values
        
        # Dictionaries to store counts
        left_counts = {}
        right_counts = {}
        total_counts = {}

        # Determine side of each data point
        for i, point in enumerate(X):
            side = self.point_side_of_line(point, angle)
            class_name = y[i]
            
            if side == 'left':
                left_counts[class_name] = left_counts.get(class_name, 0) + 1
            else:
                right_counts[class_name] = right_counts.get(class_name, 0) + 1
                
            total_counts[class_name] = total_counts.get(class_name, 0) + 1

        # Compute percentages
        left_percentages = {k: (v / total_counts[k]) * 100 for k, v in left_counts.items()}
        right_percentages = {k: (v / total_counts[k]) * 100 for k, v in right_counts.items()}

        # Display the results
        print("Counts & Percentages:")
        for class_name in total_counts:
            print(f"Class {class_name}:")
            print(f"Left of Boundary: {left_counts.get(class_name, 0)} ({left_percentages.get(class_name, 0):.2f}%)")
            print(f"Right of Boundary: {right_counts.get(class_name, 0)} ({right_percentages.get(class_name, 0):.2f}%)")
            print("-" * 40)

        cm = confusion_matrix(y, self.y_pred)
        accuracy = np.mean(y == self.y_pred)
        accuracy_title = f"Confusion Matrix\nAccuracy: {accuracy:.2%}"

        ax2 = self.line.axes.figure.axes[1]  # assuming ax2 is the second axes in the figure
        ax2.clear()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2, cbar=False)
        ax2.set_xlabel('Predicted labels')
        ax2.set_ylabel('True labels')
        ax2.set_title(accuracy_title)

    def on_hover(self, event):
        """Called when the mouse moves over the figure."""
        info_texts = []  # List to store the hover details for all highlighted lines

        hovered_row = None  # Keep track of which data row is being hovered over
        hovered_class = None  # Keep track of which class the hovered data point belongs to

        for line, original_color, start_point, end_point, row, class_info in self.lines:
            if line.contains(event)[0]:
                hovered_row = row
                hovered_class = class_info
                break

        for line, original_color, start_point, end_point, row, class_info in self.lines:
            if row == hovered_row and class_info == hovered_class:
                line.set_color('yellow')
                line.set_alpha(1.0)
                line.set_zorder(1)  # Bring the line to the front
            else:
                line.set_color(original_color)
                line.set_alpha(0.4)
                line.set_zorder(0)

        # Retrieve the correct vector information
        if hovered_row is not None:
            class_names = self.data['class'].unique()
            class_name = class_names[hovered_class]
            vector = self.data[self.data['class'] == class_name].iloc[hovered_row].values
            info_texts.append(str(vector))

        # Fix the hover info box position to the top-left corner of the axes
        self.hover_info_box.set_position((0, 1))
        self.hover_info_box.set_ha('left')
        self.hover_info_box.set_va('top')
        self.hover_info_box.set_transform(self.ax.transAxes)  # Using the axes coordinates

        # Update the textbox with the vector representation
        self.hover_info_box.set_text("\n\n".join(info_texts))
        plt.draw()
    
    def plot(self, lda=None, dataset_name=None):
        fig = plt.figure(figsize=(12, 8))  # Adjusted the figure size for better layout
        
        # Connect mouse events to the methods
        fig.canvas.mpl_connect('button_press_event', self.on_press)
        fig.canvas.mpl_connect('button_release_event', self.on_release)
        fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        # Define gridspec to create a grid layout
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])  # Adjusted to have the confusion matrix on the right
        ax = plt.subplot(gs[0])  # The main visualization will be on the left
        ax2 = plt.subplot(gs[1])  # The confusion matrix will be on the right
        
        self.ax = ax
        self.attribute_count = self.data.shape[1] - 1
        self.circle_radius = self.attribute_count / (2 * np.pi)

        self.lines = []  # To store the plotted lines for hover effect

        if dataset_name:
            ax.set_title(f"{dataset_name} in Dynamic Circular Coordinates")

        for class_order, (positions, colors) in enumerate(zip(self.all_positions, self.all_colors)):
            positions = np.array(positions)
            ax.scatter(positions[:, 0], positions[:, 1], color=colors, s=20, alpha=0.5)

            lightened_color = lighten_color(colors[0])
            for i in range(0, len(positions) - 1, self.attribute_count):  
                for j in range(i, i + self.attribute_count - 1):  # Connect each point to the next within the row
                    start_pos = positions[j]
                    end_pos = positions[j + 1]
                    curve_points = adjusted_bezier_curve(start_pos, end_pos, class_order, self.circle_radius)
                    line, = ax.plot(curve_points[:, 0], curve_points[:, 1], color=lightened_color, alpha=0.3)
                    self.lines.append((line, lightened_color, start_pos, end_pos, j // self.attribute_count, class_order))

        # Connect the motion_notify_event to the on_hover function
        fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        circle = plt.Circle((0, 0), self.circle_radius, color='darkgrey', fill=False)
        ax.add_artist(circle)
        
        # Adjust the label positioning and color
        attributes = self.data.drop(columns='class').columns
        for i, attribute in enumerate(attributes):
            angle = 2 * np.pi * (i + 0.5) / self.attribute_count
            label_radius = (self.circle_radius + self.attribute_count / 4)
            x = label_radius * np.sin(angle)
            y = label_radius * np.cos(angle)
            if 0 <= angle <= np.pi:
                ha = 'left'
            else:
                ha = 'right'
            ax.text(x, y, attribute, ha=ha, va='center', rotation=0, color='darkgrey')
        
        # Style changes: dark grey coordinate axes and labels
        ax.spines['left'].set_color('darkgrey')
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_color('darkgrey')
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.yaxis.tick_left()
        ax.xaxis.tick_bottom()
        ax.xaxis.set_tick_params(width=0.5)
        ax.yaxis.set_tick_params(width=0.5)
        plt.xticks(color='darkgrey')
        plt.yticks(color='darkgrey')

        # Calculate the LDA discrimination line's angle
        coef = lda.coef_[0]
        m = -coef[0] / coef[1]
        theta = np.arctan(m)
        self.circle_radius *= 2

        # Draw the LDA discrimination line as a radial line
        x_end = self.circle_radius * np.cos(theta)
        y_end = self.circle_radius * np.sin(theta)
        # Replace the LDA discrimination line plotting code with this:
        lda_line, = self.ax.plot([0, x_end], [0, y_end], color='black', linestyle='--', picker=5)  # 5 points tolerance
        draggable = DraggableLine(lda_line)
        draggable.connect()
        
        # Label the LDA boundary position
        boundary_label_position_factor = 1.1  # adjust this factor to place the label slightly outside the circle
        x_label = boundary_label_position_factor * x_end
        y_label = boundary_label_position_factor * y_end
        self.ax.text(x_label, y_label, "LDA Boundary", fontsize=8, ha='center')

        # Plot the confusion matrix in ax2
        X = self.data.drop(columns='class').values
        y = self.data['class'].values
        self.y_pred = lda.predict(X)
        accuracy = np.mean(y == self.y_pred)
        cm = confusion_matrix(y, self.y_pred)
        accuracy_title = f"Confusion Matrix\nAccuracy: {accuracy:.2%}"
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2, cbar=False)
        ax2.set_xlabel('Predicted labels')
        ax2.set_ylabel('True labels')
        ax2.set_title(accuracy_title)

        # Create a textbox for hover details
        self.hover_info_box = ax.text(0.0, 0.0, '', transform=ax.transAxes, fontsize=8, 
                              bbox=dict(facecolor='whitesmoke', edgecolor='darkgrey', alpha=0.2, boxstyle='round'))

        # Add legend for class color notation
        for class_name, color in self.color_map.items():
            ax.plot([], [], ' ', label=class_name, marker='o', color=color, markersize=10, markeredgecolor="none")
        ax.legend(loc="best", frameon=False, title="Classes")

        ax.set_xlim([-0.25-self.attribute_count / 4, 0.25+self.attribute_count / 4])
        ax.set_ylim([-0.25-self.attribute_count / 4, 0.25+self.attribute_count / 4])

        ax.set_aspect('equal')
        plt.show()

def load_and_visualize():
    # Open the file-picker dialog
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title="Select a dataset", filetypes=[("CSV files", "*.csv")])

    if not filepath:
        print("No file selected!")
        return

    # Extract the dataset name from the filepath
    dataset_name = filepath.split('/')[-1].split('.')[0]

    # Load the chosen dataset
    df = pd.read_csv(filepath)
    scc_instance = SCCWithChords(df)
    # Fit LDA and predict
    X = df.drop(columns='class').values
    y = df['class'].values
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    # After fitting the LDA and predicting:
    y_pred = lda.predict(X)
    misclassified = np.where(y != y_pred)[0]
    # decision_boundary = None
    # if len(misclassified) > 0:
    #     misclassified_positions = [point[0] for idx in misclassified for point in SCCWithChords.all_positions[idx]]
    #     leftmost = min(misclassified_positions, key=lambda x: x[0])  # Assuming x is the x-coordinate
    #     rightmost = max(misclassified_positions, key=lambda x: x[0])  # Assuming x is the x-coordinate

    #     # Calculate the decision boundary
    #     decision_boundary = (leftmost[0] + rightmost[0]) / 2

    scc_instance.plot(lda, dataset_name=dataset_name)

if __name__ == "__main__":
    load_and_visualize()
