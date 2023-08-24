import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QFileDialog
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPainter, QColor

from OpenGL.GL import *

import numpy as np
import pandas as pd

import distinctipy as dp

def draw_rectangle(center, search_radius_x, search_radius_y):
    """Draw a filled rectangular region around the given center with the specified half widths."""
    x, y = center
    
    # Calculate rectangle boundaries based on the two search radii
    bottom_left_x = x - search_radius_x
    bottom_left_y = y - search_radius_y
    top_right_x = x + search_radius_x
    top_right_y = y + search_radius_y

    glColor4f(0.5, 0.5, 0.5, 0.5)  # Semi-transparent gray for shading
    glRectf(bottom_left_x, bottom_left_y, top_right_x, top_right_y)

def adjust_search_radii(key, search_radius_x, search_radius_y, step=0.01):
    """
    Adjust the search_radius_x and search_radius_y of the rectangular search region based on the key pressed.
    
    Args:
    - key (str): Key pressed ('W', 'S', 'A', or 'D').
    - search_radius_x (float): Current x search radius of the rectangular search region.
    - search_radius_y (float): Current y search radius of the rectangular search region.
    - step (float, optional): The step by which the search_radius is increased or decreased. Default is 2.5.
    
    Returns:
    - tuple: New search_radius_x and search_radius_y values.
    """
    if key == 'W':
        search_radius_y += step
    elif key == 'S':
        search_radius_y -= step
    elif key == 'A':
        search_radius_x -= step
    elif key == 'D':
        search_radius_x += step

    # Ensure that search_radius_x and search_radius_y do not go below a minimum value (for instance, 5)
    search_radius_x = max(search_radius_x, 0.01)
    search_radius_y = max(search_radius_y, 0.01)

    return search_radius_x, search_radius_y

def compute_outcode(x, y, xmin, xmax, ymin, ymax):
    """Compute the outcode for a point (x, y) against a rectangle."""
    INSIDE = 0  # 0000
    LEFT = 1    # 0001
    RIGHT = 2   # 0010
    BOTTOM = 4  # 0100
    TOP = 8     # 1000

    code = INSIDE
    if x < xmin:
        code |= LEFT
    elif x > xmax:
        code |= RIGHT
    if y < ymin:
        code |= BOTTOM
    elif y > ymax:
        code |= TOP

    return code

def cohen_sutherland_line_clip(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
    """Clip a line segment using the Cohen-Sutherland algorithm."""

    # Compute outcodes
    outcode0 = compute_outcode(x0, y0, xmin, xmax, ymin, ymax)
    outcode1 = compute_outcode(x1, y1, xmin, xmax, ymin, ymax)

    max_iterations = 10
    iterations = 0  # Initialize iteration counter

    while iterations < max_iterations:
        iterations += 1
        if not (outcode0 | outcode1):  # Trivially accept
            return True
        elif outcode0 & outcode1:     # Trivially reject
            return False
        else:
            x, y = 0.0, 0.0  # Point of intersection
            # Pick an outcode to work on
            if outcode0 != 0:
                outcode_out = outcode0
            else:
                outcode_out = outcode1

            if outcode_out & compute_outcode(0, 0, 0, 0, ymin, ymax):  # Top
                x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0)
                y = ymax
            elif outcode_out & compute_outcode(0, 0, 0, 0, 0, ymax):   # Bottom
                x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0)
                y = ymin
            elif outcode_out & compute_outcode(0, 0, xmin, 0, 0, 0):   # Right
                y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
                x = xmax
            elif outcode_out & compute_outcode(0, 0, 0, xmax, 0, 0):   # Left
                y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0)
                x = xmin

            if outcode_out == outcode0:
                x0, y0 = x, y
                outcode0 = compute_outcode(x0, y0, xmin, xmax, ymin, ymax)
            else:
                x1, y1 = x, y
                outcode1 = compute_outcode(x1, y1, xmin, xmax, ymin, ymax)

    return False  # Default return, shouldn't reach here

class OpenGLPlot(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = pd.DataFrame()
        self.classes = pd.Series()
        self.unique_classes = []
        self.colors = []
        
        # Additional storage for envelope minima and maxima
        self.envelope_min = pd.DataFrame()
        self.envelope_max = pd.DataFrame()
        self.drawn_x_min = None
        self.drawn_x_max = None
        self.drawn_y_min = None
        self.drawn_y_max = None
        self.show_envelope = False  # Initially set to not show the envelope

    def load_data(self, file_name):
        dataset = pd.read_csv(file_name)
        self.classes = dataset['class']
        self.data = dataset.drop(columns=['class'])
        
        self.dataset_name = file_name.split("/")[-1]  # Store just the filename
        
        # Combine the data and classes into a single DataFrame
        combined = pd.concat([self.data, self.classes], axis=1)
        
        # Sort the combined DataFrame by the class labels
        combined = combined.sort_values(by='class')
        
        # Split the sorted DataFrame back into data and classes
        self.classes = combined['class']
        self.data = combined.drop(columns=['class'])
        
        self.unique_classes = self.classes.unique()

        # Generate unique colors for each class
        #self.colors = np.linspace(0.1, 0.9, len(self.unique_classes))
        self.colors = dp.get_colors(len(self.unique_classes))
        
        # Compute minima and maxima for each class
        self.envelope_min = self.data.groupby(self.classes).min()
        self.envelope_max = self.data.groupby(self.classes).max()

        self.normalize_data()
        self.update()

    def normalize_data(self):
        min_val = self.data.min()
        max_val = self.data.max()

        self.data = (self.data - min_val) / (max_val - min_val)
        self.envelope_min = (self.envelope_min - min_val) / (max_val - min_val)
        self.envelope_max = (self.envelope_max - min_val) / (max_val - min_val)

    def get_color_for_class(self, class_label):
        idx = np.where(self.unique_classes == class_label)[0][0]
        return self.colors[idx]

    def initializeGL(self):
        glClearColor(0.85, 0.85, 0.85, 1)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glOrtho(-1, 1, 0, 1, 1, -1)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    def paintGL(self):
        glClearColor(0.85, 0.85, 0.85, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)

        if self.data.empty:
            return

        num_axes = self.data.shape[1]

        # Define margins
        x_margin = 0.2
        y_margin = 0.05
        x_min = -1 + x_margin
        x_max = 1 - x_margin
        y_min = y_margin
        y_max = 1 - y_margin

        # Adjust axis_gap for x_margin
        axis_gap = (x_max - x_min) / (num_axes - 1)

        # Draw axes
        glColor3f(0, 0, 0)  # Black color for axes
        for i in range(num_axes):
            glBegin(GL_LINES)
            glVertex2f(x_min + i * axis_gap, y_min)
            glVertex2f(x_min + i * axis_gap, y_max)
            glEnd()

        # Start QPainter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setPen(Qt.GlobalColor.black)
        font = painter.font()
        font.setPointSize(font.pointSize() - 1)  # Decrease font size by 1
        painter.setFont(font)

        # Positioning details
        start_x = 10
        start_y = 25
        rect_size = 15
        gap = 5

        painter.drawText(start_x, start_y - 10, self.dataset_name)
        for idx, class_label in enumerate(self.unique_classes):
            r, g, b = [int(c * 255) for c in self.get_color_for_class(class_label)]
            painter.setBrush(QColor(r, g, b))
            
            rect_x = int(start_x)
            rect_y = int(start_y + (rect_size + gap) * idx)
            painter.drawRect(rect_x, rect_y, rect_size, rect_size)

            label = f"{class_label} ({(self.classes == class_label).sum()})"
            text_x = int(start_x + rect_size + gap)
            
            text_rect = painter.fontMetrics().boundingRect(label)
            
            # Adjust the vertical positioning to be centered with the colored box
            text_y = rect_y + (rect_size - text_rect.height()) // 2 + text_rect.height() - 3
            
            painter.drawText(text_x, text_y, label)
        
        # Draw axes and attribute names
        attribute_names = self.data.columns.tolist()  # Get attribute names from the DataFrame
        for i, attribute_name in enumerate(attribute_names):
            glBegin(GL_LINES)
            glVertex2f(x_min + i * axis_gap, y_min)
            glVertex2f(x_min + i * axis_gap, y_max)
            glEnd()
            
            # Calculate the width of the text string
            text_width = painter.fontMetrics().horizontalAdvance(attribute_name)
            
            # Render attribute name below the axis using QPainter
            text_x = int((x_min + i * axis_gap) * self.width() * 0.5 + self.width() * 0.5 - text_width / 2)
            text_y = int(self.height() - 10)
            painter.drawText(text_x, text_y, attribute_name)

        # End QPainter
        painter.end()

        # Reset OpenGL states for drawing lines
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        glLineWidth(0.5)  # Set line width
        if not self.show_envelope:
            # Draw individual data lines
            for idx, row in self.data.iterrows():
                r, g, b = self.get_color_for_class(self.classes[idx])
                glColor4f(r, g, b, 0.5)
                glBegin(GL_LINE_STRIP)
                for i, value in enumerate(row):
                    x = x_min + i * axis_gap
                    y = y_min + (y_max - y_min) * value

                    # Capture bounds
                    self.drawn_x_min = min(self.drawn_x_min or x, x)
                    self.drawn_x_max = max(self.drawn_x_max or x, x)
                    self.drawn_y_min = min(self.drawn_y_min or y, y)
                    self.drawn_y_max = max(self.drawn_y_max or y, y)

                    glVertex2f(x, y)
                glEnd()
        else:
            # Draw the envelope
            for class_label in self.unique_classes:
                r, g, b = self.get_color_for_class(class_label)
                
                # Draw the shaded envelope area using triangles
                glColor4f(r, g, b, 0.1)  # Very slightly transparent for shading
                
                for i in range(self.data.shape[1] - 1):
                    # Define the four points for this segment of the envelope
                    top_left = (x_min + i * axis_gap, y_min + (y_max - y_min) * self.envelope_max.loc[class_label, self.data.columns[i]])
                    top_right = (x_min + (i + 1) * axis_gap, y_min + (y_max - y_min) * self.envelope_max.loc[class_label, self.data.columns[i + 1]])
                    bottom_left = (x_min + i * axis_gap, y_min + (y_max - y_min) * self.envelope_min.loc[class_label, self.data.columns[i]])
                    bottom_right = (x_min + (i + 1) * axis_gap, y_min + (y_max - y_min) * self.envelope_min.loc[class_label, self.data.columns[i + 1]])
                    
                    # Draw two triangles to form the shaded envelope for this segment
                    glBegin(GL_TRIANGLES)
                    glVertex2f(*top_left)
                    glVertex2f(*top_right)
                    glVertex2f(*bottom_left)
                    
                    glVertex2f(*bottom_left)
                    glVertex2f(*top_right)
                    glVertex2f(*bottom_right)
                    glEnd()
                
                # Draw the maximum envelope line
                glColor4f(r, g, b, 0.5)
                glBegin(GL_LINE_STRIP)
                for i in range(self.data.shape[1]):
                    glVertex2f(x_min + i * axis_gap, y_min + (y_max - y_min) * self.envelope_max.loc[class_label, self.data.columns[i]])
                glEnd()

                # Draw the minimum envelope line
                glColor4f(r, g, b, 0.5)
                glBegin(GL_LINE_STRIP)
                for i in range(self.data.shape[1]):
                    glVertex2f(x_min + i * axis_gap, y_min + (y_max - y_min) * self.envelope_min.loc[class_label, self.data.columns[i]])
                glEnd()

        glEnable(GL_DEPTH_TEST)

    def resizeGL(self, w, h):
        # Set the viewport to take up the full canvas
        glViewport(0, 0, w, h)


class ModifiedOpenGLPlot(OpenGLPlot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Attributes to store the mouse position and circle radius
        self.mouse_pos = None
        self.search_radius_x = 0.025
        self.search_radius_y = 0.025
        self.statistics = []

    def renormalize_search_bounds(self, left, right, bottom, top):
        """
        Convert search rectangle bounds from the drawn coordinate space to the normalized space.
        
        Parameters:
            left, right, bottom, top (float): Bounds of the search rectangle in the drawn coordinate space.
            
        Returns:
            tuple: Renormalized bounds (norm_left, norm_right, norm_bottom, norm_top) in the normalized space.
        """
        
        # Renormalize the horizontal bounds:
        norm_left = (left - self.drawn_x_min) / (self.drawn_x_max - self.drawn_x_min)
        norm_right = (right - self.drawn_x_min) / (self.drawn_x_max - self.drawn_x_min)
        
        # Renormalize the vertical bounds:
        norm_bottom = (bottom - self.drawn_y_min) / (self.drawn_y_max - self.drawn_y_min)
        norm_top = (top - self.drawn_y_min) / (self.drawn_y_max - self.drawn_y_min)
        
        return (norm_left, norm_right, norm_bottom, norm_top)

    def calculate_percentages_inside_rectangle(self):
        center_x, center_y = self.mouse_pos
        axis_count = self.data.shape[1]
        axis_positions = np.linspace(-1, 1, axis_count)
        
        # Calculate rectangle boundaries based on the two search radii
        rect_left = center_x - self.search_radius_x
        rect_right = center_x + self.search_radius_x
        rect_bottom = center_y - self.search_radius_y
        rect_top = center_y + self.search_radius_y

        # Renormalize the bounds to the normalized space
        norm_left, norm_right, norm_bottom, norm_top = self.renormalize_search_bounds(rect_left, rect_right, rect_bottom, rect_top)

        intersecting_lines = []
        
        for idx, row in self.data.iterrows():
            intersects = False
            for i in range(axis_count - 1):
                x0, y0 = axis_positions[i], row[i]
                x1, y1 = axis_positions[i + 1], row[i + 1]
                # Check if the line segment intersects the rectangle
                if cohen_sutherland_line_clip(x0, y0, x1, y1, norm_left, norm_right, norm_bottom, norm_top):
                    intersects = True
                    break

            if intersects:
                intersecting_lines.append(idx)

        # Count the data points inside the rectangle for each class
        intersecting_classes = self.classes[intersecting_lines]
        class_counts = intersecting_classes.value_counts()

        total_counts = self.classes.value_counts()
        percentages = (class_counts / total_counts).fillna(0) * 100

        self.statistics = []
        for class_name, percentage in percentages.items():
            self.statistics.append(f"{class_name}: {percentage:.2f}%")
        
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            # Code to hide or remove the search rectangle
            self.search_radius_x = 0  # This will effectively "remove" the rectangle by setting its size to zero
            self.search_radius_y = 0
            self.update()  # Refresh the OpenGL view

    def mouseMoveEvent(self, event):
        # Define margins
        x_min = -1
        x_max = 1
        y_min = 0
        y_max = 1

        # Convert pixel coordinates to the range [0, 1]
        norm_x = event.position().x() / self.width()
        norm_y = 1 - event.position().y() / self.height()

        # Map the normalized coordinates to the world space (considering the margins)
        x = x_min + norm_x * (x_max - x_min)
        y = y_min + norm_y * (y_max - y_min)
        
        self.mouse_pos = (x, y)
                
        # Trigger a repaint
        self.update()
        self.calculate_percentages_inside_rectangle()

    def paintGL(self):
        # Call the parent's paintGL method to draw the existing elements
        super().paintGL()

        # If mouse_pos is not None, draw search rectangle
        if self.mouse_pos:
            draw_rectangle(self.mouse_pos, self.search_radius_x, self.search_radius_y)
            self.draw_crosshair(self.mouse_pos)
        
        painter = QPainter(self)
    
        # Set the painter properties like font, color, etc. if needed
        painter.setPen(QColor(0, 0, 0))  # Setting the text color to black for instance
        
        # Calculate the position to start drawing the text. 
        # For this example, I'll assume you want to start drawing 10 pixels from the top right corner.
        start_x = self.width() - 100  # Assuming 100 pixels is sufficient for your text
        start_y = 20
        line_height = 15  # Adjust as needed
        
        for statistic in self.statistics:
            painter.drawText(start_x, start_y, statistic)
            start_y += line_height  # Move to the next line
        if self.mouse_pos:
            x, y = self.mouse_pos
        else:
            x, y = -1, -1
        debug_info = [
            f"({x:.2f}, {y:.2f})",
            f"{self.search_radius_x}",
            f"{self.search_radius_y}"
        ]
        for info in debug_info:
            painter.drawText(start_x, start_y, info)
            start_y += line_height  # Move to the next line

        painter.end()

    def draw_crosshair(self, center):
        """Draw a small crosshair at the center of the circle."""
        glDisable(GL_DEPTH_TEST)
        aspect_ratio = self.width() / self.height()
        size = 0.01  # Size of the crosshair lines
            
        glColor3f(1, 0, 0)  # Red color for the crosshair
        glBegin(GL_LINES)
        glVertex2f(center[0] - size, center[1])
        glVertex2f(center[0] + size, center[1])
        glVertex2f(center[0], center[1] - size)
        glVertex2f(center[0], center[1] + size)
        glEnd()
        glEnable(GL_DEPTH_TEST)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Parallel Envelope Plotter")
        self.setGeometry(0, 0, 1200, 625)
        self.center_on_screen()

        layout = QVBoxLayout()

        # Add OpenGL context
        self.opengl_widget = ModifiedOpenGLPlot(self)
        layout.addWidget(self.opengl_widget)

        # Create the buttons
        self.load_csv_btn = QPushButton("Load CSV")
        self.load_csv_btn.clicked.connect(self.load_csv_file)
        self.toggle_envelope_btn = QPushButton("Toggle Envelope")
        self.toggle_envelope_btn.clicked.connect(self.toggle_envelope)

        # Create a horizontal layout for buttons
        button_layout = QHBoxLayout()

        # Add buttons to the horizontal layout and restrict their height
        self.load_csv_btn.setMaximumHeight(25)
        self.toggle_envelope_btn.setMaximumHeight(25)
        button_layout.addWidget(self.load_csv_btn)
        button_layout.addWidget(self.toggle_envelope_btn)

        # Add the horizontal layout to the main vertical layout
        layout.addLayout(button_layout)

        # Create a QWidget to set as the main window's central widget
        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def keyPressEvent(self, event):
        # Check for Escape key or Ctrl+W
        if event.key() == Qt.Key.Key_Escape or (event.key() == Qt.Key.Key_W and event.modifiers() == Qt.KeyboardModifier.ControlModifier):
            self.close()
        elif event.key() in [Qt.Key.Key_W, Qt.Key.Key_S, Qt.Key.Key_A, Qt.Key.Key_D]:
            # Adjust the search_radius based on the key pressed
            key_map = {
                Qt.Key.Key_W: 'W',
                Qt.Key.Key_S: 'S',
                Qt.Key.Key_A: 'A',
                Qt.Key.Key_D: 'D'
            }
            pressed_key = key_map[event.key()]
            new_search_radius_x, new_search_radius_y = adjust_search_radii(pressed_key, self.opengl_widget.search_radius_x, self.opengl_widget.search_radius_y)
            self.opengl_widget.search_radius_x = new_search_radius_x
            self.opengl_widget.search_radius_y = new_search_radius_y
            self.opengl_widget.update()  # Refresh the OpenGL view

    def center_on_screen(self):
        """Center the window on the screen."""
        qt_rectangle = self.frameGeometry()
        center_point = QApplication.screens()[0].availableGeometry().center()
        qt_rectangle.moveCenter(center_point)
        self.move(qt_rectangle.topLeft())
    
    def load_csv_file(self):
        options = QFileDialog.Option.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "datasets", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            # Load your CSV data here using file_name
            print(f"Loading {file_name}")
            self.opengl_widget.load_data(file_name)

    def toggle_envelope(self):
        self.opengl_widget.show_envelope = not self.opengl_widget.show_envelope
        self.opengl_widget.update()  # Refresh the OpenGL view


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
