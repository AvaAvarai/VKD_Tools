import argparse
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pandas as pd
import numpy as np
import sys

width = 1200
height = 800

def generate_distinct_colors(n):
    if n == 2:
        return [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]  # Red and Blue
    elif n == 3:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # Red, Green, Blue
    else:
        hues = np.linspace(0, 1, n + 1)[:-1]  # Generate n evenly spaced hues between 0 and 1
        colors = []
        for hue in hues:
            color = [np.sin(hue * 6.2832), np.cos(hue * 6.2832), np.sin((hue + 0.33) * 6.2832)]
            colors.append([abs(x) for x in color])
        return colors

def window_resize_callback(window, new_width, new_height):
    global width, height  # Add this line
    width, height = new_width, new_height
    glViewport(0, 0, width, height)

# Argument parsing
parser = argparse.ArgumentParser(description='Plot parallel coordinates from a CSV file.')
parser.add_argument('--file_path', type=str, required=True, help='Path to the CSV file.')
args = parser.parse_args()
file_path = args.file_path

zoom_factor = 1.0
center_x, center_y = 0.0, 0.0

# Add a variable to keep track of the hovered polyline
hovered_polyline = None

# Initialize GLFW
if not glfw.init():
    sys.exit()

# Get the screen size
monitor = glfw.get_primary_monitor()
video_mode = glfw.get_video_mode(monitor)
screen_width, screen_height = video_mode.size

# Calculate the position to center the window
window_width, window_height = 1200, 800
pos_x = (screen_width - window_width) // 2
pos_y = (screen_height - window_height) // 2

# Create a GLFW window
window = glfw.create_window(window_width, window_height, "Parallel Coordinates", None, None)

# Check window creation
if not window:
    glfw.terminate()
    sys.exit()

# Center the window
glfw.set_window_pos(window, pos_x, pos_y)

INSIDE = 0  # 0000
LEFT = 1    # 0001
RIGHT = 2   # 0010
BOTTOM = 4  # 0100
TOP = 8     # 1000

def compute_outcode(x, y, xmin, ymin, xmax, ymax):
    outcode = INSIDE
    if x < xmin:
        outcode |= LEFT
    elif x > xmax:
        outcode |= RIGHT
    if y < ymin:
        outcode |= BOTTOM
    elif y > ymax:
        outcode |= TOP
    return outcode

def hit_test(x, y, vertex_data, num_features, num_rows, zoom_factor, zoom_center_x, zoom_center_y):
    global hovered_polyline
    min_distance = 0.01  # Minimum distance for highlighting
    hovered_polyline = None
    
    # Account for zoom in the hit test
    x = (x - zoom_center_x) / zoom_factor + zoom_center_x
    y = (y - zoom_center_y) / zoom_factor + zoom_center_y

    # Define the "rectangle" around the mouse pointer
    xmin, ymin = x - min_distance, y - min_distance
    xmax, ymax = x + min_distance, y + min_distance

    # loop over the polylines with a stride of vertices_per_polyline
    for i in range(0, num_rows * 2 * (num_features - 1), 2 * (num_features - 1)):
        for j in range(num_features - 1):
            index = i + 2 * j

            if index + 2 >= len(vertex_data):
                continue

            x1, y1 = vertex_data[index]
            x2, y2 = vertex_data[index + 2]

            outcode1 = compute_outcode(x1, y1, xmin, ymin, xmax, ymax)
            outcode2 = compute_outcode(x2, y2, xmin, ymin, xmax, ymax)

            accept = False

            while True:
                if not (outcode1 | outcode2):
                    accept = True
                    break
                elif outcode1 & outcode2:
                    break
                else:
                    outcode_out = outcode1 if outcode1 else outcode2

                    if outcode_out & TOP:
                        x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
                        y = ymax
                    elif outcode_out & BOTTOM:
                        x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
                        y = ymin
                    elif outcode_out & RIGHT:
                        y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
                        x = xmax
                    elif outcode_out & LEFT:
                        y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
                        x = xmin

                    if outcode_out == outcode1:
                        x1, y1 = x, y
                        outcode1 = compute_outcode(x1, y1, xmin, ymin, xmax, ymax)
                    else:
                        x2, y2 = x, y
                        outcode2 = compute_outcode(x2, y2, xmin, ymin, xmax, ymax)

            if accept:
                hovered_polyline = i // (2 * (num_features - 1))
                return

def cursor_position_callback(window, xpos, ypos):
    global center_x, center_y
    center_x = (xpos / width) * 2.0 - 1.0  # Convert to NDC
    center_y = 1.0 - (ypos / height) * 2.0  # Convert to NDC    
    
    # Added zoom_factor and zoom_center_x, zoom_center_y as arguments
    hit_test(center_x, center_y, vertex_data, num_features, len(df_normalized), zoom_factor, 0.0, 0.0)

glfw.set_cursor_pos_callback(window, cursor_position_callback)

def scroll_callback(window, x_offset, y_offset):
    global zoom_factor
    zoom_factor += y_offset * 0.1
    zoom_factor = max(0.1, min(15.0, zoom_factor))  # Limit zoom factor

glfw.set_scroll_callback(window, scroll_callback)

# Key callback function
def key_callback(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
    elif key == glfw.KEY_W and mods == glfw.MOD_CONTROL and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

# Set key callback
glfw.set_key_callback(window, key_callback)

glfw.set_window_size_callback(window, window_resize_callback)

# Make the window's context current
glfw.make_context_current(window)

# Enable anti-aliasing
glfw.window_hint(glfw.SAMPLES, 4)
glEnable(GL_MULTISAMPLE)

# use smoothest lines
glEnable(GL_LINE_SMOOTH)
glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

# Shader setup
vertex_shader = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;

uniform vec3 uniformColor;
uniform bool useUniformColor;
uniform float zoomFactor;
uniform vec2 zoomCenter;

out vec3 ourColor;

void main()
{
    vec2 zoomedPos = (aPos - zoomCenter) * zoomFactor + zoomCenter;
    gl_Position = vec4(zoomedPos.x, zoomedPos.y, 0.0, 1.0);
    ourColor = useUniformColor ? uniformColor : aColor;
}

"""

fragment_shader = """
#version 330 core
out vec4 FragColor;
in vec3 ourColor;

void main()
{
    FragColor = vec4(ourColor, 1.0);
}
"""

shader = compileProgram(
    compileShader(vertex_shader, GL_VERTEX_SHADER),
    compileShader(fragment_shader, GL_FRAGMENT_SHADER)
)

# Read and preprocess the CSV file
def read_and_preprocess_csv(file_path):
    df = pd.read_csv(file_path)
    df_normalized = (df.drop(columns=['class']) - df.drop(columns=['class']).min()) / (df.drop(columns=['class']).max() - df.drop(columns=['class']).min())
    df_normalized['class'] = df['class']
    unique_classes = df_normalized['class'].unique()
    n = len(unique_classes)
    color_list = generate_distinct_colors(n)
    colors = {cls: color for cls, color in zip(unique_classes, color_list)}
    return df_normalized, colors

df_normalized, colors = read_and_preprocess_csv(file_path)
margin = 0.1  # Define the margin here

# After reading the CSV and normalizing it
df_normalized, colors = read_and_preprocess_csv(file_path)
num_features = len(df_normalized.columns) - 1  # Excluding 'class' column

# Function to convert DataFrame to vertex and color data
def df_to_vertex_data(df, colors):
    vertices = []
    vertex_colors = []
    num_features = len(df.columns) - 1  # excluding 'class' column
    for _, row in df.iterrows():
        cls = row['class']
        color = colors[cls]
        for i in range(num_features - 1):
            x1 = i / (num_features - 1) * 2 - 1
            y1 = row[i] * 2 - 1
            x2 = (i + 1) / (num_features - 1) * 2 - 1
            y2 = row[i + 1] * 2 - 1
            # Apply the margin
            x1, x2 = x1 * (1 - margin), x2 * (1 - margin)
            y1, y2 = y1 * (1 - margin), y2 * (1 - margin)
            vertices.extend([(x1, y1), (x2, y2)])
            vertex_colors.extend([color, color])
    return np.array(vertices, dtype=np.float32), np.array(vertex_colors, dtype=np.float32)

vertex_data, color_data = df_to_vertex_data(df_normalized, colors)

# Create vertex buffer object (VBO)
VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

# Create color buffer object (CBO)
CBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, CBO)
glBufferData(GL_ARRAY_BUFFER, color_data.nbytes, color_data, GL_STATIC_DRAW)

# Create vertex array object (VAO)
VAO = glGenVertexArrays(1)
glBindVertexArray(VAO)

# Vertex positions
glEnableVertexAttribArray(0)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

# Vertex colors
glEnableVertexAttribArray(1)
glBindBuffer(GL_ARRAY_BUFFER, CBO)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

# Function to create axis vertex data
def create_axis_data(num_features):
    axis_vertices = []
    for i in range(num_features):
        x = i / (num_features - 1) * 2 - 1
        # Apply the margin
        x = x * (1 - margin)
        axis_vertices.extend([(x, -1 + margin), (x, 1 - margin)])
    return np.array(axis_vertices, dtype=np.float32)

# Generate axis vertex data
axis_vertex_data = create_axis_data(len(df_normalized.columns) - 1)

# Create axis vertex buffer object (Axis VBO)
axis_VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, axis_VBO)
glBufferData(GL_ARRAY_BUFFER, axis_vertex_data.nbytes, axis_vertex_data, GL_STATIC_DRAW)

# Create axis vertex array object (Axis VAO)
axis_VAO = glGenVertexArrays(1)
glBindVertexArray(axis_VAO)

# Axis vertex positions
glEnableVertexAttribArray(0)
glBindBuffer(GL_ARRAY_BUFFER, axis_VBO)
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

# Main event loop
while not glfw.window_should_close(window):
    # Clear the screen
    glClear(GL_COLOR_BUFFER_BIT)
    
    # Use shader
    glUseProgram(shader)
    
    glUniform1f(glGetUniformLocation(shader, "zoomFactor"), zoom_factor)
    glUniform2f(glGetUniformLocation(shader, "zoomCenter"), center_x, center_y)
    
    # Draw parallel coordinates
    glBindVertexArray(VAO)

    glLineWidth(1.0)  # Reset to normal line width
    glUniform1i(glGetUniformLocation(shader, "useUniformColor"), 0)
    glDrawArrays(GL_LINES, 0, len(vertex_data))

    if hovered_polyline is not None:
        # Added zoom_factor and zoom_center_x, zoom_center_y as arguments
        hit_test(center_x, center_y, vertex_data, num_features, len(df_normalized), zoom_factor, 0.0, 0.0)

        glLineWidth(3.0)  # Increase line width
        glUniform1i(glGetUniformLocation(shader, "useUniformColor"), 1)
        glUniform3f(glGetUniformLocation(shader, "uniformColor"), 1.0, 1.0, 0.0)
        # draw the hovered polyline
        glDrawArrays(GL_LINES, hovered_polyline * (num_features - 1) * 2, (num_features - 1) * 2)
        
    glLineWidth(1.0)  # Reset to normal line width
    # Draw axes in white
    glBindVertexArray(axis_VAO)
    glUniform1i(glGetUniformLocation(shader, "useUniformColor"), 1)  # Use uniform color for axes
    glUniform3f(glGetUniformLocation(shader, "uniformColor"), 1.0, 1.0, 1.0)  # Set color to white in the shader
    glDrawArrays(GL_LINES, 0, len(axis_vertex_data))
    
    # Swap front and back buffers
    glfw.swap_buffers(window)

    # Poll for and process events
    glfw.poll_events()

# Cleanup
glDeleteBuffers(1, [VBO])
glDeleteBuffers(1, [CBO])
glDeleteVertexArrays(1, [VAO])
glfw.terminate()
