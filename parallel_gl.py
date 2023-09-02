import argparse
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pandas as pd
import numpy as np
import sys

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

def window_resize_callback(window, width, height):
    glViewport(0, 0, width, height)

# Argument parsing
parser = argparse.ArgumentParser(description='Plot parallel coordinates from a CSV file.')
parser.add_argument('--file_path', type=str, required=True, help='Path to the CSV file.')
args = parser.parse_args()
file_path = args.file_path

zoom_factor = 1.0
center_x, center_y = 0.0, 0.0

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

def cursor_position_callback(window, xpos, ypos):
    global center_x, center_y
    center_x = (xpos / 800.0) * 2.0 - 1.0  # Convert to NDC
    center_y = 1.0 - (ypos / 600.0) * 2.0  # Convert to NDC

glfw.set_cursor_pos_callback(window, cursor_position_callback)

def scroll_callback(window, x_offset, y_offset):
    global zoom_factor
    zoom_factor += y_offset * 0.1
    zoom_factor = max(0.1, min(10.0, zoom_factor))  # Limit zoom factor between 0.1 and 3.0

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
    glUniform1i(glGetUniformLocation(shader, "useUniformColor"), 0)  # Use vertex colors for data lines
    glDrawArrays(GL_LINES, 0, len(vertex_data))
    
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
