import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pandas as pd
import numpy as np
import sys
import argparse

# Global zoom factor and mouse position
zoom_factor = 1.0
mouse_x, mouse_y = 0.0, 0.0

# Scroll callback for GLFW
def scroll_callback(window, xoffset, yoffset):
    global zoom_factor
    zoom_speed = 0.1
    zoom_factor += yoffset * zoom_speed
    zoom_factor = max(0.1, min(5.0, zoom_factor))

# Cursor position callback for GLFW
def cursor_position_callback(window, xpos, ypos):
    global mouse_x, mouse_y
    mouse_x, mouse_y = xpos, ypos

def read_csv(file_path):
    df = pd.read_csv(file_path)
    df['class'] = df['class'].astype(str)  # Convert 'class' column to string
    labels = df['class'].unique()

    features = df.drop('class', axis=1).columns
    
    # Normalize the data by attribute
    for feature in features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        df[feature] = (df[feature] - min_val) / (max_val - min_val)
        
    return df, labels, features

# Initialize GLFW and OpenGL
def initialize_window():
    global width, height
    
    width = 800
    height = 600
    if not glfw.init():
        return None

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(width, height, "Parallel Coordinates", None, None)
    if not window:
        glfw.terminate()
        return None

    # Make the window's context current
    glfw.make_context_current(window)
    
    # Set resize callback
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    
    return window

# Framebuffer size callback for GLFW
def framebuffer_size_callback(window, new_width, new_height):
    global width, height
    glViewport(0, 0, new_width, new_height)
    width, height = new_width, new_height

# Compile shaders
def compile_shaders():
    vertex_shader = """
    #version 330
    in vec2 position;
    uniform mat4 transform;
    void main()
    {
        gl_Position = transform * vec4(position, 0.0, 1.0);
    }
    """

    fragment_shader = """
    #version 330
    out vec4 fragColor;
    uniform vec3 color;
    void main()
    {
        fragColor = vec4(color, 1.0);
    }
    """

    shader = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )

    return shader

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

def draw(df, labels, features, shader):
    # Use the shader
    glUseProgram(shader)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Generate unique colors for each label
    distinct_colors = generate_distinct_colors(len(labels))
    label_to_color = {}
    for i, label in enumerate(labels):
        label_to_color[label] = distinct_colors[i]

    # Identity transformation matrix (4x4)
    identity_matrix = np.identity(4, dtype=np.float32)
    
    # Margin and scaling factors for graph
    margin = 0.1
    scale = 1 - 2 * margin
    
    # Calculate pan based on mouse position and zoom factor
    pan_x = (mouse_x / width - 0.5) * (1.0 - zoom_factor)
    pan_y = (mouse_y / height - 0.5) * (1.0 - zoom_factor)

    # Zoom and pan transformation matrix (4x4)
    transform_matrix = np.array([
        [zoom_factor, 0, 0, pan_x],
        [0, zoom_factor, 0, pan_y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    glUniformMatrix4fv(glGetUniformLocation(shader, "transform"), 1, GL_FALSE, transform_matrix)
    
    # Loop through each data entry and draw lines
    for index, row in df.iterrows():
        # Get the label and corresponding color
        label = row['class']
        color = label_to_color[label]
        
        # Set the color uniform
        glUniform3f(glGetUniformLocation(shader, "color"), *color)
        
        # Set the transformation uniform
        glUniformMatrix4fv(glGetUniformLocation(shader, "transform"), 1, GL_FALSE, transform_matrix)
        
        # Prepare the vertex data
        vertex_data = []
        for i, feature in enumerate(features):
            x = i / (len(features) - 1) * scale + margin   # Map i to [margin, 1-margin]
            y = row[feature] * scale + margin              # Map feature to [margin, 1-margin]
            vertex_data.extend([x * 2 - 1, y * 2 - 1])     # Map to [-1, 1]
        
        # Send the vertex data to the GPU and draw lines
        vertex_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, np.array(vertex_data, dtype=np.float32), GL_STATIC_DRAW)
        
        position = glGetAttribLocation(shader, "position")
        glEnableVertexAttribArray(position)
        glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        glDrawArrays(GL_LINE_STRIP, 0, len(features))

        
        glDeleteBuffers(1, [vertex_buffer])

    # Draw white axes
    glUniform3f(glGetUniformLocation(shader, "color"), 1.0, 1.0, 1.0)  # Set color to white
    for i, feature in enumerate(features):
        x = i / (len(features) - 1) * scale + margin  # Map i to [margin, 1-margin]
        vertex_data = [
            x * 2 - 1, margin * 2 - 1,  # Bottom point
            x * 2 - 1, (1 - margin) * 2 - 1  # Top point
        ]
        
        # Send the vertex data to the GPU and draw lines
        vertex_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, np.array(vertex_data, dtype=np.float32), GL_STATIC_DRAW)
        
        position = glGetAttribLocation(shader, "position")
        glEnableVertexAttribArray(position)
        glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        glDrawArrays(GL_LINES, 0, 2)  # Draw a line segment for the axis
        glDeleteBuffers(1, [vertex_buffer])

    # Stop using the shader
    glUseProgram(0)

def main(file_path):
    global shader, features, labels, df
    
    df, labels, features = read_csv(file_path)
    window = initialize_window()

    if window is None:
        return

    shader = compile_shaders()

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)  # Clear the frame buffer
        draw(df, labels, features, shader) # Draw here

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render parallel coordinates from a CSV file.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the CSV file')
    args = parser.parse_args()

    main(args.file_path)
