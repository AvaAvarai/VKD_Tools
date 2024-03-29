import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import webbrowser
import os
import subprocess
import sys

import pandas as pd

# Function to detect the correct Python command
def detect_python_command():
    try:
        subprocess.check_output(["python", "--version"])
        return "python"
    except subprocess.CalledProcessError:
        pass
    except FileNotFoundError:
        pass

    try:
        subprocess.check_output(["python3", "--version"])
        return "python3"
    except subprocess.CalledProcessError:
        pass
    except FileNotFoundError:
        pass

    print("Neither 'python' nor 'python3' commands are available.")
    sys.exit(1)

# Detect and set the global Python command
PYTHON_CMD = detect_python_command()

def main():
    global app
    app = tk.Tk()
    
    app.title("Visual Knowledge Discovery Tools")
    app.bind('<Escape>', close_app)
    app.bind('<Control-w>', close_app)
    
    global data_dict
    global info_label
    global tuner_var
    tuner_var = tk.StringVar()
    tuner_var.set('KNN')
    data_dict = {}
    
    window_width = 400
    window_height = 650
    screen_width = app.winfo_screenwidth()
    screen_height = app.winfo_screenheight()
    x_coordinate = int((screen_width/2) - (window_width/2))
    y_coordinate = int((screen_height/2) - (window_height/2))
    app.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    style = ttk.Style()
    style.configure('TButton', background='#3498db', foreground='black', borderwidth=0, font=('Arial', 11))
    style.map('TButton',
        background=[('active', '#2980b9'), ('disabled', '#A9A9A9')],  
        foreground=[('active', 'black'), ('disabled', '#D3D3D3')]
    )
    
    main_frame = ttk.Frame(app, relief='groove', borderwidth=4)
    main_frame.pack(pady=5, fill='x')

    program_label = ttk.Label(main_frame, text="Visual Knowledge Discovery Tools", font=("Arial", 14))
    program_label.pack(pady=10)

    data_frame = ttk.Frame(main_frame, relief='groove', borderwidth=4)
    data_frame.pack(pady=2, fill='x')

    program_label = ttk.Label(data_frame, text="Dataset Selection", font=("Arial", 11))
    program_label.pack(pady=2)

    load_csv_button = ttk.Button(data_frame, text="Load Dataset", command=load_and_process_csv)
    load_csv_button.pack(pady=2)

    program_label = ttk.Label(main_frame, text="Data Visualization Selection", font=("Arial", 11))
    program_label.pack(pady=2)

    tuner_frame = ttk.Frame(main_frame)
    tuner_frame.pack(pady=2)
    
    global tuner_button
    tuner_button = ttk.Button(tuner_frame, text="Classifier Tuner", command=launch_classifier_tuner, state=tk.DISABLED)
    tuner_button.pack(side="left", padx=5, expand=True, fill='x')    
    
    tuner_options = ['KNN', 'SVM-Linear', 'SVM-RBF','SVM-Poly', 'Naive Bayes', 'Random Forest', 'LDA', 'Decision Tree', 'Logistic Regression']
    tuner_select_dropdown = tk.OptionMenu(tuner_frame, tuner_var, *tuner_options)
    tuner_select_dropdown.config(borderwidth=0, highlightthickness=1, highlightbackground="black", highlightcolor="black")
    tuner_select_dropdown.pack(side="right", padx=5, expand=True, fill='x')
    
    # Create a frame for the parallel coordinates buttons
    parallel_coords_frame = ttk.Frame(main_frame)
    parallel_coords_frame.pack(pady=2, fill='x')

    # Add a label above the buttons
    parallel_coords_label = ttk.Label(parallel_coords_frame, text="Parallel Coordinates", font=("Arial", 11))
    parallel_coords_label.pack()

    # Create 2x2 sub-frames within the main frame
    frame_top = ttk.Frame(parallel_coords_frame)
    frame_top.pack(side="top", pady=2)

    frame_bottom = ttk.Frame(parallel_coords_frame)
    frame_bottom.pack(side="bottom", pady=2)

    # Add buttons to the sub-frames
    global envelope_button
    envelope_button = ttk.Button(frame_top, text="Envelope Search", command=parallel_envelopes, state=tk.DISABLED)
    envelope_button.pack(side="left", padx=5, expand=True, fill='x')

    global plotly_demo_button
    plotly_demo_button = ttk.Button(frame_top, text="Axes Re-Order Search", command=launch_parallel_invert, state=tk.DISABLED)
    plotly_demo_button.pack(side="right", padx=5, expand=True, fill='x')

    global parallel_hb_button
    parallel_hb_button = ttk.Button(frame_bottom, text="Pure Hyper-Block Grow", command=launch_parallel_hb, state=tk.DISABLED)
    parallel_hb_button.pack(side="right", padx=5, expand=True, fill='x')

    global parallel_gl_button
    parallel_gl_button = ttk.Button(frame_bottom, text="OpenGL: GPU Accelerated", command=launch_parallel_gl, state=tk.DISABLED)
    parallel_gl_button.pack(side="left", padx=5, expand=True, fill='x')

    global parallel_curves_button
    parallel_curves_button = ttk.Button(main_frame, text="Andrew's Curves", command=launch_parallel_curves, state=tk.DISABLED)
    parallel_curves_button.pack(padx=5)    

    misc_label = ttk.Label(main_frame, text="Miscellaneous Visualizations", font=("Arial", 11))
    misc_label.pack()

    global shifted_paired_button
    shifted_paired_button = ttk.Button(main_frame, text="Shifted Paired Coordinates", command=launch_shifted_paired, state=tk.DISABLED)
    shifted_paired_button.pack(pady=2, fill='x', padx=100)

    global tree_glyph_button
    tree_glyph_button = ttk.Button(main_frame, text="Tree Glyphs", command=launch_tree_glyph_plotter, state=tk.DISABLED)
    tree_glyph_button.pack(pady=2, fill='x', padx=100)
    
    global scc_rings_button  # Declare it as global if you need to modify its state later
    scc_rings_button = ttk.Button(main_frame, text="Static Circular Rings", command=launch_scc_rings, state=tk.DISABLED)
    scc_rings_button.pack(pady=5, fill='x', padx=100)
    
    glc_frame = ttk.Frame(main_frame)
    glc_frame.pack(side="bottom", pady=2)
    
    glc_label = ttk.Label(glc_frame, text="General Line Coordinates", font=("Arial", 11))
    glc_label.pack()
    
    global glc_button
    glc_button = ttk.Button(glc_frame, text="Linear", command=launch_glc_line_plotter, state=tk.DISABLED)
    glc_button.pack(side="left", padx=5, expand=True, fill='x')
    
    global glc_3d_rotate_button
    glc_3d_rotate_button = ttk.Button(glc_frame, text="3D Rotate", command=launch_glc_3d_rotate_plotter, state=tk.DISABLED)
    glc_3d_rotate_button.pack(side="right", padx=5, expand=True, fill='x')
    
    global circular_button
    circular_button = ttk.Button(main_frame, text="Circular Coordinates", command=launch_circular_plotter, state=tk.DISABLED)
    circular_button.pack(pady=5, fill='x', padx=100)

    # Create a frame for the GitHub buttons
    github_buttons_frame = ttk.Frame(app, relief='groove', borderwidth=4)
    github_buttons_frame.pack(side='bottom', pady=10)
    
    # Add GitHub buttons to the frame
    project_github_button = ttk.Button(github_buttons_frame, text="Project Github", command=launch_github)
    project_github_button.pack(side="left", padx=15, pady=2)
    
    lab_github_button = ttk.Button(github_buttons_frame, text="Lab Github", command=launch_lab_github)
    lab_github_button.pack(side="right", padx=15, pady=2)

    info_label = ttk.Label(app, text="", font=("Arial", 9))
    info_label.pack(pady=10)

    app.mainloop()

def close_app(event=None):
    app.destroy()

def load_and_process_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return

    df = pd.read_csv(file_path)
    df_copy = df.copy()
    labels = df.pop('class')

    dataset_name = os.path.basename(file_path).split(".")[0]
    global data_dict
    data_dict = {
        "dataset_name": dataset_name,
        "file_path": file_path,
        "original_dataframe": df_copy,
        "labels": labels
    }
    
    global plotly_demo_button, circular_button, envelope_button, glc_button, tree_glyph_button, shifted_paired_button, glc_3d_rotate_button, tuner_button, parallel_gl_button, parallel_hb_button, launch_parallel_curves, scc_rings_button
    plotly_demo_button.config(state=tk.NORMAL)
    envelope_button.config(state=tk.NORMAL)
    circular_button.config(state=tk.NORMAL)
    glc_button.config(state=tk.NORMAL)
    tree_glyph_button.config(state=tk.NORMAL)
    shifted_paired_button.config(state=tk.NORMAL)
    glc_3d_rotate_button.config(state=tk.NORMAL)
    tuner_button.config(state=tk.NORMAL)
    parallel_gl_button.config(state=tk.NORMAL)
    parallel_hb_button.config(state=tk.NORMAL)
    parallel_curves_button.config(state=tk.NORMAL)
    scc_rings_button.config(state=tk.NORMAL)
    
    display_dataset_info()

def display_dataset_info():
    global data_dict, info_label
    
    dataset_name = data_dict.get("dataset_name", "N/A")
    file_path = data_dict.get("file_path", "N/A")
    label_names = data_dict.get("labels", [])
    df = data_dict.get("original_dataframe", None)
    
    unique_label_names = ", ".join(sorted(map(str, set(label_names))))
    num_unique_labels = len(set(label_names))
    sample_count = len(label_names)
    num_attributes = df.shape[1] - 1 if df is not None else "N/A"

    info_text = f"CSV Name: {dataset_name}\nPath: {file_path}\nClass Count: {num_unique_labels}\nClass Names: {unique_label_names}\nSample Count: {sample_count}\nNumber of Attributes: {num_attributes}"
    
    info_label.config(text=info_text)

def launch_scc_rings():
    global data_dict
    app.withdraw()
    subprocess.run([PYTHON_CMD, "scc_rings.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def launch_classifier_tuner():
    global data_dict
    app.withdraw()
    subprocess.run([PYTHON_CMD, "classifier_tuner.py", "--file_path", data_dict["file_path"], "--classifier_name", tuner_var.get()])
    app.deiconify()

def launch_circular_plotter():
    global data_dict
    app.withdraw()
    subprocess.run([PYTHON_CMD, "circular_plotter.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def parallel_envelopes():
    global data_dict
    app.withdraw()
    subprocess.run([PYTHON_CMD, "parallel_envelopes.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def launch_parallel_hb():
    global data_dict
    app.withdraw()
    subprocess.run([PYTHON_CMD, "parallel_hb.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def launch_parallel_invert():
    global data_dict
    app.withdraw()
    subprocess.run([PYTHON_CMD, "parallel_invert.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def launch_parallel_gl():
    global data_dict
    app.withdraw()
    subprocess.run([PYTHON_CMD, "parallel_gl.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def launch_parallel_curves():
    global data_dict
    app.withdraw()
    subprocess.run([PYTHON_CMD, "parallel_curves.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def launch_glc_line_plotter():
    global data_dict
    app.withdraw()
    subprocess.run([PYTHON_CMD, "glc_line_plotter.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def launch_tree_glyph_plotter():
    global data_dict
    app.withdraw()
    subprocess.run([PYTHON_CMD, "tree_glyph_plotter.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def launch_shifted_paired():
    global data_dict
    app.withdraw()
    subprocess.run([PYTHON_CMD, "shifted_paired.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def launch_glc_3d_rotate_plotter():
    global data_dict
    app.withdraw()
    subprocess.run([PYTHON_CMD, "glc_3d_rotate.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def launch_github():
    webbrowser.open("https://github.com/AvaAvarai/VKD_Tools")

def launch_lab_github():
    webbrowser.open("https://github.com/CWU-VKD-LAB")

if __name__ == "__main__":
    main()
