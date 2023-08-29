import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import webbrowser
import os
import subprocess

import pandas as pd

def main():
    global app
    app = tk.Tk()
    
    app.title("Visual Knowledge Discovery Tools")
    app.bind('<Escape>', close_app)
    app.bind('<Control-w>', close_app)
    
    global data_dict
    global info_label
    data_dict = {}
    
    window_width = 425
    window_height = 550
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
    style.configure('TLabel', font=('Arial', 14, 'bold'), foreground='black')

    program_label = ttk.Label(app, text="Visual Knowledge Discovery Tools")
    program_label.pack(pady=10)

    program_label = ttk.Label(app, text="Dataset Selection", font=("Arial", 11))
    program_label.pack(pady=1)

    load_csv_button = ttk.Button(app, text="Load Dataset", command=load_and_process_csv)
    load_csv_button.pack(pady=5, fill='x', padx=100)

    program_label = ttk.Label(app, text="Data Visualization Selection", font=("Arial", 11))
    program_label.pack(pady=1)

    # Create a frame for the parallel coordinates buttons
    parallel_coords_frame = ttk.Frame(app)
    parallel_coords_frame.pack(pady=5, padx=50, fill='x')

    # Add buttons to the parallel_coords_frame
    global envelope_button
    envelope_button = ttk.Button(parallel_coords_frame, text="Parallel: Envelopes", command=launch_envelope_plotter, state=tk.DISABLED)
    envelope_button.pack(side="left", padx=5, expand=True, fill='x')

    global plotly_demo_button
    plotly_demo_button = ttk.Button(parallel_coords_frame, text="Parallel: Inversions", command=launch_plotly_demo, state=tk.DISABLED)
    plotly_demo_button.pack(side="right", padx=5, expand=True, fill='x')

    global collocated_button
    collocated_button = ttk.Button(app, text="Collocated Paired Coordinates", command=launch_collocated_plotter, state=tk.DISABLED)
    collocated_button.pack(pady=5, fill='x', padx=100)    

    global tree_glyph_button
    tree_glyph_button = ttk.Button(app, text="Tree Glyphs", command=launch_tree_glyph_plotter, state=tk.DISABLED)
    tree_glyph_button.pack(pady=5, fill='x', padx=100)
    
    global glc_button
    glc_button = ttk.Button(app, text="General Line Coordinates Linear", command=launch_glc_line_plotter, state=tk.DISABLED)
    glc_button.pack(pady=5, fill='x', padx=100)
    
    global glc_3d_rotate_button
    glc_3d_rotate_button = ttk.Button(app, text="3D GLC-L Rotate", command=launch_glc_3d_rotate_plotter, state=tk.DISABLED)
    glc_3d_rotate_button.pack(pady=5, fill='x', padx=100)
    
    global circular_button
    circular_button = ttk.Button(app, text="Circular Coordinates", command=launch_circular_plotter, state=tk.DISABLED)
    circular_button.pack(pady=5, fill='x', padx=100)

    # Create a frame for the GitHub buttons
    github_buttons_frame = ttk.Frame(app)
    github_buttons_frame.pack(side='bottom', pady=10)
    
    # Add GitHub buttons to the frame
    project_github_button = ttk.Button(github_buttons_frame, text="Project Github", command=launch_github)
    project_github_button.pack(side="left", padx=15, pady=5)
    
    lab_github_button = ttk.Button(github_buttons_frame, text="Lab Github", command=launch_lab_github)
    lab_github_button.pack(side="right", padx=15, pady=5)

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
    
    global plotly_demo_button, circular_button, envelope_button, glc_button, tree_glyph_button, collocated_button, glc_3d_rotate_button
    plotly_demo_button.config(state=tk.NORMAL)
    envelope_button.config(state=tk.NORMAL)
    circular_button.config(state=tk.NORMAL)
    glc_button.config(state=tk.NORMAL)
    tree_glyph_button.config(state=tk.NORMAL)
    collocated_button.config(state=tk.NORMAL)
    glc_3d_rotate_button.config(state=tk.NORMAL)
    
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

def launch_circular_plotter():
    global data_dict
    app.withdraw()
    subprocess.run(["python", "circular_plotter.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def launch_envelope_plotter():
    global data_dict
    app.withdraw()
    subprocess.run(["python", "envelope_plotter.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def launch_plotly_demo():
    global data_dict
    app.withdraw()
    subprocess.run(["python", "plotly_demo.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def launch_glc_line_plotter():
    global data_dict
    app.withdraw()
    subprocess.run(["python", "glc_line_plotter.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def launch_tree_glyph_plotter():
    global data_dict
    app.withdraw()
    subprocess.run(["python", "tree_glyph_plotter.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def launch_collocated_plotter():
    global data_dict
    app.withdraw()
    subprocess.run(["python", "collocated_plotter.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def launch_glc_3d_rotate_plotter():
    global data_dict
    app.withdraw()
    subprocess.run(["python", "glc_3d_rotate.py", "--file_path", data_dict["file_path"]])
    app.deiconify()

def launch_github():
    webbrowser.open("https://github.com/AvaAvarai/VKD_Tools")

def launch_lab_github():
    webbrowser.open("https://github.com/CWU-VKD-LAB")

if __name__ == "__main__":
    main()
