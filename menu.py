import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import webbrowser
import os
import subprocess

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def main():
    global app
    app = tk.Tk()
    
    app.title("Visual Knowledge Discovery Tools")
    app.bind('<Escape>', close_app)
    app.bind('<Control-w>', close_app)
    
    global data_dict
    global info_label
    data_dict = {}
    
    window_width = 450
    window_height = 500
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
    program_label.pack(pady=20)

    load_csv_button = ttk.Button(app, text="Load Dataset", command=load_and_process_csv)
    load_csv_button.pack(pady=5, fill='x', padx=50)

    global circular_button
    circular_button = ttk.Button(app, text="Run Circular Plotter", command=launch_circular_plotter, state=tk.DISABLED)
    circular_button.pack(pady=5, fill='x', padx=50)

    global envelope_button
    envelope_button = ttk.Button(app, text="Run Envelope Plotter", command=launch_envelope_plotter, state=tk.DISABLED)
    envelope_button.pack(pady=5, fill='x', padx=50)

    global plotly_demo_button
    plotly_demo_button = ttk.Button(app, text="Run Plotly Demo", command=launch_plotly_demo, state=tk.DISABLED)
    plotly_demo_button.pack(pady=5, fill='x', padx=50)

    global glyph_2d_button
    glyph_2d_button = ttk.Button(app, text="Run GLC line Plotter", command=launch_glc_line_plotter, state=tk.DISABLED)
    glyph_2d_button.pack(pady=5, fill='x', padx=50)

    global tree_glyph_button
    tree_glyph_button = ttk.Button(app, text="Run Tree-Glyph Plotter", command=launch_tree_glyph_plotter, state=tk.DISABLED)
    tree_glyph_button.pack(pady=5, fill='x', padx=50)
    
    global collocated_button
    collocated_button = ttk.Button(app, text="Run Collocated Plotter", command=launch_collocated_plotter, state=tk.DISABLED)
    collocated_button.pack(pady=5, fill='x', padx=50)

    github_button = ttk.Button(app, text="Visit GitHub Page", command=launch_github)
    github_button.pack(pady=5, fill='x', padx=50)

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
    
    global plotly_demo_button, circular_button, envelope_button, glyph_2d_button, tree_glyph_button, collocated_button
    plotly_demo_button.config(state=tk.NORMAL)
    envelope_button.config(state=tk.NORMAL)
    circular_button.config(state=tk.NORMAL)
    glyph_2d_button.config(state=tk.NORMAL)
    tree_glyph_button.config(state=tk.NORMAL)
    collocated_button.config(state=tk.NORMAL)
    
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

def launch_github():
    webbrowser.open("https://github.com/CWU-VKD-LAB")

if __name__ == "__main__":
    main()
