import tkinter as tk
from tkinter import ttk
import webbrowser
import os
import subprocess

def launch_circular_plotter():
    app.withdraw()  # Hide the main menu
    subprocess.run(["python", "circular_plotter.py"])
    app.deiconify()  # Show the main menu again

def launch_envelope_plotter():
    app.withdraw()  # Hide the main menu
    subprocess.run(["python", "envelope_plotter.py"])
    app.deiconify()  # Show the main menu again

def launch_plotly_demo():
    app.withdraw()  # Hide the main menu
    subprocess.run(["python", "plotly_demo.py"])
    app.deiconify()  # Show the main menu again

def launch_github():
    webbrowser.open("https://github.com/CWU-VKD-LAB")

def get_dataset_count_and_location():
    datasets_directory = "datasets"
    if os.path.exists(datasets_directory):
        dataset_count = len(os.listdir(datasets_directory))
        dataset_path = os.path.abspath(datasets_directory)
        return dataset_count, dataset_path
    return 0, "Not Found"

app = tk.Tk()
app.title("Visual Knowledge Discovery Tools")

# Center the window on the screen
window_width = 400
window_height = 300
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
x_coordinate = int((screen_width/2) - (window_width/2))
y_coordinate = int((screen_height/2) - (window_height/2))
app.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

# Styling
style = ttk.Style()
style.configure('TButton', background='#3498db', foreground='black', borderwidth=0, font=('Arial', 12))
style.map('TButton', background=[('active', '#2980b9')], foreground=[('active', 'black')])
style.configure('TLabel', font=('Arial', 14, 'bold'), foreground='black')

# Display program collection's name
program_label = ttk.Label(app, text="Visual Knowledge Discovery Tools")
program_label.pack(pady=20)

# Buttons
circular_button = ttk.Button(app, text="Run Circular Plotter", command=launch_circular_plotter)
circular_button.pack(pady=5, fill='x', padx=50)

envelope_button = ttk.Button(app, text="Run Envelope Plotter", command=launch_envelope_plotter)
envelope_button.pack(pady=5, fill='x', padx=50)

plotly_demo_button = ttk.Button(app, text="Run Plotly Demo", command=launch_plotly_demo)
plotly_demo_button.pack(pady=5, fill='x', padx=50)

github_button = ttk.Button(app, text="Visit GitHub Page", command=launch_github)
github_button.pack(pady=5, fill='x', padx=50)

# Display dataset count and location
dataset_count, dataset_location = get_dataset_count_and_location()
datasets_info_label = ttk.Label(app, text=f"Datasets loaded: {dataset_count}\nDataset Location: {dataset_location}", foreground='black', font=('Arial', 10))
datasets_info_label.pack(pady=10)

app.mainloop()
