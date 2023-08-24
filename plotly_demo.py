import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from tkinter import filedialog
from tkinter import Tk

# Create a Tkinter root window (it won't be shown)
root = Tk()
root.withdraw()

# Open a file picker dialog and get the selected file path
file_path = filedialog.askopenfilename(title="Select a dataset", filetypes=[("CSV files", "*.csv")])

if file_path:
    # Load the dataset
    df = pd.read_csv(file_path)

    # Convert the 'class' column to numeric values
    label_encoder = LabelEncoder()
    df['class'] = label_encoder.fit_transform(df['class'])

    # Dynamically get dimensions (excluding 'class')
    dimensions = [col for col in df.columns if col != 'class']

    # Create an interactive parallel coordinates plot
    fig = px.parallel_coordinates(df, color='class',
                                  labels={col: col for col in df.columns},
                                  color_continuous_scale="Viridis",
                                  dimensions=dimensions)

    fig.show()
else:
    print("No file selected.")
