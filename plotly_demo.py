import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import argparse

# Create an ArgumentParser object to handle command line arguments
parser = argparse.ArgumentParser(description="Generate an interactive parallel coordinates plot from a dataset dictionary")
parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
parser.add_argument("--file_path", type=str, required=True, help="File path of the dataset")
args = parser.parse_args()

# Load the dataset from the provided dictionary
df = pd.read_csv(args.file_path)

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
