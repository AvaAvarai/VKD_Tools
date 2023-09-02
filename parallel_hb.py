import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import seaborn as sns

# Initialize HyperBlocks (HBs) with single data points
def initialize_hbs(df):
    hbs = []
    feature_cols = df.columns[df.columns != 'class']  # Get feature column names
    for index, row in df.iterrows():
        min_series = row.drop('class').reindex(feature_cols)
        max_series = row.drop('class').reindex(feature_cols)
        hb = {'min': min_series, 'max': max_series, 'class': row['class'], 'points': {index}}
        hbs.append(hb)
    return hbs

# Check if two HBs can be combined into a single pure HB
def can_combine(hb1, hb2):
    return hb1['class'] == hb2['class']

# Combine two HBs
def combine_hbs(hb1, hb2):
    feature_cols = hb1['min'].index  # Assuming hb1 and hb2 have the same feature columns
    new_min = hb1['min'].combine(hb2['min'], np.minimum).reindex(feature_cols)
    new_max = hb1['max'].combine(hb2['max'], np.maximum).reindex(feature_cols)
    new_hb = {
        'min': new_min,
        'max': new_max,
        'class': hb1['class'],
        'points': hb1['points'].union(hb2['points'])
    }
    return new_hb

# Optimized version of create_pure_hbs function
def create_pure_hbs_optimized(df):
    # Initialize HyperBlocks (HBs) with single data points
    hbs = [{'min': row.drop('class'), 'max': row.drop('class'), 'class': row['class'], 'points': {index}}
           for index, row in df.iterrows()]
    
    checked_pairs = set()  # To keep track of checked pairs of HBs
    changed = True
    
    while changed:
        changed = False
        new_hbs = []
        to_remove = set()  # Indices of HBs that should be removed because they were combined
        
        for i, hb1 in enumerate(hbs):
            if i in to_remove:
                continue
            
            combined = False
            for j, hb2 in enumerate(hbs[i + 1:]):
                j = i + 1 + j  # Adjust index relative to original list
                
                if j in to_remove:
                    continue
                
                # Create a unique identifier for each pair to avoid redundant work
                pair_id = frozenset([id(hb1), id(hb2)])
                
                if pair_id in checked_pairs:
                    continue
                
                checked_pairs.add(pair_id)
                
                # Check if the HBs can be combined
                if hb1['class'] == hb2['class']:
                    joint_hb = {
                        'min': hb1['min'].combine(hb2['min'], np.minimum),
                        'max': hb1['max'].combine(hb2['max'], np.maximum),
                        'class': hb1['class'],
                        'points': hb1['points'].union(hb2['points'])
                    }
                    
                    # Check if joint_hb is still pure (all points have the same class)
                    unique_classes = df.loc[list(joint_hb['points']), 'class'].unique()
                    
                    if len(unique_classes) == 1 and unique_classes[0] == hb1['class']:
                        new_hbs.append(joint_hb)
                        to_remove.add(i)
                        to_remove.add(j)
                        changed = True
                        combined = True
                        break
            
            if not combined:
                new_hbs.append(hb1)
        
        if changed:
            hbs = [hb for i, hb in enumerate(hbs) if i not in to_remove] + new_hbs
            
    return hbs

def main(file_path):
    # Load and normalize the dataset
    df = pd.read_csv(file_path)
    df_normalized = (df.drop('class', axis=1) - df.drop('class', axis=1).min()) / (df.drop('class', axis=1).max() - df.drop('class', axis=1).min())
    df_normalized['class'] = df['class']

    # Create pure HyperBlocks
    pure_hbs = create_pure_hbs_optimized(df_normalized)

    # Visualize
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    palette = sns.color_palette("husl", len(df_normalized['class'].unique()))

    class_color_map = {}

    for hb in pure_hbs:
        feature_columns = df_normalized.columns[df_normalized.columns != 'class']
        for i in range(len(feature_columns) - 1):
            feature1 = feature_columns[i]
            feature2 = feature_columns[i + 1]
            color = palette[df_normalized['class'].unique().tolist().index(hb['class'])]
            class_color_map[hb['class']] = color
            poly_points = [[i, hb['min'][feature1]], [i + 1, hb['min'][feature2]], 
                           [i + 1, hb['max'][feature2]], [i, hb['max'][feature1]]]
            polygon = plt.Polygon(poly_points, closed=True, facecolor=color, edgecolor=color, alpha=0.3)
            ax.add_patch(polygon)

    # Create a custom legend
    custom_legend = [plt.Line2D([0], [0], color=color, lw=4) for color in class_color_map.values()]
    plt.legend(custom_legend, class_color_map.keys(), title="Classes")

    data_name = file_path.split('/')[-1].split('.')[0]
    plt.xticks(range(len(df_normalized.columns) - 1), df_normalized.columns[:-1])
    plt.title(f'Pure HyperBlocks of {data_name} on Parallel Coordinates')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render pure HyperBlocks on parallel coordinates from a CSV file.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the CSV file')
    args = parser.parse_args()

    main(args.file_path)
