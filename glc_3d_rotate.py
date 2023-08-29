import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import argparse
from concurrent.futures import ThreadPoolExecutor
import cProfile
from collections import Counter

# Function for profiling
def profile_func(func):
    def wrapper(*args, **kwargs):
        profile = cProfile.Profile()
        profile.enable()
        result = func(*args, **kwargs)
        profile.disable()
        profile.print_stats()
        return result
    return wrapper

# Parse command line arguments
parser = argparse.ArgumentParser(description="3D GLC-L Visualization with SVM Boundary Curve")
parser.add_argument("--file_path", required=True, help="Path to the CSV file containing the dataset")
args = parser.parse_args()

# Load dataset from CSV
df = pd.read_csv(args.file_path)
data = df.drop(columns=['class']).values  # Dropping the 'class' column to get features
labels = df['class'].values  # Getting the labels from the 'class' column

# Normalize the data
data_normalized = normalize(data, axis=0, norm='max')

# Function to calculate angle from coefficient
def calculateAngle(coefficient):
    return np.arctan(coefficient)

# Function to evaluate coefficients using LDA classifier
def evaluateCoefficientsLDA(X, y, coefficients):
    X_projected = X.dot(coefficients)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_projected.reshape(-1, 1), y)
    accuracy = lda.score(X_projected.reshape(-1, 1), y)
    return accuracy

# Function to run one epoch and find coefficients
def one_epoch(data, labels):
    coefficients = np.random.uniform(-1, 1, data.shape[1])
    current_accuracy = evaluateCoefficientsLDA(data, labels, coefficients)
    return coefficients, current_accuracy

# Function to find the best coefficients using GLC-AL with SVM and ThreadPoolExecutor
@profile_func
def coefficients_search_svm_parallel(data, labels, epochs=100):
    best_coefficients = None
    best_accuracy = 0
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(one_epoch, [data]*epochs, [labels]*epochs))
        for coefficients, current_accuracy in results:
            if current_accuracy > best_accuracy:
                best_coefficients = coefficients
                best_accuracy = current_accuracy
    return best_coefficients, best_accuracy

# Ensure results are cached so we don't re-run this part
if 'best_coefficients' not in globals():
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.2, random_state=42)

    # Find the best coefficients using GLC-AL with SVM
    best_coefficients, best_accuracy = coefficients_search_svm_parallel(X_train, y_train)

# Once you have found the best coefficients, train the final LDA model
X_projected = data_normalized.dot(best_coefficients).reshape(-1, 1)
lda = LinearDiscriminantAnalysis()
lda.fit(X_projected, labels)

# Get the LDA predictions for the entire dataset
predictions = lda.predict(X_projected)

# Unique labels in the dataset
unique_labels = np.unique(labels)

# Color map
color_map = {label: plt.cm.jet(i/float(len(unique_labels)-1)) for i, label in enumerate(unique_labels)}

# Function to update the plot for each rotation angle
def update(num):
    ax.cla()  # Clear the previous plot
    ax.view_init(azim=1*num, elev=30)
    z_offset_factor = 0.0 # How much to lift the first class along the Z-axis

    line_alpha = 0.1  # Lowering the opacity of the lines (set between 0 and 1)
    x_coords, y_coords, z_coords = [], [], []

    for i in range(data_normalized.shape[0]):
        x, y, z = 0, 0, 0
        
        for j in range(data_normalized.shape[1]):
            radius = data_normalized[i, j]
            angle = calculateAngle(best_coefficients[j])
            new_x = x + radius * np.cos(angle)
            new_y = y + radius * np.sin(angle)
            new_z = z + radius * np.tan(angle)
            
            if labels[i] == unique_labels[0]:  # If this point belongs to the first class
                new_z += z_offset_factor  # Lift the endpoint along the Z-axis
            else:
                new_z -= z_offset_factor  # Lower the endpoint along the Z-axis
            
            ax.plot([x, new_x], [y, new_y], [z, new_z], color=color_map[labels[i]], alpha=line_alpha)
            
            # Show the endpoints
            if j == data_normalized.shape[1] - 1:
                ax.scatter(new_x, new_y, new_z, color='black', s=10, marker='s')  # Larger size for end points

            x, y, z = new_x, new_y, new_z

        # Store the endpoint coordinates for each vector
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)

    # Identify misclassified points based on SVM predictions
    misclassified_indices = np.where(predictions != labels)[0]
    misclassified_points = np.array([[x, y, z] for idx, (x, y, z) in enumerate(zip(x_coords, y_coords, z_coords)) if idx in misclassified_indices])
    
    # Find the furthest left and right misclassified points based on x-coordinate
    if len(misclassified_points) > 0:
        sorted_misclassified = sorted(misclassified_points, key=lambda x: x[0])
        furthest_left = sorted_misclassified[0]
        furthest_right = sorted_misclassified[-1]
        
        # Calculate midpoint between furthest left and right misclassified points
        midpoint = (furthest_left + furthest_right) / 2.0

        # Create a classification boundary plane through the midpoint, parallel to the yz-plane
        yy, zz = np.meshgrid(np.linspace(min(y_coords), max(y_coords), 50),
                             np.linspace(min(z_coords), max(z_coords), 50))
        xx = np.full_like(yy, midpoint[0])  # The plane is at the x-coordinate of the midpoint
        
        # Plot the classification boundary plane
        ax.plot_surface(xx, yy, zz, color='c', alpha=0.4)
        
        # Count points for each class on both sides of the boundary plane
        left_counts = Counter()
        right_counts = Counter()
        
        for x, label in zip(x_coords, labels):
            if x < midpoint[0]:
                left_counts[label] += 1
            else:
                right_counts[label] += 1
                
        # Calculate and display percentages
        annotation_text = "Boundary plane:\n"
        for label in unique_labels:
            total_count = left_counts[label] + right_counts[label]
            left_percentage = (left_counts[label] / total_count) * 100 if total_count else 0
            right_percentage = (right_counts[label] / total_count) * 100 if total_count else 0
            annotation_text += f"{left_percentage:.2f}% of {label} class on the left side\n"
            annotation_text += f"{right_percentage:.2f}% of {label} class on the right side\n"
        
        # Display the text outside the plot
        plt.annotate(annotation_text, xy=(-0.25, 0.95), xycoords='axes fraction', fontsize=10)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D GLC-L (Rotation: {1*num} degrees)')

    # Add a legend outside the plot area with class names
    legend_handles = [plt.Line2D([0], [0], marker='s', color='w', label=str(label), markerfacecolor=color_map[label], markersize=5) for label in unique_labels]
    ax.legend(handles=legend_handles, title="Classes", loc='upper left', bbox_to_anchor=(1, 1))

# Initialize the figure and axis for the animation
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create an animation rotating the plot by 1 degree at each frame
ani = FuncAnimation(fig, update, frames=range(360), repeat=False)

# Save the animation
#ani.save('glcl_3d_rotation_1_degree_svm.gif', writer='pillow', fps=10)

# Or show the animation (uncomment the line below)
plt.show()
