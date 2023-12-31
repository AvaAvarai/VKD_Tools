from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import argparse

# Initialize the classifiers and their parameter grids
classifiers = {
    'KNN': KNeighborsClassifier(),
    'SVM-Linear': SVC(kernel='linear'),
    'SVM-RBF': SVC(kernel='rbf'),
    'SVM-Poly': SVC(kernel='poly'),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(),
    'LDA': LinearDiscriminantAnalysis(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression()
}

param_grids = {
    'KNN': {'n_neighbors': [1, 3, 5, 7, 9, 11]},
    'SVM-Linear': {'C': [0.1, 1, 10]},
    'SVM-RBF': {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]},
    'SVM-Poly': {'C': [0.1, 1, 10], 'degree': [2, 3, 4]},
    'Naive Bayes': {},
    'Random Forest': {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]},
    'LDA': {'solver': ['svd', 'lsqr', 'eigen']},
    'Decision Tree': {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30]},
    'Logistic Regression': {'C': [0.1, 1, 10], 'solver': ['newton-cg', 'lbfgs', 'liblinear']}
}

def run_classifier(file_path, classifier_name):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # get csv file name without extension
    dataset_name = file_path.split('/')[-1].split('.')[0]
    
    # Extract features and target
    X = df.drop('class', axis=1).values
    
    # Label Encoding for the target variable
    le = LabelEncoder()
    y = le.fit_transform(df['class'].values)
    
    # Get the classifier and its parameter grid
    clf = classifiers[classifier_name]
    param_grid = param_grids[classifier_name]
    
    # Perform hyperparameter tuning using GridSearchCV
    if param_grid:
        grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=5)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        clf = grid_search.best_estimator_
    
    # Fit the classifier
    clf.fit(X, y)
    
    # Perform 5-fold cross-validation
    scores = cross_val_score(clf, X, y, cv=5)
    avg_score = np.mean(scores)
    
    feature_names = df.columns[:-1]
    plot_decision_boundaries(X, y, clf, dataset_name, feature_names, classifier_name, le)

def plot_decision_boundaries(X, y, clf, dataset_name, feature_names, classifier_name, le):
    n_features = X.shape[1]
    h = .02  # Step size in the mesh
    unique_labels = np.unique(y)
    n_classes = len(unique_labels)
    cmap_bold = ListedColormap(plt.cm.rainbow(np.linspace(0, 1, n_classes)))
    accuracies = []
    
    plt.figure(figsize=(14, 8))
    
    for i in range(n_features):
        for j in range(i):
            plt.subplot(n_features, n_features, i * n_features + j + 1)
            
            # We only take two corresponding features
            X_sub = X[:, [j, i]]
            
            # Fit the classifier and make predictions
            clf.fit(X_sub, y)
            y_pred = clf.predict(X_sub)
            
            # Compute accuracy
            acc = accuracy_score(y, y_pred)
            accuracies.append(acc)
            
            # Display accuracy as the title of each subplot
            plt.title(f'Acc: {acc:.2f}')
            
            # Plot the decision boundary
            x_min, x_max = X_sub[:, 0].min() - 1, X_sub[:, 0].max() + 1
            y_min, y_max = X_sub[:, 1].min() - 1, X_sub[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.pcolormesh(xx, yy, Z, cmap=cmap_bold, alpha=0.33)
            
            # Plot the training points
            plt.scatter(X_sub[:, 0], X_sub[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
            
            # Set axis labels
            plt.xlabel(feature_names[j])
            plt.ylabel(feature_names[i])
            
    # Create handles for the legend
    labels = le.classes_
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                           markersize=10, markerfacecolor=cmap_bold(i / (n_classes - 1))) for i, label in enumerate(le.classes_)]
    plt.figlegend(handles=list(handles), labels=list(labels), loc='upper center', bbox_to_anchor=(0.5, 0.85), ncol=len(set(y)), title="Classes", bbox_transform=plt.gcf().transFigure)

    # Display the average accuracy in the title
    avg_acc = np.mean(accuracies)
    # include the average accuracy in the title with the dataset name and classifier name and the hyperparameters that are used and not None
    params = clf.get_params()
    param_str = ', '.join(f'{key}={value}' for key, value in params.items() if value is not None and value != False and value != -1 and value != 0 and value != 0.0 and value != '0.0')

    plt.suptitle(f'{dataset_name} decisions for {classifier_name} Attribute Pairing Matrix (Avg Acc: {avg_acc:.2f}) Hyperparameters:\n{param_str}', fontsize=14, y=0.92)
    
    plt.tight_layout()
    plt.show()

# Parse command line arguments
parser = argparse.ArgumentParser(description="3D GLC-L Visualization with SVM Boundary Curve")
parser.add_argument("--file_path", required=True, help="Path to the CSV file containing the dataset")
parser.add_argument("--classifier_name", required=True, help="Name of the classifier to use")
args = parser.parse_args()

run_classifier(args.file_path, args.classifier_name)
