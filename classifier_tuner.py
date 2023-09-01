
from tkinter import Tk, Label, Button, filedialog, OptionMenu, StringVar
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Initialize the classifiers and their parameter grids
classifiers = {
    'KNN': KNeighborsClassifier(),
    'SVM-Linear': SVC(kernel='linear'),
    'SVM-RBF': SVC(kernel='rbf'),
    'SVM-Poly': SVC(kernel='poly'),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier()
}

param_grids = {
    'KNN': {'n_neighbors': [1, 3, 5, 7, 9, 11]},
    'SVM-Linear': {'C': [0.1, 1, 10]},
    'SVM-RBF': {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]},
    'SVM-Poly': {'C': [0.1, 1, 10], 'degree': [2, 3, 4]},
    'Naive Bayes': {},
    'Random Forest': {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]}
}

def run_classifier(file_path, classifier_name):
    # Load the dataset
    df = pd.read_csv(file_path)
    
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
        grid_search = GridSearchCV(clf, param_grid, cv=5)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        clf = grid_search.best_estimator_
    
    # Fit the classifier
    clf.fit(X, y)
    
    # Perform 5-fold cross-validation
    scores = cross_val_score(clf, X, y, cv=5)
    avg_score = np.mean(scores)
    
    # Generate the scatterplot matrix with decision boundaries
    # This is a simplified visualization and works best with 2D or 3D data
    plt.figure(figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.title(f"{classifier_name} (5-Fold CV Score: {avg_score:.2%})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

class ClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("Classifier Tool")

        self.label = Label(master, text="Select Classifier and Dataset")
        self.label.pack()

        self.classifier_var = StringVar(master)
        self.classifier_var.set("KNN")  # default value

        classifier_options = list(classifiers.keys())
        self.classifier_menu = OptionMenu(master, self.classifier_var, *classifier_options)
        self.classifier_menu.pack()

        self.select_button = Button(master, text="Select Dataset", command=self.select_file)
        self.select_button.pack()

        self.run_button = Button(master, text="Run Classifier", command=self.run)
        self.run_button.pack()

    def select_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    def run(self):
        run_classifier(self.file_path, self.classifier_var.get())

root = Tk()
app = ClassifierApp(root)
root.mainloop()
