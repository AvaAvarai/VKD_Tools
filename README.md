# VKD_Tools

Tools made for visual knowledge discovery of multidimensional classification data.

Launch with the menu.py script.

## Overview

These scripts are designed for visualizing and exploring multidimensional data. Particularly data used for machine learning classification tasks. We aim to produce models which are inherently explainable.

## Libraries

These python libraries are required to run these scripts.

### Data Manipulation and Analysis

- pandas
- numpy
- scikit-learn

### Data Visualization

- matplotlib
- plotly
- PyOpenGL

### User Interface and System Interaction

- tkinter
- argparse
- subprocess
- webbrowser

### Main Menu Script

- menu.py: Provides a Tkinter-based graphical user interface as a main menu for launching the visualization scripts.

![Menu after data loaded](screenshots/menu.png)

### Visualization Scripts

1. envelope_plotter.py: Creates an interactive application for plotting envelope-like structures.
    - Utilizes PyQt6 for the graphical user interface.
    - Employs OpenGL for rendering graphical elements.
    - Drag and drop searchable hyper-rectangle with WASD resizing, right-click to clear.

    ![Envelope Demo](screenshots/envelope1.png)

2. circular_plotter.py: Produces circular plots using Matplotlib and scikit-learn.
    - Processes data with Linear Discriminant Analysis and plots discriminant line.
    - Displays classification confusion matrix.
    - Handles data preprocessing using Pandas and NumPy.
    - Draggable LDA discriminant line.

    ![Circular Demo](screenshots/circular1.png)

3. plotly_demo.py: Focuses on data visualization using Plotly.
    - Plots the data in draggable axis parallel coordinates plot.
    - Distinctly displays classes with heatmap legend.

    ![Plotly Demo](screenshots/plotly1.png)

4. glc_line_plotter.py: Generates GLC line glyphs.
    - Generates a multidimensional GLC line graph.
    - Projects last glyph per class to x axis.
    - Processes data with Linear Discriminant Analysis and sorts by coefficient array.

    ![GLC Lines Demo](screenshots/glc_lines.png)

5. tree_glyph_plotter.py: Generates high-dimensional data visualization using tree-like glyphs.
    - Lossless visualization of high-dimensional data.
    - Plots a permutation of the feature vecture in tree glyphs.
    - Plotted permutation can be cycled with the mouse wheel.

    ![Tree Glyph Output Demo](screenshots/wheat_seeds_combinatorial_trees_1.png)

---

### Aknowledgements

- CWU Visual Knowledge Discovery and Imaging Lab at <https://github.com/CWU-VKD-LAB>
