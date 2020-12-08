# Visualizations Generators
The Visualizations are built in jupyter notebooks. A running jupyter kernel(Python 3) is required to access them.

## Data:
All the csv files downloaded from [Zillow Economics Data](https://www.kaggle.com/zillow/zecon) should be kept in data folder. This is very important.

## Required Installation:
+   Python 3
+   jupyter

## Packages:
Once jupyter and python are installed, install the other packages in requirements.txt

    pip install -r requirements.txt

The jupyter kernel can be started by running

    jupyter notebook

Once a kernel is running, open ipynb files in the browser.

+   Cross Generator.ipynb generates a simplified State-County relationships and abbriviations used in code for UI

+   Level 1 Plots Generator.ipynb generates the US country chloropleth heatmap

+   Level 2 Plots Generator.ipynb generates chloropleth heatmaps for all the states in US (This script will run for a considerable amount of time)

+   Level 3 Plots Generator.ipynb This notebook trains the LSTM model for individual counties and generates csv files with Date,Price format each county.
