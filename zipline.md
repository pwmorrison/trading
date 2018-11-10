# Installing
Needs Python 3.5.

https://www.zipline.io/install.html#conda
Best to use conda install:

conda install -c Quantopian zipline

https://github.com/quantopian/zipline/issues/2186

Downgrade pandas-datareader to get rid of an error message when calling "zipline" at the command line:

conda install pandas-datareader==0.2.1 

# Jupyter notebook
Since it needs Python 3.5, need to create it in a new environment with Python 3.5, "conda install ipykernel", and run something like "python -m ipykernel install --user --name myenv --display-name "Python (myenv)"", with myenv being the environment name. This will make sure there is a Python 3.5 kernel available within Jupyter Notebook.

Then, inside Jupyter Notebook browser tab, can select this Python 3.5 kernel.

https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments

# Data

Set the Quandl key as an environment variable:
sytzpzH7YG9Y_xhzXGY8

Have zipline injest the data, by running this at the command line:
zipline ingest -b quandl

Can download the Quandl data from here: https://www.quandl.com/databases/WIKIP