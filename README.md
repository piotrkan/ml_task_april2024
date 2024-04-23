# README
This repo encapsulates my workflow & my solutions to the provided ML task. 

My workflow & solutions to the problem can be found under main.ipynb - in this notebook I tried to create a report of 
my ways of thinking, including coded justifications. The code utilities accompanying this notebook can be found within src folder - the functions were briefly described & commented so they should be quite self-explanatory
If you want to check specific functions source code from the notebook, run function?? or function?

'main.ipynb' encapsulates my workflow, ways of thinking and coded justifications, however the main codebase is present in src folder which is imported as a module. The outputs of cells are saved thus you can see my work withour actually running the notebook (but if you would like to reproduce it, follow the setup below). 

*Note: when working on the problem, I was using BERT transformers from hugging face to tackle the sequence data. While it worked at the time, when I was reproducing the notebook (to ensure everything works OK), huggingface server is not available some cells might be missing*

## Setup
I am using python 3.9.7 & pip 21.2.3. I decided to use pythons virtual environment for this project for good reproducibility (easier to setup than conda)
~~~
python -m venv .venv_name #create a virtual env
#windows activation
.\.venv_name\Scripts\activate
#linux or mac activation
source ./.venv_name/bin/activate

#once activated, install dependencies
pip install -r requirements.txt #install all dependencies

#install the venv as a kernel for the notebook
python -m ipykernel install --user --name=venv_name

#if you want to use jupyter notebook, run the following. I used VSCode for the development but jupyter notebook works just as fine
jupyter notebook
~~~
You are ready to reproduce the notebook now.

## Structure
~~~
ml_challenge/
├── data/          # Data directory;
├── docs/          # Documentation
│   ├── ML engineer # instructions provided
│
├── src/           # Source code
│   ├── models.py # neural network model
│   ├── train.py  # training environment
│   ├── data.py   # data-related utilities
│   └── utils.py/ # misc
│
├── test_src/           # Test files (pytest)
│   └── data_test.py    # Tests data.py module (I didnt have time to write tests
│                       # for all scripts so I chose the most important one)
│
├── main.ipynb       # main notebook with my solutions done in time      
├── models/          # directory with saved models (will be created when running models)
└── README.md      # Project README
~~~
