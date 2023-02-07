# gauss
Welcome to the code repository of the paper "Gaussian Siwtchsampling: A Second-Order Approach to Active Learning".

## Environment Setup
First, create a virtual environment, and install poetry
```
python3 -m venv .venv/
. .venv/bin/activate
pip install -U pip
pip install poetry
```
Second, run poetry to install dependencies
```
poetry install
```

If you run into an error similar to "Failed to unlock collection", I suggest the solution 
[here](https://stackoverflow.com/questions/74438817/poetry-failed-to-unlock-the-collection). 

Third, export the python path 
```
export PYTHONPATH=${PYTHONPATH}:${PWD}
```

## Running Experiments
ALl configurations are managed in the example_config.toml file. Here, you can set different strategies, query sizes, 
etc. under the active_learning bracket. To see the available strategies, check out the __init__.py file under 
activelearning/qustrategies/. For hyperparameters related to training (e.g. learning rate) are under the 
classification bracket.

To run an active learning experiment run
```
python3 training/classification/run.py --config example_config.toml 
```

To train a network outside of an active learning setting (with the full training set) run

```
python3 training/classification/run.py --config example_config.toml 
```

## Citation

If you find our code/paper insightful please consider citing us!
