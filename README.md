## DL4IA Challenge
Model used to predict whether images of cells come from cancer
patients or healthy patients.

## Data
The data is not included in the repo. Download it from
https://www.kaggle.com/c/cancer-cell-challange/data
and add the ``train`` and ``test`` folders, as well as the ``train.csv`` file
to the ``data`` folder.

## Usage

### Training
To train a model, use ``train.py``:

``
python train.py --model=model_name
``

The following options are available:
* ``--model``: Name of the model for saving the network and figures __(required)__
* ``--batch_size``: Number of data points to read in each batch, default 32
* ``--epochs``: Number of epochs to train the model, default 5
* ``--lr``: Learning rate, default 0.001
* ``--n_cpu``: Number of CPUs used to load data, default 0 (only main process)
* ``--validate_epochs``: How frequently validation results should be reported
  to figures, default 1 (every epoch)

### Testing
To test a model, use ``test.py``:

``
python test.py --model=model_name
``

The following options are available:
* ``--model``: Name of the model to load __(required)__
* ``--batch_size``: Number of data points to read in each batch, default 32
* ``--n_cpu``: Number of CPUs used to load data, default 0 (only main process)

## Models
There is a pretrained model available, ``basic_model``. Test it using

``python test.py --model=basic_model``