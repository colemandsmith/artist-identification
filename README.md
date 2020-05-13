# artist-identification
Project aiming to perform artist identification from paintings via a neural network.

As a note, my computer currently has a GTX 980 installed, which I normally use to play video games, that I leverage for training my neural networks. For use without cuda, one simply needs to remove the .cuda() calls sprinkled throughout the code.


## Setup
I use a conda environment for this project. Assuming you have conda installed, installing the dependencies should look like:

```
conda env create -f environment.yml
conda activate art
```

Then we install the package itself via pip:

```
pip install -e .
```
or 
```
python setup.py install
```
