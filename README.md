# GRACE

## Installation

```bash
python setup.py install
```

## Requirements (automatically installed by installation)
* TensorFlow (1.0 or later)
* python 3.8.12
* networkx
* scikit-learn
* scipy

## Extra Requirements
* pip install umap-learn


## Data

Set the root. Have a look at the `my_data(data_file)` function in `input_data_my.py`
```
file = pd.read_csv('.../{}.txt'.format(data_file), sep='\t')
```
```
file_label = pd.read_csv('.../{}_label.txt'.format(data_file), names=['order', 'target'], sep='\t')
```
You can specify a dataset as follows:
```
python GRACE.py --dataset usoskin
```



## Run the demo

```bash
python GRACE.py
```


## Models

You can choose between the following models: 
* `gcn_ae`: Graph Auto-Encoder (with GCN encoder)
* `gcn_vae`: Variational Graph Auto-Encoder (with GCN encoder)


## Cite

Please cite our paper if you use this code in your own work:

```
@article{},
  title={},
  author={},
  journal={},
  year={}
}
```
