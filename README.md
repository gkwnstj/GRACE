## Installation

```bash
python setup.py install
```

## Requirements
* TensorFlow (1.0 or later)
* python 2.7
* networkx
* scikit-learn
* scipy

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
@article{kipf2016variational,
  title={Variational Graph Auto-Encoders},
  author={Kipf, Thomas N and Welling, Max},
  journal={NIPS Workshop on Bayesian Deep Learning},
  year={2016}
}
```
