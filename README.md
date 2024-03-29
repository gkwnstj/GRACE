# GRACE

## Installation

```bash
python setup.py install
* After installation, change form numpy == 1.19.2 to numpy in  ../gae.egg-info/requires.txt (remove '==1.19.2')
* and then try installation again => python setup.py install
```


## Requirements (automatically installed by installation)
* TensorFlow (1.0 or later)
* python 3.8.15
* networkx
* scikit-learn
* scipy

## Extra Requirements
* pip install umap-learn
* pip install scipy==1.8.0

## conda 
```bash
conda create -n GRACE python=3.8.15 pip
conda activate GRACE
conda install -c anaconda tensorflow-gpu==2.6.0
```



## Run the demo

```bash
python GRACE.py
```

## Data

Set the route. Have a look at the `my_data(data_file)` function in `input_data_my.py`
```
file = pd.read_csv('.../{}.txt'.format(data_file), sep='\t')
```
```
file_label = pd.read_csv('.../{}_label.txt'.format(data_file), names=['order', 'target'], sep='\t')
```

Set the route. Have a look at the `svm_m(names, PRED_elbow_kmean)` function in `support_vector_machine.py`. File `Norm_data_{}` is provided.
```
svm_file = pd.read_csv('.../Norm_data_Umap/Norm_data_{}.txt'.format(names), sep='\t')
```
You can adjust a data(single-cell RNA seqeuncing datasets) as follows:
```
python GRACE.py --dataset usoskin
```

## Models

You can choose between the following models: 
* `gcn_ae`: Graph Auto-Encoder (with GCN encoder)
* `gcn_vae`: Variational Graph Auto-Encoder (with GCN encoder)


## Cite

Please cite our paper if you use this code in your own work:

```
@article{ha2023grace,
  title={GRACE: Graph autoencoder based single-cell clustering through ensemble similarity learning},
  author={Ha, Jun Seo and Jeong, Hyundoo},
  journal={Plos one},
  volume={18},
  number={4},
  pages={e0284527},
  year={2023},
  publisher={Public Library of Science San Francisco, CA USA}
}
```
