from __future__ import division
from __future__ import print_function

import time
import os

import tensorflow as tf        
import tensorflow.compat.v1 as tf      

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.input_data_my import my_data
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
from gae.wcss import wcss_by_cluster
from gae.estimate_cluster import elbow_method
from gae.support_vector_machine import svm_m

from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score

import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import umap
import seaborn as sns
import warnings
import argparse
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ""
tf.compat.v1.disable_eager_execution()
warnings.filterwarnings(action='ignore')


result1 = []
result2 = []
result3 = []
result4 = []
result5 = []
result6 = []
result7 = []
result8 = []


Elbow_list = []
result_elbow_kmean_ARI = []
result_elbow_kmean_NMI = []
WCSS=[]
svm_result = []
PRED_results = []
result_time = []


sns.set()
sns.set(rc={"figure.figsize": (5, 4)})


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')    
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('model', 'gcn_ae', 'Model string.')       # Choosing model gcn_ae, gcn_vae
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')   # out data features are not binary
flags.DEFINE_integer('SVM_run', 1, 'Whether to run SVM (1) or not (0)')    
flags.DEFINE_integer('WCSS_run', 1, 'Whether to run WCSS (1) or not (0)')   
flags.DEFINE_integer('sampling_count', 20, 'Number of samples')    # default 20
flags.DEFINE_integer('repeat_count', 1, 'Number of runs')    # User can get the averages of each performance results through multiple runs
flags.DEFINE_string('dataset', 'kolod', 'Dataset string.')

dataset_str = FLAGS.dataset
model_str = FLAGS.model
repeat_count = FLAGS.repeat_count
sampling_count = FLAGS.sampling_count
SVM_run = FLAGS.SVM_run
WCSS_run = FLAGS.WCSS_run


print("############################# {} ##############################".format(dataset_str))
    
for j in range(0,repeat_count):
      
    start = time.time()


    # Load data
    adj, features, n_clusters, true_label, points_for_WCSS = my_data(dataset_str, sampling_count)
    
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()


    adj_train = mask_test_edges(adj) 
    adj = adj_train


    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]


    # Create model
    model = None                      
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                           validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())



    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        feed_dict.update({placeholders['dropout']: 0})     
        emb = sess.run(model.z_mean, feed_dict=feed_dict)  


    print("Optimization Finished!")
    print("avg_cost : ", avg_cost)

    ########################### Elbow Method for K means ####################

    elbow = elbow_method(emb)

    Elbow_list.append(elbow)

    ############################ Clustering #####################################

    KMEANS_elbow = KMeans(elbow)

    KMEANS_elbow.fit(emb)

    PRED_elbow_kmean = KMEANS_elbow.predict(emb)

    PRED_results.append(PRED_elbow_kmean)

    ###################################### ARI, NMI ##################################
    
    result_elbow_kmean_ARI.append(adjusted_rand_score(true_label,PRED_elbow_kmean))

    result_elbow_kmean_NMI.append(normalized_mutual_info_score(true_label,PRED_elbow_kmean))

    print("time : ", time.time() - start)
    
    result_time.append(time.time() - start)

    #######################################      SVM        ###################################

    if SVM_run == 1:
        
        svm_accuracy = svm_m(dataset_str, PRED_elbow_kmean)

        svm_result.append(svm_accuracy)


    ############################################   WCSS     ##############################################

    if WCSS_run == 1:
        
        WCSSresult = wcss_by_cluster(points_for_WCSS, PRED_elbow_kmean)

        WCSS.append(WCSSresult)


print("#################################################################################################")
print("Elbow_list : ", Elbow_list)
print("time_list : ", result_time)
print("result_elbow_kmeans_ARI_list : ", result_elbow_kmean_ARI)
print("result_elbow_kmeans_NMI_list : ", result_elbow_kmean_NMI)
print("result_elbow_kmeans_svm_list : ", svm_result)
print("result_elbow_kmeans_WCSS_list : ", WCSS)

print("AVERAGE_elbow_kmean_ARI : ", sum(result_elbow_kmean_ARI)/repeat_count)
print("AVERAGE_elbow_kmean_NMI : ", sum(result_elbow_kmean_NMI)/repeat_count)
print("AVERAGE_elbow_kmean_svm : ", sum(svm_result)/repeat_count)
print("AVERAGE_elbow_kmean_WCSS : ", sum(WCSS)/repeat_count)
print("Average_time : ", sum(result_time)/repeat_count)








