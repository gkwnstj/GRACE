from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.optimizer import OptimizerAE, OptimizerVAE
##from gae.input_data import load_data
from gae.input_data_my import my_data
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges


import umap



##
tf.compat.v1.disable_eager_execution()
##



###################################################################################
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
###################################################################################

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets

###########################################################

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import seaborn as sns

###########################################################


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--plot_dimension', type=int, default=2, help='number of dimension.') 
args = parser.parse_args()
###########################################################

########################################################################################################

result1 = []
result2 = []
result3 = []
result4 = []
result5 = []
result6 = []
result7 = []
result8 = []
########################################################################################################

########################################################################################################


Elbow_list = []
result_elbow_kmean_ARI = []
result_elbow_kmean_NMI = []
WCSS=[]
svm_result = []
PRED_results = []
result_time = []


########################################################################################################
sns.set()
sns.set(rc={"figure.figsize": (5, 4)})
########################################################################################################


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')    #####################
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')       ################### Choosing model gcn_ae, gcn_vae
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')   ############ out data features are not binary

model_str = FLAGS.model
dataset_str = FLAGS.dataset



##names = ['klein', 'zeisel', 'usoskin', 'kolod', 'baron_h1', 'baron_h2', 'baron_h3', 'baron_h4', 'baron_m1', 'baron_m2']

names = ['baron_m2']





for k in range(0,len(names)):
    
    file_name = names[k]
    print("############################# {} ##############################".format(names[k]))
        
    for j in range(0,1):
          



        start = time.time()


        # Load data
        adj, features, n_clusters, true_label, points_for_WCSS = my_data(file_name)
        
        # Store original adjacency matrix (without diagonal entries) for later
        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()


        ##adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj) #######
        adj_train = mask_test_edges(adj) #######
        adj = adj_train


        if FLAGS.features == 0:
            features = sp.identity(features.shape[0])  # featureless

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
        model = None                      # None
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

        cost_val = []
        acc_val = []


        def get_roc_score(edges_pos, edges_neg, emb=None):
            if emb is None:
                feed_dict.update({placeholders['dropout']: 0})
                emb = sess.run(model.z_mean, feed_dict=feed_dict)
                #print(emb) #########

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            # Predict on test set of edges
            adj_rec = np.dot(emb, emb.T)
        ##    print(adj_rec)
            preds = []
            pos = []
            for e in edges_pos:
                preds.append(sigmoid(adj_rec[e[0], e[1]]))
                pos.append(adj_orig[e[0], e[1]])

            preds_neg = []
            neg = []
            for e in edges_neg:
                preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
                neg.append(adj_orig[e[0], e[1]])

            preds_all = np.hstack([preds, preds_neg])
            labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
            roc_score = roc_auc_score(labels_all, preds_all)
            ap_score = average_precision_score(labels_all, preds_all)

            return roc_score, ap_score


        cost_val = []
        acc_val = []
        val_roc_score = []

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
            ###################################################################################################
            feed_dict.update({placeholders['dropout']: 0})     
            emb = sess.run(model.z_mean, feed_dict=feed_dict)  
    ##        print(emb)
            #####################################################################################################    
        ##    roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
        ##    val_roc_score.append(roc_curr)

    ##        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
    ##              "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
    ##              "val_ap=", "{:.5f}".format(ap_curr),
    ##              "time=", "{:.5f}".format(time.time() - t))

        print("Optimization Finished!")
        print("avg_cost : ", avg_cost)


        ##roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
        ##print('Test ROC score: ' + str(roc_score))
        ##print('Test AP score: ' + str(ap_score))




        ###################
        

        ############################# F I L E _ L O A D ###################################

    ##    file = pd.read_csv('usoskin.txt', sep='\t')
    ##    X6 = np.array(file.transpose())
    ##    X7 = X6.sum(axis=1)
    ##    X8 = X7.reshape(len(X7), 1)
    ##    normalized_array = np.log2(1 + (X6 / X8 * (10 ** 6)))
    ##    file2 = normalized_array
    ##    cluster_df = np.array(file2)
        cluster_df = emb


        ####################################################################################







        ############################################### Elbow Method for K means        2        


    ##    start = time.time()

        # Import ElbowVisualizer
        from yellowbrick.cluster import KElbowVisualizer
        model = KMeans()
        # k is range of number of clusters.
        visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
        visualizer.fit(cluster_df)        # Fit data to visualizer
        print("Elbow Method : ", visualizer.elbow_value_)

        elbow = visualizer.elbow_value_

        Elbow_list.append(elbow)
        
##        visualizer.show(outpath="{}_Elbow_Method_for_K_means.pdf".format(names[j]))        # Finalize and render figure
    ##    print("time :", time.time() - start)

        plt.close()


        ########################################################################




        ########################################################################################################



    ##    KMEANS = KMeans(n_clusters)
    ##    KMEANS.fit(emb)
    ##    PRED = KMEANS.predict(emb)
    ##    print(adjusted_rand_score(true_label,PRED))
    ##    print(normalized_mutual_info_score(true_label,PRED))
    ##
    ##
    ##
    ##
    ##    clustering = SpectralClustering(n_clusters).fit(emb)
    ##    print(adjusted_rand_score(true_label,clustering.labels_))
    ##    print(normalized_mutual_info_score(true_label,clustering.labels_))
    ##
    ##    
    ##
    ##    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    ##
    ##
    ##
    ##    result1.append(adjusted_rand_score(true_label,PRED))
    ##    result2.append(adjusted_rand_score(true_label,clustering.labels_))
    ##    result3.append(time.time() - start)
    ##    result4.append(avg_cost)
    ##    result5.append(normalized_mutual_info_score(true_label,PRED))
    ##    result6.append(normalized_mutual_info_score(true_label,clustering.labels_))
    ##    result7.append(PRED)
    ##    result8.append(clustering.labels_)


        ########################################################################################################




        KMEANS_elbow = KMeans(elbow)

        KMEANS_elbow.fit(emb)

        PRED_elbow_kmean = KMEANS_elbow.predict(emb)

        PRED_results.append(PRED_elbow_kmean)

        result_elbow_kmean_ARI.append(adjusted_rand_score(true_label,PRED_elbow_kmean))

        result_elbow_kmean_NMI.append(normalized_mutual_info_score(true_label,PRED_elbow_kmean))

        print("time : ", time.time() - start)
        
        result_time.append(time.time() - start)



        ############################################   WCSS     ###############################################
    
        norm_data = pd.DataFrame(points_for_WCSS)
        norm_data['label'] = PRED_elbow_kmean                 #########################
        Extract = norm_data.sort_values(by ='label',ascending=True)



        
        WCSS_result = []
        for i in range(0,len(Counter(PRED_elbow_kmean).keys())):     #########################


            Extract_0 = Extract[Extract['label']==i]
            #print(len(Extract_0))

            Extract_del = Extract_0.drop(columns=['label'],axis=1)
            num_arr = np.array(Extract_del)
            #print(num_arr)

            WCSS_kmeans = KMeans(n_clusters=1, init = "k-means++")
            WCSS_kmeans.fit_predict(num_arr)
            inertia = WCSS_kmeans.inertia_
            print(inertia)
            
            WCSS_result.append(inertia)


        a = sum(WCSS_result)
        print("WCSS : ", a)
        WCSS.append(a)
        

        WCSS_result.clear()



    
        ########################################################################################################




        #######################################      SVM        ########################################################


        svm_file = pd.read_csv('D:/FINAL_AUTO_PROJECT/WCSS_coodinate/Norm_data_{}.txt'.format(names[k]), sep='\t')
        df_set = svm_file[['0','1']]

        svm_lb = du = pd.DataFrame(PRED_elbow_kmean, columns = ['label'])

        # Split data to train and test on 80-20 ratio
        X_train, X_test, y_train, y_test = train_test_split(df_set, svm_lb, test_size = 0.2, random_state = 0)   ### auto shuffled

        print("Displaying data. Close window to continue.")
        # Plot data
        #plot_data(X_train, y_train, X_test, y_test)

        # make a classifier and fit on training data
        clf = svm.SVC(kernel='linear')

        # Train classifier 
        clf.fit(X_train, y_train)

        print("Displaying decision function. Close window to continue.")  
        # Plot decision function on training and test data
        #plot_decision_function(X_train, y_train, X_test, y_test, clf)

        #plt.scatter(X_train['0'],X_train['1'], c=y_train['label'], cmap=CMAP, s=30)
        #plt.show(block =True)
        #plt.scatter(X_test['0'],X_test['1'], c=y_test['label'], cmap=CMAP, s=30)
        #plt.show(block =True)

        # Make predictions on unseen test data
        clf_predictions = clf.predict(X_test)
        print("Accuracy: {}%".format(clf.score(X_test, y_test) * 100 ))
        svm_result.append(clf.score(X_test, y_test) * 100 )


        ##################################################################################################################### 




##########################################################################################################

    print("#################################################################################################")

    print("Elbow_list : ", Elbow_list)
    print("time_list : ", result_time)
    print("result_elbow_kmeans_ARI_list : ", result_elbow_kmean_ARI)
    print("result_elbow_kmeans_NMI_list : ", result_elbow_kmean_NMI)
    print("result_elbow_kmeans_WCSS : ", WCSS)
    print("result_elbow_kmeans_svm : ", svm_result)
##########################################################################################################
    
    print("AVERAGE_elbow_kmean_ARI : ", sum(result_elbow_kmean_ARI)/10)
    print("AVERAGE_elbow_kmean_NMI : ", sum(result_elbow_kmean_NMI)/10)
    print("AVERAGE_elbow_kmean_WCSS : ", sum(WCSS)/10)
    print("AVERAGE_elbow_kmean_svm : ", sum(svm_result)/10)

    print("#################################################################################################")

##########################################################################################################
    
    print("time_average : ", sum(result_time)/10)



    CSV_pred = pd.DataFrame(PRED_results)
    CSV_pred_ = CSV_pred.to_csv("pred_gae_{}.csv".format(names[k]))


    ############################################### Scatter_start ################################################

    PALETTE_true = sns.color_palette('deep', n_colors=len(true_label.unique()))
    CMAP_true = ListedColormap(PALETTE_true.as_hex())

    ################################### emb_2_true_label  ######################################
    
    plt.figure(1)
    pca = PCA(n_components = 2)
    emb_PCA = pca.fit_transform(emb)
##    plt.scatter(x=emb_PCA[:,0], y=emb_PCA[:,1], c = true_label, cmap=CMAP_true)
##    plt.show(block=False)

    pca_emb=pd.DataFrame(emb_PCA)
    pca_emb.to_csv("PCA_with_16_embedded_vector_{}.csv".format(names[k]))

    ################################# T-SNE_emb_2_pred_label  ######################################


    plt.figure(2)
    tsne = TSNE(n_components=2, learning_rate = 200, init = 'pca')
    emb_tsne = tsne.fit_transform(emb)
##    plt.scatter(x=emb_tsne[:,0], y=emb_tsne[:,1], c = true_label, cmap=CMAP_true)
##    plt.show(block=False)


    t_sne_emb=pd.DataFrame(emb_tsne)
    t_sne_emb.to_csv("TSNE_with_16_embedded_vector_{}.csv".format(names[k]))


    ################################# Umap_emb_2_pred_label  ######################################


    plt.figure(3)

    reducer = umap.UMAP()


    points_for_umap = reducer.fit_transform(emb)

    umap_emb = pd.DataFrame(points_for_umap)

    umap_emb.to_csv("Umap_with_16_embedded_vector_{}.csv".format(names[k]))

    







    Elbow_list.clear()
    result_elbow_kmean_ARI.clear()
    result_elbow_kmean_NMI.clear()
    PRED_results.clear()
    WCSS.clear()
    svm_result.clear()
    result_time.clear()




    



    




    








##########################################################################################################


    ##print("KMEANS_ARI : ",result1)
    ##print("ARI_kmeans average : ",sum(result1)/10)
    ##print("KMEANS_NMI : ",result5)
    ##print("NMI_kmeans average : ",sum(result5)/10)
    ##
    ##
    ##print("SPECTRAL_ARI : ",result2)
    ##print("ARI_spectral average : ",sum(result2)/10)
    ##print("SPECTRAL_NMI : ",result6)
    ##print("NMI_spectral average : ",sum(result6)/10)
    ##
    ##print("time average : ",sum(result3)/10)
    ##
    ##print("avg_cost(loss) : ", result4)
    ##print("avg_cost(loss) average : ",sum(result4)/10 )
    ##
    ##
    ##
    ##CSV_pred = pd.DataFrame(result7)
    ##CSV_pred_ = CSV_pred.to_csv("pred_kmeans_.csv")
    ##
    ##CSV_pred1 = pd.DataFrame(result8)
    ##CSV_pred_1 = CSV_pred1.to_csv("pred_spectral_.csv")


##########################################################################################################




    ##if args.plot_dimension == 2:
    ##    points_frame=pd.DataFrame(emb, columns=['x','y'])
    ##    plt.scatter(x=points_frame['x'], y=points_frame['y'])
    ##else:
    ##    points_frame=pd.DataFrame(emb, columns=['x','y','z'])
    ##    x_coor = list(np.array(points_frame['x'].tolist()))
    ##    y_coor = list(np.array(points_frame['y'].tolist()))
    ##    z_coor = list(np.array(points_frame['z'].tolist()))
    ##    fig = plt.figure()
    ##    ax = fig.gca(projection='3d')
    ##    ax.scatter(x_coor,y_coor,z_coor, marker='o', s=15, c='darkgreen')
    ##
    ##
    ##plt.show(block = False)
