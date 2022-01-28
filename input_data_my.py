import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


#######
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans  # model
from itertools import combinations
##import networkx as nx
import copy
from networkx.algorithms import community
import matplotlib.animation as animation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.manifold import TSNE
import time
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import jaccard_score
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import umap

import torch

#######


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features




def my_data(data_file):


    ################################
    print("############################# {} ##############################".format(data_file))



    file = pd.read_csv('D:/FINAL_AUTO_PROJECT/my_data/{}.txt'.format(data_file), sep='\t')
    file_label = pd.read_csv('D:/FINAL_AUTO_PROJECT/my_data/{}_label.txt'.format(data_file), names=['order', 'target'], sep='\t')





##    def plot_buettner_2d(x, y, title, xlabel="1st eigenvector", ylabel="2nd eigenvector"):
##        sns.set_style("darkgrid")
##        plt.scatter(x, y, c=file_label_num['target'], cmap=CMAP, s=30)
##        plt.title(title, fontsize=20, y=1.03)
##
##        plt.xlabel(xlabel, fontsize=16)
##        plt.ylabel(ylabel, fontsize=16)




    #########################
    num_list=list(range(0,len(file_label['target'].unique())))

    cell_type=file_label['target'].unique()
    num_label = list(file_label['target'])

    for index, value in enumerate(num_label):
        for i in range(0,len(num_list)):
            if value == cell_type[i]:       
                num_label[index] = num_list[i]

    #print(cell_type)
    #print(num_label)
    file_label_num = pd.DataFrame(num_label,columns=['target'])
    true_label = file_label_num['target']

    #########################


    sns.set()
    sns.set(rc={"figure.figsize": (5, 4)})
    PALETTE = sns.color_palette('deep', n_colors=len(file_label_num['target'].unique()))
    CMAP = ListedColormap(PALETTE.as_hex())

################################################## Scatter ##################################################

##    plt.figure(1)
##    pca = PCA(n_components = 2)
##    orig_PCA = pca.fit_transform(file.transpose())
##    plt.scatter(x=orig_PCA[:,0], y=orig_PCA[:,1], c=file_label_num['target'], cmap=CMAP, s=30)
##    plt.show(block=False)


################################################## Scatter ##################################################



    X6 = np.array(file.transpose())
    X7 = X6.sum(axis=1)
    X8 = X7.reshape(len(X7), 1)
    normalized_array = np.log2(1 + (X6 / X8 * (10 ** 6)))
    file2 = normalized_array
    T = pd.DataFrame(file2, columns=list(file.index))
    T1 = T.transpose()  ###normalized data

    ###Variance
    file3 = file2.transpose()
    variance = file3.var(axis=1)
    variance1 = variance.reshape(len(variance), 1)
    pd_variance = pd.DataFrame(variance, columns=['variance'], index=list(file.index))
    Sort = pd_variance.sort_values(by='variance', ascending=False)
    Select = Sort.head(int(0.05 * len(pd_variance)))                   ################################# Sampling of variance

    sns.set()
    sns.set(rc={"figure.figsize": (5, 4)})
    PALETTE = sns.color_palette('deep', n_colors=len(file_label['target'].unique()))
    CMAP = ListedColormap(PALETTE.as_hex())

    file1 = pd.DataFrame(file2, columns=list(file.index))
    Select_file_T = file1[Select.index]




    reducer = umap.UMAP()
    points_for_WCSS = reducer.fit_transform(T1.transpose())


############################################## NO LOAD adj_START #######################################################



    #pca = PCA(n_components = 2)
    #points_for_WCSS = pca.fit_transform(T1.transpose())
    #plot_buettner_2d(x=points_for_WCSS[:, 0], y=points_for_WCSS[:, 1], title='dataset visualized with TSNE of variance of 5%')
    #plt.show(block=False)








        
    start = time.time() # 시작 시간 저장



      
        
##    tsne = TSNE(n_components=2, learning_rate = 200, init = 'pca', random_state = 0)
##    tsne = TSNE(n_components=2, learning_rate = 200, init = 'random')
    tsne = TSNE(n_components=2, learning_rate = 200, init = 'pca')


    points = tsne.fit_transform(Select_file_T)

        


    fig, ax = plt.subplots(2, 10, figsize=(20, 5))
    fig, ax1 = plt.subplots(2, 10, figsize=(20, 5)) #####true label with PCA
    adj_matrs_kmean=[]

        


    for i in range(0, 20):
##        for j in range(0, 10):
        Select_file = Select_file_T.transpose()
        
##        Sample = Select_file.sample(int(len(Select)*0.7), random_state = i)  ########################################## feature Sampling
        Sample = Select_file.sample(int(len(Select)*0.7))
        
        Sample_T = Sample.transpose()
        points = tsne.fit_transform(Sample_T)
##        print(points)
##            ax1[i, j].scatter(x=points[:, 0], y=points[:, 1], c=file_label_num['target'], cmap=CMAP, s=30) ########### True label
        tSNE_points = pd.DataFrame(points, columns=['x', 'y'])
        
##        kmeans = KMeans(n_clusters=30, random_state = 0)   ########30_similar#######  #n_clusters=len(file_label['target'].unique())
        kmeans = KMeans(n_clusters=30)

        kmeans.fit(tSNE_points)
        pred = kmeans.predict(tSNE_points)

        ###################
        new_label=pred.reshape(len(pred),1)
        #print(new_label)
        new_label_pd = pd.DataFrame(new_label, columns=['new_label'])
        #print(new_label_pd)
        label_sample0 = new_label_pd[new_label_pd['new_label']==0]
        label_sample1 = new_label_pd[new_label_pd['new_label']==1]
        label_sample2 = new_label_pd[new_label_pd['new_label']==2]
        label_sample3 = new_label_pd[new_label_pd['new_label']==3]
        label_sample4 = new_label_pd[new_label_pd['new_label']==4]
        label_sample5 = new_label_pd[new_label_pd['new_label']==5]
        label_sample6 = new_label_pd[new_label_pd['new_label']==6]
        label_sample7 = new_label_pd[new_label_pd['new_label']==7]
        label_sample8 = new_label_pd[new_label_pd['new_label']==8]
        label_sample9 = new_label_pd[new_label_pd['new_label']==9]
        label_sample10 = new_label_pd[new_label_pd['new_label']==10]
        label_sample11 = new_label_pd[new_label_pd['new_label']==11]
        label_sample12 = new_label_pd[new_label_pd['new_label']==12]
        label_sample13 = new_label_pd[new_label_pd['new_label']==13]
        label_sample14 = new_label_pd[new_label_pd['new_label']==14]
        label_sample15 = new_label_pd[new_label_pd['new_label']==15]
        label_sample16 = new_label_pd[new_label_pd['new_label']==16]
        label_sample17 = new_label_pd[new_label_pd['new_label']==17]
        label_sample18 = new_label_pd[new_label_pd['new_label']==18]
        label_sample19 = new_label_pd[new_label_pd['new_label']==19]
        label_sample20 = new_label_pd[new_label_pd['new_label']==20]
        label_sample21 = new_label_pd[new_label_pd['new_label']==21]
        label_sample22 = new_label_pd[new_label_pd['new_label']==22]
        label_sample23 = new_label_pd[new_label_pd['new_label']==23]
        label_sample24 = new_label_pd[new_label_pd['new_label']==24]
        label_sample25 = new_label_pd[new_label_pd['new_label']==25]
        label_sample26 = new_label_pd[new_label_pd['new_label']==26]
        label_sample27 = new_label_pd[new_label_pd['new_label']==27]
        label_sample28 = new_label_pd[new_label_pd['new_label']==28]
        label_sample29 = new_label_pd[new_label_pd['new_label']==29]
   

    
        
        index0=list(map(str,label_sample0.index))
        index1=list(map(str,label_sample1.index))
        index2=list(map(str,label_sample2.index))
        index3=list(map(str,label_sample3.index))
        index4=list(map(str,label_sample4.index))
        index5=list(map(str,label_sample5.index))
        index6=list(map(str,label_sample6.index))
        index7=list(map(str,label_sample7.index))
        index8=list(map(str,label_sample8.index))
        index9=list(map(str,label_sample9.index))
        index10=list(map(str,label_sample10.index))
        index11=list(map(str,label_sample11.index))
        index12=list(map(str,label_sample12.index))
        index13=list(map(str,label_sample13.index))
        index14=list(map(str,label_sample14.index))
        index15=list(map(str,label_sample15.index))
        index16=list(map(str,label_sample16.index))
        index17=list(map(str,label_sample17.index))
        index18=list(map(str,label_sample18.index))
        index19=list(map(str,label_sample19.index))
        index20=list(map(str,label_sample20.index))
        index21=list(map(str,label_sample21.index))
        index22=list(map(str,label_sample22.index))
        index23=list(map(str,label_sample23.index))
        index24=list(map(str,label_sample24.index))
        index25=list(map(str,label_sample25.index))
        index26=list(map(str,label_sample26.index))
        index27=list(map(str,label_sample27.index))
        index28=list(map(str,label_sample28.index))
        index29=list(map(str,label_sample29.index))
       
        
        nodes=list(map(str,new_label_pd.index))
        combination0 = list(combinations(index0, 2))
        combination1 = list(combinations(index1, 2))
        combination2 = list(combinations(index2, 2))
        combination3 = list(combinations(index3, 2))
        combination4 = list(combinations(index4, 2))
        combination5 = list(combinations(index5, 2))
        combination6 = list(combinations(index6, 2))
        combination7 = list(combinations(index7, 2))
        combination8 = list(combinations(index8, 2))
        combination9 = list(combinations(index9, 2))
        combination10 = list(combinations(index10, 2))
        combination11 = list(combinations(index11, 2))
        combination12 = list(combinations(index12, 2))
        combination13 = list(combinations(index13, 2))
        combination14 = list(combinations(index14, 2))
        combination15 = list(combinations(index15, 2))
        combination16 = list(combinations(index16, 2))
        combination17 = list(combinations(index17, 2))
        combination18 = list(combinations(index18, 2))
        combination19 = list(combinations(index19, 2))
        combination20 = list(combinations(index20, 2))
        combination21 = list(combinations(index21, 2))
        combination22 = list(combinations(index22, 2))
        combination23 = list(combinations(index23, 2))
        combination24 = list(combinations(index24, 2))
        combination25 = list(combinations(index25, 2))
        combination26 = list(combinations(index26, 2))
        combination27 = list(combinations(index27, 2))
        combination28 = list(combinations(index28, 2))
        combination29 = list(combinations(index29, 2))

        graph = {'nodes': nodes,
         'edges': combination0+combination1+combination2+combination3+combination4+combination5+combination6+combination7
                 +combination8+combination9+combination10+combination11+combination12+combination13+combination14+combination15
                 +combination16+combination17+combination18+combination19+combination20+combination21+combination22+combination23
                 +combination24+combination25+combination26+combination27+combination28+combination29}
        n = len(graph['nodes'])

        ###

        
        adj_matr_kmean = pd.DataFrame(0, columns=graph['nodes'], index=graph['nodes'])


        for k in graph['edges']:
            adj_matr_kmean.at[k[0], k[1]] = 1
            adj_matr_kmean.at[k[1], k[0]] = 1
        #print(adj_matr_kmean)
        
        adj_matrs_kmean.append(adj_matr_kmean)

##        print(adj_matr_kmean)
       
        

        ###################
        
        
        centers = kmeans.cluster_centers_
##            ax[i, j].scatter(x=tSNE_points['x'], y=tSNE_points['y'], c=pred)
##            ax[i, j].scatter(x=centers[:, 0], y=centers[:, 1], marker='D', c='b')

        #print(sum(adj_matrs_kmean))
        #plt.show(block=False)   ######### Kmean plot

        ### Until KMeans
        ######################################################################################################
    fig, ax2 = plt.subplots(2, 10, figsize=(20, 5))
    fig, ax3 = plt.subplots(2, 10, figsize=(20, 5))
    adj_matrs_hier=[]
    for i in range(0, 20):
##        for j in range(0, 10):
        Select_file = Select_file_T.transpose()
        
##        Sample = Select_file.sample(int(len(Select)*0.7), random_state = i)  ########################################## feature Sampling
        Sample = Select_file.sample(int(len(Select)*0.7))  ########################################## feature Sampling

        Sample_T = Sample.transpose()
        points = tsne.fit_transform(Sample_T)
##        print(points)
##            ax3[i, j].scatter(x=points[:, 0], y=points[:, 1], c=file_label_num['target'], cmap=CMAP, s=30) ########### True label

        #####################################################################
        
        tSNE_points = pd.DataFrame(points, columns=['x', 'y'])
        '''
        kmeans = KMeans(n_clusters=20, random_state=0, algorithm='auto')     #n_clusters=len(file_label['target'].unique())
        kmeans.fit(PCA_Kmeans)
        pred = kmeans.predict(PCA_Kmeans)
        '''
        #####################################################################
        hierchical_cluster = AgglomerativeClustering(n_clusters=30)           ########30_similar####### 
        hier_label = hierchical_cluster.fit_predict(points)

        #####################################################################
        new_label=hier_label.reshape(len(hier_label),1)
        #print(new_label)
        new_label_pd = pd.DataFrame(new_label, columns=['new_label'])
        #print(new_label_pd)
        label_sample0 = new_label_pd[new_label_pd['new_label']==0]
        label_sample1 = new_label_pd[new_label_pd['new_label']==1]
        label_sample2 = new_label_pd[new_label_pd['new_label']==2]
        label_sample3 = new_label_pd[new_label_pd['new_label']==3]
        label_sample4 = new_label_pd[new_label_pd['new_label']==4]
        label_sample5 = new_label_pd[new_label_pd['new_label']==5]
        label_sample6 = new_label_pd[new_label_pd['new_label']==6]
        label_sample7 = new_label_pd[new_label_pd['new_label']==7]
        label_sample8 = new_label_pd[new_label_pd['new_label']==8]
        label_sample9 = new_label_pd[new_label_pd['new_label']==9]
        label_sample10 = new_label_pd[new_label_pd['new_label']==10]
        label_sample11 = new_label_pd[new_label_pd['new_label']==11]
        label_sample12 = new_label_pd[new_label_pd['new_label']==12]
        label_sample13 = new_label_pd[new_label_pd['new_label']==13]
        label_sample14 = new_label_pd[new_label_pd['new_label']==14]
        label_sample15 = new_label_pd[new_label_pd['new_label']==15]
        label_sample16 = new_label_pd[new_label_pd['new_label']==16]
        label_sample17 = new_label_pd[new_label_pd['new_label']==17]
        label_sample18 = new_label_pd[new_label_pd['new_label']==18]
        label_sample19 = new_label_pd[new_label_pd['new_label']==19]
        label_sample20 = new_label_pd[new_label_pd['new_label']==20]
        label_sample21 = new_label_pd[new_label_pd['new_label']==21]
        label_sample22 = new_label_pd[new_label_pd['new_label']==22]
        label_sample23 = new_label_pd[new_label_pd['new_label']==23]
        label_sample24 = new_label_pd[new_label_pd['new_label']==24]
        label_sample25 = new_label_pd[new_label_pd['new_label']==25]
        label_sample26 = new_label_pd[new_label_pd['new_label']==26]
        label_sample27 = new_label_pd[new_label_pd['new_label']==27]
        label_sample28 = new_label_pd[new_label_pd['new_label']==28]
        label_sample29 = new_label_pd[new_label_pd['new_label']==29]
    
        
        index0=list(map(str,label_sample0.index))
        index1=list(map(str,label_sample1.index))
        index2=list(map(str,label_sample2.index))
        index3=list(map(str,label_sample3.index))
        index4=list(map(str,label_sample4.index))
        index5=list(map(str,label_sample5.index))
        index6=list(map(str,label_sample6.index))
        index7=list(map(str,label_sample7.index))
        index8=list(map(str,label_sample8.index))
        index9=list(map(str,label_sample9.index))
        index10=list(map(str,label_sample10.index))
        index11=list(map(str,label_sample11.index))
        index12=list(map(str,label_sample12.index))
        index13=list(map(str,label_sample13.index))
        index14=list(map(str,label_sample14.index))
        index15=list(map(str,label_sample15.index))
        index16=list(map(str,label_sample16.index))
        index17=list(map(str,label_sample17.index))
        index18=list(map(str,label_sample18.index))
        index19=list(map(str,label_sample19.index))
        index20=list(map(str,label_sample20.index))
        index21=list(map(str,label_sample21.index))
        index22=list(map(str,label_sample22.index))
        index23=list(map(str,label_sample23.index))
        index24=list(map(str,label_sample24.index))
        index25=list(map(str,label_sample25.index))
        index26=list(map(str,label_sample26.index))
        index27=list(map(str,label_sample27.index))
        index28=list(map(str,label_sample28.index))
        index29=list(map(str,label_sample29.index))
       
        
        nodes=list(map(str,new_label_pd.index))
        combination0 = list(combinations(index0, 2))
        combination1 = list(combinations(index1, 2))
        combination2 = list(combinations(index2, 2))
        combination3 = list(combinations(index3, 2))
        combination4 = list(combinations(index4, 2))
        combination5 = list(combinations(index5, 2))
        combination6 = list(combinations(index6, 2))
        combination7 = list(combinations(index7, 2))
        combination8 = list(combinations(index8, 2))
        combination9 = list(combinations(index9, 2))
        combination10 = list(combinations(index10, 2))
        combination11 = list(combinations(index11, 2))
        combination12 = list(combinations(index12, 2))
        combination13 = list(combinations(index13, 2))
        combination14 = list(combinations(index14, 2))
        combination15 = list(combinations(index15, 2))
        combination16 = list(combinations(index16, 2))
        combination17 = list(combinations(index17, 2))
        combination18 = list(combinations(index18, 2))
        combination19 = list(combinations(index19, 2))
        combination20 = list(combinations(index20, 2))
        combination21 = list(combinations(index21, 2))
        combination22 = list(combinations(index22, 2))
        combination23 = list(combinations(index23, 2))
        combination24 = list(combinations(index24, 2))
        combination25 = list(combinations(index25, 2))
        combination26 = list(combinations(index26, 2))
        combination27 = list(combinations(index27, 2))
        combination28 = list(combinations(index28, 2))
        combination29 = list(combinations(index29, 2))

        graph = {'nodes': nodes,
         'edges': combination0+combination1+combination2+combination3+combination4+combination5+combination6+combination7
                 +combination8+combination9+combination10+combination11+combination12+combination13+combination14+combination15
                 +combination16+combination17+combination18+combination19+combination20+combination21+combination22+combination23
                 +combination24+combination25+combination26+combination27+combination28+combination29}
        n = len(graph['nodes'])

        ###

        
        adj_matr_hier = pd.DataFrame(0, columns=graph['nodes'], index=graph['nodes'])


        for k in graph['edges']:
            adj_matr_hier.at[k[0], k[1]] = 1
            adj_matr_hier.at[k[1], k[0]] = 1
        #print(adj_matr_hier)
        
        adj_matrs_hier.append(adj_matr_hier)
##        print(adj_matr_hier)
       
        

        ###################
        
        
        #centers = kmeans.cluster_centers_
        
##            ax2[i, j].scatter(x=tSNE_points['x'], y=tSNE_points['y'], c=hier_label)
##            ax[i, j].scatter(x=centers[:, 0], y=centers[:, 1], marker='D', c='b')

      
    adj_matrs=sum(adj_matrs_hier)+sum(adj_matrs_kmean)

##    print(adj_matrs)

       

    plt.close('all')

    A = np.array(adj_matrs)


    print(A)

    ###########################################################################

##    B = np.array(adj_matrs_hier[1])
        



##    G = nx.from_numpy_matrix(A)             ############## A is weighted adjacency matrix
##    adj = nx.adjacency_matrix(G)
    #adj = nx.adjacency_matrix(nx.from_numpy_matrix(A))

    #######################IDENTIFY FEATURE#########################
##    features = np.identity(adj.shape[0])
##    features = nx.adjacency_matrix(nx.from_numpy_matrix(features))
##    features = torch.FloatTensor(np.array(features.todense()))
    ##################################################################


############################################## NO LOAD adj_END #######################################################

    
##    A = np.load('D:/FINAL_PROJECT/adj_30_cluster/#######New_PROJECT######/adj_list/usoskin_adj0.npy')

##    A0 = np.load('D:/PR_FINAL_PROJECT/adj_30_cluster/usoskin_adj0.npy')
##    A1 = np.load('D:/PR_FINAL_PROJECT/adj_30_cluster/usoskin_adj1.npy')
##    A2 = np.load('D:/PR_FINAL_PROJECT/adj_30_cluster/usoskin_adj2.npy')
##    A3 = np.load('D:/PR_FINAL_PROJECT/adj_30_cluster/usoskin_adj3.npy')
##    A4 = np.load('D:/PR_FINAL_PROJECT/adj_30_cluster/usoskin_adj4.npy')
##    A5 = np.load('D:/PR_FINAL_PROJECT/adj_30_cluster/usoskin_adj5.npy')
##    A6 = np.load('D:/PR_FINAL_PROJECT/adj_30_cluster/usoskin_adj6.npy')
##    A7 = np.load('D:/PR_FINAL_PROJECT/adj_30_cluster/usoskin_adj7.npy')
##    A8 = np.load('D:/PR_FINAL_PROJECT/adj_30_cluster/usoskin_adj8.npy')
##    A9 = np.load('D:/PR_FINAL_PROJECT/adj_30_cluster/usoskin_adj9.npy')
##
##    A = A0 #+ A1+ A2+ A3+ A4+ A5+ A6+ A7+ A8+ A9 

    
    G = nx.from_numpy_matrix(A)
    adj = nx.adjacency_matrix(G)

    ################### Tensorflow ################################
    
    Select_feature = Sort.head(10)        ######## default 20
    densed_file = T[Select_feature.index]
    features_np = np.array(densed_file).astype(np.int8)
    features = sp.csr_matrix(features_np)
##    features = nx.adjacency_matrix(nx.from_numpy_matrix(features_np))

    ################################################################


    ################### PYTORCH ################################

##    #good_features => Sort
##    Select_feature = Sort.head(20)
##    densed_file = T[Select_feature.index]
##    features_np = np.array(densed_file).astype(np.int8)
##    features_float = torch.from_numpy(features_np)
##    features = features_float.float()

    ################################################################

    ###### Solve float error ######
##    features_np = np.array(densed_file)
##    features_float = torch.from_numpy(features_np)
##    features = features_float.float()


    n_clusters = len(file_label['target'].unique())
##    true_label = list(file_label['target'])


    #########################  COLOR SCATTER_TRUE_LABEL If load the adj   ########################
##    num_list=list(range(0,len(file_label['target'].unique())))
##
##    cell_type=file_label['target'].unique()
##    num_label = list(file_label['target'])
##
##    for index, value in enumerate(num_label):
##        for i in range(0,len(num_list)):
##            if value == cell_type[i]:       
##                num_label[index] = num_list[i]
##
##    #print(cell_type)
##    #print(num_label)
##    file_label_num = pd.DataFrame(num_label,columns=['target'])
##    true_label = file_label_num['target']
    #########################################################################




    return adj, features, n_clusters, true_label, points_for_WCSS





