import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter


def wcss_by_cluster(points_for_WCSS, PRED_elbow_kmean):
    
    norm_data = pd.DataFrame(points_for_WCSS)
    norm_data['label'] = PRED_elbow_kmean                 #########################
    Extract = norm_data.sort_values(by ='label',ascending=True)




    WCSS_result = []
    for i in range(0,len(Counter(PRED_elbow_kmean).keys())):     #########################

        Extract_0 = Extract[Extract['label']==i]
        Extract_del = Extract_0.drop(columns=['label'],axis=1)
        num_arr = np.array(Extract_del)
        WCSS_kmeans = KMeans(n_clusters=1, init = "k-means++")
        WCSS_kmeans.fit_predict(num_arr)
        inertia = WCSS_kmeans.inertia_
        WCSS_result.append(inertia)

    WCSSresult = sum(WCSS_result)
    WCSS_result.clear()

    return WCSSresult
