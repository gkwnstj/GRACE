from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


def elbow_method(emb):
    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
    visualizer.fit(emb)        # Fit data to visualizer
    print("Elbow Method : ", visualizer.elbow_value_)
    elbow = visualizer.elbow_value_

    return elbow
