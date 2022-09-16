import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.svm import SVC



def svm_m(names, PRED_elbow_kmean):
    svm_file = pd.read_csv('.../Norm_data_Umap/Norm_data_{}.txt'.format(names), sep='\t')
    df_set = svm_file[['0','1']]
    svm_lb = du = pd.DataFrame(PRED_elbow_kmean, columns = ['label'])
    X_train, X_test, y_train, y_test = train_test_split(df_set, svm_lb, test_size = 0.2, random_state = 0)   ### auto shuffled
    clf = svm.SVC(kernel='linear')
    # Train classifier 
    clf.fit(X_train, y_train)
    clf_predictions = clf.predict(X_test)
    print("Accuracy: {}%".format(clf.score(X_test, y_test) * 100 ))
    svm_accuracy = clf.score(X_test, y_test) * 100

    
    return svm_accuracy
