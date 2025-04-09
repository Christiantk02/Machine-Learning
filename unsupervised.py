from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import confusion_matrix, silhouette_score

def unsupervised(x_train, x_test, y_test):

    print("\n<-------------------unsupervised--------------------->")

    #Train the KMeans model
    KMmodel1 = KMeans(n_clusters=2).fit(x_train)
    KMmodel2 = KMeans(n_clusters=3).fit(x_train)
    KMmodel3 = KMeans(n_clusters=4).fit(x_train)
    KMmodel4 = KMeans(n_clusters=5).fit(x_train)
    KMmodel5 = KMeans(n_clusters=6).fit(x_train)

    #Train the AgglomerativeClustering model
    ACmodel1 = AgglomerativeClustering(n_clusters=2).fit(x_train)
    ACmodel2 = AgglomerativeClustering(n_clusters=4).fit(x_train)
    ACmodel3 = AgglomerativeClustering(n_clusters=6).fit(x_train)

    # Predict the clusters for the test set using KMeans and calculate silhouette scores
    print("\nKMeans Silhouette Scores:")
    
    y_kmeans = KMmodel1.predict(x_test)
    KMcm1 = confusion_matrix(y_test, y_kmeans)
    print("KMmodel1 (k=2):", silhouette_score(x_test, y_kmeans))

    y_kmeans = KMmodel2.predict(x_test)
    KMcm2 = confusion_matrix(y_test, y_kmeans)
    print("KMmodel2 (k=3):", silhouette_score(x_test, y_kmeans))

    y_kmeans = KMmodel3.predict(x_test)
    KMcm3 = confusion_matrix(y_test, y_kmeans)
    print("KMmodel3 (k=4):", silhouette_score(x_test, y_kmeans))

    y_kmeans = KMmodel4.predict(x_test)
    KMcm4 = confusion_matrix(y_test, y_kmeans)
    print("KMmodel4 (k=5):", silhouette_score(x_test, y_kmeans))

    y_kmeans = KMmodel5.predict(x_test)
    KMcm5 = confusion_matrix(y_test, y_kmeans)
    print("KMmodel5 (k=6):", silhouette_score(x_test, y_kmeans))

    # Predict the clusters for the test set using AgglomerativeClustering
    print("\nAgglomerativeClustering Silhouette Scores:")
    y_agg = ACmodel1.fit_predict(x_test)
    ACcm1 = confusion_matrix(y_test, y_agg)
    print("ACmodel1 (k=2):", silhouette_score(x_test, y_agg))

    y_agg = ACmodel2.fit_predict(x_test)
    ACcm2 = confusion_matrix(y_test, y_agg)
    print("ACmodel2 (k=4):", silhouette_score(x_test, y_agg))

    y_agg = ACmodel3.fit_predict(x_test)
    ACcm3 = confusion_matrix(y_test, y_agg)
    print("ACmodel3 (k=6):", silhouette_score(x_test, y_agg))

    print("\n<-------------------------------------------------->")

    return KMcm1, KMcm2, KMcm3, KMcm4, KMcm5, ACcm1, ACcm2, ACcm3