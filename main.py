from preprocessing import preprocessing
from supervised import supervised
from unsupervised import unsupervised
from plots import eda, heatmap

#Preprocessing
data, x_train, x_test, y_train, y_test = preprocessing()

#Data Exploration Plots
eda(data)

#Supervised Learning
DTcm1, DTcm2, DTcm3, KNcm1, KNcm2, KNcm3 = supervised(x_train, x_test, y_train, y_test)

#Unsupervised Learning
KMcm1, KMcm2, KMcm3, KMcm4, KMcm5, ACcm1, ACcm2, ACcm3 = unsupervised(x_train, x_test, y_test)

#Plotting Confusion Matrices Of Best Of Each Model
heatmap([DTcm1, KNcm1, KMcm1, ACcm1], ["Decision Tree", "KNN", "KMeans", "AgglomerativeClustering"])   

