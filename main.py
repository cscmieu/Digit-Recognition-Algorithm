import numpy as np # type: ignore
from sklearn.cluster import KMeans # type: ignore
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

from ucimlrepo import fetch_ucirepo  # type: ignore
import matplotlib.pyplot as plt 
# fetch dataset 
optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80) 

# data (as pandas dataframes) 
X = optical_recognition_of_handwritten_digits.data.features 
y = optical_recognition_of_handwritten_digits.data.targets


data = np.concatenate((X, y.values.reshape(-1, 1)), axis=1)

pca = PCA(2)
 
#Transform the data
df = pca.fit_transform(data)
 
df.shape

# print(X)
# print(y.values.reshape(-1,1))
# # Preprocess the data
# data = np.concatenate((X, y.values.reshape(-1, 1)), axis=1)
# print("============ data concatenate =============")
# print(data)
# data.shape

#Initialize the class object
kmeans = KMeans(n_clusters= 10)
 
#predict the labels of clusters.
label = kmeans.fit_predict(df)

centroids = kmeans.cluster_centers_
u_labels = np.unique(label)
 
#plotting the results:
 
# for i in u_labels:
    # plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)


# plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'black')
# plt.legend()
# plt.show()

prediction_cluster = kmeans.predict(df)
print("===================DATA====================")
print(data)

print("=================== DF ====================")
print(df)


print("=================== TARGET ====================")
print(y)


print("=================== PREDICT ====================")
print(prediction_cluster)
print(prediction_cluster.shape[0])

for j in range(10):
    list_cluster = np.zeros(10)
    for i in range(prediction_cluster.shape[0]):

        if prediction_cluster[i] == j :
            list_cluster[y.values[i]] += 1
    plt.clf()
    plt.title(f"histogramme du cluster n°{j}")
    
    plt.stairs(list_cluster,[i for i in range(11)])
    plt.savefig(f"C:\\Users\\alexa\\Documents\\GitHub\\Digit-Recognition-Algorithm\\Cluster{j}")

