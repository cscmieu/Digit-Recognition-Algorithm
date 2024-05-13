import numpy as np # type: ignore
from sklearn.cluster import KMeans # type: ignore
from ucimlrepo import fetch_ucirepo  # type: ignore
  
# fetch dataset 
optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80) 
  
# data (as pandas dataframes) 
X = optical_recognition_of_handwritten_digits.data.features 
y = optical_recognition_of_handwritten_digits.data.targets 

# Preprocess the data
data = np.concatenate((X, y.values.reshape(-1, 1)), axis=1)

# Normalize the data
data = (data - data.mean()) / data.std()

# Remove the target column
X = data[:, :-1]

# Create an instance of the KMeans model and fit the data
kmeansInstance = KMeans().fit(X)

# Get the cluster labels for each data point
labels = kmeansInstance.labels_

# Get the cluster centers
centers = kmeansInstance.cluster_centers_

# Print the cluster labels and centers
print("Cluster Labels:", labels)
print("Cluster Centers:", centers)