# K-Means Clustering
# Customer Segmentation with K-Means

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D 


'''

Imagine that you have a customer dataset, and you need to apply customer segmentation on this historical data. 
Customer segmentation is the practice of partitioning a customer base into groups of individuals that have similar characteristics. 
It is a significant strategy as a business can target these specific groups of customers and effectively allocate marketing resources. 
For example, one group might contain customers who are high-profit and low-risk, that is, more likely to purchase products, or subscribe for a service. 
A business task is to retain those customers. 
Another group might include customers from non-profit organizations and so on.
'''

path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv'

cust_df = pd.read_csv(path)
print(cust_df.head(6))


# Pre-processing - As you can see, Address in this dataset is a categorical variable. 
# The k-means algorithm isn't directly applicable to categorical variables because the Euclidean distance function isn't really meaningful for discrete variables. 
# So, let's drop this feature and run clustering.

df = cust_df.drop('Address', axis=1)


# Normalizing over the standard deviation
# Now let's normalize the dataset. But why do we need normalization in the first place? 
# Normalization is a statistical method that helps mathematical-based algorithms to interpret features with different magnitudes and distributions equally. 
# We use StandardScaler() to normalize our dataset.


X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet

'''
Modeling
In our example (if we didn't have access to the k-means algorithm), it would be the same as guessing that each customer group would have certain age, income, 
education, etc, with multiple tests and experiments. 
However, using the K-means clustering we can do all this process much easier.
Let's apply k-means on our dataset, and take a look at cluster labels.
'''

clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

# Insights - We assign the labels to each row in the dataframe.

df["Clus_km"] = labels
print(df.head(5))

# We can easily check the centroid values by averaging the features in each cluster.

df.groupby('Clus_km').mean()

# Now, let's look at the distribution of customers based on their age and income:


area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()


fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(float))


'''
k-means will partition your customers into mutually exclusive groups, for example, into 3 clusters. The customers in each cluster are similar to each other demographically. 
Now we can create a profile for each group, considering the common characteristics of each cluster. For example, the 3 clusters can be:
AFFLUENT, EDUCATED AND OLD AGED
MIDDLE AGED AND MIDDLE INCOME
YOUNG AND LOW INCOME
'''
