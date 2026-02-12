# clusterTools
The package “clusterTools” contains functions for cluster analysis (include Diana Clustering Algoritm, Hybrid hierarchical k-means clustering algoritm and Searching for optimal count of clusters for a Data Set).

In particular, the functions are provided for identifying in the data of the trends to clustering: for сalculate the Hopkins’ statistic and for visualize Principal component analysis (PCA).
Function 'dist' computes and returns the distance matrix computed by using the specified distance measure to compute the distances between the rows of a dataframe.
Identification and visualize the optimal number of clusters using different methods: within cluster sums of squares, average silhouette and gap statistics. NbClust function provides 30 indices for determining the number of clusters and proposes to user the best clustering scheme from the different results obtained by varying all combinations of number of clusters, distance measures, and clustering methods.
The package also includes a functions for Heuristic Identification of Noisy Variables (HINoV) method for clustering and Identification of differences between signs of data set clusters based on the T-test.
Create linkage matrix for plot the dendrogram and plot the hierarchical clustering as a dendrogram. Also may be returned of mean within-cluster distances and mean inter-cluster distances.
Two dendrograms can be compared using coefficient of measures entanglement between two dendrograms and draw a tanglegram plot of a side by side trees.
It's finally, possible visualization of partitioning methods including K-means, K-medoids, CLARA, AGNES, DIANA. An ellipse is drawn around each cluster.
