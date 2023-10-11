#!/usr/bin/env python
"""
Tools for cluster analysis (include Diana Clustering Algoritm and Searching for optimal count of clusters for a Data Set)

Copyright (c) 2023, A.A. Bessonov (bestallv@mail.ru)

Routines in this module:

classAgreement(tab)
clusGap(x,FUNcluster,kmax,B=10,dpower=1,spaceH0='scaledPCA',showits=None)
cluster_features(data, k, distance=None, method='kmeans', index='cRAND',
          plot=False, labels=False, figsize=(7,5), **kwarg)
coef_entanglement(dend1, dend2, L=1.5, leaves_matching_method='order')
compact_hypothesis(X,cl)
DianaClustering()
diff_clust(X,cl,alpha=0.05,**kwargs)
dist(data,metric='euclidean',diag=True,upper=False,nadel=False)
fviz_cluster(data,cl,colors=None,showclcent=False,ellipse=None,ellipsetype='confidence',
          ellipselevel=2.0, title=None,grid=True,annotlab=None,pointsize=30,labelsize=10,
          alpha=0.2,figsize=(8,8),**kwargs)
fviz_dist(datadist,order=True,show_labels=False,lab_size=10,figsize=(9,8),cmap='coolwarm')
fviz_gap_stat(gap_stat,method='firstSEmax',SEfactor=1)
fviz_nbclust(data,FUNcluster,type_test='silhouette',kmax=10,nboot=10,showits=True,**kwargs)
fviz_pca_ind(X,func,figsize=(8,8),alpha=1,**kwargs)
fviz_silhouette(X,FUNcluster,n_clust,figsize=(9,7))
gap_stat(x,kmax=10,B=10,showits=None)
hKMeans(data, k, halgoritm='agnes', method='average', metric='euclidean',
          algorithm='auto', max_iter=300)
hopkins_pval(x,n)
hopkins_stat(X)
hopkins_test(X,m=None)
linkage_matrix(model)
NbClust(data=None,diss=None,distance ="euclidean",min_nc=2,max_nc=15,method=None,
          index=['all'],alphaBeale=0.1,plotInd=False,PrintRes=True,n_init=10)
NbClustViz(res,figsize = (10,6))
plot_dendrogram(Z,k=None,annotate=False,fontsize=10,figsize=(10,7),title=None,labelsize=12,**kwargs)
plot_tanglegram(model1, model2, L=1.5, leaves_matching_method='order', figsize=(12,7),
          fontsize=12, **kwargs)
scale(data, center=True, scale=True)


"""
from __future__ import division, absolute_import, print_function

__all__ = ['classAgreement', 'clusGap', 'cluster_features', 'coef_entanglement',
           'compact_hypothesis', 'DianaClustering', 'diff_clust', 'dist', 'fviz_cluster',
           'fviz_dist', 'fviz_gap_stat','fviz_nbclust', 'fviz_pca_ind', 'fviz_silhouette',
           'gap_stat', 'hKMeans', 'hopkins_pval', 'hopkins_stat', 'hopkins_test',
           'linkage_matrix', 'NbClust', 'NbClustViz', 'plot_dendrogram', 'plot_tanglegram',
           'scale']

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from random import sample
from numpy.random import uniform
from sklearn import metrics
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import sys
import time
import math
from scipy.stats import f
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from scipy.stats import uniform
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm
from typing import Iterable
from itertools import chain
from scipy.stats import beta, uniform
from numpy.linalg import eig
from scipy import cluster
from sklearn_extra.cluster import KMedoids
from sklearn_extra.cluster import CLARA
from scipy.spatial.distance import pdist
import warnings
from scipy.stats import beta


def classAgreement(tab):
    """
    Computes several coefficients of agreement between the columns and rows of a 2-dimensional array.
   
    Parameters
    ----------
    tab : [array] A 2-dimensional array.
    
    Returns
    ----------
    Dictionary with the following elements:
    diag : Percentage of data points in the main diagonal of tab.
    kappa : diag corrected for agreement by chance.
    rand : Rand index.
    crand : Rand index corrected for agreement by chance.
    """
    n=int(np.sum(tab))
    ni=pd.Series(tab.sum(axis=1),dtype='int')
    nj=pd.Series(tab.sum(axis=0),dtype='int')
    p0=np.sum(np.diag(tab))/n
    pc=np.sum(ni*nj)/n**2
    n2=math.comb(n, 2)
    rand=1+(np.sum(tab**2)-(np.sum(ni**2)+np.sum(nj**2))/2)/n2
    nis2=np.sum(ni[ni>1].agg(lambda x: math.comb(x, 2)))
    njs2=np.sum(nj[nj>1].agg(lambda x: math.comb(x, 2)))
    tkk = pd.Series([x for l in tab for x in l]).astype('int')
    crand=(np.sum(tkk[tkk>1].agg(lambda x: math.comb(x, 2)))-(nis2*njs2)/n2)/((nis2+njs2)/2-(nis2*njs2)/n2)
    res={'diag':p0,'kappa':(p0-pc)/(1-pc),'rand':rand,'crand':crand}
    return res


def clusGap(x,FUNcluster,kmax,B=10,dpower=1,spaceH0='scaledPCA',showits=None):
    """
    Calculates a goodness of clustering measure, the “gap” statistic. For each number 
    of clusters k, it compares log(W(k)) with E*[log(W(k))] where the latter is defined 
    via bootstrapping, i.e., simulating from a reference (H_0) distribution, a uniform 
    distribution on the hypercube determined by the ranges of x, after first centering, 
    and then svd (aka ‘PCA’)-rotating them when (as by default) spaceH0 = "scaledPCA".

    Parameters
    ----------
    x : [DataFrame] dataframe for calculates a goodness of clustering measure, the "gap" statistic.
    FUNcluster : a function which returns a list with a component named cluster which is a vector 
        of length n = nrow(x) of integers in 1:k determining the clustering or grouping of the n observations.
    kmax : [int] the maximum number of clusters to consider, must be at least two.
    B : [int] number of Monte Carlo ("bootstrap") samples.
    dpower : [int] a positive integer specifying the power p which is applied to the euclidean 
        distances (dist) before they are summed up to give W(k). The default, dpower = 1.
    spaceH0 : [str] string specifying the space of the H_0 distribution (of no cluster). 
        Both 'scaledPCA' (default) and 'original' use a uniform distribution in a hyper cube.
    showits : [bool] if True, reports the iterations of Bootstrapping so the user can monitor 
        the progress of the algorithm (default showits=None).

    Returns
    ----------
    a dictionary with components with 'kmax' values of the simulated ("bootstrapped"):
    'gap':   the gap statistics.
    'logW':  logarithm of the sum of the within-dispersion measures for data set.
    'ElogW': logarithm of the sum of the within-dispersion measures for uniform distribution data sets.
    'SEsim': the standard deviation.
    """
    if kmax<2:
        raise ValueError('kmax must be > = 2')
    if B<=0:
        raise ValueError('B has to be a positive integer')
    if x.ndim<=1 | x.ndim<=1:
        raise ValueError('x must be shape is row > 1 and columns > 1')
    if isinstance(x, pd.DataFrame):
        x=x.values
    if isinstance(x, np.ndarray):
        x=x
    n=x.shape[0]

    def scale(y,c=True,sc=True):
        """
        Generic function whose default method centers and/or 
        scales the columns of a numeric matrix.
        """
        x=y.copy()
        xsc=x.mean()
        if c:
            x -= x.mean()
        if sc and c:
            x /= x.std()
        elif sc:
            x /= np.sqrt(x.pow(2).sum().div(x.count() - 1))
        return x, xsc

    def Wk(X,K,FUNcluster,dpower):
        n=X.shape[0]
        if K>1:
            cluster=FUNcluster
            cluster.n_clusters=K
            clus=cluster.fit_predict(X)
        else:
            clus=np.array([0]*n)
        sxs=[]
        for i in range(K):
            xs=X[np.where(clus==i)[0]]
            sxs.append(np.sum(dist(xs)**dpower/xs.shape[0]))
        res=np.sum(sxs)*0.5
        return res

    def progress(it,total,buffer=30):
        """
        A progress bar is used to display the progress of a long running Iterations of 
        Bootstrapping, providing a visual cue that processing is underway.
        """
        percent = 100.0*it/(total+1)
        sys.stdout.write('\r')
        sys.stdout.write("Bootstrapping: [\033[34m{:{}}] {:>3}% ".format('█'*int(percent/(100.0/buffer)),buffer, int(percent)))
        sys.stdout.flush()
        time.sleep(0.001)

    # Clustering original Data
    logW=[]
    for k in range(1,kmax+1):
        logW.append(np.log(Wk(x,k,FUNcluster,dpower)))
    # Scale 'x' into hypercube -- later fill with H0-generated data
    xs=scale(pd.DataFrame(x),sc=False)
    mx=xs[1]
    if spaceH0=='scaledPCA':
        Vsx=np.linalg.svd(xs[0])[2]
        xs=np.dot(xs[0],Vsx.T)
    if spaceH0=='original': # do nothing, use 'xs'
        xs=xs[0]
    # Clustering original and Reference Data
    rngx1=np.stack((np.amin(xs, 0),np.max(xs, 0)),axis=0)
    logWks=np.zeros((B, kmax))
    it=0
    for b in range(B):
        # Generate "H0"-data as "parametric bootstrap sample":
        z1=uniform.rvs(rngx1.min(axis=0),rngx1.max(axis=0),size=(n,xs.shape[1]))
        if spaceH0=='scaledPCA':
            z=np.dot(z1,Vsx)
            z=z+mx.values
        if spaceH0=='original':
            z=z1
            z=z+mx.values
        for k in range(1,kmax+1):
            logWks[b,k-1]=np.log(Wk(z,k,FUNcluster,dpower))
        it=it+1
        if showits==True:
            progress(it+1,B)
    ElogW=np.mean(logWks, axis = 0)
    SEsim=np.sqrt((1+1/B)*np.var(logWks, axis = 0))
    gap=ElogW-logW
    res={'gap':gap,'logW':np.array(logW),'ElogW':ElogW,'SEsim':SEsim}
    return res


def cluster_features(data, k, distance=None, method='kmeans', index='cRAND', plot=False, labels=False, figsize=(7,5), **kwarg):
    """
    Heuristic Identification of Noisy Variables (HINoV) method for clustering.

    Parameters
    ----------
    data : [DataFrame] the dataframe or array that has been used for clustering.
    k : [int] The number of clusters to form.
    distance : [str] the distance measure to be used to compute the dissimilarity matrix. 
                   This must be one of: 'euclidean' (default), 'cityblock', 'cosine', 'l1', 
                   'l2', 'manhattan' and other from scipy.spatial.distance.               
    method : [str] clustering method: 'kmeans' (default), 'pam','clara','single','complete',
                   'average','weighted','centroid','ward','median','diana'.
    index : [str] 'cRAND' - corrected Rand index (default), 'RAND' - Rand index.
    plot : [bool] if True, visualize results of ranked values of topri in decreasing order on plot. By default, plot=False.
    labels : [bool] If True, set the current names of columns of data as labels of the x-axis.
                    Use only for DataFrame. By default, labels=False. 
    figsize : [int, int] a method used to change the dimension of plot window, width, height in inches 
                   (default figsize=(7, 5)).
    **kwargs : other arguments for function matplotlib.pyplot.plot.

    Returns
    ----------
    Dictionary with the following elements:
    parim : m x m symmetric matrix (m - number of variables). Matrix contains pairwise corrected
            Rand (Rand) indices for partitions formed by the j-th variable with partitions formed
            by the l-th variable.
    topri : sum of rows of parim.
    stopri : ranked values of topri in decreasing order.

    If plot=True, visualize results of ranked values of topri in decreasing order on plot.
    """
    methods=['kmeans','pam','clara','single','complete','average','weighted','centroid','ward','median','diana']
    method_=list(x for x in range(len(methods)) if methods[x] in [method])
    if any(map(lambda v: v not in methods, [method])):
        raise ValueError('invalid method')
    if k<2:
        raise ValueError('The number of clusters must be at least equal to 2')
    if (3 in method_ or 4 in method_ or 5 in method_ or 6 in method_ or 7 in method_ or 8 in method_ or 9 in method_):
        if distance is None:
            raise ValueError('distance must be')
    if isinstance(data, pd.DataFrame):
        listf=data.columns.to_list()
        df=data.dropna()
        df=df.reset_index(drop=True)
        df=df.values
    if isinstance(data, np.ndarray):
        df=data[~np.isnan(data).any(axis=1)]
        if labels is True:
            raise ValueError('labels=true only for DataFrame')
    z=df.copy()
    klasyfikacje=[]
    for i in range(z.shape[1]):
        x=z[:,[i]]
        if distance is not None:
            d=dist(x,metric=distance)
        if (0 in method_):
            cl=KMeans(n_clusters=k,random_state=123,n_init='auto').fit(x).labels_
        if (1 in method_):
            if distance is not None:
                cl=KMedoids(n_clusters=k,random_state=123).fit(d).labels_
            if distance is None:
                cl=KMedoids(n_clusters=k,method='pam',random_state=123).fit(x).labels_
        if (2 in method_):
            if distance is not None:
                cl=CLARA(n_clusters=k,random_state=123).fit(d).labels_
            if distance is None:
                cl=CLARA(n_clusters=k,random_state=123).fit(x).labels_
        if (3 in method_ or 4 in method_ or 5 in method_ or 6 in method_ or 7 in method_ or 8 in method_ or 9 in method_):
            cl=cluster.hierarchy.cut_tree(cluster.hierarchy.linkage(d,method=method),n_clusters=k)
            cl=np.array([item for sublist in cl for item in sublist])
        if (10 in method_):
            if distance is not None:
                cl=DianaClustering(d,metric='euclidean').fit(k).cluster_labels_
            if distance is None:
                cl=DianaClustering(x,metric='euclidean').fit(k).cluster_labels_
        klasyfikacje.append(cl)
    klasyfikacje=pd.DataFrame(klasyfikacje).T
    wyn=np.zeros((z.shape[1],z.shape[1]))
    for i in range(z.shape[1]):
        for j in range(z.shape[1]):
            if i==j:
                wyn[i,j]=1
            else:
                tk=klasyfikacje.pivot_table(columns=[i,j],index=[i],aggfunc='size').values
                tk=np.nan_to_num(tk)
                w=classAgreement(tk)
                if index=='cRAND':
                    wyn[i,j]=w['crand']
                else:
                    wyn[i,j]=w['rand']
    pos=np.zeros((2,z.shape[1]))
    for k in range(z.shape[1]):
        pos[0,k]=k
        pos[1,k]=np.sum(wyn[k])-1
    topri=pos.copy()
    if z.shape[1]>1:
        for i in range(z.shape[1]):
            for j in range(z.shape[1]-1):
                if pos[1,j]<pos[1,j+1]:
                    p1=pos[0,j+1]
                    p2=pos[1,j+1]
                    pos[0,j+1]=pos[0,j]
                    pos[1,j+1]=pos[1,j]
                    pos[0,j]=p1
                    pos[1,j]=p2
    res={'parim':wyn,'topri':topri,'stopri':pos}
    if plot is True:
        x_=res.get('stopri')[0].astype('int')
        y=res.get('stopri')[1]
        ax=plt.figure(figsize=figsize,facecolor='w').gca()
        if labels is True:
            label=np.array(listf)[x_]
            plt.xticks(range(len(x_)), label, rotation=90)
        else:
            plt.xticks(range(len(x_)), x_)
        plt.plot(range(len(x_)), y, 'o-r', **kwarg)
        plt.xlim([min(x_)-0.1, max(x_)+0.1])
        plt.ylim([min(y)-0.1, max(y)+0.1])
        plt.xlabel('features')
        plt.ylabel('stopri')
        plt.grid(True)
    return res


def coef_entanglement(dend1, dend2, L=1.5, leaves_matching_method='order'):
    """
    Measures entanglement between two dendrograms.

    Return the measures the entanglement between two trees. Entanglement is a measure
    between 1 (full entan- glement) and 0 (no entanglement). The exact behavior of the
    number depends on the L norm which is chosen.

    Parameters
    ----------
    dend1, dend2 : [object] a dendrograms for comparison.
    L : [float, int] The distance norm to use for measuring the distance between the 
                    two dendrograms. It can be any positive number, often one will 
                    want to use 0, 1, 1.5 (default), 2.
    leaves_matching_method : [str] If 'order' (default) then use the old leaves order 
                    value for matching the leaves order value, if using "labels", then
                    use the labels for matching the leaves order value. 

    Returns
    ----------
    the value of measures the entanglement between two trees.
    """
    def mobl(dend1, dend2, check_match=True):
        """
        Takes one dendrogram and adjusts its order leaves valeus based on the order 
        of another dendrogram.

        Parameters
        ----------
        dend1, dend2 : [array] Cluster labels for each point or labels corresponding 
                        to the leaf nodes.
        check_match : [bool] If True to check that the labels in the two dendrogram match
                        (if they do not, the function aborts).
        Returns
        ----------
        dend1 after adjusting its order values to be like dend_template. 
        """
        tch_labels=dend1
        ttemp_labels=dend2
        if check_match:
            if sorted(tch_labels)!=sorted(ttemp_labels):
                raise ValueError('labels do not match in both trees')
        match = lambda a, b: [ b.index(x)+1 if x in b else None for x in a ]
        order_change=np.array(match(list(tch_labels),list(ttemp_labels)))
        tree_new_leaf_numbers=np.array(ttemp_labels)[order_change-1]
        return tree_new_leaf_numbers

    def modbo(dend1, dend2, dend_change, check_match=False):
        """
        Takes one dendrogram and adjusts its order leaves valeus based on the order
        of another dendrogram. The values are matached based on the order of
        the two dendrograms.

        Parameters
        ----------
        dend1, dend2 : [array] Cluster labels for each point or labels corresponding 
                        to the leaf nodes.
        dend_change : [array] The array with the order of leaves in dend1 (at least 
                        before it was changes for some reason). It is based on which adjust 
                        the new values of dend1.
        check_match : [bool] If True to check that the orders in the two dendrogram match
                        (if they do not, the function aborts).
        Returns
        ----------
        Returns dend1 after adjusting its order values to be like dend2.
        """
        tcho_labels=dend1
        tto_labels=dend2
        if check_match:
            if sorted(tcho_labels)!=sorted(tto_labels):
                raise ValueError('labels do not match in both trees')
        match = lambda a, b: [ b.index(x)+1 if x in b else None for x in a ]
        order_change=np.array(match(list(tcho_labels),list(dend_change)))
        new_leaves_order=np.array(tto_labels)[order_change-1]
        return new_leaves_order
    def sum_abs_diff_L(x, y, L):
        """
        Return sum of the absolute difference (each one in the power of L)

        Parameters
        ----------
        x, y : [array] Cluster labels for each point or labels corresponding 
                        to the leaf nodes.
        L : [float, int] The distance norm to use for measuring the distance between the 
                    two dendrograms. It can be any positive number, often one will 
                    want to use 0, 1, 1.5 (default), 2.

        Returns
        ----------
        Value of sum of the absolute difference between cluster labels for each point 
        or labels corresponding to the leaf nodes, of two dendrograms.
        """
        return sum(abs(np.array(x) - np.array(y))**L)

    n_leaves = len(dend1.labels_)
    one_to_n_leaves=np.array(list(range(n_leaves)))
    if leaves_matching_method=='order':
        Z=linkage_matrix(dend1)
        Z2=linkage_matrix(dend2)
        dn=dendrogram(Z,no_plot=True)
        dn2=dendrogram(Z2,no_plot=True)
        dend1_old_order=np.array(dn['leaves'])
        dend1=one_to_n_leaves
        dend2=modbo(np.array(dn2['leaves']), dend1, dend1_old_order)
    if leaves_matching_method=='labels':
        dend1=dend1.labels_
        dend2=mobl(dend2.labels_, dend1)
    entanglement_result=sum_abs_diff_L(dend1, dend2, L)
    worse_entanglement_result=sum_abs_diff_L(one_to_n_leaves, np.array(list(reversed(one_to_n_leaves))), L)
    ner = round(entanglement_result / worse_entanglement_result, 4)
    return ner


def compact_hypothesis(X,cl):
    """
    Return of mean within-cluster distances and mean inter-cluster distances.
    
    Parameters
    ----------
    X : [DataFrame, array] dataframe or array.
    cl : [array] vector of labels of clusters for X.
    
    Returns
    ----------
    Matrix of mean within-cluster distances and mean inter-cluster distances.
    """
    if isinstance(X, pd.DataFrame):
        X=X.values
    if isinstance(X, np.ndarray):
        X=X
    md=dist(X)
    n=max(cl)+1
    separation_matrix=np.zeros((n,n))
    for i in range(n):
        dmat=md[np.where(cl==i)[0]]
        dmat=dmat[:,np.where(cl==i)[0]]
        separation_matrix[i,i]=np.mean(dmat)
        for j in range(n):
            if i!=j:
                suv=md[np.where(cl==i)[0]]
                suv=suv[:,np.where(cl==j)[0]]
                separation_matrix[i,j]=np.mean(suv)
                separation_matrix[j,i]=np.mean(suv)
    SM=pd.DataFrame(separation_matrix)
    return SM


class DianaClustering:
    """
    Divisive Clustering.

    Computes a divisive hierarchical clustering of the dataset returning an linkage matrix
    and other results.

    Parameters
    ----------
    data : [DataFrame, array] dataframe or array for computes a divisive hierarchical clustering.
    metric : [str] the distance measure to be used to compute the dissimilarity matrix. 
                   This must be one of: 'euclidean' (default), 'cityblock', 'cosine', 'l1', 'l2', 
                   'manhattan' and other from scipy.spatial.distance.
    trace_lev : [bool] if True specifying a trace level for printing diagnostics during 
                   the algorithm. Default False does not print anything, 
                   by default trace_lev=False.
    keepdiss : [bool] if True the dissimilarities matrix should be kept in the result,
                   by default keepdiss=False.

    Attributes
    ----------
    linkage_matrix_ : ndarray of shape (n_samples-1, 4)
        linkage_matrix.
    merge_ : ndarray of shape (n_samples-1, 2)
        an (n-1) by 2 matrix, where n is the number of observations. Row i of merge
        describes the split at step n-i of the clustering. If a number j in row r 
        is negative, then the single observation |j| is split off at stage n-r. 
        If j is positive, then the cluster that will be splitted at stage n-j (described
        by row j), is split off at stage n-r.
    children_ : array-like of shape (n_samples-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.
    labels_ : ndarray of shape (n_samples)
        Array giving a permutation of the original observations to allow 
        for plotting, in the sense that the branches of a clustering tree 
        will not cross.
    cluster_labels_ : ndarray of shape (n_samples)
        Cluster labels for each point.
    distances_: array-like of shape (n_nodes-1,)
        Distances between nodes in the corresponding place in `children_`.
        Only computed if `distance_threshold` is used or `compute_distances`
        is set to `True`.
    feature_names_in_ : list of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `data`
        has feature names that are all strings.
    diss_ : array-like of shape (n_samples, n_samples)
        The dissimilarities matrix, representing the total dissimilarity 
        matrix of the dataset.
    divisive_coefficient_ : float
        The divisive coefficient, measuring the clustering structure of the 
        dataset. For each observation i, denote by d(i) the diameter of the 
        last cluster to which it belongs (before being split off as a single 
        observation), divided by the diameter of the whole dataset. The dc 
        is the average of all 1 − d(i). It can also be seen as the average 
        width (or the percentage filled) of the banner plot. Because dc grows 
        with the number of observations, this measure should not be used to 
        compare datasets of very different sizes.
    """
    from itertools import chain
    def __init__(self,data,metric='euclidean',trace_lev=False,keepdiss=False):
        self.data = data
        self.metric=metric
        self.trace_lev=trace_lev
        self.keepdiss=keepdiss
        self.n_samples, self.n_features = data.shape
    def __repr__(self):
        return f"DianaClustering(metric='{self.metric}', trace_lev={self.trace_lev}, keepdiss={self.keepdiss})"
    def get_params(self):
        args=['metric','trace_lev','keepdiss']
        args_names = inspect.getfullargspec(DianaClustering)[3]
        args_dict = {**dict(zip(args,args_names)),}
        return args_dict
    def ltuti(self,n):
        """
        Returns indexes for convert lower to upper matrix of pairwise distances
                between observations.

        Parameters
        ----------
        n : [int] The number of observations.

        Returns
        -------
        self : list
            Indexes for convert matrix.
        """
        def cumsum(n,k):
            rr=np.cumsum(np.array(range((n-k),(n-1)))[::-1])
            arr = np.append(np.array(0), rr)
            return arr
        a=list(np.repeat(range(n), range(n), axis=0))
        b=list(np.append(np.array(0),list(chain.from_iterable(map(cumsum,[n]*n,range(2,n))))))
        c=list(map(sum, zip(a,b)))
        return c
    def min_dis(self,dys, ka, kb, ner):
        """
        Сalculates the minimum distance between two data points.

        Parameters
        ----------
        dys : [array] The dissimilarities matrix.
        ka, kb : [int] Variables for iterations.
        ner : [array] Array giving a permutation of the original observations.

        Returns
        -------
        self : int
            The value of the minimum distance between two data points.
        """
        dm = 0.0
        k = ka-1
        while k < kb-1:
            ner_k = ner[k]
            j = k+1
            while j < kb:
                k_j = self.ind_2(ner_k, ner[j])
                if dm < dys[k_j]:
                    dm = dys[k_j]
                j += 1
            k += 1
        return dm    
    def ind_2(self,l,j):
        """
        Returns the indices of original observations for the matrix of distances.

        Parameters
        ----------
        l, j : [array] The array of original observations.

        Returns
        -------
        self : int
            Index for the distance matrix.
        """
        if(l > j):
            return int((l-2)*(l-1)/2 + j)
        if(l == j):
            return int(0)
        else:
            return int((j-2)*(j-1)/2 + l)
    def bncoef(self,n, ban):
        """
        Compute divisive coefficient from the distance matrix.

        Parameters
        ----------
        n : [int] The number of observations in data.
        ban : [array] The distance matrix.

        Returns
        -------
        self : float
            Divisive Coefficient.
        """
        n_1 = n-1
        sup = 0. # sup := max_k ban[k]
        for k in range(1, n):
            if sup < ban[k]:
                sup = ban[k]
        cf = 0.
        k = 0
        while k < n:
            kearl = k if k > 0 else 1
            kafte = k+1 if k+1 < n else n_1
            syze = min(ban[kearl], ban[kafte])
            cf += (1. - syze / sup)
            k += 1
        return cf / n
    def fit(self,n_clusters=None):
        """
        Fit the hierarchical clustering from distance matrix.

        Parameters
        ----------
        n_clusters : [int] The number of clusters for computes a divisive 
                hierarchical clustering, by default n_clusters=None.

        Returns
        -------
        self : dict
            Returns the fitted instance.
        """
        from scipy.spatial.distance import pdist
        from scipy.cluster.hierarchy import cut_tree
        if isinstance(self.data, pd.DataFrame):
            feature_names=self.data.columns.to_list()
            self.data=self.data.values
        if isinstance(self.data, np.ndarray):
            self.data=self.data
            feature_names=list(range(self.data.shape[1]))
        v=pdist(self.data,metric=self.metric)
        dv=v[np.array(self.ltuti(self.data.shape[0]))-1]
        dys=np.append(np.array(0),dv)
        nn=self.n_samples
        n_1=nn-1
        if self.keepdiss == True:
            diss = np.full((nn, nn), 0, dtype = 'float')
            diss[np.tril_indices(diss.shape[0], k = -1)] = dys[1:]
            self.diss_ = diss
        # Parameter adjustments
        kwan=np.array([0]*(nn+1))
        ner=np.array([0]*(nn+1))
        ban=np.array([0]*(nn+1),dtype=float)
        merge=np.zeros((nn-1,2))
        merge_2=np.zeros((nn-1,2))
        mergefin=np.zeros((nn-1,2))
        # initialization
        nclu = 1
        nhalf = int(nn * n_1 / 2+1)
        for l in range(nn+1):
            kwan[l] = 0
            ban[l] = 0.
            ner[l] = l
        kwan[1] = nn
        ja = 1
        # cs := diameter of data set
        cs = 0.
        for k in range(nhalf):
            if cs < dys[k]:
                cs = dys[k]
        if self.trace_lev:
            print("C diana(): ndist= %d, diameter = %g\n"%( nhalf, cs))       
        # prepare for splitting
        split = True
        while split:        
            jb = ja + kwan[ja] - 1
            jma = jb
            if kwan[ja] == 2:
                # special case of a pair of objects
                kwan[ja] = 1
                kwan[jb] = 1
                ban[jb] = dys[self.ind_2(ner[ja], ner[jb])]
            else:
                # finding first object to be shifted
                bygsd = -1.
                lndsd = -1
                for l in range(ja, jb+1):
                    lner = ner[l]
                    sd = 0.
                    for j in range(ja, jb+1):
                        sd += dys[self.ind_2(lner, ner[j])]
                    if bygsd < sd:
                        bygsd = sd
                        lndsd = l
                # shifting the first object
                kwan[ja] -= 1
                kwan[jb] = 1
                if jb != lndsd:
                    lchan = ner[lndsd]
                    lmm = jb - 1
                    for lmma in range(lndsd, lmm+1):
                        lmmb = lmma + 1
                        ner[lmma] = ner[lmmb]
                    ner[jb] = lchan
                splyn = 0
                jma = jb - 1
                # finding the next object to be shifted
                loop_condition = True
                while loop_condition:
                    splyn += 1
                    rest = (jma - ja)
                    jaway = -1
                    bdyff = -1.
                    for l in range(ja, jma+1):
                        lner = ner[l]
                        da = 0.
                        db = 0.
                        for j in range(ja, jma+1):
                            da += dys[self.ind_2(lner, ner[j])]
                        da /= rest
                        for j in range(jma + 1, jb+1):
                            db += dys[self.ind_2(lner, ner[j])]
                        db /= splyn
                        dyff = da - db
                        if bdyff < dyff:
                            bdyff = dyff
                            jaway = l
                    jmb = jma + 1
                    # shifting the next object when necessary
                    if bdyff <= 0.:
                        break # out of "object shifting" while loop
                    if jma != jaway:
                        lchan = ner[jaway]
                        lmz = jma - 1
                        for lxx in range(jaway, lmz+1):
                            ner[lxx] = ner[lxx + 1]
                        ner[jma] = lchan
                    for lxx in range(jmb, jb+1):
                        l_1 = lxx - 1
                        if ner[l_1] < ner[lxx]:
                            break
                        lchan = ner[l_1]
                        ner[l_1] = ner[lxx]
                        ner[lxx] = lchan
                    kwan[ja] -= 1
                    kwan[jma] = kwan[jmb] + 1
                    kwan[jmb] = 0
                    jma -= 1
                    jmb = jma + 1
                    loop_condition = jma != ja        
                if ner[ja] >= ner[jmb]:
                    lxxa = ja
                    for lgrb in range(jmb, jb+1):
                        lxxa += 1
                        lchan = ner[lgrb]
                        lxg = -1
                        for ll in range(lxxa, lgrb+1):
                            lxf = lgrb - ll + lxxa
                            lxg = lxf - 1
                            ner[lxf] = ner[lxg]        
                        ner[lxg] = lchan
                    llq = kwan[jmb]
                    kwan[jmb] = 0
                    jma = ja + jb - jma - 1
                    jmb = jma + 1
                    kwan[jmb] = kwan[ja]
                    kwan[ja] = llq  
                # compute level for banner
                if nclu == 1:
                    ban[jmb] = cs
                else:
                    ban[jmb] = self.min_dis(dys, ja, jb, ner[1:])
            # continue splitting until all objects are separated
            if nclu<nn:
                if jb != nn:
                    loop_cond = True
                    while loop_cond:
                        ja += kwan[ja]
                        if ja>nn:
                            break
                        if ja <= nn:
                            if kwan[ja] <= 1:
                                loop_cond=True
                            else:
                                split = True
                                break
                ja = 1
                if kwan[ja] == 1:
                    loop_cond = True
                    while loop_cond:
                        if ja>nn:
                            break
                        ja += kwan[ja]
                        if ja <= nn:

                            if kwan[ja] <= 1:
                                loop_cond=True
                            else:
                                split = True
                                break        
            nclu += 1
            if nclu==nn:
                break
        # merge-structure for plotting tree
        clust=np.array(range(1,nn))+nn
        mergefin=np.zeros((nn-1,2))
        for nmerge in range(n_1):
            nj = -1
            dmin = cs
            for j in range(2,nn+1):
                if kwan[j] >= 0 and dmin >= ban[j]:
                    dmin = ban[j]
                    nj = j
            kwan[nj] = -1
            l1 = -ner[nj - 1]
            l2 = -ner[nj]
            for j in range(nmerge):
                if merge[j, 0] == l1 or merge[j, 1] == l1:
                    l1 = j+1
                if merge[j, 0] == l2 or merge[j, 1] == l2:
                    l2 = j+1
            merge[nmerge, 0] = l1
            merge[nmerge, 1] = l2
        for i in range(nn-1):
            if merge[i, 0] < 0:
                l1 = -merge[i, 0]
            else:
                l1 = clust[int(merge[i, 0] - 1)]
            mergefin[i, 0] = l1
            if merge[i, 1] < 0:
                l2 = -merge[i, 1]
            else:
                l2 = clust[int(merge[i, 1] - 1)]
            mergefin[i, 1] = l2
        mergefin=(mergefin-1).astype('int')
        for i in range(merge.shape[0]):
            if merge[i,0]<0:
                merge_2[i,0]=merge[i,0]+1
            if merge[i,1]<0:
                merge_2[i,1]=merge[i,1]+1
            if merge[i,0]>0:
                merge_2[i,0]=merge[i,0]-1
            if merge[i,1]>0:
                merge_2[i,1]=merge[i,1]-1
        # linkage matrix for plotting dendrogram
        counts=np.zeros(mergefin.shape[0])
        n_samples=len(ner[1:])
        for i, m in enumerate(mergefin):
            current_count=0
            for child_idx in m:
                if child_idx<n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx-n_samples]
            counts[i]=current_count
        linkage_matrix=np.column_stack([mergefin, sorted(ban[2:]),
                                  counts]).astype(float)
        # Divisive Coefficient
        dc=round(self.bncoef(nn, ban),6)
        # cluster labels
        if n_clusters != None:
            if (n_clusters)<2:
                raise ValueError('The number of clusters must be at least equal to 2')
            cl=cut_tree(linkage_matrix,n_clusters=n_clusters)
            cl=np.array([item for sublist in cl for item in sublist])
        if n_clusters == None:
            cl=np.array(list(range(self.n_samples)))
        self.linkage_matrix_ = linkage_matrix
        self.merge_ = merge_2
        self.children_ = mergefin
        self.labels_ = ner[1:]-1
        self.cluster_labels_ = cl
        self.distances_ = ban[2:]
        self.feature_names_in_ = feature_names
        self.divisive_coefficient_ = dc
        return self


def diff_clust(X,cl,alpha=0.05,**kwargs):
    """
    Identification of differences between signs of data set clusters based on the T-test.
    
    Parameters
    ----------
    X : [DataFrame, array] dataframe or array.
    cl : [array] vector of labels of clusters for X.
    alpha : [float] probability that the returned confidence interval contains the true parameter.
    **kwargs : additional arguments for function scipy.stats.ttest_ind.

    Returns
    ----------
    Differences table between signs of data set clusters.
    """
    m=max(cl)+1
    colnames=X.columns.tolist()
    n=len(colnames)
    res=[]
    ind=[]
    for j in range(m):
        for k in range(1,m):
            if j!=k and j<(m-1):
                ind.append('cl%d_cl%r'%(j,k))
                cl1=X.loc[np.where(cl==j)]
                cl2=X.loc[np.where(cl==k)]
                resV=[0]*n
                for i in range(n):
                    if ttest_ind(cl1[colnames[i]],cl2[colnames[i]],nan_policy='omit',**kwargs)[1]>alpha:
                        resV[i]='нет'
                    else:
                        resV[i]='есть'
                resV=np.array(resV)
                res.append(resV)
                resF=pd.DataFrame(res)
                resF.columns=colnames
    resF.index=ind
    return resF


def dist(data,metric='euclidean',diag=True,upper=False,nadel=False):
    """
    This function computes and returns the distance matrix computed by using the specified 
    distance measure to compute the distances between the rows of a dataframe.

    Parameters
    ----------
    data : [DataFrame] dataframe for calculations a distance matrix.
    metric: [str] the distance measure to be used. This must be one of 'euclidean' (default),
    'cityblock', 'cosine', 'l1', 'l2', 'manhattan' and other from scipy.spatial.distance.
    diag : [bool] if True (default), the diagonal of the distance matrix must be printed.
    upper : [bool] if True, the upper triangle of the distance matrix should be printed
    (default upper=False).
    nadel : [bool] if True, remove missing values (default nadel=False).

    Returns
    ----------
    the lower triangle of the distance matrix or the the lower and upper triangle of the distance matrix.
    """
    if nadel==True:
        indexList = [np.any(i) for i in np.isnan(data)]
        data=np.delete(data, indexList, axis=0)
    if nadel==False:
        data=data
    d=np.full((data.shape[0],data.shape[0]),0,dtype = float)
    for i in range(data.shape[0]):
        for k in range(data.shape[0]):
            if k>=i:
                x=np.nan_to_num(np.reshape(data[i],(1,data.shape[1])))
                y=np.nan_to_num(np.reshape(data[k],(1,data.shape[1])))
                d[k,i] = pairwise_distances(x,y,metric=metric)
    if upper==True:
        d=np.tril(d,1)+d.T
    if diag==True:
        d=d
    if diag==False:
        d=np.delete(d,0,axis=0)
        d=np.delete(d,len(d[0])-1,axis=1)
    return d


def fviz_cluster(data,cl,colors=None,showclcent=False,ellipse=None,ellipsetype='confidence',ellipselevel=2.0, title=None,grid=True,annotlab=None,pointsize=30,labelsize=10,alpha=0.2,figsize=(8,8),**kwargs):
    """
    Visualization of partitioning methods including KMeans. Observations are represented by points 
    in the plot, using principal components if columns(data) > 2. An ellipse is drawn around each cluster.
    
    Parameters
    ----------
    data : [DataFrame] the dataframe or array that has been used for clustering.
    cl : [array] vector of labels of clusters for data.
    colors : [list] list of named colors for points and ellipse.
    showclcent : [bool] if True, shows cluster centers. By default, showclcent=False.
    ellipse : [bool] if True, draws outline around points of each cluster. By default, ellipse=None.
    ellipsetype : [str] string specifying type of ellipse, possible values are 'confidence' (default) or 'convex'.
    ellipselevel : [float] the size of the concentration ellipse. Default value is 2.0.
    title : [str] plot main title.
    grid : [bool] if True to show the grid lines.
    annotlab : [bool] if True annotate the point xy with text.
    pointsize : [float, int] the size of points.
    labelsize : [float, int] font size for the labels.
    alpha : [float, int] the alpha blending value, between 0 (transparent) and 1 (opaque).
    figsize : [int, int] a method used to change the dimension of plot window, width, height in inches 
        (default figsize=(8,8)).
    **kwargs : other arguments for function matplotlib.pyplot.scatter.
    
    Returns
    ----------
    Visualize results of clustering on plot.
    """
    if data.shape[1]<2:
        raise ValueError('the number columnes of data must be > = 2')

    def confidence_ellipse(x,y,ax,n_std=3.0,facecolor='none',**kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.
        """
        if x.size!=y.size:
            raise ValueError("x and y must be the same size")
        cov=np.cov(x,y)
        pearson=cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
        ell_radius_x=np.sqrt(1+pearson)
        ell_radius_y=np.sqrt(1-pearson)
        ellipse=Ellipse((0,0),width=ell_radius_x*ellipselevel,height=ell_radius_y*ellipselevel,
                        facecolor=facecolor,**kwargs)
        scale_x=np.sqrt(cov[0,0])*n_std
        mean_x=np.mean(x)
        scale_y=np.sqrt(cov[1,1])*n_std
        mean_y=np.mean(y)
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        ellipse.set_transform(transf+ax.transData)
        return ax.add_patch(ellipse)
    # prepare the data for plotting
    ncl=max(cl)+1
    if data.shape[1]>2:
        pca=PCA(n_components=2)
        pComp=pca.fit_transform(data)
        eig=pca.explained_variance_ratio_*100
        pDf=pd.DataFrame(data=pComp,columns=['pc1','pc2'])
    if data.shape[1]==2:
        pDf=pd.DataFrame(data.values,columns=['pc1','pc2'])
        pComp=data.values
        var=np.cov(data.values.T)
        explained_var=var.diagonal()
        all_var=np.cov(data.values.T)
        sum_all_var=np.sum(all_var.diagonal())
        eig=explained_var/sum_all_var*100
    pDf['cl']=cl
    # plot
    fig=plt.figure(figsize=figsize)
    ax=fig.add_subplot(1,1,1)
    ax.set_xlabel("Dim1(%.1f%%)" % eig[0],fontsize=14)
    ax.set_ylabel("Dim2(%.1f%%)" % eig[1],fontsize=14)
    if title is None:
        title='Cluster plot'
    else:
        title=title
    ax.set_title(title,fontsize=16)
    targets=list(range(ncl))
    if colors is None:
        colors=['#0000FF','#FF0000','#32CD32','#8A2BE2','#A52A2A','#00008B','#A9A9A9','#00BFFF',
                '#FFD700','#FF69B4','#F08080','#FF00FF','#191970','#FF4500','#708090','#00FF7F',
                '#FAFAD2','#8FBC8F','#FF8C00','#8B008B']
    else:
        colors=colors
    centers1=[]
    centers2=[]
    for i in range(ncl):
        centers1.append(np.mean(pDf['pc1'].loc[np.where(cl==i)]))
        centers2.append(np.mean(pDf['pc2'].loc[np.where(cl==i)]))
    for target, color in zip(targets,colors):
        indicesToKeep=pDf['cl']==target
        ax.scatter(pDf.loc[indicesToKeep,'pc1'],pDf.loc[indicesToKeep,'pc2'],c=color,s=pointsize,label=target,**kwargs)
        if annotlab is True:
            text=pDf.loc[indicesToKeep,'pc1'].index.tolist()
            for i in range(pDf.loc[indicesToKeep,'pc1'].shape[0]):
                ax.annotate(text[i],(pDf.loc[indicesToKeep,'pc1'].values[i],pDf.loc[indicesToKeep,'pc2'].values[i]+0.05),ha='center',va='bottom',size=labelsize)
        if ellipse is True:
            if ellipsetype=='confidence':
                confidence_ellipse(pDf.loc[indicesToKeep,'pc1'],pDf.loc[indicesToKeep,'pc2'],ax,facecolor=color,edgecolor='red',alpha=alpha,zorder=0)
            if ellipsetype=='convex':
                hull = ConvexHull(pComp[np.where(indicesToKeep)])
                for simplex in hull.simplices:
                    ax.plot(pComp[np.where(indicesToKeep)][simplex,0],pComp[np.where(indicesToKeep)][simplex,1],c=color)
                    ax.fill(pComp[np.where(indicesToKeep)][hull.vertices,0],pComp[np.where(indicesToKeep)][hull.vertices,1],c=color,alpha=0.01)
    if showclcent is True:
        ax.scatter(centers1,centers2,s=pointsize,c='black',marker='^',label='Centroids')
    ax.legend()
    if grid is True:
        ax.grid()
    plt.show()


def fviz_dist(datadist,order=True,show_labels=False,lab_size=10,figsize=(9,8),cmap='coolwarm'):
    """
    Visualizes a distance matrix.
    
    Parameters
    ----------
    datadist : [array] a distance matrix for visualizes.
    order : [bool] if True (default) the ordered dissimilarity image (ODI) is shown.
    show_labels : [bool] If True, the labels are displayed (default show_labels=False).
    lab_size : [int] the size of labels (default lab_size=10).
    figsize : [int, int] a method used to change the dimension of plot window, width,
    height in inches (default figsize=(9,8)).
    cmap : [str] value as a matplotlib colormap name or object, or list of colors
    (default cmap='coolwarm').
    
    Returns
    ----------
    plot of the data to check clusterability.
    """
    if order == True:
        Z=linkage(datadist,'ward')
        dn=dendrogram(Z,no_plot=True)
        d=np.tril(datadist,1)+datadist.T
        DFd=pd.DataFrame(d)
        DFd=DFd.loc[DFd.index[dn['leaves']]]
        DFd=DFd[dn['leaves']]
    if order==False:
        d=datadist
        DFd=pd.DataFrame(d)
    x=DFd.columns.to_list()
    y=DFd.columns.to_list()
    plt.figure(figsize=figsize)
    if show_labels==True:
        heatmap=sns.heatmap(DFd,cmap=cmap)
        heatmap.tick_params(labelsize=lab_size)
    if show_labels==False:
        heatmap=sns.heatmap(DFd,cmap=cmap,xticklabels=False,yticklabels=False)


def fviz_gap_stat(gap_stat,method='firstSEmax',SEfactor=1):
    """
    Visualize the gap statistic generated by the function clusGap().
    The optimal number of clusters is specified using the "firstmax" and others methods.

    Parameters
    ----------
    gap_stat : [DataFrame] an object returned by the function clusGap().
    method : [str] method for determining how the "optimal" number of clusters,
    k^, is computed from the gap statistics (and their standard deviations),
    or more generally how the location k^ of the maximum of f[k] should be determined:
    - 'firstmax': gives the location of the first local maximum.
    - 'globalmax': simply corresponds to the global maximum.
    - 'Tibs2001SEmax': uses the criterion, Tibshirani et al (2001) proposed:
      "the smallest k such that f(k)≥f(k+1)-s_{k+1}". Note that this chooses
      k = 1 when all standard deviations are larger than the differences f(k+1)-f(k).
    - 'firstSEmax': location of the first f() value which is not smaller than
      the first local maximum minus SEfactor*SEf[], i.e, within an "f S.E."
      range of that maximum (see also SEfactor). This, the default, has been proposed
      by Martin Maechler in 2012, when adding clusGap() to the cluster package, after 
      having seen the 'globalSEmax' proposal (in code) and read the 'Tibs2001SEmax' proposal.
    - 'globalSEmax': (used in Dudoit and Fridlyand (2002), supposedly following Tibshirani's 
      proposition): location of the first f() value which is not smaller than the global 
      maximum minus SEfactor*SEf[], i.e, within an "f S.E." range of that maximum 
      (see also SEfactor).
    SEfactor : [int] Determining the optimal number of clusters, Tibshirani et al. proposed
    the "1 S.E."-rule. Using an SE.factor f, the "f S.E."-rule is used, more generally.

    Returns
    ----------

    Plot of the Optimal Number of Clusters determined of the gap statistic.
    """
    def maxSE(f,SEf,method='firstmax',SEfactor=1):
        # Determines the location of the maximum of f
        # f: numeric vector containing the gap statistic
        K=len(f)-1
        fSE=SEfactor*SEf
        if method=='firstmax':
            decr=np.array(np.diff(f)<=0)
            if any(decr):
                res=decr.argmax()+1
            else:
                res=K
        if method=='globalmax':
            res=f.argmax()+1
        if method=='Tibs2001SEmax':
            gs=f-fSE
            if any(f[:K-1]>=gs[1:]):
                mp=f[:K-1]
                res=mp.argmax()+1
            else:
                res=K
        if method=='firstSEmax':
            decr=np.array(np.diff(f)<=0)
            if any(decr):
                nc=decr.argmax()+1
            else:
                nc=K
            which=lambda lst:list(np.where(lst)[0])
            if any(f[:nc-1]>=(f[nc]-fSE[nc])):
                mp=f[:nc-1]
                lst=list(map(lambda x:x>=(f[nc]-fSE[nc]),mp))
                res=(which(lst)[0])+1
            else:
                res=nc
        if method=='globalSEmax':
            nc=f.argmax()
            which = lambda lst:list(np.where(lst)[0])
            if any(f[:nc-1]>=(f[nc]-fSE[nc])):
                mp=f[:nc-1]
                lst=list(map(lambda x:x>=(f[nc]-fSE[nc]),mp))
                res=(which(lst)[0])+1
            else:
                res=nc
        return res
    gap=gap_stat.get('gap')
    se=gap_stat.get('SEsim')
    k=maxSE(gap,se,method=method,SEfactor=SEfactor)
    xticks=range(1,len(gap)+1)
    plt.figure(figsize = (8,6))
    plt.errorbar(xticks,gap,yerr=se,marker='o',color='blue')
    plt.axvline(x=k, color="red", linestyle="--")
    plt.xticks(xticks)
    plt.title('Optimal number of clusters k = {}'.format(k),fontsize=16)
    plt.ylabel('Gap statistic (k)',fontsize=14)
    plt.xlabel('Number of clusters k',fontsize=14)
    plt.show()


def fviz_nbclust(data,FUNcluster,type_test='silhouette',kmax=10,nboot=10,showits=True,**kwargs):
    """
    Dertemines and visualize the optimal number of clusters using different methods:
    within cluster sums of squares, average silhouette and gap statistics.

    Parameters
    ----------
    data : [DataFrame] dataframe for calculates a goodness of clustering measure.
    FUNcluster : a function which accepts as first argument a dataframe like x, second argument,
    say k, k>=2, the number of clusters desired, and returns a list with a component named
    (or shortened to) cluster which is a vector of length n = nrow(x) of integers in 1:k
    determining the clustering or grouping of the n observations.
    type_test : [str] the method to be used for estimating the optimal number of clusters.
    Possible values are 'silhouette' (for average silhouette width), 'wcss' (for total within sum of square)
    and 'gap_stat' (for gap statistics).
    kmax : [int] the maximum number of clusters to consider, must be at least two.
    nboot : [int] number of Monte Carlo ("bootstrap") samples.
    dpower : [int] a positive integer specifying the power p which is applied to the euclidean 
    distances (dist) before they are summed up to give W(k). The default, dpower = 1.
    spaceH0 : [str] string specifying the space of the H_0 distribution (of no cluster). 
    Both 'scaledPCA' (default) and 'original' use a uniform distribution in a hyper cube.
    showits : [bool] if True, reports the iterations of Bootstrapping so the user can monitor 
    the progress of the algorithm (default showits=None).
    **kwargs : other arguments for the function fviz_gap_stat().

    Returns
    ----------
    plot of the optimal number of clusters using different methods: within cluster sums of squares,
    average silhouette and gap statistics.
    """
    data=data.dropna()
    data=data.values
    if kmax<2:
        raise ValueError('kmax must be > = 2')
    if kmax>data.shape[0]:
        raise ValueError('kmax must be < = nobs data')
    if type_test=='gap_stat':
        gap_stat=clusGap(pd.DataFrame(data),FUNcluster,kmax=kmax,B=nboot,showits=showits)
        p=fviz_gap_stat(gap_stat,**kwargs)
        return p
    if type_test=='silhouette':
        res={1:0}
        for k in range(2,kmax+1):
            FUNcluster.n_clusters = k
            FUNcluster.fit(data)
            res[k]=metrics.silhouette_score(data,FUNcluster.labels_)
    if type_test=='wcss':
        res={}
        for k in range(2,kmax+1):
            FUNcluster.n_clusters = k
            FUNcluster.fit(data)
            res[k]=FUNcluster.inertia_
    plt.figure(figsize = (8,6))
    plt.plot(res.keys(), res.values(), 'bs-')
    if type_test=='silhouette':
        l=max(res, key=res.get)
        plt.axvline(x=l, color="red", linestyle="--")
        ytitle='Average silhouette width'
    if type_test=='wcss':
        ytitle='Total Within Sum of Square'
    plt.locator_params(axis="both", integer=True, tight=True)
    plt.xlabel('Number of clusters k')
    plt.ylabel(ytitle)
    plt.title('Optimal number of clusters')
    plt.show()


def fviz_pca_ind(X,func,figsize=(8,8),alpha=1,**kwargs):
    """
    Visualize Principal component analysis (PCA)
    
    Parameters
    ----------
    X : [DataFrame] dataframe for Principal component analysis.
    func : a function for Principal component analysis.
    figsize : [int, int] a method used to change the dimension of plot window, width,
    height in inches (default figsize=(8,8)).
    alpha : [int] the alpha blending value, between 0 (transparent) and 1 (opaque).
    **kwargs : other arguments for the function seaborn.scatterplot().

    Returns
    ----------
    plot of variables as result of PCA.
    """
    pca=func(n_components=2)
    pC=pca.fit_transform(X)
    dfpC=pd.DataFrame(pC,columns=['x','y'])
    eig=pca.explained_variance_ratio_*100
    title=func.__name__
    fig=plt.figure(figsize=figsize)
    ax=fig.add_subplot(1,1,1)
    plt.axhline(y=0,color="black",linestyle="--")
    plt.axvline(x=0,color="black",linestyle="--")
    ax.grid(axis='both')
    ax.set_xlabel("Dim1(%.1f%%)" % eig[0],fontsize=14)
    ax.set_ylabel("Dim2(%.1f%%)" % eig[1],fontsize=14)
    ax.set_title(title,fontsize=20)
    sns.scatterplot(data=dfpC,x='x',y='y',ax=ax,alpha=alpha,**kwargs)
    plt.show()


def fviz_silhouette(X,FUNcluster,n_clust,figsize=(9,7)):
    """
    The plot of silhouette analysis for clustering on data.
    
    Parameters
    ----------
    X : [DataFrame] dataframe.
    FUNcluster : a function which returns a list with a component named cluster which is a vector 
        of length n = nrow(x) of integers in 1:k determining the clustering or grouping of the n observations.
    n_clust : [int] the number of clusters, must be at least two.
    figsize : [int, int] a method used to change the dimension of plot window, width,
        height in inches (default figsize=(9,7)).
    
    Returns
    ----------
    The plot of results of the silhouette analysis for clustering on data.
    """
    clusterer=FUNcluster(n_clusters=n_clust, random_state=123)
    cluster_labels=clusterer.fit_predict(X)
    silhouette_avg=metrics.silhouette_score(X,cluster_labels)
    sample_silhouette_values=silhouette_samples(X, cluster_labels)
    fig, ax=plt.subplots(figsize=figsize)
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0,len(X)+(n_clust+1)*10])
    y_lower=10
    for i in range(n_clust):
        ith_cluster_silhouette_values=sample_silhouette_values[cluster_labels==i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i=ith_cluster_silhouette_values.shape[0]
        y_upper=y_lower+size_cluster_i
        color=cm.nipy_spectral(float(i)/n_clust)
        ax.fill_betweenx(np.arange(y_lower,y_upper),0,ith_cluster_silhouette_values,facecolor=color,edgecolor=color,alpha=0.7)
        ax.text(-0.05,y_lower+0.5*size_cluster_i,str(i))
        y_lower=y_upper+10
        ax.set_title("The silhouette plot for %d clusters"%(n_clust),fontsize=14)
        ax.set_xlabel("The silhouette coefficient values",fontsize=14)
        ax.set_ylabel("Cluster label",fontsize=14)
        ax.axvline(x=silhouette_avg,color="red",linestyle="--")
        ax.set_yticks([])
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()


def gap_stat(x,kmax=10,B=10,showits=None):
    """
    Calculates and visualize a goodness of clustering measure the "gap" statistic.

    Parameters
    ----------
    x : [DataFrame, array] dataframe or array for calculates a goodness of clustering measure
        the "gap" statistic.
    kmax : [int] the maximum number of clusters to consider, must be at least two.
        (default kmax=10)
    B : [int] number of Monte Carlo ("bootstrap") samples. By default, B=10.
    showits : [bool] if True, reports the iterations of Bootstrapping so the user can monitor 
        the progress of the algorithm (default showits=None).

    Returns
    ----------
    Plot of the Optimal Number of Clusters determined of the gap statistic.
    """
    if kmax<2:
        raise ValueError('kmax must be > = 2')
    if B<=0:
        raise ValueError('B has to be a positive integer')
    if x.ndim<=1 | x.ndim<=1:
        raise ValueError('x must be shape is row > 1 and columns > 1')
    if isinstance(x, pd.DataFrame):
        x=x.values
    if isinstance(x, np.ndarray):
        x=x

    def progress(it,total,buffer=30):
            """
            A progress bar is used to display the progress of a long running Iterations of 
            Bootstrapping, providing a visual cue that processing is underway.
            """
            percent = 100.0*it/(total+1)
            sys.stdout.write('\r')
            sys.stdout.write("Bootstrapping: [\033[34m{:{}}] {:>3}% ".format('█'*int(percent/(100.0/buffer)),buffer, int(percent)))
            sys.stdout.flush()
            time.sleep(0.001)

    def kmeans(num_centers,X):
        """
        The initialization of the centroids and clusters labels
        """
        centers=np.random.uniform(X.min(axis=0),X.max(axis=0),size=(num_centers,X.shape[1]))
        for _ in range(10):
            assigned_to_clusters=np.zeros(len(X))
            for i, x in enumerate(X):
                closest_center=np.argmin([np.linalg.norm(x-center) for center in centers]) 
                assigned_to_clusters[i]=closest_center
            new_centers=np.array([np.mean(X[assigned_to_clusters==i],axis=0) for i in range(num_centers)])    
            convergence=np.sum(np.abs(new_centers-centers))
            centers=new_centers
        return np.array(new_centers),np.array(assigned_to_clusters)

    def Wk(Ks,X):    
        Ks = np.arange(1,Ks+1)
        within_cluster_dists=[]
        for K in Ks:
            dists=0
            centroids,points=kmeans(K,X)
            for i in range(K):
                cluster_array=X[points==i]
                centroid_dist=[] 
                dist=0
                if len(cluster_array)>0:
                    for j in range(len(cluster_array)):
                        centroid_dist.append(np.linalg.norm(centroids[i]-cluster_array[j]))
                dist+=np.sum(centroid_dist)
                dists+=dist
            within_cluster_dists.append(np.log(((dists)/K)))
            normalized_wcd=within_cluster_dists-np.max(within_cluster_dists)
        return normalized_wcd
    
    def shift_to_one(shift,to_range):
        shifted=np.zeros(to_range)
        shifted[1:]=shift
        shifted[0]=np.nan 
        return shifted

    simulated_Wk=np.zeros((B,kmax))
    simulated_sk=np.zeros((B,kmax))
    it=0
    for i in range(B):
        temp_wk=[]
        temp_sk=[]
        X=np.random.uniform(0,1,size=(x.shape[0],x.shape[1]))
        within_cluster_dists=Wk(kmax,X)
        simulated_Wk[i]=within_cluster_dists
        it=it+1
        if showits==True:
            progress(it+1,B)
    Wks=np.mean(simulated_Wk,axis=0)
    se=np.std(simulated_Wk,axis=0)*np.sqrt(1+1/B)
    Wcd=Wk(kmax,x)
    gap=Wks-Wcd
    k=0
    for i in range(len(gap)-1):
        if(gap[i]>=gap[i+1]+se[i+1]):
            k=i
            break
    if k==0:
        k=1
    xticks=range(1,len(gap)+1)
    plt.figure(figsize=(8,6))      
    plt.plot(shift_to_one(gap,len(gap)+1),marker='o',color='blue')
    plt.axvline(x=k, color="red", linestyle="--")
    plt.xticks(xticks)
    plt.title('Optimal number of clusters k = {}'.format(k),fontsize=14)
    plt.ylabel('Gap statistic (k)',fontsize=14)
    plt.xlabel('Number of clusters k',fontsize=14)
    plt.show()


def hKMeans(data, k, halgoritm='agnes', method='average', metric='euclidean', algorithm='auto', max_iter=300):
    """
    Hybrid hierarchical k-means clustering for optimizing clustering outputs.

    This function provides a solution using an hybrid approach by combining 
    the hierarchical clustering and the k-means methods.

    Parameters
    ----------
    data : [DataFrame, array] dataframe or array for computes a hybrid hierarchical
                   k-means clustering.
    k : [int]      The number of clusters to find.
    halgoritm : [str] The algorithm of hierarchical clustering: 'agnes' - Agglomerative
                   Clustering (default), 'diana' - Divisive Clustering.
    method : [str] Which linkage criterion to use. The linkage criterion determines 
                   which distance to use between sets of observation. This must be 
                   one of: 'ward', 'average' (default), 'complete' or 'maximum', 'single'.
    metric : [str] the distance measure to be used to compute the dissimilarity matrix. 
                   This must be one of: 'euclidean' (default), 'cityblock', 'cosine', 'l1', 'l2', 
                   'manhattan'. Also it must be one of the options allowed by 
                   sklearn.metrics.pairwise_distances for its metric parameter.
    algorithm : [str] K-means algorithm to use. This must be one of: 'auto'(default),
                   'elkan' or 'full'.
    max_iter : [int] Maximum number of iterations of the k-means algorithm for a single run,
                   max_iter=300.

    Returns
    ----------
    Dictionary with the following elements:
    'cluster_labels': Cluster labels for each point.
    'cluster_size' : The number of points in each cluster.
    'centers' : A array of cluster centres.
    'totss' : The total sum of squares.
    'tot_withinss' : Total within-cluster sum of squares, i.e. 'sum(withinss)'.
    'withinss' : Vector of within-cluster sum of squares, one component per cluster.
    'betweenss' : The between-cluster sum of squares, i.e. 'totss-tot.withinss'.
    'between_SS/total_SS' : The ratio the between-cluster sum of squares to the 
                            total sum of squares.
    'model' : an object containing attributes necessary for the plot of a dendrogram.
    """
    from collections import Counter, OrderedDict
    if isinstance(data, pd.DataFrame):
        data=data.values.astype(float)
    if halgoritm=='agnes':
        model=AgglomerativeClustering(n_clusters=k,linkage=method,metric=metric,compute_distances=True).fit(data)
        clCentres=np.array([np.mean(data[model.labels_==i],axis=0) for i in np.unique(model.labels_)])
    if halgoritm=='diana':
        model=DianaClustering(data,metric=metric,keepdiss=True).fit(k)
        clCentres=np.array([np.mean(data[model.cluster_labels_==i],axis=0) for i in np.unique(model.cluster_labels_)])
    cl = KMeans(n_clusters=k, init=clCentres, n_init=1, algorithm=algorithm, max_iter=max_iter).fit(data).labels_
    сlust_сount=dict(OrderedDict(sorted(Counter(cl).items())))
    totss=np.sum(scale(data,scale=False)[0]**2)
    tot_withinss=np.around(np.sum([np.sum(scale(data[cl==i],scale=False)[0]**2) for i in np.unique(cl)]),5)
    withinss=list(np.around([np.sum(scale(data[cl==i],scale=False)[0]**2) for i in np.unique(cl)],5))
    betweenss=round(totss-tot_withinss,4)
    bss_tss=betweenss/totss*100
    res={'cluster_labels':cl,'cluster_size':str(сlust_сount)[1:-1],'centers':clCentres,'totss':totss,'tot_withinss':tot_withinss,'withinss':withinss,'betweenss':betweenss,'between_SS/total_SS':"{r:.2f}%".format(r=bss_tss),'model':model}
    return res


def hopkins_pval(x,n):
    """
    Calculate the p-value for Hopkins statistic.
    
    Parameters
    ----------
    x : [float] observed value of Hopkins statistic.
    n : [int] number of events/points sampled.
    
    Returns
    ----------
    a p-value between 0 and 1.
    """
    if x>0.5:
        hpv=1-(beta.cdf(x,n,n)-beta.cdf(1-x,n,n))
    else:
        hpv=1-(beta.cdf(1-x,n,n)-beta.cdf(x,n,n))
    return hpv


def hopkins_stat(X):
    """
    Calculate the Hopkins’ statistic.
    
    Parameters
    ----------
    X : [DataFrame] dataframe for calculates the Hopkins’ statistic.
    
    Returns
    ----------
    the value returned is actually Hopkins’ statistic.
    """
    X=X.dropna()
    X=X.values
    np.random.seed(123)
    p=np.random.uniform(X.min(axis=0),X.max(axis=0),size=(X.shape[0],X.shape[1]))
    p=np.array(p)
    k=sample(range(0,X.shape[0],1),X.shape[0])
    q=X[k]
    neigh=NearestNeighbors(n_neighbors=2)
    nbrs=neigh.fit(X)
    u_distances,u_indices=nbrs.kneighbors(p,n_neighbors=2)
    u_distances=u_distances[:,0]
    w_distances,w_indices=nbrs.kneighbors(q,n_neighbors=2)
    w_distances=w_distances[:,1]
    u_sum=np.sum(u_distances)
    w_sum=np.sum(w_distances)
    H=u_sum/(u_sum+w_sum)
    return H


def hopkins_test(X,m=None):
    """
    Calculate the Hopkins’ statistic.

    Parameters
    ----------
    X : [DataFrame] dataframe for calculates the Hopkins’ statistic.
    m : [int] number of rows to sample from X. Default is 1/10th the number of rows of X.   

    Returns
    ----------
    the value returned is actually Hopkins’ statistic.
    """
    if isinstance(X, pd.DataFrame):
        X=X.values
    else:
        X=X
    if m==None:
        m=int(round(X.shape[0]/10,0))
    else:
        m=m
    d=X.shape[1]
    colmin=X.min(axis=0)
    colmax=X.max(axis=0)
    np.random.seed(123)
    U=np.random.uniform(colmin,colmax,size=(m,X.shape[1]))
    j=np.round(np.random.uniform(0,X.shape[0],m),0).astype('int')
    W=X[j]
    dwx=pairwise_distances(W,X)
    dwx[dwx==0]='Inf'
    dwx=dwx.min(axis=1)
    dux=pairwise_distances(U,X)
    dux=dux.min(axis=1)
    H=sum(dux**d)/sum(dux**d+dwx**d)
    return H


def linkage_matrix(model):
    """
    Create linkage matrix for plot the dendrogram.
    
    Parameters
    ----------
    model : Object returned by AgglomerativeClustering.
    
    Returns
    ----------
    linkage matrix the corresponding results of the AgglomerativeClustering.
    """
    counts=np.zeros(model.children_.shape[0])
    n_samples=len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count=0
        for child_idx in merge:
            if child_idx<n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx-n_samples]
        counts[i]=current_count
    linkage_matrix=np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    return linkage_matrix


def NbClust(data=None,diss=None,distance ="euclidean",min_nc=2,max_nc=15,method=None,index=['all'],alphaBeale=0.1,plotInd=False,PrintRes=True,n_init=10):
    """
    NbClust function provides 30 indices for determining the number of clusters and proposes
    to user the best clustering scheme from the different results obtained by varying all
    combinations of number of clusters, distance measures, and clustering methods.

    Parameters
    ----------
    data : [DataFrame, array] dataframe or array for calculates a goodness the number of clusters 
                   (default data=None).
    diss : [array] dissimilarity matrix to be used. By default, diss=None, but if it is replaced 
                   by a dissimilarity matrix, distance should be "None".
    distance : [str] the distance measure to be used to compute the dissimilarity matrix. 
                   This must be one of: 'euclidean' (default), 'cityblock', 'cosine', 'l1', 'l2', 
                   'manhattan' and other from scipy.spatial.distance.
    min_nc : [int] minimal number of clusters, between 1 and (number of objects - 1).
    max_nc : [int] maximal number of clusters, between 2 and (number of objects - 1),
                   greater or equal to min_nc. By default, max_nc=15.
    method : [str] the cluster analysis method to be used. This should be one of:
                   'KMeans','single','complete','average','weighted','centroid','ward','median'.
    index : [list, str] the index to be calculated. This should be one or some of : "kl","ch",
                   "hartigan","ccc","scott","marriot","trcovw","tracew","friedman","rubin",
                   "cindex","db","silhouette","duda","pseudot2","beale","ratkowsky","ball",
                   "ptbiserial","gap","frey","mcclain","gamma","gplus","tau","dunn","hubert",
                   "sdindex","dindex", "sdbw", "all" (all indices except GAP, Gamma, 
                   Gplus and Tau),"alllong" (all indices). By default, index=['all'].
    alphaBeale : [float] significance value for Beale’s index.
    plotInd : [bool] if True plot Hubert index and\or D index. By default, plotInd=False.
    PrintRes : [bool] if True print of results of among all indices proposed as the best number 
                    of clusters and best number of clusters.
    n_init : [int] number of times the k-means algorithm is run with different centroid seeds.
                    By default, n_init=10.

    Returns
    ----------
    Dictionary with the following elements:
    'Allindex'          Values of indices for each partition of the dataset obtained with a number
                        of clusters between min_nc and max_nc.
    'AllCriticalValues' Critical values of some indices for each partition obtained with a number
                        of clusters between min_nc and max_nc.
    'BestNc'            Best number of clusters proposed by each index and the corresponding index value.
    'BestPartition'     Partition that corresponds to the best number of clusters.
    """

    indexes=["kl","ch","hartigan","ccc","scott","marriot","trcovw","tracew","friedman",
          "rubin","cindex","db","silhouette","duda","pseudot2","beale","ratkowsky","ball",
          "ptbiserial","gap", "frey", "mcclain",  "gamma", "gplus", "tau", "dunn",  
          "hubert", "sdindex", "dindex", "sdbw", "all","alllong"]
    indice=list(x for x in range(len(indexes)) if indexes[x] in index)
    if any(map(lambda v: v not in indexes, index)):
        raise ValueError('invalid clustering index')

    if method is None:
        raise ValueError('method is None')
    methods=['KMeans','single','complete','average','weighted','centroid','ward','median']
    method_=list(x for x in range(len(methods)) if methods[x] in [method])
    if any(map(lambda v: v not in methods, [method])):
        raise ValueError('invalid method')

    if (2 in indice or 4 in indice or 5 in indice or 6 in indice or 7 in indice or 8 in indice or 9 in indice or 10 in indice or 17 in indice or 26 in indice or 28 in indice or 30 in indice or 31 in indice):
        if (max_nc-min_nc)<2:
            raise ValueError('The difference between the minimum and the maximum number of clusters must be at least equal to 2')

    if data is None:
        if 0 in method_:
            raise ValueError("method = KMeans, data matrix is needed")
        else:
            if 0 in indice or 1 in indice or 2 in indice or 3 in indice or 4 in indice or 5 in indice or 6 in indice or 7 in indice or 8 in indice or 9 in indice or 11 in indice or 12 in indice or 13 in indice or 14 in indice or 15 in indice or 16 in indice or 17 in indice or 18 in indice or 19 in indice or 22 in indice or 23 in indice or 24 in indice or 26 in indice or 27 in indice or 28 in indice or 29 in indice or 30 in indice or 31 in indice:
                raise ValueError("Dataframe is needed. Only frey, mcclain, cindex and dunn can be computed")
            if diss is None:
                raise ValueError("Dataframe and dissimilarity matrix are both null")
            else:
                print("Only frey, mcclain, cindex and dunn can be computed. To compute the other indices, dataframe is needed")

    if isinstance(data, pd.DataFrame):
        df=data.dropna()
        df=df.reset_index(drop=True)
        jeu=df.values
    if isinstance(data, np.ndarray):
        jeu=data[~np.isnan(data).any(axis=1)]
    if data is not None:
        nn=jeu.shape[0]
        pp=jeu.shape[1]
        TT=np.dot(jeu.T,jeu)
        sizeEigenTT=len(eig(TT)[0])
        eigenValues=eig(TT/(nn-1))[0]

    if 3 in indice or 4 in indice or 5 in indice or 6 in indice or 7 in indice or 8 in indice or 9 in indice or 30 in indice or 31 in indice:
        for i in range(sizeEigenTT):
            if eigenValues[i]<0:
                raise ValueError("The TSS matrix is indefinite. There must be too many missing values. The index cannot be calculated.")
        s1=np.sqrt(eigenValues)
        ss=list(filter(lambda x: (x!=0), s1))
        vv=np.prod(ss)

    if diss is None:
        md=dist(jeu,metric=distance)
    else:
        md=diss

    x_axis=list(range(min_nc,max_nc+1))

    res=np.zeros((max_nc-min_nc+1,30))
    rownames=list(range(min_nc,max_nc+1))
    colnamesR=["KL","CH","Hartigan","CCC","Scott","Marriot","TrCovW",
   		   "TraceW","Friedman","Rubin","Cindex","DB","Silhouette",
   		   "Duda","Pseudot2","Beale","Ratkowsky","Ball","Ptbiserial",
   		   "Gap","Frey","McClain","Gamma","Gplus","Tau","Dunn",
   		   "Hubert","SDindex","Dindex","SDbw"]
    res=pd.DataFrame(res,index=rownames,columns=colnamesR)

    resCritical=np.zeros((max_nc-min_nc+1,4))
    colnamesRC=["CritValue_Duda","CritValue_PseudoT2","Fvalue_Beale","CritValue_Gap"]
    resCritical=pd.DataFrame(resCritical,index=rownames,columns=colnamesRC)

    if (1 in method_ or 2 in method_ or 3 in method_ or 4 in method_ or 5 in method_ or 6 in method_ or 7 in method_):
        hc=linkage(md,method=method)

    # SD and SDbw

    def centers(cl,x):
        k=np.max(cl)+1
        centers=np.zeros((k,x.shape[1]))
        for i in range(k):
            for j in range(x.shape[1]):
                centers[i,j]=x[np.where(cl==i)[0]].mean(axis=0)[j]
        return centers

    def average_scattering(cl,x):
        n=len(cl)
        k=np.max(cl)+1
        cluster_size=[0]*k
        centers_matrix=centers(cl,x)
        variance_clusters=np.zeros((k,x.shape[1]))
        var=np.zeros((k,x.shape[1]))
        for u in range(k):
            cluster_size[u]=x[np.where(cl==u)[0]].shape[0]
        for u in range(k):
            for j in range(x.shape[1]):
                for i in range(n):
                    if cl[i]==u:
                        variance_clusters[u,j]=variance_clusters[u,j]+(x[i,j]-centers_matrix[u,j])**2
        for u in range(k):
            for j in range(x.shape[1]):
                variance_clusters[u,j]=variance_clusters[u,j]/cluster_size[u]
        variance_matrix=[0]*x.shape[1]
        for j in range(x.shape[1]):
            variance_matrix[j]=np.var(x[:,[j]],ddof=1)*(n-1)/n
        Somme_variance_clusters=[0]*k
        for u in range(k):
            Somme_variance_clusters+=np.sqrt(np.dot(variance_clusters[u],variance_clusters[u]))
            Somme_variance_clusters=np.unique(Somme_variance_clusters)[0]
        stdev=(1/k)*np.sqrt(Somme_variance_clusters)
        scat=(1/k)*(Somme_variance_clusters/np.sqrt(np.dot(variance_matrix,variance_matrix)))
        scat={'stdev':stdev,'centers':centers_matrix,'variance_intraclusters':variance_clusters,'scatt':scat}
        return scat

    def density_clusters(cl,x):
        n=len(cl)
        k=np.max(cl)+1
        distance=[0]*n
        density=[0]*k
        centers_matrix=centers(cl,x)
        stdev=average_scattering(cl,x)['stdev']
        for i in range(n):
            u=0
            while cl[i]!=u:
                u=u+1
            for j in range(x.shape[1]):
                distance[i]=distance[i]+(x[i,j]-centers_matrix[u,j])**2
            distance[i]=np.sqrt(distance[i])
            if distance[i]<=stdev:
                density[u]=density[u]+1
        dens={'distance':distance,'density':density}
        return dens

    def density_bw(cl,x):
        n=len(cl)
        k=np.max(cl)+1
        centers_matrix=centers(cl,x)
        stdev=average_scattering(cl,x)['stdev']
        density_bw=np.zeros((k,k))
        u=0
        for u in range(k):
            for v in range(k):
                if v!=u:
                    distance=[0]*n
                    moy=(centers_matrix[u]+centers_matrix[v])/2
                    for i in range(n):
                        if cl[i]==u or cl[i]==v:
                            for j in range(x.shape[1]):
                                distance[i]=distance[i]+(x[i,j]-moy[j])**2
                            distance[i]=np.sqrt(distance[i])
                            if distance[i]<=stdev:
                                density_bw[u,v]=density_bw[u,v]+1
        density_clust=density_clusters(cl,x)['density']
        S=0
        for u in range(k):
            for v in range(k):
                if max(density_clust[u],density_clust[v])!=0:
                    S=S+(density_bw[u,v]/max(density_clust[u],density_clust[v]))
        density_bw=S/(k*(k-1))
        return density_bw

    def Dis(cl,x):
        k=np.max(cl)+1
        centers_matrix=centers(cl,x)
        Distance_centers=dist(centers_matrix,upper=True)
        np.fill_diagonal(Distance_centers,0)
        Dmin=np.amin(Distance_centers[Distance_centers!= 0])
        Dmax=np.max(Distance_centers)
        s2=0
        for u in range(k):
            s1=0
            for j in range(Distance_centers.shape[1]):
                s1=s1+Distance_centers[u,j]
            s2=s2+1/s1
        dis=(Dmax/Dmin)*s2
        return dis

    def flatten(lis):
        for item in lis:
            if isinstance(item, Iterable) and not isinstance(item, str):
                for x in flatten(item):
                    yield x
            else:        
                yield item

    # Hubert index
    def index_hubert(x,cl):
        n=x.shape[0]
        k=np.max(cl)+1
        y=np.zeros((n,x.shape[1]))
        P=dist(x,upper=True)
        meanP=np.mean(P)
        variance_matrix=[0]*n
        for j in range(n):
            variance_matrix[j]=np.var(P[:,[j]],ddof=1)*(n-1)/n
        varP=np.sqrt(np.dot(variance_matrix,variance_matrix))
        centers_clusters=centers(cl,x)
        for i in range(n):
            for u in range(k):
                if cl[i]==u:
                    y[i]=centers_clusters[u]
        Q=dist(y,upper=True)
        meanQ=np.mean(Q)
        for j in range(n):
            variance_matrix[j]=np.var(Q[:,[j]],ddof=1)*(n-1)/n
        varQ=np.sqrt(np.dot(variance_matrix,variance_matrix))
        M=n*(n-1)/2
        S=0
        n1=n-1
        for i in range(n1):
            for j in range(1,n):
                if j<=n:
                    S=(S+(P[i,j]-meanP)*(Q[i,j]-meanQ))
        S=S*0.5
        gamma=S/(M*varP*varQ)
        return gamma

    # Gamma, Gplus and Tau
    def Index_sPlussMoins(cl1,jeu):
        cn1=np.max(cl1)+1
        n1=len(cl1)
        separation_matrix=np.zeros((cn1,cn1))
        cluster_size=[0]*cn1
        average_distance=[0]*cn1
        median_distance=[0]*cn1
        between_dist1=[0]*cn1
        within_dist1=[]
        for u in range(cn1):
            cluster_size[u]=np.sum(cl1==u)
            du = pdist(jeu[cl1 == u, :])
            within_dist1.extend(du)
            average_distance[u]=np.mean(du)
            median_distance[u]=np.median(du)
            for v in range(cn1):
                if v!=u:
                    suv = pairwise_distances(jeu[cl1 == u, :], jeu[cl1 == v, :])
                    bv=list(suv)
                    if u<v:
                        separation_matrix[v,u]=np.min(suv)
                        separation_matrix[u,v]=np.min(suv)
                        np.fill_diagonal(separation_matrix,0)
        between_dist1=suv.flatten().tolist()
        nwithin1 = len(within_dist1)
        nbetween1 = len(between_dist1)
        meanwithin1 = np.mean(within_dist1)
        meanbetween1 = np.mean(between_dist1)
        s_plus=[]
        s_moins=[]
        for k in range(nwithin1):
            s_plus.append(np.sum(np.array(between_dist1)>within_dist1[k]))
            s_moins.append(np.sum(np.array(between_dist1)<within_dist1[k]))
        s_plus=sum(s_plus)
        s_moins=sum(s_moins)
        Index_Gamma=(s_plus-s_moins)/(s_plus+s_moins)
        Index_Gplus=(2*s_moins)/(n1*(n1-1))
        t_tau=(nwithin1*nbetween1)-(s_plus+s_moins)
        with np.errstate(invalid='ignore'):
            Index_Tau=(s_plus-s_moins)/(((n1*(n1-1)/2-t_tau)*(n1*(n1-1)/2))**(1/2))
        result={'gamma':Index_Gamma,'gplus':Index_Gplus,'tau':Index_Tau}
        return result

    # Frey and McClain
    def Index_15and28(cl1,cl2,md):
        cl1=np.array(list(flatten(cl1)))
        cl2=np.array(list(flatten(cl2)))
        cn1=np.max(cl1)+1
        n1=len(cl1)
        dmat=md
        within_dist1=[]
        between_dist1=[]
        for u in range(cn1):
            dmat1=dmat[cl1==u]
            du=dmat1[:,cl1==u]
            if len(du)!=0:
                du=np.delete(du,0,axis=0)
            if len(du)==0:
                du=du
            if len(du)!=0:
                du=np.delete(du,len(du[0])-1,axis=1)
            if len(du)==0:
                du=du    
            within_dist1.append(du[np.tril_indices(du.shape[0])])
            for v in range(cn1):
                if v!=u:
                    suv1=dmat[cl1==v]
                    suv=suv1[:,cl1==u]
                    if u<v:
                        between_dist1.append(suv)
        within_dist1=list(flatten(within_dist1))
        between_dist1=list(flatten(between_dist1))
        cn2=np.max(cl2)+1
        n2=len(cl2)
        within_dist2=[]
        between_dist2=[]
        for w in range(cn2):
            dmat1_=dmat[cl2==w]
            dw=dmat1_[:,cl2==w]
            if len(dw)!=0:
                dw=np.delete(dw,0,axis=0)
            if len(dw)==0:
                dw=dw
            if len(dw)!=0:
                dw=np.delete(dw,len(dw[0])-1,axis=1)
            if len(dw)==0:
                dw=dw
            within_dist2.append(dw[np.tril_indices(dw.shape[0])])
            for x in range(cn2):
                if x!=w:
                    swx1=dmat[cl2==x]
                    swx=swx1[:,cl2==w]            
                    if w<x:
                        between_dist2.append(swx)
        within_dist2=list(flatten(within_dist2))
        between_dist2=list(flatten(between_dist2))
        nwithin1=len(within_dist1)
        nbetween1=len(between_dist1)
        meanwithin1=np.mean(within_dist1)
        meanbetween1=np.mean(between_dist1)
        meanwithin2=np.mean(within_dist2)
        meanbetween2=np.mean(between_dist2)
        Index_15=(meanbetween2-meanbetween1)/(meanwithin2-meanwithin1)
        Index_28=(meanwithin1/nwithin1)/(meanbetween1/nbetween1)
        results={'frey':Index_15,'mcclain':Index_28}
        return results

    # Point-biserial 
    def Indice_ptbiserial(x,md,cl1):
        def biserial_cor(x,y,level=1):
            if len(x)!=len(y):
                raise ValueError("'x' and 'y' do not have the same length")
            y=pd.Series(flatten(y),dtype="category")
            levs=y.cat.categories
            df=pd.DataFrame({'x':flatten(x),'y':y})
            if len(np.unique(df.y))>2:
                raise ValueError("'y' must be a dichotomous variable")
            df.dropna(axis = 0, how = 'any', inplace = True)
            ind=y==levs[level]
            diff_mu=np.mean(x[ind==0])-np.mean(x[ind==1])
            prob=np.mean(ind)
            res=diff_mu*np.sqrt(prob*(1-prob))/np.std(x)
            return res
        nn=x.shape[0]
        pp=x.shape[1]
        md2=md
        m01=np.tile(np.nan,(nn,nn))
        nbr=int((nn*(nn-1))/2)
        pb=np.zeros((nbr,2))
        m3=0
        for m1 in range(nn):
            m12=m1
            for m2 in range(m12):
                if cl1[m1]==cl1[m2]:
                    m01[m1,m2]=0
                if cl1[m1]!=cl1[m2]:
                    m01[m1,m2]=1
                pb[m3,0]=m01[m1,m2]
                pb[m3,1]=md2[m1,m2]
                m3=m3+1
        ptbiserial=biserial_cor(x=pb[:,[1]],y=pb[:,[0]],level=0)
        return ptbiserial

    # Duda, pseudot2 and beale
    def Indices_WKWL(x,cl1,cl2):
        dim2=x.shape[1]
        def wss(x):
            centers=np.zeros(x.shape[1])
            if x.shape[1]==1:
                centers[centers==0]=np.mean(x)
            if any(x.shape)==0:
                centers[centers==0]=list(flatten(x[0:x.shape[1]:,[0]]))
            else:
                centers=x.mean(axis=0)
            x_2=x-centers[np.newaxis,:]
            withins=np.sum(x_2**2)
            wss=np.sum(withins)
            return wss
        ncg1=0
        ncg1max=np.max(cl1)+1
        while sum(cl1==ncg1)==sum(cl2==ncg1) and ncg1<=ncg1max:
            ncg1+=1
        g1=ncg1
        ncg2=np.max(cl2)
        nc2g2=ncg2-1
        ncg1max=np.max(cl1)+1
        while sum(cl1==nc2g2)==sum(cl2==ncg2) and nc2g2>=1:
            ncg2-=1
            nc2g2-=1
        g2=ncg2
        NK=np.sum(cl2==g1)
        WK_x=x[list(flatten(cl2==g1))]
        WK=wss(x=WK_x)
        NL=np.sum(cl2==g2)
        WL_x=x[list(flatten(cl2==g2))]
        WL=wss(x=WL_x)
        NM=np.sum(cl1==g1)
        WM_x=x[list(flatten(cl1==g1))]
        WM=wss(x=WM_x)
        duda=(WK+WL)/WM
        BKL=WM-WK-WL
        pseudot2=BKL/((WK+WL)/(NK+NL-2))
        beale=(BKL/(WK+WL))/(((NM-1)/(NM-2))*(2**(2/dim2)-1))
        results={'duda':duda,'pseudot2':pseudot2,'NM':NM,'NK':NK,'NL':NL,'beale':beale}
        return results

    # ccc, scott, marriot, trcovw, tracew, friedman and rubin
    def Indices_WBT(x,cl,P,s,vv):
        n=x.shape[0]
        pp=x.shape[1]
        qq=np.max(cl)+1
        z=np.zeros((n,qq))
        clX=list(flatten(cl))
        for i in range(n):
            for j in range(qq):
                z[i,j]=0
                if clX[i]==j:
                    z[i,j]=1
        xbar=np.dot(np.dot(np.linalg.inv(np.dot((z).T,z)),(z).T),x)
        B=np.dot(np.dot(np.dot(xbar.T,z.T),z),xbar)
        W=P-B
        marriot=(qq**2)*np.linalg.det(W)
        trcovw=np.sum(np.diag(np.cov(W)))
        tracew=np.sum(np.diag(W))
        if np.linalg.det(W)!=0:
            scott=n*np.log(np.linalg.det(P)/np.linalg.det(W))
        else:
            scott='Error: division by zero!'
        friedman=np.sum(np.diag(np.linalg.inv(W)*B))
        rubin=np.sum(np.diag(P))/np.sum(np.diag(W))
        R2=1-np.sum(np.diag(W))/np.sum(np.diag(P))
        v1=1
        u=[0]*pp
        c=(vv/(qq))**(1/pp)
        u=s/c
        k1=np.sum((u>=1)==True)
        p1=min(k1,qq-1)
        if p1>0 and p1<pp:
            for i in range(p1):
                v1=v1*s[i]
                c=(v1/(qq))**(1/p1)
                u=s/c
                b1=np.sum(1/(n+u[0:p1]))
                b2=np.sum(u[p1+0:pp]**2/(n+u[p1+0:pp]))
                E_R2=1-((b1+b2)/np.sum(u**2))*((n-qq)**2/n)*(1+4/n)
                ccc=np.log((1-E_R2)/(1-R2))*(np.sqrt(n*p1/2)/((0.001+E_R2)**1.2))
        else:
            b1=np.sum(1/(n+u))
            E_R2=1-(b1/np.sum(u**2))*((n-qq)**2/n)*(1+4/n)
            ccc=np.log((1-E_R2)/(1-R2))*(np.sqrt(n*pp/2)/((0.001+E_R2)**1.2))
        results={'ccc':ccc,'scott':scott,'marriot':marriot,'trcovw':trcovw,'tracew':tracew,'friedman':friedman,'rubin':rubin}
        return results

    # Kl, Ch, Hartigan, Ratkowsky and Ball 
    def Indices_Traces(data,d,clall,index="all"):
        x=data
        cl0=clall[0]
        cl1=clall[1]
        cl2=clall[2]
        nbcl0=pd.DataFrame(np.unique(cl0,return_counts=True))
        nbcl1=pd.DataFrame(np.unique(cl1,return_counts=True))
        nbcl2=pd.DataFrame(np.unique(cl2,return_counts=True))
        nb1cl0=sum(nbcl0.loc[1]==1)
        nb1cl1=sum(nbcl1.loc[1]==1)
        nb1cl2=sum(nbcl2.loc[1]==1)

        def gss(x,cl):
            n=len(cl)
            k=np.max(cl)+1
            centers=np.tile(np.nan,(k,x.shape[1]))
            for i in range(k):
                if x.shape[1]==1:
                    centers[i]=np.mean(x[cl==i])
                if x[cl==0].shape is None:
                    bb=x[cl==i][0:x.shape[1]][:,[0]].reshape((1,x.shape[1]))
                    centers[i]=bb.mean(axis=0)
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        centers[i]=x[cl==i].mean(axis=0)
            allmean=x.mean(axis=0)
            dmean=x-allmean
            allmeandist=np.sum(dmean**2)
            withins=[0]*k
            x2=(x-centers[cl])**2
            for i in range(k):
                withins[i]=np.sum(x2[cl==i])
            wgss=np.sum(withins)
            bgss=allmeandist-wgss
            results={'wgss':wgss,'bgss':bgss,'centers':centers}
            return results
    
        def varwithinss(x,centers,cl):
            nrow=centers.shape[0]
            nvar=x.shape[1]
            varwithins=np.zeros((nrow,nvar))
            x1=np.array([((x[np.where(cl==i)[0]]-centers[i])**2) for i in range(nrow)],dtype='object')
            for j in range(nrow):
                varwithins[j]=x1[j].sum(axis=0)
            return varwithins
    
        def vargss(x,clsize,varwithins):
            nvar=x.shape[1]
            n=sum(clsize)
            k=len(clsize)
            vardmean=x-x.mean(axis=0)
            varallmeandist=(vardmean**2).sum(axis=0)
            varwgss=varwithins.sum(axis=0)
            varbgss=varallmeandist-varwgss
            vartss=varbgss+varwgss
            zvargss={'vartss':vartss,'varbgss':varbgss}
            return zvargss
    
        def indice_kl(x,clall,nb1cl1):
            if nb1cl1 > 0:
                KL=np.nan
            m=x.shape[1]
            g=np.max(clall[1])+1
            KL=abs((g-1)**(2/m)*gss(x,clall[0])['wgss']-g**(2/m)*gss(x,clall[1])['wgss'])/abs(g**(2/m)*gss(x,clall[1])['wgss']-(g+1)**(2/m)*gss(x,clall[2])['wgss'])
            return KL
    
        def indice_ch(x,cl,nb1cl1):
            if nb1cl1>0:
                CH=np.nan
            n=len(cl)
            k=np.max(cl)+1
            CH=(gss(x,cl)['bgss']/(k-1))/(gss(x,cl)['wgss']/(n-k))
            return CH
    
        def indice_ball(x,cl):
            wgssB=gss(x,cl)['wgss']
            qq=np.max(cl)+1
            ball=wgssB/qq
            return ball
    
        def indice_hart(x,clall):
            n=x.shape[0]
            g=np.max(clall[0])+1
            HART=(gss(x,clall[1])['wgss']/gss(x,clall[2])['wgss']-1)*(n-g-1)
            return HART
    
        def indice_ratkowsky(x,cl):
            qq=np.max(cl)+1
            clsize=list(pd.DataFrame(np.unique(cl,return_counts=True)).loc[1])
            centers=gss(x,cl)['centers']
            varwithins=varwithinss(x,centers,cl)
            zvargss=vargss(x,clsize,varwithins)
            ratio=np.mean(np.sqrt(zvargss['varbgss']/zvargss['vartss']))
            ratkowsky=ratio/np.sqrt(qq)
            return ratkowsky
    
        indexes=["kl","ch","hart","ratkowsky","ball","all"]
        indice=list(x for x in range(len(indexes)) if indexes[x] in index)
        vecallindex=[0]*5
        if 0 in indice or 5 in indice:
            vecallindex[0]=indice_kl(x,clall,nb1cl1)
        if 1 in indice or 5 in indice:
            vecallindex[1]=indice_ch(x,clall[1],nb1cl1)
        if 2 in indice or 5 in indice:
            vecallindex[2]=indice_hart(x,clall)
        if 3 in indice or 5 in indice:
            vecallindex[3]=indice_ratkowsky(x,cl1)
        if 4 in indice or 5 in indice:
            vecallindex[4]=indice_ball(x,cl1)
        if 5 in indice:
            vecallindex=vecallindex
            colnames=indexes[0:5]
        else:
            vecallindex=[i for i in vecallindex if vecallindex.index(i) in indice]
            colnames=[indexes[x] for x in indice]
        res=pd.DataFrame(np.array([vecallindex]),columns=colnames)
        return res

    # C-index
    def Indice_cindex(d,cl):
        v_max=[1]*(np.max(cl)+1)
        v_min=[1]*(np.max(cl)+1)
        DU=0
        r=0
        for i in range(np.max(cl)+1):
            n=np.sum(cl==i)
            if n>1:
                t1=d[list(np.where(cl==i)[0])]
                t=t1[:,list(np.where(cl==i)[0])]
                DU=DU+np.sum(t)/2
                v_max[i]=np.max(t)
                if np.sum(t==0)==n:
                    v_min[i]=np.min(t[t!=0])
                else:
                    v_min[i]=0
                r=r+n*(n-1)/2
        Dmin = np.min(v_min)
        Dmax = np.max(v_max)
        if Dmin==Dmax:
            result=NaN
        else:
            result=(DU-r*Dmin)/(Dmax*r-Dmin*r)
        return result

    # DB
    def Indice_DB(x,cl,p=2,q=2):
        n=len(cl)
        k=np.max(cl)+1
        centers=np.zeros((k,x.shape[1]))
        for i in range(k):
            for j in range(x.shape[1]):
                centers[i,j]=x[np.where(cl==i)[0]].mean(axis=0)[j]
        S=[0]*k
        for i in range(k):
            if np.sum(cl==i)>1:
                S[i]=np.mean(np.sqrt(((x[np.where(cl==i)[0]]-centers[i])**2).sum(axis=1))**q)**(1/q)
            else:
                S[i]=0
        M=dist(centers,upper=True)
        M[M==0.0]=np.nan
        R=np.tile(np.nan,(k,k))
        r=[0]*k
        for i in range(k):
            for j in range(k):
                R[i,j]=(S[i]+S[j])/M[i,j]
            r[i]=np.max(R[i][np.isfinite(R[i])])
        DB=np.mean(r)
        return DB

    # Silhouette 
    # used function sklearn.metrics.silhouette_score

    # Gap
    def Indice_Gap(X,clall,reference_distribution='unif',B=10,method='ward'):
        def pcsm(X):
                Xmm=x.mean(axis=0)
                X=X-Xmm
                VT = np.linalg.svd(X)[2]
                Xs=X.dot(VT.T)
                ma=Xs.max(axis=0)
                mi=Xs.min(axis=0)
                np.random.seed(123)
                Xnew=uniform.rvs(mi,ma,size=(Xs.shape[0],Xs.shape[1]))
                Xt=Xnew@VT
                Xt=Xt+Xmm
                return Xt
        def Gap(X,cl,reference_distribution,B,method):
            ClassNr=np.max(cl)+1
            Wk0=0
            WkB=[0]*B
            for bb in range(B):
                if reference_distribution=="unif":
                    ma=X.max(axis=0)
                    mi=X.min(axis=0)
                    np.random.seed(123)
                    Xnew=uniform.rvs(mi,ma,size=(X.shape[0],X.shape[1]))
                if reference_distribution=="pc":
                    Xnew=pcsm(X)   
                if bb==0:
                    pp=cl
                    if ClassNr==len(cl):
                        pp2=list(range(ClassNr))
                    if method=='KMeans':
                        kmeans=KMeans(n_clusters=ClassNr,random_state=123,max_iter=100,n_init=n_init).fit(Xnew)
                        pp2=kmeans.labels_
                    if method=='single' or method=='complete' or method=='average' or method=='weighted' or method=='centroid' or method=='ward' or method=='median':
                        md_=dist(Xnew)
                        hc_=linkage(md_,method=method)
                        pp2=cluster.hierarchy.cut_tree(hc_,n_clusters=ClassNr)        
                    if ClassNr>1:
                        for zz in range(ClassNr):
                            Xuse=X[np.where(pp==zz)[0]]
                            Wk0+=np.sum(np.var(Xuse,axis=0))*(len(pp[pp==zz])-1)/(X.shape[0]-ClassNr)
                            Xuse2=Xnew[np.where(pp2==zz)[0]]
                            WkB[bb]+=np.sum(np.var(Xuse2,axis=0))*(len(pp2[pp2==zz])-1)/(X.shape[0]-ClassNr)
                    if ClassNr==1:
                        Wk0=np.sum(np.var(X,axis=0))
                        WkB[bb]=np.sum(np.var(Xnew,axis=0))
                if bb>=1:
                    if ClassNr==len(cl):
                        pp2=list(range(ClassNr))
                    if method=='KMeans':
                        kmeans=KMeans(n_clusters=ClassNr,random_state=123,max_iter=100,n_init=n_init).fit(Xnew)
                        pp2=kmeans.labels_
                    if method=='single' or method=='complete' or method=='average' or method=='weighted' or method=='centroid' or method=='ward' or method=='median':
                        md_=dist(Xnew)
                        hc_=linkage(md_,method=method)
                        pp2=cluster.hierarchy.cut_tree(hc_,n_clusters=ClassNr)     
                    if ClassNr>1:
                        for zz in range(ClassNr):
                            Xuse2=Xnew[np.where(pp2==zz)[0]]
                            WkB[bb]+=np.sum(np.var(Xuse2,axis=0))*(len(pp2[pp2==zz]))/(X.shape[0]-ClassNr)
                    if ClassNr==1:
                        WkB[bb]=np.sum(np.var(Xnew,axis=0))
            Sgap=np.mean(np.log(WkB))-np.log(Wk0)
            Sdgap=np.sqrt(1+1/B)*np.sqrt(np.var(np.log(WkB)))*np.sqrt((B-1)/B)
            result={'Sgap':Sgap,'Sdgap':Sdgap}
            return result
        gap1=Gap(X,clall[0],reference_distribution=reference_distribution,B=B,method=method)
        gap=gap1['Sgap']
        gap2=Gap(X,clall[1],reference_distribution=reference_distribution,B=B,method=method)
        diffu=gap-(gap2['Sgap']-gap2['Sdgap'])
        res={'gap':gap,'diffu':diffu}
        return res

    # SD, sdbw
    def Index_sdindex(x,clmax,cl):
        Alpha=Dis(clmax,x)
        Scatt=average_scattering(cl,x)['scatt']
        Dis0=Dis(cl,x)
        SD_indice=Alpha*Scatt+Dis0
        return SD_indice

    def Index_SDbw(x,cl):
        Scatt=average_scattering(cl,x)['scatt']
        Dens_bw=density_bw(cl,x)
        SDbw=Scatt+Dens_bw
        return SDbw

    # D index
    def Index_Dindex(cl,x):
        distance=density_clusters(cl,x)['distance']
        n=len(distance)
        S=0
        for i in range(n):
            S+=distance[i]
        inertieIntra=S/n
        return inertieIntra

    # Dunn index
    def Index_dunn(md,clusters):
        nc=np.max(clusters)+1
        interClust=np.tile(np.nan,(nc,nc))
        intraClust=[np.nan]*nc
        for i in range(nc):
            c1=np.where(clusters==i)[0]
            for j in range(i,nc):
                if j==i:
                    dist1=md[c1]
                    dist2=dist1[:,[c1]]
                    intraClust[i]=np.max(dist2)
                if j>i:
                    c2=np.where(clusters==j)[0]
                    dist_=md[c2]
                    dist_=dist_[:,[c1]]
                    dist_[dist_==0]=np.nan
                    interClust[i,j]=np.nanmin(dist_)
        dunn=np.nanmin(interClust)/np.nanmax(intraClust)
        return dunn

    # The calculation of statistics to determine the best number of clusters for the data
    for nc in range(min_nc,max_nc+1):
        if (1 in method_ or 2 in method_ or 3 in method_ or 4 in method_ or 5 in method_ or 6 in method_ or 7 in method_):
            cl1=cluster.hierarchy.cut_tree(hc,n_clusters=nc)
            cl2=cluster.hierarchy.cut_tree(hc,n_clusters=nc+1)
            clall=np.vstack((list(flatten(cl1)),list(flatten(cl2)))).astype('int')
            clmax=cluster.hierarchy.cut_tree(hc,n_clusters=max_nc)
            if nc>=2:
                cl0=cluster.hierarchy.cut_tree(hc,n_clusters=nc-1)
                clall1=np.vstack((list(flatten(cl0)),list(flatten(cl1)),list(flatten(cl2)))).astype('int')
            if nc==1:
                cl0=np.tile(np.nan,(1,nn))
                clall1=np.vstack((list(flatten(cl0)),list(flatten(cl1)),list(flatten(cl2))))
        if (0 in method_):
            cl2=KMeans(n_clusters=nc+1,random_state=123,n_init=n_init).fit(jeu).labels_
            clmax=KMeans(n_clusters=max_nc,random_state=123,n_init=n_init).fit(jeu).labels_
            if nc>2:
                cl1=KMeans(n_clusters=nc,random_state=123,n_init=n_init).fit(jeu).labels_
                clall=np.vstack((list(flatten(cl1)),list(flatten(cl2)))).astype('int')
                cl0=KMeans(n_clusters=nc-1,random_state=123,n_init=n_init).fit(jeu).labels_
                clall1=np.vstack((list(flatten(cl0)),list(flatten(cl1)),list(flatten(cl2)))).astype('int')
            if nc==2:
                cl1=KMeans(n_clusters=nc,random_state=123,n_init=n_init).fit(jeu).labels_
                clall=np.vstack((list(flatten(cl1)),list(flatten(cl2)))).astype('int')
                cl0=[1]*jeu.shape[0]
                clall1=np.vstack((list(flatten(cl0)),list(flatten(cl1)),list(flatten(cl2)))).astype('int')
            if nc==1:
                raise ValueError('Number of clusters must be higher than 2')
        j=pd.DataFrame(np.unique(cl1,return_counts=True))
        s=sum(j.loc[1]==1)
        j2=pd.DataFrame(np.unique(cl2,return_counts=True))
        s2=sum(j2.loc[1]==1)
        if (2 in indice or 30 in indice or 31 in indice):
            res.iloc[nc-min_nc,2]=Indices_Traces(jeu,md,clall1,index=["hart"]).values
        if (3 in indice or 30 in indice or 31 in indice):
            res.iloc[nc-min_nc,3]=Indices_WBT(jeu,cl1,TT,ss,vv)['ccc']
        if (4 in indice or 30 in indice or 31 in indice):
            res.iloc[nc-min_nc,4]=Indices_WBT(jeu,cl1,TT,ss,vv)['scott']
        if (5 in indice or 30 in indice or 31 in indice):
            res.iloc[nc-min_nc,5]=Indices_WBT(jeu,cl1,TT,ss,vv)['marriot']
        if (6 in indice or 30 in indice or 31 in indice):
            res.iloc[nc-min_nc,6]=Indices_WBT(jeu,cl1,TT,ss,vv)['trcovw']
        if (7 in indice or 30 in indice or 31 in indice):
            res.iloc[nc-min_nc,7]=Indices_WBT(jeu,cl1,TT,ss,vv)['tracew']
        if (8 in indice or 30 in indice or 31 in indice):
            res.iloc[nc-min_nc,8]=Indices_WBT(jeu,cl1,TT,ss,vv)['friedman']
        if (9 in indice or 30 in indice or 31 in indice):
            res.iloc[nc-min_nc,9]=Indices_WBT(jeu,cl1,TT,ss,vv)['rubin']
        if (13 in indice or 30 in indice or 31 in indice):
            res.iloc[nc-min_nc,13]=Indices_WKWL(jeu,cl1,cl2)['duda']
        if (14 in indice or 30 in indice or 31 in indice):
            res.iloc[nc-min_nc,14]=Indices_WKWL(jeu,cl1,cl2)['pseudot2']
        if (15 in indice or 30 in indice or 31 in indice):
            res.iloc[nc-min_nc,15]=beale=Indices_WKWL(jeu,cl1,cl2)['beale']
        if (13 in indice or 14 in indice or 15 in indice or 30 in indice or 31 in indice):
            NM=Indices_WKWL(jeu,cl1,cl2)['NM']
            NK=Indices_WKWL(jeu,cl1,cl2)['NK']
            NL=Indices_WKWL(jeu,cl1,cl2)['NL']
            zz=3.20 # Best standard score in Milligan and Cooper 1985
            zzz=zz*np.sqrt(2*(1-8/((math.pi**2)*pp))/(NM*pp))
        if (13 in indice or 30 in indice or 31 in indice):
            resCritical.iloc[nc-min_nc,0]=critValue=1-(2/(math.pi*pp))-zzz
        if (14 in indice or 30 in indice or 31 in indice):
            critValue=1-(2/(math.pi*pp))-zzz
            resCritical.iloc[nc-min_nc,1]=((1-critValue)/critValue)*(NK+NL-2)
        if (15 in indice or 30 in indice or 31 in indice):
            df2=(NM-2)*pp
            resCritical.iloc[nc-min_nc,2]=1-f.cdf(beale,pp,df2)
        if (17 in indice or 30 in indice or 31 in indice):
            res.iloc[nc-min_nc,17]=Indices_Traces(jeu,md,clall1,index=["ball"]).values
        if (18 in indice or 30 in indice or 31 in indice):
            res.iloc[nc-min_nc,18]=Indice_ptbiserial(jeu,md,cl1)
        if (19 in indice or 31 in indice):
            resultSGAP=Indice_Gap(jeu,clall,reference_distribution='unif',B=10,method=method)
            res.iloc[nc-min_nc,19]=resultSGAP['gap']
            resCritical.iloc[nc-min_nc,3]=resultSGAP['diffu']
        if nc>=2:
            if (0 in indice or 30 in indice or 31 in indice):
                res.iloc[nc-min_nc,0]=Indices_Traces(jeu,md,clall1,index=["kl"]).values
            if (1 in indice or 30 in indice or 31 in indice):
                res.iloc[nc-min_nc,1]=Indices_Traces(jeu,md,clall1,index=["ch"]).values
            if (10 in indice or 30 in indice or 31 in indice):
                res.iloc[nc-min_nc,10]=Indice_cindex(md,cl1)
            if (11 in indice or 30 in indice or 31 in indice):
                res.iloc[nc-min_nc,11]=Indice_DB(jeu,cl1)
            if (12 in indice or 30 in indice or 31 in indice):  
                res.iloc[nc-min_nc,12]=metrics.silhouette_score(jeu,list(flatten(cl1)))
            if (16 in indice or 30 in indice or 31 in indice):
                res.iloc[nc-min_nc,16]=Indices_Traces(jeu,md,clall1,index=["ratkowsky"]).values
            if (20 in indice or 30 in indice or 31 in indice):
                res.iloc[nc-min_nc,20]=Index_15and28(cl1,cl2,md)['frey']
            if (21 in indice or 30 in indice or 31 in indice):
                res.iloc[nc-min_nc,21]=Index_15and28(cl1,cl2,md)['mcclain']
            if (22 in indice or 31 in indice):
                res.iloc[nc-min_nc,22]=Index_sPlussMoins(cl1,jeu)['gamma']
            if (23 in indice or 31 in indice):
                res.iloc[nc-min_nc,23]=Index_sPlussMoins(cl1,jeu)['gplus']
            if (24 in indice or 31 in indice):
                res.iloc[nc-min_nc,24]=Index_sPlussMoins(cl1,jeu)['tau']
            if (25 in indice or 30 in indice or 31 in indice):  
                res.iloc[nc-min_nc,25]=Index_dunn(md,cl1)
            if (26 in indice or 30 in indice or 31 in indice):  
                res.iloc[nc-min_nc,26]=index_hubert(jeu,cl1)
            if (27 in indice or 30 in indice or 31 in indice):  
                res.iloc[nc-min_nc,27]=Index_sdindex(jeu,clmax,cl1)
            if (28 in indice or 30 in indice or 31 in indice):  
                res.iloc[nc-min_nc,28]=Index_Dindex(cl1,jeu)
            if (29 in indice or 30 in indice or 31 in indice):  
                res.iloc[nc-min_nc,29]=Index_SDbw(jeu,cl1)
        else:
            res.iloc[nc-min_nc,0]=np.nan
            res.iloc[nc-min_nc,1]=np.nan
            res.iloc[nc-min_nc,10]=np.nan
            res.iloc[nc-min_nc,11]=np.nan
            res.iloc[nc-min_nc,12]=np.nan
            res.iloc[nc-min_nc,16]=np.nan
            res.iloc[nc-min_nc,20]=np.nan
            res.iloc[nc-min_nc,21]=np.nan
            res.iloc[nc-min_nc,22]=np.nan
            res.iloc[nc-min_nc,23]=np.nan
            res.iloc[nc-min_nc,24]=np.nan
            res.iloc[nc-min_nc,25]=np.nan
            res.iloc[nc-min_nc,26]=np.nan
            res.iloc[nc-min_nc,27]=np.nan
            res.iloc[nc-min_nc,28]=np.nan
            res.iloc[nc-min_nc,29]=np.nan

    # Best Number of Clusters 
    nCl=[]
    nInd=[]
    colNames=[]
    if (0 in indice or 30 in indice or 31 in indice):
        nc_KL=res.iloc[:,0].idxmax(axis=0)
        indice_KL=res.iloc[:,0].max()
        nCl.append(nc_KL)
        nInd.append(indice_KL)
        colNames.append("KL")
    if (1 in indice or 30 in indice or 31 in indice):
        nc_CH=res.iloc[:,1].idxmax(axis=0)
        indice_CH=res.iloc[:,1].max()
        nCl.append(nc_CH)
        nInd.append(indice_CH)
        colNames.append("CH")
    if (3 in indice or 30 in indice or 31 in indice):
        nc_CCC=res.iloc[:,3].idxmax(axis=0)
        indice_CCC=res.iloc[:,3].max()
        nCl.append(nc_CCC)
        nInd.append(indice_CCC)
        colNames.append("CCC")
    if (11 in indice or 30 in indice or 31 in indice):
        nc_DB=res.iloc[:,11].idxmin(axis=0)
        indice_DB=res.iloc[:,11].min()
        nCl.append(nc_DB)
        nInd.append(indice_DB)
        colNames.append("DB")
    if (12 in indice or 30 in indice or 31 in indice):
        nc_Silhouette=res.iloc[:,12].idxmax(axis=0)
        indice_Silhouette=res.iloc[:,12].max()
        nCl.append(nc_Silhouette)
        nInd.append(indice_Silhouette)
        colNames.append("Silhouette")
    if (19 in indice or 31 in indice):
        found=False
        for ncG in range(min_nc,max_nc):
            if resCritical.iloc[ncG-min_nc,3]>=0 and not(found):
                ncGap=ncG
                indiceGap=res.iloc[ncG-min_nc,19]
                found=True
        if found:
            nc_Gap=ncGap
            indice_Gap=indiceGap
            nCl.append(nc_Gap)
            nInd.append(indice_Gap)
            colNames.append("Gap")
        else:
            nc_Gap=np.nan
            indice_Gap=np.nan
            nCl.append(nc_Gap)
            nInd.append(indice_Gap)
            colNames.append("Gap")
    if (13 in indice or 30 in indice or 31 in indice):
        foundDuda=False
        for ncD in range(min_nc,max_nc):
            if res.iloc[ncD-min_nc,13]>=resCritical.iloc[ncD-min_nc,0] and not(foundDuda):
                ncDuda=ncD
                indiceDuda=res.iloc[ncD-min_nc,13]
                foundDuda=True
        if foundDuda:
            nc_Duda=ncDuda
            indice_Duda=indiceDuda
            nCl.append(nc_Duda)
            nInd.append(indice_Duda)
            colNames.append("Duda")
        else:
            nc_Duda=np.nan
            indice_Duda=np.nan
            nCl.append(nc_Duda)
            nInd.append(indice_Duda)
            colNames.append("Duda")
    if (14 in indice or 30 in indice or 31 in indice):
        foundPseudo=False
        for ncP in range(min_nc,max_nc):
            if res.iloc[ncP-min_nc,14]<=resCritical.iloc[ncP-min_nc,1] and not(foundPseudo):
                ncPseudo=ncP
                indicePseudo=res.iloc[ncP-min_nc,14]
                foundPseudo=True
        if foundPseudo:
            nc_Pseudo=ncPseudo
            indice_Pseudo=indicePseudo
            nCl.append(nc_Pseudo)
            nInd.append(indice_Pseudo)
            colNames.append("PseudoT2")
        else:
            nc_Pseudo=np.nan
            indice_Pseudo=np.nan
            nCl.append(nc_Pseudo)
            nInd.append(indice_Pseudo)
            colNames.append("PseudoT2")
    if (15 in indice or 30 in indice or 31 in indice):
        foundBeale=False
        for ncB in range(min_nc,max_nc):
            if resCritical.iloc[ncB-min_nc,2]>=alphaBeale and not(foundBeale):
                ncBeale=ncB
                indiceBeale=res.iloc[ncB-min_nc,15]
                foundBeale=True
        if foundBeale:
            nc_Beale=ncBeale
            indice_Beale=indiceBeale
            nCl.append(nc_Beale)
            nInd.append(indice_Beale)
            colNames.append("Beale")
        else:
            nc_Beale=np.nan
            indice_Beale=np.nan
            nCl.append(nc_Beale)
            nInd.append(indice_Beale)
            colNames.append("Beale")
    if (18 in indice or 30 in indice or 31 in indice):
        nc_ptbiserial=res.iloc[:,18].idxmax(axis=0)
        indice_ptbiserial=res.iloc[:,18].max()
        nCl.append(nc_ptbiserial)
        nInd.append(indice_ptbiserial)
        colNames.append("PtBiserial")
    if (20 in indice or 30 in indice or 31 in indice):
        foundFrey=False
        foundNC=[]
        foundIndice=[]
        i=1
        for ncF in range(min_nc,max_nc):
            if res.iloc[ncF-min_nc,20]<1:
                ncFrey=ncF-1
                indiceFrey=res.iloc[ncF-min_nc,20]
                foundFrey=True
                foundNC.append(ncFrey)
                foundIndice.append(indiceFrey)
                i+=1
        if foundFrey:
            nc_Frey=foundNC[0]
            indice_Frey=foundIndice[0]
            nCl.append(nc_Frey)
            nInd.append(indice_Frey)
            colNames.append("Frey")
        else:
            nc_Frey=np.nan
            indice_Frey=np.nan
            nCl.append(nc_Frey)
            nInd.append(indice_Frey)
            colNames.append("Frey")
            print("Frey index : No clustering structure in this data set")
    if (21 in indice or 30 in indice or 31 in indice):
        nc_McClain=res.iloc[:,21].idxmin(axis=0)
        indice_McClain=res.iloc[:,21].min()
        nCl.append(nc_McClain)
        nInd.append(indice_McClain)
        colNames.append("McClain")
    if (22 in indice or 30 in indice or 31 in indice):
        nc_Gamma=res.iloc[:,22].idxmax(axis=0)
        indice_Gamma=res.iloc[:,22].max()
        nCl.append(nc_Gamma)
        nInd.append(indice_Gamma)
        colNames.append("Gamma")
    if (23 in indice or 30 in indice or 31 in indice):
        nc_Gplus=res.iloc[:,23].idxmin(axis=0)
        indice_Gplus=res.iloc[:,23].min()
        nCl.append(nc_Gplus)
        nInd.append(indice_Gplus)
        colNames.append("Gplus")
    if (24 in indice or 30 in indice or 31 in indice):
        nc_Tau=res.iloc[:,24].idxmax(axis=0)
        indice_Tau=res.iloc[:,24].max()
        nCl.append(nc_Tau)
        nInd.append(indice_Tau)
        colNames.append("Tau")

    DiffLev=np.zeros((max_nc-min_nc+1,12))
    DiffLev[::,0]=np.array(range(min_nc,max_nc+1))

    if (2 in indice or 4 in indice or 5 in indice or 6 in indice or 7 in indice or 8 in indice or 9 in indice or 17 in indice or 26 in indice or 10 in indice or 28 in indice or 30 in indice or 31 in indice):
        for nc3 in range(min_nc,max_nc):
            if nc3==min_nc:
                DiffLev[nc3-min_nc,1]=abs(res.iloc[nc3-min_nc,2]-np.nan) # Hartigan
                DiffLev[nc3-min_nc,2]=abs(res.iloc[nc3-min_nc,4]-np.nan) #Scott
                DiffLev[nc3-min_nc,3]=abs(res.iloc[nc3-min_nc,5]-np.nan) #Marriot
                DiffLev[nc3-min_nc,4]=abs(res.iloc[nc3-min_nc,6]-np.nan) #Trcovw
                DiffLev[nc3-min_nc,5]=abs(res.iloc[nc3-min_nc,7]-np.nan) #Tracew
                DiffLev[nc3-min_nc,6]=abs(res.iloc[nc3-min_nc,8]-np.nan) #Friedman
                DiffLev[nc3-min_nc,7]=abs(res.iloc[nc3-min_nc,9]-np.nan) #Rubin
                DiffLev[nc3-min_nc,8]=abs(res.iloc[nc3-min_nc,17]-np.nan) #Ball
                DiffLev[nc3-min_nc,9]=abs(res.iloc[nc3-min_nc,26]-np.nan) #Hubert
                DiffLev[nc3-min_nc,11]=abs(res.iloc[nc3-min_nc,28]-np.nan) #D index
            else:
                if nc3==max_nc:
                    DiffLev[nc3-min_nc,1]=abs(res.iloc[nc3-min_nc,2]-res.iloc[nc3-min_nc-1,2]) # Hartigan
                    DiffLev[nc3-min_nc,2]=abs(res.iloc[nc3-min_nc,4]-res.iloc[nc3-min_nc-1,4]) #Scott
                    DiffLev[nc3-min_nc,3]=abs(res.iloc[nc3-min_nc,5]-np.nan) #Marriot
                    DiffLev[nc3-min_nc,4]=abs(res.iloc[nc3-min_nc,6]-res.iloc[nc3-min_nc-1,6]) #Trcovw
                    DiffLev[nc3-min_nc,5]=abs(res.iloc[nc3-min_nc,7]-np.nan) #Tracew
                    DiffLev[nc3-min_nc,6]=abs(res.iloc[nc3-min_nc,8]-res.iloc[nc3-min_nc-1,8]) #Friedman
                    DiffLev[nc3-min_nc,7]=abs(res.iloc[nc3-min_nc,9]-np.nan) #Rubin
                    DiffLev[nc3-min_nc,8]=abs(res.iloc[nc3-min_nc,17]-res.iloc[nc3-min_nc-1,17]) #Ball
                    DiffLev[nc3-min_nc,9]=abs(res.iloc[nc3-min_nc,26]-np.nan) #Hubert
                    DiffLev[nc3-min_nc,11]=abs(res.iloc[nc3-min_nc,28]-np.nan) #D index
                else:
                    DiffLev[nc3-min_nc,1]=abs(res.iloc[nc3-min_nc,2]-res.iloc[nc3-min_nc-1,2]) # Hartigan
                    DiffLev[nc3-min_nc,2]=abs(res.iloc[nc3-min_nc,4]-res.iloc[nc3-min_nc-1,4]) #Scott
                    DiffLev[nc3-min_nc,3]=abs(res.iloc[nc3-min_nc,5]-res.iloc[nc3-min_nc,5]-res.iloc[nc3-min_nc,5]-res.iloc[nc3-min_nc-1,5]) #Marriot
                    DiffLev[nc3-min_nc,4]=abs(res.iloc[nc3-min_nc,6]-res.iloc[nc3-min_nc-1,6]) #Trcovw
                    DiffLev[nc3-min_nc,5]=abs(res.iloc[nc3-min_nc+1,7]-res.iloc[nc3-min_nc,7]-res.iloc[nc3-min_nc,7]-res.iloc[nc3-min_nc-1,7]) #Tracew
                    DiffLev[nc3-min_nc,6]=abs(res.iloc[nc3-min_nc,8]-res.iloc[nc3-min_nc-1,8]) #Friedman
                    DiffLev[nc3-min_nc,7]=abs(res.iloc[nc3-min_nc+1,9]-res.iloc[nc3-min_nc,9]-res.iloc[nc3-min_nc,9]-res.iloc[nc3-min_nc-1,9]) #Rubin
                    DiffLev[nc3-min_nc,8]=abs(res.iloc[nc3-min_nc,17]-res.iloc[nc3-min_nc-1,17]) #Ball
                    DiffLev[nc3-min_nc,9]=abs(res.iloc[nc3-min_nc,26]-res.iloc[nc3-min_nc-1,26]) #Hubert
                    DiffLev[nc3-min_nc,11]=abs(res.iloc[nc3-min_nc+1,28]-res.iloc[nc3-min_nc,28]-res.iloc[nc3-min_nc,28]-res.iloc[nc3-min_nc-1,28]) #D index

    DiffLev=pd.DataFrame(DiffLev)
    DiffLev=DiffLev.set_index(0)
    DiffLev.index.names=[None]

    if (2 in indice or 30 in indice or 31 in indice):
        nc_Hartigan=int(DiffLev.iloc[:,0].idxmax(axis=0))
        indice_Hartigan=DiffLev.iloc[:,0].max()
        nCl.append(nc_Hartigan)
        nInd.append(indice_Hartigan)
        colNames.append("Hartigan")
    if (16 in indice or 30 in indice or 31 in indice):
        nc_Ratkowsky=res.iloc[:,16].idxmax(axis=0)
        indice_Ratkowsky=res.iloc[:,16].max()
        nCl.append(nc_Ratkowsky)
        nInd.append(indice_Ratkowsky)
        colNames.append("Ratkowsky")
    if (10 in indice or 30 in indice or 31 in indice):
        nc_cindex=res.iloc[:,10].idxmin(axis=0)
        indice_cindex=res.iloc[:,10].min()
        nCl.append(nc_cindex)
        nInd.append(indice_cindex)
        colNames.append("Cindex")
    if (4 in indice or 30 in indice or 31 in indice):
        nc_Scott=int(DiffLev.iloc[:,1].idxmax(axis=0))
        indice_Scott=DiffLev.iloc[:,1].max()
        nCl.append(nc_Scott)
        nInd.append(indice_Scott)
        colNames.append("Scott")
    if (5 in indice or 30 in indice or 31 in indice):
        nc_Marriot=int(DiffLev.iloc[:,2].idxmax(axis=0))
        indice_Marriot=round(DiffLev.iloc[:,2].max(),1)
        nCl.append(nc_Marriot)
        nInd.append(indice_Marriot)
        colNames.append("Marriot")
    if (6 in indice or 30 in indice or 31 in indice):
        nc_TrCovW=int(DiffLev.iloc[:,3].idxmax(axis=0))
        indice_TrCovW=round(DiffLev.iloc[:,3].max(),1)
        nCl.append(nc_TrCovW)
        nInd.append(indice_TrCovW)
        colNames.append("TrCovW")
    if (7 in indice or 30 in indice or 31 in indice):
        nc_TraceW=int(DiffLev.iloc[:,4].idxmax(axis=0))
        indice_TraceW=round(DiffLev.iloc[:,4].max(),2)
        nCl.append(nc_TraceW)
        nInd.append(indice_TraceW)
        colNames.append("TraceW")
    if (8 in indice or 30 in indice or 31 in indice):
        nc_Friedman=int(DiffLev.iloc[:,5].idxmax(axis=0))
        indice_Friedman=round(DiffLev.iloc[:,5].max(),2)
        nCl.append(nc_Friedman)
        nInd.append(indice_Friedman)
        colNames.append("Friedman")
    if (9 in indice or 30 in indice or 31 in indice):
        nc_Rubin=int(DiffLev.iloc[:,6].idxmin(axis=0))
        indice_Rubin=DiffLev.iloc[:,6].min()
        nCl.append(nc_Rubin)
        nInd.append(indice_Rubin)
        colNames.append("Rubin")
    if (17 in indice or 30 in indice or 31 in indice):
        nc_Ball=int(DiffLev.iloc[:,7].idxmax(axis=0))
        indice_Ball=DiffLev.iloc[:,7].max()
        nCl.append(nc_Ball)
        nInd.append(indice_Ball)
        colNames.append("Ball")
    if (25 in indice or 30 in indice or 31 in indice):
        nc_Dunn=res.iloc[:,25].idxmax(axis=0)
        indice_Dunn=res.iloc[:,25].max()
        nCl.append(nc_Dunn)
        nInd.append(indice_Dunn)
        colNames.append("Dunn")
    if (27 in indice or 30 in indice or 31 in indice):
        nc_sdindex=res.iloc[:,27].idxmin(axis=0)
        indice_sdindex=res.iloc[:,27].min()
        nCl.append(nc_sdindex)
        nInd.append(indice_sdindex)
        colNames.append("SDindex")
    if (29 in indice or 30 in indice or 31 in indice):
        nc_SDbw=res.iloc[:,29].idxmin(axis=0)
        indice_SDbw=res.iloc[:,29].min()
        nCl.append(nc_SDbw)
        nInd.append(indice_SDbw)
        colNames.append("SDbw")
    if (26 in indice or 30 in indice or 31 in indice):
        nc_Hubert=0
        indice_Hubert=0.0
        nCl.append(nc_Hubert)
        nInd.append(indice_Hubert)
        colNames.append("Hubert")
        if plotInd is True:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7))
            ax1.plot(x_axis, res.iloc[:,26], 'bs-')
            ax1.set_ylabel('Hubert Statistic values',fontsize=14)
            ax1.set_xlabel('Number of clusters',fontsize=14)
            ax2.plot(x_axis, DiffLev.iloc[:,8], 'rs-')
            ax2.set_ylabel('Hubert statistic second differences',fontsize=14)
            ax2.set_xlabel('Number of clusters',fontsize=14)
            plt.show()
            print("*** : The Hubert index is a graphical method of determining the number of clusters.\n      In the plot of Hubert index, we seek a significant knee that corresponds to a\n      significant increase of the value of the measure i.e the significant peak in\n      Hubert index second differences plot.")
    if (28 in indice or 30 in indice or 31 in indice):
        nc_Dindex=0
        indice_Dindex=0.0
        nCl.append(nc_Dindex)
        nInd.append(indice_Dindex)
        colNames.append("Dindex")
        if plotInd is True:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7))
            ax1.plot(x_axis, res.iloc[:,28], 'bs-')
            ax1.set_ylabel('Dindex Values',fontsize=14)
            ax1.set_xlabel('Number of clusters',fontsize=14)
            ax2.plot(x_axis, DiffLev.iloc[:,10], 'rs-')
            ax2.set_ylabel('Second differences Dindex Values',fontsize=14)
            ax2.set_xlabel('Number of clusters',fontsize=14)
            plt.show()
            print("*** : The D index is a graphical method of determining the number of clusters.\n      In the plot of D index, we seek a significant knee (the significant peak\n      in Dindex second differences plot) that corresponds to a significant\n      increase of the value of the measure.")

    dfIndex=["Number clusters","Value Index"]

    # Displaying results 
    res=res.round(4)    
    dictResCrit=dict([(13,0),(14,1),(15,2),(19,3)])
    if (31 in indice):
        resCritical=resCritical
    if (30 in indice):
        resCritical=resCritical.iloc[:,0:3]
    if (all(x < 30 for x in indice)):
        indResCrit=[i for i in [dictResCrit.get(key) for key in indice] if i is not None]
        resCritical=resCritical.iloc[:,indResCrit]

    if (31 in indice):
        res=res
        nCl=nCl
        nInd=nInd
        colNames=colNames
    if (30 in indice):
        indRes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,25,26,27,28,29]
        res=res.iloc[:,indRes]
        nCl=nCl
        nInd=nInd
        colNames=colNames
    if (all(x < 30 for x in indice)):
        res=res.iloc[:,indice]
        nCl=nCl
        nInd=nInd
        colNames=colNames
    result=np.stack((nCl,nInd))
    resultats=pd.DataFrame(result,index=dfIndex,columns=colNames)
    resultats=resultats.T.astype({'Number clusters': 'int'},errors='ignore')
    resultats['Value Index']=resultats['Value Index'].round(3)

    # Summary results
    if PrintRes==True:
        resultFin=resultats['Number clusters'].value_counts()
        print("*******************************************************************")
        print("* Among all indices:")
        for i in range(len(resultFin)):
            k=resultFin.index[i].astype('int')
            p=resultFin.values[i]
            print("* %d  proposed %r as the best number of clusters" % (p,k))
        print("                  ***** Conclusion *****                           ")
        print("* According to the majority rule, the best number of clusters is %d"% (resultFin.index[0].astype('int') if resultFin.index.size != 0 else 0))
        print("*******************************************************************")

    # The Best partition
    resultFin=resultats['Number clusters'].value_counts()
    if resultFin.index.size != 0:
        v=resultFin.index[0].astype('int')
    if resultFin.index.size == 0:
        raise ValueError('No clustering structure in this data set')
    if (1 in method_ or 2 in method_ or 3 in method_ or 4 in method_ or 5 in method_ or 6 in method_ or 7 in method_):
        partition=cluster.hierarchy.cut_tree(hc,n_clusters=v)
    if (0 in method_):
        partition=KMeans(n_clusters=v,random_state=123,n_init=n_init).fit(jeu).labels_
    if (0 in indice or 1 in indice or 2 in indice or 3 in indice or 4 in indice or 5 in indice or 6 in indice or 7 in indice or 8 in indice or 9 in indice or 10 in indice or 11 in indice or 12 in indice or 13 in indice or 14 in indice or 15 in indice or 16 in indice or 17 in indice or 18 in indice or 19 in indice or 20 in indice or 21 in indice or 22 in indice or 23 in indice or 24 in indice or 25 in indice or 27 in indice or 29 in indice):
        if (1 in method_ or 2 in method_ or 3 in method_ or 4 in method_ or 5 in method_ or 6 in method_ 
            or 7 in method_):
            partition=cluster.hierarchy.cut_tree(hc,n_clusters=v)
        if (0 in method_):
            partition=KMeans(n_clusters=v,random_state=123,n_init=n_init).fit(jeu).labels_
    partition=np.array(list(flatten(partition)))

    # Final summary results 
    if (13 in indice or 14 in indice or 15 in indice or 19 in indice or 30 in indice or 31 in indice):
        resultsfinal={'Allindex':res,'AllCriticalValues':resCritical,'BestNc':resultats,'BestPartition':partition}
    if (26 in indice or 28 in indice):
        resultsfinal={'Allindex':res}
    if (0 in indice or 1 in indice or 2 in indice or 3 in indice or 4 in indice or 5 in indice or 6 in indice or 7 in indice or 8 in indice or 9 in indice or 10 in indice or 11 in indice or 12 in indice or 16 in indice or 17 in indice or 18 in indice or 20 in indice or 21 in indice or 22 in indice or 23 in indice or 24 in indice or 25 in indice or 27 in indice or 29 in indice):
        resultsfinal={'Allindex':res,'BestNc':resultats,'BestPartition':partition}
    return resultsfinal


def NbClustViz(res,figsize = (10,6)):
    """
    Dertemines and visualize the optimal number of clusters using different methods.

    Parameters
    ----------
    res : [dict] object returned function NbClust.
    figsize : [int, int] a method used to change the dimension of plot window, width,
        height in inches (default figsize=(10,6)).

    Returns
    ----------
    plot of the optimal number of clusters using different methods.
    """
    x=res['BestNc']['Number clusters'].value_counts().index
    y=res['BestNc']['Number clusters'].value_counts().values
    dfres=pd.DataFrame({'cl':x,'ccl':y})
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=dfres,x='cl',y='ccl',color='royalblue')
    ax.set_title('Optimal number of clusters - k = %d'%(x[0]),fontsize=16)
    plt.grid(visible=True, axis='y')
    plt.xlabel('Number of clusters k',fontsize=14)
    plt.ylabel('Frequency among all indices',fontsize=14)
    plt.show()


def plot_dendrogram(Z,k=None,annotate=False,fontsize=10,figsize=(10,7),title=None,labelsize=12,**kwargs):
    """
    Plot the hierarchical clustering as a dendrogram.

    Parameters
    ----------  
    Z : [ndarray] The linkage matrix encoding the hierarchical clustering
                  to render as a dendrogram.
    k : [int] The number of clusters to form. By, default None.
    annotate: [bool] if True annotate the distance along each branch 
                  of the tree. By, default False.
    fontsize : [float, int] font size for the annotate the distance along 
                  each branch of the tree. By, default fontsize=10.
    figsize : [int, int] a method used to change the dimension of plot window, 
                  width, height in inches. By default, figsize=(10,7).
    title : [str] plot main title. By, default None.
    labelsize : [int] the size of x-axis labels.
    **kwargs : other arguments for function scipy.cluster.hierarchy.dendrogram.

    Returns
    ----------
    Plot the dendrogram.
    """
    if k is not None:
        if k<=1:
            raise ValueError('k must be > = 2')
        ct=Z[-(k-1),2]  
        ct1=np.mean([Z[-(k-1),2],Z[-(k),2]])
        plt.figure(figsize=figsize,facecolor='w')
        if title is None:
            title='Hierarchical Clustering Dendrogram'
        else:
            title=title
        plt.title(title)
        plt.ylabel('distance',fontsize=14)
        plt.xlabel('sample index or (cluster size)',fontsize=14)
        Dd=dendrogram(Z,color_threshold=ct,**kwargs)
        plt.tick_params(axis='x',which='major',labelsize=labelsize)
        plt.axhline(ct1,c='r',linestyle="--")
        if annotate is True:
            for i, d, c in zip(Dd['icoord'],Dd['dcoord'],Dd['color_list']):
                x=0.5*sum(i[1:3])
                y=d[1]
                plt.plot(x,y,'o',c=c)
                plt.annotate("%.3g" % y,(x,y),xytext=(0,-5),textcoords='offset points',va='top',ha='center',fontsize=fontsize)
    else:
        plt.figure(figsize=figsize,facecolor='w')
        if title is None:
            title='Hierarchical Clustering Dendrogram'
        else:
            title=title
        plt.title(title)
        plt.ylabel('distance',fontsize=14)
        plt.xlabel('sample index or (cluster size)',fontsize=14)
        Dd=dendrogram(Z,**kwargs)
        plt.tick_params(axis='x',which='major',labelsize=labelsize)
        if annotate is True:
            for i, d, c in zip(Dd['icoord'],Dd['dcoord'],Dd['color_list']):
                x=0.5*sum(i[1:3])
                y=d[1]
                plt.plot(x,y,'o',c=c)
                plt.annotate("%.3g" % y,(x,y),xytext=(0,-5),textcoords='offset points',va='top',ha='center',fontsize=fontsize)
    plt.show()


def plot_tanglegram(model1, model2, L=1.5, leaves_matching_method='order', figsize=(12,7), fontsize=12, **kwargs):
    """
    Plots a tanglegram plot of a side by side trees.

    Parameters
    ----------
    model1 : [object] Tree object (dendrogram), plotted on the left.
    model2 : [object] Tree object (dendrogram), plotted on the right.
    L : [float, int] The distance norm to use for measuring the distance between the 
                    two dendrograms for function coef_entanglement. It can be any 
                    positive number, often one will want to use 0, 1, 1.5 (default), 2.
    leaves_matching_method : [str] If 'order' (default) then use the old leaves order 
                    value for matching the leaves order value, if using "labels", then
                    use the labels for matching the leaves order value.
    figsize : [int, int] a method used to change the dimension of plot window, width, height in inches 
                    (default figsize=(12,7)).
    fontsize : [float, int] font size for the title.
    **kwargs : other arguments for function scipy.cluster.hierarchy.dendrogram.

    Returns
    ----------
    Plot of the tanglegram for comparing two dendrograms.
    """
    if type(model1).__name__=='DianaClustering':
        title1='{}'.format(type(model1).__name__)
    if type(model2).__name__=='DianaClustering':
        title2='{}'.format(type(model2).__name__)
    if type(model1).__name__=='AgglomerativeClustering':
        title1=r'{}''\n'r'with linkage = {}'.format(type(model1).__name__,model1.get_params()['linkage'])
    if type(model2).__name__=='AgglomerativeClustering':
        title2=r'{}''\n'r'with linkage = {}'.format(type(model2).__name__,model2.get_params()['linkage'])
    Z=linkage_matrix(model1)
    Z2=linkage_matrix(model2)
    fig = plt.figure(figsize=figsize,facecolor='w')
    coef_entang=coef_entanglement(model1, model2, L=L, leaves_matching_method=leaves_matching_method)
    coef_entang=round(coef_entang, 3)
    # Plot of dendrograms with use the old leaves order value
    if leaves_matching_method=='order':
        plt.subplot (1, 3, 1)
        if type(model1).__name__=='DianaClustering':
            dn=dendrogram(model1.linkage_matrix_,orientation='left', **kwargs)
        if type(model1).__name__=='AgglomerativeClustering':
            dn=dendrogram(Z,orientation='left', **kwargs)
        plt.title(title1,fontsize=fontsize)
        plt.style.use('classic')
        plt.subplot (1, 3, 3)
        if type(model2).__name__=='DianaClustering':
            dn2=dendrogram(model2.linkage_matrix_,orientation='right', **kwargs)
        if type(model2).__name__=='AgglomerativeClustering':
            dn2=dendrogram(Z2,orientation='right', **kwargs)
        plt.title(title2,fontsize=fontsize)
        plt.style.use('classic')
        x = dn['leaves']
        x2 = dn2['leaves']
        y = [0]*len(dn['leaves'])
        y2 = [1]*len(dn2['leaves'])
    # Plot of dendrograms with use the labels
    if leaves_matching_method=='labels':
        plt.subplot (1, 3, 1)
        if type(model1).__name__=='DianaClustering':
            dn=dendrogram(model1.linkage_matrix_,orientation='left',labels=model2.labels_, **kwargs)
        if type(model1).__name__=='AgglomerativeClustering':
            dn=dendrogram(Z,orientation='left',labels=model2.labels_, **kwargs)
        plt.title(title1,fontsize=fontsize)
        plt.style.use('classic')
        plt.subplot (1, 3, 3)
        if type(model2).__name__=='DianaClustering':
            dn2=dendrogram(model2.linkage_matrix_,orientation='right',labels=model2.labels_, **kwargs)
        if type(model2).__name__=='AgglomerativeClustering':
            dn2=dendrogram(Z2,orientation='right',labels=model2.labels_, **kwargs)
        plt.title(title2,fontsize=fontsize)
        plt.style.use('classic')
        x = model1.labels_
        x2 = model2.labels_
        y = [0]*len(model1.labels_)
        y2 = [1]*len(model2.labels_)
    # Plot of measures entanglement between two dendrograms.
    ax = plt.subplot (1, 3, 2)
    num_plots=len(model1.labels_)
    a = np.vstack((y, x)).T
    b = np.vstack((y2, x2)).T
    colormap = plt.cm.gist_ncar  
    colors = [colormap(i) for i in np.linspace(0, 1,len(ax.lines))]
    for i,j in enumerate(ax.lines):
        j.set_color(colors[i])
    ab_pairs = np.c_[a, b]
    ab_args = ab_pairs.reshape(-1, 2, 2).swapaxes(1, 2).reshape(-1, 2)
    plt.ylim([0, num_plots])
    plt.title(r'The coefficient''\n''of entanglement = {}'.format(coef_entang),fontsize=fontsize)
    plt.plot(*ab_args)
    plt.axis('off')
    plt.show()


def scale(data, center=True, scale=True):
    """
    This is a generic function which centers and scales the columns
    of a dataframe or array.
    
    Parameters
    ----------
    data :   [DataFrame, array] dataframe or array for centering, scaling.
    center : [bool] If True (default) then centering is done by subtracting
              the column means of data from their corresponding columns, 
              and if center=False, no centering is done.
    scale :  [bool] If True (default) then scentered columns of the 
              dataframe/array is divided by the root mean square. 
              If scale=False, no scaling is done.
    
    Returns
    ----------
    Dateframe or array which scaled and/or centered and mean values by columns.
      
    """
    x = data.copy()
    xsc=np.mean(x,axis=0)
    if center:
        x -= np.mean(x,axis=0)
    if scale and center:
        x /= np.std(x,axis=0)
    elif scale:
        x /= np.sqrt(np.sum(np.power(x,2),axis=0)/(x.shape[0]-1))
    return x, xsc