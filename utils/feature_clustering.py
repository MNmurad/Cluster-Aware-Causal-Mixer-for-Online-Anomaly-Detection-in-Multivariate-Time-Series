
import numpy as np
# import networkx as nx
from sklearn.cluster import KMeans, SpectralClustering
# from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import spectral_embedding
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
import numbers
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

def features_clustering(data, m, featureSimilarity):
    # data = data[:int(data.shape[0]*0.4), :] # selecting first 40%
    
    if m == 1:
        cluster_labels = np.zeros(data.shape[1], dtype = np.int32)
        return list(cluster_labels)
    
    if featureSimilarity == 0:
        ''' Consider the Distance matrix to cluster using Kmeans'''
        cluster_labels = kmeans_distance(data, m)
    
    elif featureSimilarity == 1:
        ''' Consider the feature's correlation profile to calculate the similarity matix'''
        # cluster_labels = profile_based_clustering(data, m)
        cluster_labels = detailedProfileBasedClustering(data, m, norm_laplacian = True)
    
    elif featureSimilarity == 2:
        ''' to get the similarity matrix, I consider the cosine among the features directly'''
        cluster_labels = featuresCosine(data, m, norm_laplacian = True)
    
    elif featureSimilarity == 3:
        ''' Similarity_matrix = |Corre. coeff matrix|'''
        cluster_labels = Rsimilarity(data, m, norm_laplacian = True)
    
    elif featureSimilarity == 4:
        ''' KMeans on |R|'''
        cluster_labels = kmeans_absCorrelation(data, m)
        
    elif featureSimilarity == 5:
        ''' Randomly assign the cluster index '''
        cluster_labels = randomClustering(data, m)
    
    else:
        raise NotImplementedError('Check Feature similarity type')
        
    return cluster_labels


def randomClustering(data, m):
    ''' Random Clustering'''
    ''' data: [L, C] numpy array'''
    channel = data.shape[1]
    n_clusters = m
    # rng = np.random.default_rng(seed=42)
    # clusters = rng.integers(low=0, high=n_clusters, size=channel, dtype=np.int16)
    clusters = np.random.randint(low = 0, high = n_clusters, size = channel, dtype = np.int16, seed = 42)
    # print(clusters)
    return list(clusters)


def featuresCosine(data, m, norm_laplacian = True):
    data = data.numpy()
    data_norm = data / np.linalg.norm(data, axis = 0, keepdims = True)
    
    _Similarity = data_norm.T @ data_norm
    _Similarity[_Similarity < 0] = 0
    
    # some features are compltelely constant, so they are disconnected
    disconnected_idx = np.where(np.all(_Similarity == 0, axis=1))[0]
    connected_idx = np.where(~np.all(_Similarity == 0, axis=1))[0]
    
    similarity_matrix = _Similarity[np.ix_(connected_idx, connected_idx)]
    
    if len(disconnected_idx) == 0: # No disconnected features
        embedding = __spectral_embedding__(similarity_matrix, n_components = m, norm_laplacian = norm_laplacian)
        kmeans = KMeans(n_clusters = m, random_state = 42)# , n_init = 'auto')
    else: # one cluster will be for disconnected features
        assert m >= 2 # total cluster should be greater or equal 2
        assert len(connected_idx) >= (m - 1) # number of connected channels can not be less than (m-1)
        
        embedding = __spectral_embedding__(similarity_matrix, n_components = m - 1, norm_laplacian = norm_laplacian)
        kmeans = KMeans(n_clusters = m - 1, random_state = 42)#, n_init = 'auto')
        print('one clusters for the disconnected features')
        
    clusters_connected = kmeans.fit_predict(embedding)
    clusters = np.empty(_Similarity.shape[0], dtype = int)
    clusters[connected_idx] = clusters_connected  # from KMeans or other method
    clusters[disconnected_idx] = max(clusters_connected) + 1  # new cluster
    
    # __figuresPlot__(m, clusters, _Similarity)
    return list(clusters)


def Rsimilarity(data, m, norm_laplacian = True):
    ''' Here I directly used abs Correl. Coeff. as the similarity matrix'''
    df = pd.DataFrame(data)
    correlation_matrix = df.corr()
    correlation_matrix = correlation_matrix.fillna(0)
    abs_correlation_matrix = np.abs(correlation_matrix).values.astype(np.float32)
    
    # some features are compltelely constant, so they are disconnected
    disconnected_idx = np.where(np.all(abs_correlation_matrix == 0, axis=1))[0]
    connected_idx = np.where(~np.all(abs_correlation_matrix == 0, axis=1))[0]
    
    connected_abs_correlation_matrix = abs_correlation_matrix[np.ix_(connected_idx, connected_idx)]
    similarity_matrix = connected_abs_correlation_matrix # __cosine_similarity__(connected_abs_correlation_matrix)
    
    if len(disconnected_idx) == 0: # No disconnected features
        embedding = __spectral_embedding__(similarity_matrix, n_components = m, norm_laplacian = norm_laplacian)
        kmeans = KMeans(n_clusters = m, random_state = 42)# , n_init = 'auto')
    else: # one cluster will be for disconnected features
        assert m >= 2 # total cluster should be greater or equal 2
        assert len(connected_idx) >= (m - 1) # number of connected channels can not be less than (m-1)
        
        embedding = __spectral_embedding__(similarity_matrix, n_components = m - 1, norm_laplacian = norm_laplacian)
        kmeans = KMeans(n_clusters = m - 1, random_state = 42)#, n_init = 'auto')
        print('one clusters for the disconnected features')
        
    clusters_connected = kmeans.fit_predict(embedding)
    clusters = np.empty(abs_correlation_matrix.shape[0], dtype = int)
    clusters[connected_idx] = clusters_connected  # from KMeans or other method
    clusters[disconnected_idx] = max(clusters_connected) + 1  # new cluster
    
    # __figuresPlot__(m, clusters, abs_correlation_matrix)
    return list(clusters)




def detailedProfileBasedClustering(data, m, norm_laplacian = True):
    ''' Normalized spectral clustering according to Shi and Malik '''
    df = pd.DataFrame(data)
    correlation_matrix = df.corr()
    correlation_matrix = correlation_matrix.fillna(0)
    abs_correlation_matrix = np.abs(correlation_matrix).values.astype(np.float32)
    
    # some features are compltelely constant, so they are disconnected
    disconnected_idx = np.where(np.all(abs_correlation_matrix == 0, axis=1))[0]
    connected_idx = np.where(~np.all(abs_correlation_matrix == 0, axis=1))[0]
    
    connected_abs_correlation_matrix = abs_correlation_matrix[np.ix_(connected_idx, connected_idx)]
    similarity_matrix = __cosine_similarity__(connected_abs_correlation_matrix)
    
    if len(disconnected_idx) == 0: # No disconnected features
        embedding = __spectral_embedding__(similarity_matrix, n_components = m, norm_laplacian = norm_laplacian)
        kmeans = KMeans(n_clusters = m, random_state = 42)# , n_init = 'auto')
    else: # one cluster will be for disconnected features
        assert m >= 2 # total cluster should be greater or equal 2
        assert len(connected_idx) >= (m - 1) # number of connected channels can not be less than (m-1)
        
        embedding = __spectral_embedding__(similarity_matrix, n_components = m - 1, norm_laplacian = norm_laplacian)
        kmeans = KMeans(n_clusters = m - 1, random_state = 42)#, n_init = 'auto')
        print('one clusters for the disconnected features')
        
    clusters_connected = kmeans.fit_predict(embedding)
    clusters = np.empty(abs_correlation_matrix.shape[0], dtype = int)
    clusters[connected_idx] = clusters_connected  # from KMeans or other method
    clusters[disconnected_idx] = max(clusters_connected) + 1  # new cluster
    # if 1 in Counter(clusters).values():
    #     print('single member', m, silhouette_score(1 - abs_correlation_matrix, clusters))
    # else:
    #     print(m, silhouette_score(1 - abs_correlation_matrix, clusters))
            
    # __figuresPlot__(m, clusters, abs_correlation_matrix)
    return list(clusters)


def __cosine_similarity__(matrix): # matrix: channel x channel
    row_magnitude = np.sqrt((matrix ** 2).sum(-1, keepdims = True))
    norm_matrix = matrix / row_magnitude
    similarity = norm_matrix @ norm_matrix.T
    return similarity
    

def __spectral_embedding__(adjacency_matrix, n_components = None, norm_laplacian = True, random_state = 42):
    random_state = check_random_state(random_state)
    
    W = adjacency_matrix # channels x channels
    np.fill_diagonal(W, 0)
    
    # Laplacian Matrix
    L, D = csgraph_laplacian(W, normed = norm_laplacian, return_diag = True)
    
    
    '''
    Finding the smallest eigenvalues directly is hard.
    It's slower and less stable, especially for large matrices.
    Instead of computing the eigenvalues λ of L directly, we transform the problem into:
    ((L−σI)^-1)v = μv
    Now, μ = 1 / (λ−σ)​
    If λi == σ, then ∣μi∣ is large.
    So, if we ask for the largest eigenvalue with sigma = 1,
    it will give effectively largest λ closer to sigma = 1.
    
    However, λi ranges from 0 to 2 for Normalized laplacian.
    So????
    We transform L to -L. This will transform λi to -λi (= -2 to -0)
    So, if we want to get the largest eigenvalue closer to sigma 1,
    This will effectively give the largest eigenvalue closer to -0, because these
    are the most closest to sigma=1.
    However, λi closer to 0, are effectively the smallest eigenvalue of original L.
    ---
    For more details, you can review the 'spectral_embedding' from 'sklearn.manifold'
    '''
    
    L *= -1
    v0 = _init_arpack_v0(L.shape[0], random_state)
    _, diffusion_map = eigsh(L, k = n_components + 1, sigma = 1.0, which = "LM", tol = 0, v0 = v0)
    embedding = diffusion_map.T[n_components + 1::-1]
    
    if norm_laplacian:
        # recover u = D^-1/2 x from the eigenvector output x
        embedding = embedding / D
    embedding = _deterministic_vector_sign_flip(embedding)
    
    return embedding[1:n_components + 1].T



def profile_based_clustering(data, m):
    ''' data: [L, C] numpy array'''
    df = pd.DataFrame(data)
    
    correlation_matrix = df.corr()
    correlation_matrix = correlation_matrix.fillna(0)
    abs_correlation_matrix = np.abs(correlation_matrix).values.astype(np.float32)
    
    # some features are compltelely constant, so they are disconnected
    disconnected_idx = np.where(np.all(abs_correlation_matrix == 0, axis=1))[0]
    connected_idx = np.where(~np.all(abs_correlation_matrix == 0, axis=1))[0]
    
    connected_abs_correlation_matrix = abs_correlation_matrix[np.ix_(connected_idx, connected_idx)]
    
    similarity_matrix = cosine_similarity(connected_abs_correlation_matrix)
    
    if len(disconnected_idx) == 0: # No disconnected features
        embedding = spectral_embedding(similarity_matrix, n_components = m, random_state=42)
        kmeans = KMeans(n_clusters = m, random_state = 42)
    else: # one cluster will be for disconnected features
        assert m >= 2 # total cluster should be greater or equal 2
        assert len(connected_idx) >= (m - 1) # number of connected channels can not be less than (m-1)
        
        embedding = spectral_embedding(similarity_matrix, n_components = m - 1, random_state=42)
        kmeans = KMeans(n_clusters = m - 1, random_state = 42)
        print('one clusters for the disconnected features')
        
    clusters_connected = kmeans.fit_predict(embedding)
    clusters = np.empty(abs_correlation_matrix.shape[0], dtype = int)
    clusters[connected_idx] = clusters_connected  # from KMeans or other method
    clusters[disconnected_idx] = max(clusters_connected) + 1  # new cluster
    
    # silhouette_score(distance_matrix, clusters)
    return list(clusters)



def kmeans_distance(data, m):
    ''' data: [L, C] numpy array'''
    df = pd.DataFrame(data)
    n_clusters = m
    
    correlation_matrix = df.corr()
    # replaceing nan causing by constant column
    correlation_matrix = correlation_matrix.fillna(0)
    distance_matrix = 1 - np.abs(correlation_matrix)
    # Perform k-means clustering
    kmeans = KMeans(n_clusters = n_clusters, random_state = 42)
    clusters = kmeans.fit_predict(distance_matrix)
    return list(clusters)


def kmeans_absCorrelation(data, m):
    ''' data: [L, C] numpy array'''
    df = pd.DataFrame(data)
    n_clusters = m
    
    correlation_matrix = df.corr()
    # replaceing nan causing by constant column
    correlation_matrix = correlation_matrix.fillna(0)
    similarity_matrix = np.abs(correlation_matrix)
    # Perform k-means clustering
    kmeans = KMeans(n_clusters = n_clusters, random_state = 42)
    clusters = kmeans.fit_predict(similarity_matrix)
    return list(clusters)


def __figuresPlot__(m, clusters, abs_correlation_matrix):
    # m: number of the clusters
    # clusters: cluster index
    # abs_correlation_matrix: abs. correlation coeff matrix
    
    rows = (m + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 3 * rows))
    axes = axes.flatten()  # Flatten in case of single row
    for i in range(m):
        cidx = np.where(clusters == i)[0]
        group_score = abs_correlation_matrix[cidx, :]
        im = axes[i].matshow(group_score)
        axes[i].set_yticks(np.arange(len(cidx)))
        axes[i].set_yticklabels(cidx)
        axes[i].set_xlabel('Scores')
        axes[i].set_ylabel('Channel ID')
        axes[i].set_title(f'Cluster {i}')
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04, label="Correlation Score")
    # Hide any unused subplots
    for j in range(m, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()
    return None


def _deterministic_vector_sign_flip(u):
    """Modify the sign of vectors for reproducibility.

    Flips the sign of elements of all the vectors (rows of u) such that
    the absolute maximum element of each vector is positive.

    Parameters
    ----------
    u : ndarray
        Array with vectors as its rows.

    Returns
    -------
    u_flipped : ndarray with same shape as u
        Array with the sign flipped vectors as its rows.
    """
    max_abs_rows = np.argmax(np.abs(u), axis=1)
    signs = np.sign(u[range(u.shape[0]), max_abs_rows])
    u *= signs[:, np.newaxis]
    return u


def _init_arpack_v0(size, random_state):
    '''Pytorch Scipy builtin'''
    """Initialize the starting vector for iteration in ARPACK functions.

    Initialize a ndarray with values sampled from the uniform distribution on
    [-1, 1]. This initialization model has been chosen to be consistent with
    the ARPACK one as another initialization can lead to convergence issues.

    Parameters
    ----------
    size : int
        The size of the eigenvalue vector to be initialized.

    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator used to generate a
        uniform distribution. If int, random_state is the seed used by the
        random number generator; If RandomState instance, random_state is the
        random number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    Returns
    -------
    v0 : ndarray of shape (size,)
        The initialized vector.
    """
    random_state = check_random_state(random_state)
    v0 = random_state.uniform(-1, 1, size)
    return v0

def check_random_state(seed):
    '''Scipy Builtin'''
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )