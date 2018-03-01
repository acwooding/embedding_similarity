import math
import scipy
import numpy as np
import sys
import os
import json
import pickle
import sklearn.neighbors
from functools import partial
import tqdm
import multiprocessing as mp
import time
from itertools import combinations as comb
from collections import defaultdict

import logging

# Timing and Performance
def log_message(str, line_break=False):
    '''Display a message. If line_break is set, add a line break'''
    if line_break:
        print("{}\n".format(str))
    else:
        print("{}".format(str))

LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"

logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, datefmt=DATE_FORMAT)        
logger = logging.getLogger()

message_handler = logging.StreamHandler(sys.stdout)
message_handler.setLevel(logging.INFO)
message_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(message_handler)

def timing_info(method):
    def wrapper(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        logger.info("timing_info: {}@{} ms".format(method.__name__, round((end_time-start_time)*1000,1)))
        return result

    return wrapper

def record_time_interval(section, start_time, line_break=False):
    """Record a time interval since the last timestamp"""
    end_time = time.time()
    delta = end_time - start_time
    if delta < 1:
        delta *= 1000
        units = "ms"
    else:
        units = "s"
    if line_break:
        logger.info("PROCESS_TIME:{:>36}    {} {}\n".format(section, round(delta, 1), units))
    else:
        logger.info("PROCESS_TIME:{:>36}    {} {}".format(section, round(delta, 1), units))
    return end_time

def save_embedding(basefilename, embedding=None, labels=None, embedding_algorithm=None, dataset=None, 
                   other_info=None, run_number=0, data_path=None):
    """
    Save off a vector space embedding of a data set in a common format.
    
    Parameters
    ----------
    basefilename: base for the filenames
    embedding: 2d-numpy array representing embedding of points as rows (a row gives the coordinates for a point)
    labels: 1d-numpy array labeling the rows (points)
    embedding_algorithm: (str) name of the algorithm used for the given vector space embedding
    dataset: (str) name of the dataset that was embedded
    other_info: (str) any other information to note about how the embedding was done 
        (eg. variables for the embedding algorithm)
    run_number: (int) attempt number via the same embedding parameters
    data_path: (path) base path for save the embedding to
    """
    
    if embedding is None or labels is None or embedding_algorithm is None or dataset is None or data_path is None:
        raise ValueError("embedding, labels, embedding_algorithm, dataset, and data_path are all required")
    filename = basefilename + "_" + str(run_number)
    embedding_filename = os.path.join(data_path, filename + ".embedding")
    labels_filename = os.path.join(data_path, filename + ".labels")
    metadata_filename = os.path.join(data_path, filename + ".metadata")

    embedding_shape = embedding.shape
    assert(len(embedding_shape) == 2)
    assert(embedding_shape[0] == labels.shape[0])

    # save the embedding and labels
    np.save(embedding_filename, embedding)
    np.save(labels_filename, labels)

    metadata = {"Embedding Algorithm": embedding_algorithm, "Dataset": dataset, "Run Number":str(run_number),
                "Other Information": other_info}
    #save metadata
    with open(metadata_filename, "w") as outfile:
        outfile.write(json.dumps(metadata, indent=4))
    
def read_embedding(basefilename, run_number=0, data_path=None):
    """
    Companion function for reading in a vector space embedding to go with save_embedding
    
    Returns: (embedding, labels, metadata)
    
    Parameters
    ----------
    basefilename: 
        base for the filenames
    run_number: (int)
        attempt number via the same embedding parameters
    data_path: (path)
        base path for save the embedding to
    
    Returns
    -------
    (embedding, labels, metdata)
    """
    filename = basefilename + "_" + str(run_number)
    embedding_filename = os.path.join(data_path, filename + ".embedding.npy")
    labels_filename = os.path.join(data_path, filename + ".labels.npy")
    metadata_filename = os.path.join(data_path, filename + ".metadata")

    log_message("Reading embedding {}".format(embedding_filename))    
    embedding = np.load(embedding_filename)
    labels = np.load(labels_filename)
    
    with open(metadata_filename, "r") as infile:
        metadata = json.load(infile)

    assert(metadata['Run Number'] == str(run_number))

    return embedding, labels, metadata

def save_neighbors(basefilename, neighbors=None, metric=None, n_neighbors=None, 
                   other_info=None, metadata=None, run_number=0, 
                   data_path=None):
    """
    Save off neighbor sets in a common format.
    
    Parameters
    ----------
    basefilename: (path)
        base for the filenames
    neighbors: 
        output from get_neighbors
    other_info: (str) 
        any other information to note to include in the metadata
    run_number: (int)
        attempt number via the same embedding parameters
    data_path: (path)
        base path for save the embedding to
    """
    
    if metric is None or n_neighbors is None or metadata is None or neighbors is None:
        raise ValueError(" metric, n_neighbors, metadata, neighbors  are all required")
    filename = basefilename + "_" + metric + "_" + str(run_number)
    neighbors_filename = os.path.join(data_path, filename + ".neighbors")
    metadata_filename = os.path.join(data_path, filename + ".neighbors_metadata")

    log_message("Saving neighbours to {}".format(neighbors_filename))
    with open(neighbors_filename, "wb") as outfile:
        pickle.dump(neighbors, outfile)

    metadata["k-nn Metric"] = metric
    metadata["Number of Neighbors"] = str(n_neighbors)
    metadata["Other Neighbors Information"] = other_info

    log_message("Saving neighbours metadata to {}".format(metadata_filename))
    #save metadata
    with open(metadata_filename, "w") as outfile:
        outfile.write(json.dumps(metadata, indent=4))
        
def read_neighbors(basefilename, run_number=0, metric=None, data_path=None):
    """
    Read neighbor sets from a common format.
    
    Parameters
    ----------
    basefilename: base for the filenames
    neighbors: output from get_neighbors
    other_info: (str) any other information to note about how the embedding was done 
        (eg. variables for the embedding algorithm)
    run_number: (int) attempt number via the same embedding parameters
    data_path: (path) base path for save the embedding to

    Returns
    -------
    (neighbors, metadata)
    """
    if metric is None or data_path is None:
        raise ValueError("metric and data_path are required")
    filename = basefilename + "_" + metric + "_" + str(run_number)
    neighbors_filename = os.path.join(data_path, filename + ".neighbors")
    metadata_filename = os.path.join(data_path, filename + ".neighbors_metadata")
    
    neighbors = pickle.load(open(neighbors_filename, "rb"))

    with open(metadata_filename, "r") as infile:
        metadata = json.load(infile)

    return neighbors, metadata

def get_neighbors(embedding, labels, data=None, n_neighbors=5, algorithm='brute', metric='cosine'):
    '''
    Parameters
    ----------
    embedding: 
        a scipy sparse matrix representing an embedding where each row is the embedding of 
        the index into a embedding.shape[1] dimensional vector space
    data: 
        submatrix of embedding to get the neighbours of
    index2word: dict labelling the indices of embedding
    n_neighbours: size of the neighbour sets to generate

    algorithm: choice of algorithm for sklearn.neighbors.NearestNeighbors
    metric: choice of metric for sklearn.neighbors.NearestNeighbors

    Returns
    -------
    (embedding_neighbors, embedding_distances)
    '''
    if n_neighbors > len(labels):
        raise ValueError("Expected n_neighbors < number of labels, but n_neighbors = {} and len(labels) = {}".format(n_neighbors, len(labels)))
    if data is None:
        data = embedding

    neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric).fit(embedding)
    embedding_neighbors = neighbors.kneighbors(data)

    index2word = dict(zip(list(range(len(labels))), labels))

    embedding_label_neighbors = {}
    embedding_label_distances = {}
    for i, row in enumerate(embedding_neighbors[1]):
        nneighbors = [index2word[index] for index in row]
        embedding_label_neighbors[nneighbors[0]] = nneighbors
        distances = embedding_neighbors[0][i]
        
        # check that the nearest neighbours are ordered from closest to farthest
        for j, d in enumerate(distances):
            if j == 0:
                test = True
            elif d < distances[j-1]:
                logger.debug("Distances: {} aren't ordered correctly with respect to {}".format(distances, d))
                assert(False)
        embedding_label_distances[nneighbors[0]] = distances
        
    return embedding_label_neighbors, embedding_label_distances


def make_neighbors_from_files(basefilename, run_numbers=None, n_neighbors=None, n=None,
                       knn_metrics=['minkowski', 'cosine', 'l1', 'hamming'],
                       other_info=None, data_path=None):
    """
    Reads all the embedding files corresponding to a given basefilename (for the embedding)
    and specified run numbers, and computes the k-nearest neighbor sets (as per get_neighbors).
    It then saves the neighbour sets to the data_path as basefilename.neighbors and 
    basefilename.neighbors_metadata files.

    Parameters
    ----------
    basefilename: (path)
        basefilename for the embeddings
    run_numbers: (list of integers)
        the run numbers to get neighbor sets for
    n: (int)
        size of the subspace of the embedding to compute neighbors for (by default the first n entries)
        if n is None, default to using the entire embedding
    n_neighbors: (int)
        number of neighbors to compute
    knn_metrics: 
        list of metrics to use for computing the k-nearest neighbors.
    data_path: (path) 
        base path to the data files
    other_info: (str) 
        any other information to note to include in the metadata
    """
    if run_numbers is None:
        raise ValueError("run_numbers is required")

    for knn_metric in tqdm.tqdm(knn_metrics, total=len(knn_metrics)):
        for i in run_numbers:

            embedding, labels, metadata = read_embedding(basefilename, run_number=i, data_path=data_path)

            # make sure n <= embedding.shape[0]
            if n is None:
                n = embedding.shape[0]
            else:
                if n > embedding.shape[0]:
                    n = embedding.shape[0]

            log_message("Computing neighbours for {}, run number {}".format(basefilename, i))
            neighbors = get_neighbors(embedding, labels, data=embedding[:n], n_neighbors=n_neighbors, metric=knn_metric)

            save_neighbors(basefilename=basefilename, neighbors=neighbors, metric=knn_metric,
                           n_neighbors=n_neighbors, metadata=metadata, run_number=i, other_info=other_info,
                           data_path=data_path)



def jaccard(A, B):
    '''
    Jaccard similarity of two sets.
    
    Parameters
    ----------
    A, B: (sets)

    Returns
    -------
    similarity in [0, 1]
    '''
    if A and B:
        return len(A.intersection(B))/len(A.union(B))
    else:
        return 0

def adaptedKendallTau(X, Y):
    '''
    Parameters
    ----------
    X, Y: 
        2 lists X and Y of arbitrary length, consisting of unique elements (ie.
        X and Y represent posets).
    Returns
    -------
    similarity in [0,1]
    '''
    if not X or not Y:
        return 0
    elif len(set(X)) != len(X) or len(set(Y)) != len(Y):
        raise ValueErrror("There are repeated elements in X or Y")
    else:
        SI = [x for x in X if x in Y] ## intersection - list to keep the order as in X
        SX = set(X).difference(SI)    ## in X only
        SY = set(Y).difference(SI)    ## in Y only
        tau = 0
        ## SI: pairs in intersection
        l = len(SI)
        y = [Y.index(z) for z in SI]
        s = np.sum([a<b for a,b in comb(y,2)])
        tau += 2*s - l*(l-1)/2 ## pos-neg pairs a la Kendall-tau.
        ## SX, resp. SY and SI: one element in intersection
        x = [X.index(z) for z in SX]
        y = [Y.index(z) for z in SY]
        i = [X.index(z) for z in SI]
        s = 2*np.sum([a>b for a in x for b in i]) - len(x)*len(i)
        tau += s
        s = 2*np.sum([a>b for a in y for b in i]) - len(y)*len(i)
        tau += s
        ## ze rest
        tau -= len(SX)*len(SY)
        tau += l*(l+1)/2
        tau /= (len(X)*len(Y))
        return ((tau+1)/2)

def list_similarity(label, list1, list2, how='jaccard'):
    """
    Compute similarity score of two lists.

    Parameters
    ----------
    label:
        name of the point
    list1, list2: 
        lists of k-nearest neighbors of label
    how: {'jaccard', 'adapted-ktau'}
        similarity score method
    
    Returns: (label, score)
    """
    if how == 'jaccard':
        score = jaccard(set(list1), set(list2))
    elif how == 'adapted-ktau':
        score = adaptedKendallTau(list1, list2)
    else:
        raise ValueError("Unknown method: {}".format(how))
    return label, score

"""
def old_compare_neighbors(D1, D2, n_neighbors=None, how='jaccard'):
    '''
    Parameters
    ----------
    D1, D2: dictionaries with {label:neighbors} to be compared where neighbours is
        a list of neighbors for the label where neighbors[i] is the i-th nearest
        neighbor to the label. (the 0-th nearest neighbour is itself)
    n_neighbors: (int)
        the number of neighbors to compare. If None, compare the full list of neighbours given.
    how: how to compare the neighbours
        jaccard: return the jaccard distances between the neighbours as sets
        adapted-ktau: return the adapted Kendall Tau similarity between the neighbours

    Returns
    -------
    (labels, similarity)
        where similarity[i] is the neighbourhood similarity score of labels[i]
    '''
    same_labels = set(D1.keys()).intersection(D2.keys())
    similarity = []
    labels = []
    for label in same_labels:
        labels.append(label)
        # remove the matching first element
        assert(D1[label][0]==label)
        assert(D2[label][0]==label)
        if n_neighbors:
            nghbrs1 = D1[label][1:n_neighbors]
            nghbrs2 = D2[label][1:n_neighbors]
        else:
            nghbrs1 = D1[label][1:]
            nghbrs2 = D2[label][1:]
        if how == 'jaccard':
            similarity.append(jaccard(set(nghbrs1), set(nghbrs2)))
        elif how == 'adapted-ktau':
            similarity.append(adaptedKendallTau(nghbrs1, nghbrs2))
        else:
            raise ValueError("Invalid similarity score:{}".format(how))
    return labels, similarity
"""

def compare_neighbors(D1, D2, n_neighbors=None, how='jaccard'):
    '''
    Parameters
    ----------
    D1, D2: dictionaries with {label:neighbors} to be compared where neighbours is
        a list of neighbors for the label where neighbors[i] is the i-th nearest
        neighbor to the label. (the 0-th nearest neighbour is itself)
    n_neighbors: (int)
        the number of neighbors (minus 1) to compare (as we don't include the
        the 0-th neighbour in the comparison as it is the label itself).
        If None, compare the full list of neighbours given (except the 0-th neighbour).
    how: how to compare the neighbours
        jaccard: return the jaccard distances between the neighbours as sets
        adapted-ktau: return the adapted Kendall Tau similarity between the neighbours

    Returns
    -------
    (labels, similarity)
        where similarity[i] is the neighbourhood similarity score of labels[i]
    '''
    same_labels = set(D1.keys()).intersection(D2.keys())
    similarity = []
    labels = []
    for label in same_labels:
        labels.append(label)
        # remove the matching first element
        assert(D1[label][0]==label)
        assert(D2[label][0]==label)
        if n_neighbors:
            neighbors1 = D1[label][1:n_neighbors]
            neighbors2 = D2[label][1:n_neighbors]
        else:
            neighbors1 = D1[label][1:]
            neighbors2 = D2[label][1:]

        _, score = list_similarity(label, neighbors1, neighbors2, how=how)
        similarity.append(score)

    return labels, similarity

def parallel_compare_neighbors(D1, D2, n_neighbors=None, how='jaccard', n_proc=2, chunksize=None):
    '''
    Parameters
    ----------
    D1, D2: (dict)
        dictionaries with {label:neighbors} to be compared where neighbours is
        a list of neighbors for the label where neighbors[i] is the i-th nearest
        neighbor to the label. (the 0-th nearest neighbour is itself)
    n_neighbors: (int)
        the number of neighbors (minus 1) to compare (as we don't include the
        the 0-th neighbour in the comparison as it is the label itself).
        If None, compare the full list of neighbours given (except the 0-th neighbour).
    how:  {'jaccard', 'adapted-ktau'}
        how to compare the neighbours
        jaccard: return the jaccard distances between the neighbours as sets
        adapted-ktau: return the adapted Kendall Tau similarity between the neighbours
    n_proc: (int)
        number of processors to use
    chunksize: (int)
        size of chunk to send to each process at a time

    Returns
    -------
    (labels, similarity)
        where similarity[i] is the neighbourhood similarity score of labels[i]
    '''
    same_labels = set(D1.keys()).intersection(D2.keys())
    if n_neighbors is None:
        triples = [(word, D1[word][1:], D2[word][1:]) for word in same_labels]
    else:
        triples = [(word, D1[word][1:n_neighbors], D2[word][1:n_neighbors]) for word in same_labels]
    list_similarity_partial = partial(list_similarity, how=how)
    
    with mp.Pool(processes=n_proc) as pool:
        if chunksize is not None:
            ret = pool.starmap(list_similarity_partial, triples, chunksize=chunksize)
        else:
            ret = pool.starmap(list_similarity_partial, triples)
    
    return list(zip(*ret))


# helper for get_pairwise_comparisons 
def do_comparisons(indices, neighbors_list=None, n_neighbors=None, how=None):
    """
    Helper function for get_pairwise_comparisons.
    """
    i1, i2 = indices
    logger.debug("length of neighbors_list: {}, indices: {}, {}".format(len(neighbors_list), i1, i2))
    neighbor_set_1 = neighbors_list[i1][0]
    neighbor_set_2 = neighbors_list[i2][0]
    return [i1, i2, compare_neighbors(neighbor_set_1, neighbor_set_2, n_neighbors=n_neighbors, how=how)]

def get_pairwise_comparisons(neighbors_list, n_neighbors=None, how=None):
    """
    Parameters
    ----------
    neighbors_list:
        list of neighbors_list
    n_neighbors: (int)
        the number of neighbors (minus 1) to compare (as we don't include the
        the 0-th neighbour in the comparison as it is the label itself).
        If None, compare the full list of neighbours given (except the 0-th neighbour).
    how:
        metric to use as a similarity score between the n_neighbors-nearest neighbors
        as per compare_neighbors

    Returns
    -------
    comparisons:
        a list elements of the form: [index1, index2, comparison] where comparison is the result
        of compare_neighbors(neighbors_list[index1], nieghbors_list[index2])
    
    """
    if how is None:
        raise ValueError("how is required")

    partial_do_comparisons = partial(do_comparisons, neighbors_list=neighbors_list, n_neighbors=n_neighbors, 
                                     how=how)

    index_list = []
    size_nset = len(neighbors_list)
    for i in range(size_nset):
        for j in range(size_nset):
            if i<j:
                index_list.append((i, j))
                
    comparisons = []
    for index in tqdm.tqdm(index_list, total=len(index_list)):
        comparisons.append(partial_do_comparisons(index))
    return comparisons

# helper functions for parallel version of comparisons
def knn_iterator(neighbors_list, n_neighbors=None):
    """
    Produce all pairs of knn-list from neighbors_list
    
    Parameters
    ----------
    neighbors_list:
        list of dictionaries with {word:neighbour_set}
    n_neighbors: (int)
        number of neighbors (minus 1) to include in the comparisons 
    
    Returns: (a, b), word, neighbors_sets[a][word][1:n_neighbors], neighbors_sets[b][word][1:n_neighbors]
       where a, b are indices into neighbors_list, word are all the keys present in
       the intersection
    """
    pairs = [(a, b) for a, b in comb(range(len(neighbors_list)), 2) if a < b]
    for a, b in pairs:
        knn_dict_a, _ = neighbors_list[a]
        knn_dict_b, _ = neighbors_list[b]

        label_intersection = set(knn_dict_a.keys()).intersection(knn_dict_b.keys())
        for word in label_intersection:
            word_list_a = knn_dict_a[word]
            word_list_b = knn_dict_b[word]
            ## sanity check that my nearest neighbour is myself
            assert(word_list_a[0] == word_list_b[0])
            if n_neighbors is not None:
                yield (a, b), word, word_list_a[1:n_neighbors], word_list_b[1:n_neighbors]
            else:
                yield (a, b), word, word_list_a[1:], word_list_b[1:]

def knn_iterator_varying_k(neighbors_list, max_k):
    """
    Produce all pairs of knn-list from neighbors_list
    
    Parameters
    ----------
    neighbors_list:
        list of dictionaries with {word:neighbour_set}
    
    Returns: (a, b, k), word, neighbors_sets[a][word][1:k], neighbors_sets[b][word][1:k]
       where a, b are indices into neighbors_list, word are all the keys present in
       the intersection
    """
    pairs = [(a, b) for a, b in comb(range(len(neighbors_list)), 2) if a < b]
    for a, b in pairs:
        knn_dict_a, _ = neighbors_list[a]
        knn_dict_b, _ = neighbors_list[b]

        label_intersection = set(knn_dict_a.keys()).intersection(knn_dict_b.keys())
        for word in label_intersection:
            word_list_a = knn_dict_a[word]
            word_list_b = knn_dict_b[word]
            ## sanity check that my nearest neighbour is myself
            assert(word_list_a[0] == word_list_b[0])
            for k in range(2, max_k):
                yield (a, b, k), word, word_list_a[1:k], word_list_b[1:k]

def pairs_list_similarity(context, label, list1, list2, how='jaccard'):
    """
    Compute similarity score of two lists.

    Parameters
    ----------
    context: 
        contextual data associated with the input.
        Not used by the computation, but is returned to caller for establishing
        context when parallelizing
    label:
        name of the point
    list1, list2: 
        lists of k-nearest neighbors of label
    how: {'jaccard', 'adapted-ktau'}
        similarity score method
    
    Returns: (context, (label, score))
    """
    if how == 'jaccard':
        score = jaccard(set(list1), set(list2))
    elif how == 'adapted-ktau':
        score = adaptedKendallTau(list1, list2)
    else:
        raise ValueError("Unknown method: {}".format(how))
    return context, (label, score)

def reduce(kv_pair):
    key, value = kv_pair
    w, s = [],[]
    for x in value:
        w += [x[0]]
        s += [x[1]]
    comparison = (*key, (w, s))
    return comparison

def parallel_get_pairwise_comparisons(neighbors_list, n_neighbors=None, how=None, n_proc=2, chunksize=None):
    """
    Use a map reduce approach to parallelizing the overall pairwise similarity comparisons
    of the neighbors_list.
    
    Parameters
    ----------
    neighbors_list: (list)
        list of neighbors
    n_neighbors: (int)
        the number of neighbors (minus 1) to compare (as we don't include the
        the 0-th neighbour in the comparison as it is the label itself).
        If None, compare the full list of neighbours given (except the 0-th neighbour).
    how:
        metric to use as a similarity score between the k-nearest neighbors
    n_proc:
        number of processes to use
    chunksize:
        number of items to send to each process at a time. If None, will default to
        total items/(n_proc *4).

    Returns
    -------
    comparisons:
        a list elements of the form: [index1, index2, comparison] where comparison is the result
        of compare_neighbors(neighbors_list[index1], nieghbors_list[index2])
    """
    if how is None:
        raise ValueError("how is required")
    pairs_list_similarity_partial = partial(pairs_list_similarity, how=how)

    log_message("Map step starting...")
    start_time = time.time()

    with mp.Pool(processes=n_proc) as pool:
        similarity_map = pool.starmap(pairs_list_similarity_partial,
                                      knn_iterator(neighbors_list, n_neighbors=n_neighbors),
                                      chunksize=chunksize)

    ts = record_time_interval("Map Step", start_time)
    
    log_message("Partition step...")
    ws_dict = defaultdict(list)
    for x, y in similarity_map:
        ws_dict[x].append(y)
    ts = record_time_interval("Partition Step", ts)

    log_message("Reduce step...")
    comparisons = list(map(reduce, ws_dict.items()))
    ts = record_time_interval("Reduce Step", ts)

    record_time_interval("Total Time", start_time)
    return comparisons


def parallel_get_pairwise_comparisons_varying_k(neighbors_list, max_k, n_proc=2, how=None, chunksize=None):
    """
    Use a map reduce approach to parallelizing the overall pairwise similarity comparisons
    of the neighbors_list.
    
    Parameters
    ----------
    neighbors_list:
        list of neighbors_list
    how:
        metric to use as a similarity score between the k-nearest neighbors
    max_k: (int)
        compare all neighbours from neighbour list of size 2 up to max_k
    n_proc:
        number of processes to use
    chunksize:
        chunksize to send to each process at a time.

    Returns
    -------
    comparisons:
        a list elements of the form: [index1, index2, k, comparison] where comparison is the result
        of compare_neighbors(neighbors_list[index1][:k], nieghbors_list[index2][:k]) where k varies
        from 2 to max_k.

    """
    if how is None:
        raise ValueError("how is required")
    pairs_list_similarity_partial = partial(pairs_list_similarity, how=how)
    
    log_message("Map step starting...")
    start_time = time.time()
    with mp.Pool(processes=n_proc) as pool:
        similarity_map = pool.starmap(pairs_list_similarity_partial,
                                      knn_iterator_varying_k(neighbors_list, max_k),
                                      chunksize=chunksize)

    ts = record_time_interval("Map Step", start_time)

    log_message("Partition step...")
    ws_dict = defaultdict(list)
    for x, y in similarity_map:
        ws_dict[x].append(y)
    ts = record_time_interval("Partition Step", ts)

    log_message("Reduce step...")
    comparisons = list(map(reduce, ws_dict.items()))
    ts = record_time_interval("Reduce Step", ts)

    record_time_interval("Total Time", start_time)
    return comparisons

def save_comparisons(basefilename, comparisons=None, metadata=None, max_k=None,
                     data_path=None):
    """
    Save off list of pairwise comparisons of neighbors_list generated by one of
        get_pairwise_comparisons 
        parallel_get_pairwise_comparisons 
        parallel_get_pairwise_comparisons_varying_k
    
    Parameters
    ----------
    basefilename:
        base for the filenames
    comparisons:
        output from get_pairwise_comparisons or one of its variations
    metadata: 
        list of metadata corresponding the neighbors_list used as input to generate
        the comparisons
    max_k:
        set to the value of max_k used in parallel_get_pairwise_comparisons
    data_path: (path) 
        base path to save to
    """
    
    if comparisons is None or metadata is None:
        raise ValueError("comparisons and metadata are required")

    comparisons_filename = os.path.join(data_path, basefilename + ".comparisons")
    metadata_filename = os.path.join(data_path, basefilename + ".comparisons_metadata")
    
    # check that the lengths of the comparisons and metadata match
    n = len(metadata)
    if max_k:
        num_comparisons = n*(n-1)/2*(max_k-2)
    else:
        num_comparisons = n*(n-1)/2
    assert(len(comparisons) == num_comparisons)
    
    # save the comparisons and metadata
    with open(comparisons_filename, "wb") as file:
        pickle.dump(comparisons, file)
    with open(metadata_filename, "wb") as file:
        pickle.dump(metadata, file)


def read_comparisons(basefilename, data_path=None):
    """
    Read in a list of pairwise comparisons of neighbors_list generated by one of
        get_pairwise_comparisons 
        parallel_get_pairwise_comparisons 
        parallel_get_pairwise_comparisons_varying_k
    along with its accompanying metadata.

    Parameters
    ----------
    basefilename: (str)

    data_path: (path)
        base path

    Returns
    -------
    (comparisons, metdata)
    """
    comparisons_filename = os.path.join(data_path, basefilename + ".comparisons")
    metadata_filename = os.path.join(data_path, basefilename + ".comparisons_metadata")
    
    # read the comparisons and metadata
    with open(comparisons_filename, "rb") as file:
        comparisons = pickle.load(file)
    with open(metadata_filename, "rb") as file:
        metadata = pickle.load(file)

    return comparisons, metadata


def aggregate_comparison_stats(comparisons, max_k=None):
    """
    Parameters
    ----------
    comparisons: list of comparisons as generated by get_pairwise_comparisons, parallel_get_pariwise_comparsions,
       get_pairwise_comparisons_varying_k, or parallel_get_pairwise_comparisons_varying_k
    
    max_k: (int)
       if max_k in not None, then it should be an integer greater than 2 that corresponds the max_k used in
       get_pairwise_comparisons_varying_k or parallel_get_pairwise_comparisons_varying_k. 
    Returns
    -------
    (results, results_mean, results_variance)    
    """
    if max_k is None:
        if len(comparisons) > 1:
            n_neighbors = int((math.sqrt(8*len(comparisons) + 1)+1)/2)
            logger.debug("computed n_neighbors: {}, size of comparisons: {}".format(n_neighbors, len(comparisons)))
            assert(n_neighbors*(n_neighbors - 1)/2 == len(comparisons))
        else:
            n_neighbors = 2

        results = {}
        results_mean = np.ones((n_neighbors, n_neighbors))
        results_variance = np.zeros((n_neighbors, n_neighbors))

        for i, j, vectors in comparisons:
            describe = scipy.stats.describe(vectors[1])
            results[i, j] = describe
            results_mean[i, j] = describe.mean
            results_variance[i, j] = describe.variance

            # symmetry of matrix
            results_mean[j, i] = describe.mean
            results_variance[j, i] = describe.variance
    elif not (max_k > 1):
        raise ValueError("max_k: {} must be an integer greater than 1".format(max_k))
    else:
        if max_k > 2: # don't want to divide by zero
            n_neighbors = int((math.sqrt(8*len(comparisons)/(max_k-2) + 1)+1)/2)
            logger.debug("computed n_neighbors: {}, size of comparisons: {}".format(n_neighbors, len(comparisons)))
            assert(n_neighbors*(n_neighbors - 1)/2*(max_k-2) == len(comparisons))
        elif len(comparisons) > 1:
            n_neighbors = int((math.sqrt(8*len(comparisons) + 1)+1)/2)
            logger.debug("computed n_neighbors: {}, size of comparisons: {}".format(n_neighbors, len(comparisons)))
            assert(n_neighbors*(n_neighbors - 1)/2 == len(comparisons))
        else:
            n_neighbors = 2

        results = {}
        results_mean = np.ones((n_neighbors, n_neighbors, max_k))
        results_variance = np.zeros((n_neighbors, n_neighbors, max_k))    
        for i, j, k, vectors in comparisons:
            describe = scipy.stats.describe(vectors[1])
            results[i, j, k] = describe
            results_mean[i, j, k] = describe.mean
            results_variance[i, j, k] = describe.variance

            # symmetry of matrix
            results_mean[j, i, k] = describe.mean
            results_variance[j, i, k] = describe.variance
    return results, results_mean, results_variance
