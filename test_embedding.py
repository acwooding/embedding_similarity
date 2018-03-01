import embedding as embed
import os
import numpy as np
import scipy
import random

# hypothesis testing framework
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from hypothesis import given, note, settings, HealthCheck
from itertools import combinations as comb

from collections import defaultdict
import logging

LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"

logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, datefmt=DATE_FORMAT)
logger = logging.getLogger()

test_dir = "./tests"
fixed_examples_path = os.path.join(test_dir, "static")

# all possible valid knn metrics for sklearn nn algorithm
# except for metrics inteded for boolean-valued vector spaces
KNN_METRICS = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
               'manhattan', 'braycurtis', 'canberra', 'chebyshev',
               'correlation', 'hamming', 'mahalanobis', 'minkowski',
               'seuclidean', 'sqeuclidean']

SIMILARITY_METRICS = ['jaccard', 'adapted-ktau']


class Embedding(object):
    '''
    Create an object that we can use for testing that has all the information of an embbedding
    that would be read by `embed.read_embedding`.
    '''
    def __init__(self, data, labels, embedding_algorithm, dataset, other_info, run_number):
        # for nearest neighbors to work, need dimension < number of observations
        dimension = data.draw(st.integers(min_value=5, max_value=len(labels)-1))
        self.embedding = np.random.rand(len(labels), dimension)
        self.labels = np.array(labels)
        self.embedding_algorithm = embedding_algorithm
        self.dataset = dataset
        self.other_info = other_info
        self.run_number = run_number
        self.metadata = {"Embedding Algorithm": embedding_algorithm, "Dataset": dataset, "Run Number":str(run_number),
                "Other Information": other_info}

class Neighbors(object):
    '''
    Neighbors object that contains an Embedding object and the corresponding neighbor sets. On the full embedding.
    '''
    def __init__(self, embedding_object, data, other_info, metric):
        self.n_neighbors = data.draw(st.integers(min_value=2, max_value=len(embedding_object.labels)))
        self.embedding_object = embedding_object
        embedding = embedding_object.embedding
        labels = embedding_object.labels
        neighbors = embed.get_neighbors(embedding, labels, n_neighbors=self.n_neighbors, algorithm='brute', metric=metric)
        self.neighbors = neighbors

        ## add to existing metadata
        metadata = embedding_object.metadata
        metadata["k-nn Metric"] = metric
        metadata["Number of Neighbors"] = str(self.n_neighbors)
        metadata["Other Neighbors Information"] = other_info
        self.metadata = metadata


EmbeddingStrategy = st.builds(Embedding,
                              st.data(),
                              st.shared(st.lists(elements=st.text(average_size=10, min_size=1), min_size=7, max_size=200, unique=True), key='shared_labels'),
                              st.text(average_size=10),
                              st.text(average_size=10),
                              st.text(average_size=10),
                              st.integers(min_value=0, max_value=1000))

NeighborsStrategy = st.builds(Neighbors,
                              EmbeddingStrategy,
                              st.data(),
                              st.text(max_size=20),
                              st.sampled_from(KNN_METRICS))


def comparison_equality(comparison1, comparison2):
    """
    Equality test for the outputs of compare_neighbors.
    """
    labels1, similarity1 = comparison1
    labels2, similarity2 = comparison2

    dd1, dd2 = defaultdict(list), defaultdict(list)

    for i, label in enumerate(labels1):
        dd1[label] = similarity1[i]
        dd2[labels2[i]] = similarity2[i]
    note((dd1, dd2))
    return dd1 == dd2


@given(EmbeddingStrategy)
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_save_and_read_embedding(embedding_object):
    """
    Check that save_embedding and then read_embedding gives back the same data in the expected format.
    """
    basefilename = "test_embedding"
    data_path = test_dir

    embed.save_embedding(basefilename, embedding=embedding_object.embedding, labels=embedding_object.labels,
                         embedding_algorithm=embedding_object.embedding_algorithm, dataset=embedding_object.dataset,
                         other_info=embedding_object.other_info, run_number=embedding_object.run_number, data_path=data_path)
    read_embedding, read_labels, read_metadata = embed.read_embedding(basefilename, run_number=embedding_object.run_number, data_path=data_path)
    ## clean up saved file
    filename = basefilename + "_" + str(embedding_object.run_number)
    embedding_filename = os.path.join(data_path, filename + ".embedding.npy")
    labels_filename = os.path.join(data_path, filename + ".labels.npy")
    metadata_filename = os.path.join(data_path, filename + ".metadata")
    os.remove(embedding_filename)
    os.remove(labels_filename)
    os.remove(metadata_filename)

    # check the embedding and labels are the same
    assert((read_embedding == embedding_object.embedding).all())
    assert((read_labels == embedding_object.labels).all())
    # check the metadata is the same
    assert(read_metadata == embedding_object.metadata)

@given(st.data(), st.sampled_from(KNN_METRICS), EmbeddingStrategy)
def test_get_neighbors(data, metric, embedding_object):
    """
    Check that get_neighbors returns valid output
    """
    ## don't ask for more neighbors than there are in the embedding
    n_neighbors = data.draw(st.integers(min_value=2, max_value=embedding_object.embedding.shape[0]))
    run_number = embedding_object.run_number
    embedding, labels, metadata = embedding_object.embedding, embedding_object.labels, embedding_object.metadata

    neighbors, distances = embed.get_neighbors(embedding, labels, n_neighbors=n_neighbors, algorithm='brute')

    # check everything has a neighbor and distance
    assert(np.in1d(labels, list(distances.keys())).all())
    assert(np.in1d(labels, list(neighbors.keys())).all())

    for k, v in neighbors.items():
        ## expected number of neighbors and distances
        assert(len(v) == n_neighbors)
        assert(len(distances[k]) == n_neighbors)
        ## check distances are all non-zero
        assert((distances[k] >= 0).all())

''' Leave this out for now...example files missing...redo this differently.
@given(st.integers(min_value=2, max_value=30), st.text(max_size=20))
def test_make_neighbors_from_files(n_neighbors, other_info):
    """Check that the neighbors come back in the expected format."""
    basefilename = "make_neighbours_test_embedding"
    embedding_dir = test_dir
    run_numbers = [858]
    knn_metrics = ['l1', 'l2', 'cosine']

    ## cannot ask for more neighbors than the number of elements in the embedding
    embedding, labels, metadata = embed.read_embedding(basefilename, run_number=run_numbers[0], data_path=embedding_dir)

    if n_neighbors >= len(labels):
        n_neighbors = len(labels) - 1

    embed.make_neighbors_from_files(basefilename, run_numbers=run_numbers, n_neighbors=n_neighbors,
                                    knn_metrics=knn_metrics, other_info=other_info, data_path=embedding_dir)

    for metric in knn_metrics:
        read_neighbors, read_metadata = embed.read_neighbors(basefilename, metric=metric, run_number=run_numbers[0], data_path=embedding_dir)
        assert(len(read_neighbors)==2) # neighbors and distances
        neighbors = read_neighbors[0]
        distances = read_neighbors[1]f

        # check everything has a neighbor and distance
        assert(np.in1d(labels, list(distances.keys())).all())
        assert(np.in1d(labels, list(neighbors.keys())).all())

        for k, v in neighbors.items():
            ## expected number of neighbors and distances
            assert(len(v)==n_neighbors)
            assert(len(distances[k])==n_neighbors)
            ## check distances are all non-zero
            assert((distances[k] >= 0).all())

        ## check the metadata matches the input
        assert(read_metadata["k-nn Metric"] == metric)
        assert(read_metadata["Number of Neighbors"] == str(n_neighbors))
        assert(read_metadata["Other Neighbors Information"] == other_info)

'''

@given(NeighborsStrategy)
@settings(deadline=None)
def test_save_and_read_neighbors(neighbors_object):
    basefilename = "test_embedding"
    data_path = test_dir
    metric = neighbors_object.metadata["k-nn Metric"]
    run_number= neighbors_object.embedding_object.run_number
    other_info = neighbors_object.metadata["Other Neighbors Information"]

    note("metric {}".format(metric))
    note("embedding {}".format(neighbors_object.embedding_object.embedding))

    embed.save_neighbors(basefilename, neighbors=neighbors_object.neighbors, metric=metric,
                         n_neighbors=neighbors_object.n_neighbors, other_info=other_info,
                         metadata=neighbors_object.embedding_object.metadata,
                         run_number=run_number,
                         data_path=data_path)

    neighbors, metadata = embed.read_neighbors(basefilename, run_number=run_number, metric=metric, data_path=data_path)
    # clean up after the test
    filename = basefilename + "_" + metric + "_" + str(run_number)
    neighbors_filename = os.path.join(data_path, filename + ".neighbors")
    metadata_filename = os.path.join(data_path, filename + ".neighbors_metadata")
    os.remove(neighbors_filename)
    os.remove(metadata_filename)

    assert(len(neighbors) == 2)

    note("neighbors dict: {} \n distances dict: {}".format(neighbors[0], neighbors[1]))
    labels = neighbors_object.embedding_object.labels
    # check everything has a neighbor and distance
    assert(np.in1d(labels, list(neighbors[0].keys())).all())
    assert(np.in1d(labels, list(neighbors[1].keys())).all())
    for k in neighbors_object.embedding_object.labels:
        assert(neighbors[0][k] == neighbors_object.neighbors[0][k])
        assert((neighbors[1][k] == neighbors_object.neighbors[1][k]).all())
    assert(metadata == neighbors_object.metadata)


@given(st.data())
def test_jaccard_text(data):
    set1 = data.draw(st.sets(elements=st.text(average_size=5), average_size=50, max_size=100))
    overlap_size = data.draw(st.integers(min_value=0, max_value=len(set1)))
    if set1:
        set2 = set(random.sample(set1, overlap_size))
    else:
        set2 = set([])
    set2 = set2.union(data.draw(st.sets(elements=st.text(average_size=5), average_size=50, max_size=100)))
    if set1 and set2:
        score = len(set1.intersection(set2))/len(set1.union(set2))*1.0
    else:
        score = 0
    note("{}, {}".format(embed.jaccard(set1, set2), score))
    assert(embed.jaccard(set1, set2) == score)
    assert(score <= 1)
    assert(score >= 0)


@given(st.data())
def test_jaccard_integers(data):
    set1 = data.draw(st.sets(elements=st.integers(), average_size=100, min_size=1))
    overlap_size = data.draw(st.integers(min_value=0, max_value=len(set1)))
    if set1:
        set2 = set(random.sample(set1, overlap_size))
    else:
        set2 = set([])
    set2 = set2.union(data.draw(st.sets(elements=st.integers(), average_size=50, min_size=1)))
    score = len(set1.intersection(set2))/len(set1.union(set2))*1.0
    note("{}, {}".format(embed.jaccard(set1, set2), score))
    assert(embed.jaccard(set1, set2) == score)
    assert(score <= 1)
    assert(score >= 0)

@given(st.data())
def test_adaptedKendallTau_integers(data):
    list1 = data.draw(st.lists(elements=st.integers(), unique=True, average_size=100))

    # check akt score with myself is 1
    if list1:
        self_score = embed.adaptedKendallTau(list1, list1)
        note("akt score of list with itself: {}".format(self_score))
        note(list1)
        assert(self_score == 1)

    # check akt score is always between 0 and 1
    overlap_size = data.draw(st.integers(min_value=0, max_value=len(list1)))
    if list1 and overlap_size:
        list2 = random.sample(list1, overlap_size)
        min_value = max(list1) + 1
    else:
        list2 = []
        min_value = None
    list2 = list2 + data.draw(st.lists(elements=st.integers(min_value=min_value), unique=True, average_size=50))

    akt_score = embed.adaptedKendallTau(list1, list2)
    note(akt_score)
    assert(akt_score <= 1)
    assert(akt_score >= 0)


@given(st.data())
def test_akt_vs_jaccard(data):
    list1 = data.draw(st.lists(elements=st.integers(), unique=True, average_size=100))

    # top elements from list 1 are the ones that that overlap with list 2, akt >=jac
    # only overlap on up to %90 of the elements
    overlap_size = data.draw(st.integers(min_value=0, max_value=int(len(list1)*.8)))

    if list1 and overlap_size:
        list2 = list1[:overlap_size]
        random.shuffle(list2)
        min_value = max(list1) + 1
    elif not list1:
        list2 = []
        min_value = None
    else:
        list2 = []
        min_value = max(list1) + 1

    list2 = list2 + data.draw(st.lists(elements=st.integers(min_value=min_value), unique=True, average_size=50))
    note((list1, list2, list1==list2))

    jac_score = embed.jaccard(set(list1), set(list2))
    akt_score = embed.adaptedKendallTau(list1, list2)
    if (overlap_size > 0):
        assert(akt_score >= jac_score)
    else:
        assert(akt_score <= jac_score)

    # check that shuffling elements in the same list results in an akt_score of <= 1 and jac score of 1
    list3 = list2
    random.shuffle(list3)
    jac_score = embed.jaccard(set(list3), set(list2))
    akt_score = embed.adaptedKendallTau(list3, list2)
    assert(akt_score <= jac_score)
    if list2:
        assert(jac_score == 1)
    else:
        assert(jac_score == 0)

@given(st.text(average_size=10), st.data(), st.sampled_from(SIMILARITY_METRICS))
def test_list_similarity(label, data, metric):
    list1 = data.draw(st.lists(elements=st.integers(), unique=True, average_size=100))

    # check score with myself is 1
    if list1:
        _, self_score = embed.list_similarity(label, list1, list1, how=metric)
        note("{} score of list with itself: {}".format(metric, self_score))
        note(list1)
        assert(self_score == 1)

    # check score is always between 0 and 1
    overlap_size = data.draw(st.integers(min_value=0, max_value=len(list1)))
    if list1 and overlap_size:
        list2 = random.sample(list1, overlap_size)
        min_value = max(list1) + 1
    else:
        list2 = []
        min_value = None
    list2 = list2 + data.draw(st.lists(elements=st.integers(min_value=min_value), unique=True, average_size=50))

    ret_label, score = embed.list_similarity(label, list1, list2, how=metric)
    assert(label == ret_label)
    assert(score <= 1)
    assert(score >= 0)

    # check that we computed the correct score
    if metric == 'jaccard':
        assert(score == embed.jaccard(set(list1), set(list2)))
    elif metric == 'adapted-ktau':
        assert(score == embed.adaptedKendallTau(list1, list2))


@given(NeighborsStrategy, NeighborsStrategy, st.data(), st.sampled_from(SIMILARITY_METRICS))
@settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_compare_neighbors(neighbors1, neighbors2, data, metric):
    D1 = neighbors1.neighbors[0]
    D2 = neighbors2.neighbors[0]
    n_neighbors = data.draw(st.integers(min_value=2, max_value=min([neighbors1.n_neighbors, neighbors2.n_neighbors])))
    labels, similarity = embed.compare_neighbors(D1, D2, n_neighbors=n_neighbors, how=metric)

    assert(set(labels) == set(neighbors1.embedding_object.labels))
    assert(len(labels) == len(similarity))
    assert((np.array(similarity) <= 1).all())
    assert((np.array(similarity) >= 0).all())


@given(NeighborsStrategy, NeighborsStrategy, st.data(), st.sampled_from(SIMILARITY_METRICS))
@settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_parallel_compare_neighbors(neighbors1, neighbors2, data, metric):
    n_proc = 2 # the examples we're generating are quite small

    D1 = neighbors1.neighbors[0]
    D2 = neighbors2.neighbors[0]

    chunksize = len(D1)/n_proc + 1

    # avoid hanging on small problems
    if chunksize < 1000:
        n_proc = 1
        chunksize = None

    n_neighbors = data.draw(st.integers(min_value=2, max_value=min([neighbors1.n_neighbors, neighbors2.n_neighbors])))
    labels, similarity = embed.compare_neighbors(D1, D2, n_neighbors=n_neighbors, how=metric)
    p_labels, p_similarity = embed.parallel_compare_neighbors(D1, D2, n_neighbors=n_neighbors, n_proc=n_proc, chunksize=chunksize, how=metric)

    # check that the results match
    # note that they don't come out in the same order.
    s_dict = defaultdict(list)
    p_dict = defaultdict(list)
    for i, label in enumerate(labels):
        s_dict[label] = similarity[i]
        p_dict[p_labels[i]] = p_similarity[i]

    assert(s_dict == p_dict)


@given(NeighborsStrategy, NeighborsStrategy, st.data(), st.sampled_from(SIMILARITY_METRICS))
@settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
def test_parallel_compare_neighbors(neighbors1, neighbors2, data, metric):
    n_proc = 2 # the example we're generating are quite small

    D1 = neighbors1.neighbors[0]
    D2 = neighbors2.neighbors[0]

    chunksize = len(D1)/n_proc + 1

    # avoid hanging on small problems
    if chunksize < 1000:
        n_proc = 1
        chunksize = None

    n_neighbors = data.draw(st.integers(min_value=2, max_value=min([neighbors1.n_neighbors, neighbors2.n_neighbors])))
    comparison = embed.compare_neighbors(D1, D2, n_neighbors=n_neighbors, how=metric)
    p_comparison = embed.parallel_compare_neighbors(D1, D2, n_neighbors=n_neighbors, n_proc=n_proc, chunksize=chunksize, how=metric)

    # check that the results match
    # note that they don't come out in the same order.
    assert(comparison_equality(comparison, p_comparison))

@given(st.lists(elements=NeighborsStrategy, min_size=1, max_size=10), st.data(), st.sampled_from(SIMILARITY_METRICS))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_get_pairwise_comparisons(neighbors_object_list, data, metric):
    n_neighbors_list = []
    neighbors_list = []
    for neighbors_object in neighbors_object_list:
        n_neighbors_list.append(neighbors_object.n_neighbors)
        neighbors_list.append(neighbors_object.neighbors)
    n_neighbors = data.draw(st.integers(min_value=2, max_value=min(n_neighbors_list)))
    comparisons = embed.get_pairwise_comparisons(neighbors_list, n_neighbors=n_neighbors, how=metric)

    # check that it's giving back expected results
    for i1, i2, comparison in comparisons:
        s_comparison= embed.compare_neighbors(neighbors_list[i1][0], neighbors_list[i2][0], n_neighbors=n_neighbors, how=metric)
        assert(comparison_equality(comparison, s_comparison))


@given(st.lists(elements=NeighborsStrategy, min_size=1, max_size=10), st.data())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_knn_iterate(neighbors_object_list, data):
    n_neighbors_list = []
    neighbors_list = []
    for neighbors_object in neighbors_object_list:
        n_neighbors_list.append(neighbors_object.n_neighbors)
        neighbors_list.append(neighbors_object.neighbors)
    n_neighbors = data.draw(st.integers(min_value=2, max_value=min(n_neighbors_list)))

    knn_iter = embed.knn_iterator(neighbors_list, n_neighbors=n_neighbors)
    knn_iter_list = list(knn_iter)

    # compute the expected length of knn_iter_list
    pairs = [(a, b) for a, b in comb(range(len(neighbors_list)), 2) if a < b]
    knn_length = 0
    for a, b in pairs:
        knn_dict_a, _ = neighbors_list[a]
        knn_dict_b, _ = neighbors_list[b]
        knn_length += len(set(knn_dict_a.keys()).intersection(knn_dict_b.keys()))

    assert(knn_length == len(knn_iter_list))

    # check elements satisfy basic expected properties
    for item in knn_iter_list:
        (i1, i2), word, n1, n2 = item
        assert(i1 < i2)
        note(neighbors_list[i1][0][word])
        note(n1)
        assert(neighbors_list[i1][0][word][1:n_neighbors] == n1)
        assert(neighbors_list[i2][0][word][1:n_neighbors] == n2)


@given(st.lists(elements=NeighborsStrategy, min_size=1, max_size=10), st.integers(min_value=2, max_value=300))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_knn_iterator_varying_k(neighbors_object_list, max_k):
    neighbors_list = []
    for neighbors_object in neighbors_object_list:
        neighbors_list.append(neighbors_object.neighbors)

    knn_iter = embed.knn_iterator_varying_k(neighbors_list, max_k)
    knn_iter_list = list(knn_iter)

    # compute the expected length of knn_iter_list
    pairs = [(a, b) for a, b in comb(range(len(neighbors_list)), 2) if a < b]
    knn_length = 0
    for a, b in pairs:
        knn_dict_a, _ = neighbors_list[a]
        knn_dict_b, _ = neighbors_list[b]
        knn_length += len(set(knn_dict_a.keys()).intersection(knn_dict_b.keys()))*(max_k-2)

    assert(knn_length == len(knn_iter_list))

    # check elements satisfy basic expected properties
    for item in knn_iter_list:
        (i1, i2, k), word, n1, n2 = item
        assert(i1 < i2)
        note(neighbors_list[i1][0][word])
        note(n1)
        assert(neighbors_list[i1][0][word][1:k] == n1)
        assert(neighbors_list[i2][0][word][1:k] == n2)
        assert(k <= max_k)


@given(st.text(average_size=10), st.text(average_size=10), st.data(), st.sampled_from(SIMILARITY_METRICS))
def test_pairs_list_similarity(context, label, data, metric):
    list1 = data.draw(st.lists(elements=st.integers(), unique=True, average_size=100))

    # check score with myself is 1
    if list1:
        _, (_, self_score) = embed.pairs_list_similarity(context, label, list1, list1, how=metric)
        note("{} score of list with itself: {}".format(metric, self_score))
        note(list1)
        assert(self_score == 1)

    # check score is always between 0 and 1
    overlap_size = data.draw(st.integers(min_value=0, max_value=len(list1)))
    if list1 and overlap_size:
        list2 = random.sample(list1, overlap_size)
        min_value = max(list1) + 1
    else:
        list2 = []
        min_value = None
    list2 = list2 + data.draw(st.lists(elements=st.integers(min_value=min_value), unique=True, average_size=50))

    ret_context, (ret_label, score) = embed.pairs_list_similarity(context, label, list1, list2, how=metric)
    assert(label == ret_label)
    assert(context == ret_context)

    assert(score <= 1)
    assert(score >= 0)

    # check that we computed the correct score
    if metric == 'jaccard':
        assert(score == embed.jaccard(set(list1), set(list2)))
    elif metric == 'adapted-ktau':
        assert(score == embed.adaptedKendallTau(list1, list2))


@given(st.lists(elements=NeighborsStrategy, min_size=1, max_size=10), st.data(), st.sampled_from(SIMILARITY_METRICS))
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_parallel_get_pairwise_comparisons(neighbors_object_list, data, metric):
    neighbors_list = []
    n_neighbors_list = []
    for neighbors_object in neighbors_object_list:
        neighbors_list.append(neighbors_object.neighbors)
        n_neighbors_list.append(neighbors_object.n_neighbors)

    n_neighbors = data.draw(st.integers(min_value=2, max_value=min(n_neighbors_list)))

    # set parameters so we don't hang on small examples
    num_comparisons = len(list(embed.knn_iterator(neighbors_list)))
    if num_comparisons < 2000:
        n_proc = 1
        chunksize = None
    else:
        n_proc = 2
        chunksize = 1000

    comparisons = embed.get_pairwise_comparisons(neighbors_list, n_neighbors=n_neighbors, how=metric)
    p_comparisons = embed.parallel_get_pairwise_comparisons(neighbors_list, n_neighbors=n_neighbors, how=metric, n_proc=n_proc, chunksize=chunksize)

    for comparison in comparisons:
        i1, i2 = comparison[0], comparison[1]
        for i in range(len(p_comparisons)):
            if i1 == p_comparisons[i][0] and i2 == p_comparisons[i][1]:
                assert(comparison_equality(comparison[2], p_comparisons[i][2]))

@given(st.lists(elements=NeighborsStrategy, min_size=1, max_size=10), st.data(), st.sampled_from(SIMILARITY_METRICS))
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_parallel_get_pairwise_comparisons_varying_k(neighbors_object_list, data, metric):
    neighbors_list = []
    n_neighbors_list = []
    for neighbors_object in neighbors_object_list:
        neighbors_list.append(neighbors_object.neighbors)
        n_neighbors_list.append(neighbors_object.n_neighbors)

    n_neighbors = data.draw(st.integers(min_value=2, max_value=min(n_neighbors_list)))
    max_k = data.draw(st.integers(min_value=2, max_value=n_neighbors))


    # set parameters so we don't hang on small examples
    num_comparisons = len(list(embed.knn_iterator(neighbors_list, max_k)))
    if num_comparisons < 2000:
        n_proc = 1
        chunksize = None
    else:
        n_proc = 2
        chunksize = 1000

    comparisons = embed.parallel_get_pairwise_comparisons_varying_k(neighbors_list, max_k, how=metric, n_proc=n_proc, chunksize=chunksize)

    for comparison in comparisons:
        i1, i2, k = comparison[0], comparison[1], comparison[2]
        test_comparison = embed.compare_neighbors(neighbors_list[i1][0], neighbors_list[i2][0], n_neighbors=k, how=metric)
        assert(comparison_equality(comparison[3], test_comparison))

@given(st.lists(elements=NeighborsStrategy, min_size=1, max_size=10), st.data(), st.sampled_from(SIMILARITY_METRICS))
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_save_and_read_comparisons(neighbors_object_list, data, metric):
    basefilename = "test_comparisons"
    data_path = test_dir
    neighbors_list = []
    n_neighbors_list = []
    metadata_list = []
    for neighbors_object in neighbors_object_list:
        neighbors_list.append(neighbors_object.neighbors)
        n_neighbors_list.append(neighbors_object.n_neighbors)
        metadata_list.append(neighbors_object.metadata)

    n_neighbors = data.draw(st.integers(min_value=2, max_value=min(n_neighbors_list)))

    # test out return format of get_pairwise_comparisons
    comparisons = embed.get_pairwise_comparisons(neighbors_list, n_neighbors=n_neighbors, how=metric)
    embed.save_comparisons(basefilename, comparisons=comparisons, metadata=metadata_list, data_path=data_path)
    read_comparisons, read_metadata = embed.read_comparisons(basefilename, data_path=test_dir)

    comparisons_filename = os.path.join(data_path, basefilename + ".comparisons")
    metadata_filename = os.path.join(data_path, basefilename + ".comparisons_metadata")
    os.remove(comparisons_filename)
    os.remove(metadata_filename)

    assert(comparisons == read_comparisons)
    assert(metadata_list == read_metadata)

    # test save/read from parallel_get_pairwise_comparisons_varying_k
    max_k = data.draw(st.integers(min_value=2, max_value=n_neighbors))

    # set parameters so we don't hang on small examples
    num_comparisons = len(list(embed.knn_iterator(neighbors_list, max_k)))
    if num_comparisons < 2000:
        n_proc = 1
        chunksize = None
    else:
        n_proc = 2
        chunksize = 1000

    comparisons = embed.parallel_get_pairwise_comparisons_varying_k(neighbors_list, max_k,
                                                                    n_proc=n_proc, chunksize=chunksize, how=metric)
    embed.save_comparisons(basefilename, max_k=max_k, comparisons=comparisons, metadata=metadata_list, data_path=data_path)
    read_comparisons, read_metadata = embed.read_comparisons(basefilename, data_path=test_dir)

    comparisons_filename = os.path.join(data_path, basefilename + ".comparisons")
    metadata_filename = os.path.join(data_path, basefilename + ".comparisons_metadata")
    os.remove(comparisons_filename)
    os.remove(metadata_filename)

    assert(comparisons == read_comparisons)
    assert(metadata_list == read_metadata)


@given(st.lists(elements=NeighborsStrategy, min_size=1, max_size=10), st.data(), st.sampled_from(SIMILARITY_METRICS))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_aggregate_comparisons_stats(neighbors_object_list, data, metric):
    basefilename = "test_comparisons"
    data_path = test_dir
    neighbors_list = []
    n_neighbors_list = []
    metadata_list = []
    for neighbors_object in neighbors_object_list:
        neighbors_list.append(neighbors_object.neighbors)
        n_neighbors_list.append(neighbors_object.n_neighbors)
        metadata_list.append(neighbors_object.metadata)

    n_neighbors = data.draw(st.integers(min_value=2, max_value=min(n_neighbors_list)))

    # test get_pairwise_comparisons output
    comparisons = embed.get_pairwise_comparisons(neighbors_list, n_neighbors=n_neighbors, how=metric)

    results, results_mean, results_variance = embed.aggregate_comparison_stats(comparisons)

    for i, j, (labels, similarity) in comparisons:
        # check symmetry
        assert(results_mean[i, j] == results_mean[j, i])
        assert(results_variance[i, j] == results_variance[j, i])
        describe = scipy.stats.describe(similarity)
        assert(describe == results[i, j])
        assert(results_mean[i, j] == scipy.mean(similarity))
        assert(describe.variance == results_variance[i, j])

   # test on parallel_get_pairwise_comparisons_varying_k output
    max_k = data.draw(st.integers(min_value=2, max_value=n_neighbors))

    # set parameters so we don't hang on small examples
    num_comparisons = len(list(embed.knn_iterator(neighbors_list, max_k)))
    if num_comparisons < 2000:
        n_proc = 1
        chunksize = None
    else:
        n_proc = 2
        chunksize = 1000

    comparisons = embed.parallel_get_pairwise_comparisons_varying_k(neighbors_list, max_k,
                                                                    n_proc=n_proc, chunksize=chunksize, how=metric)
    results, results_mean, results_variance = embed.aggregate_comparison_stats(comparisons, max_k=max_k)

    for i, j, k, (labels, similarity) in comparisons:
        # check symmetry
        assert(k <= max_k)
        assert(results_mean[i, j, k] == results_mean[j, i, k])
        assert(results_variance[i, j, k] == results_variance[j, i, k])
        describe = scipy.stats.describe(similarity)
        assert(describe == results[i, j, k])
        assert(results_mean[i, j, k] == scipy.mean(similarity))
        assert(describe.variance == results_variance[i, j, k])
