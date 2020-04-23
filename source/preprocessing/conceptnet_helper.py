import os
import csv
import gzip
import json
import tqdm
import math
import logging
import itertools

import numpy as np

from operator import mul
from functools import reduce
from scipy.sparse import coo_matrix, dok_matrix
from collections import defaultdict, namedtuple


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

Resource = namedtuple('Resource',
                      'index2concept, concept2index, index2relation, relation2index, edges, cooc_mat')

# we'll use infinity as a default distance to nodes.
Edge = namedtuple('Edge', 'start, end, rel, cost')

# Based on "Commonsense Knowledge Mining from Pretrained Models".
# Joshua Feldman, Joe Davison, and Alexander M. Rush. 2019.
REL_TO_TEMPLATE = {
    "relatedto": "[w1] is like [w2]",
    "externalurl": "[w1] is described at the following URL [w2]",
    "formof": "[w1] is a form of the word [w2]",
    "isa": "[w1] is a type of [w2]",
    "notisa": "[w1] is not [w2]",
    "partof": "[w1] is part of [w2]",
    "usedfor": "[w1] is used for [w2]",
    "capableof": "[w1] can [w2]",
    "atlocation": "You are likely to find [w1] in [w2]",
    "causes": "Sometimes [w1] causes [w2]",
    "hasa": "[w1] has [w2]",
    "hassubevent": "Something you do when you [w1] is [w2]",
    "hasfirstsubevent": "the first thing you do when you [w1] is [w2]",
    "haslastsubevent": "the last thing you do when you [w1] is [w2]",
    "hasprerequisite": "In order for [w1] to happen, [w2] needs to happen",
    "hasproperty": "[w1] is [w2]",
    "hascontext": "[w1] is a word used in the context of [w2]",
    "motivatedbygoal": "You would [w1] because you want to [w2]",
    "obstructedby": "[w1] can be prevented by [w2]",
    "desires": "[w1] wants [w2]",
    "createdby": "[w1] is created by [w2]",
    "synonym": "[w1] and [w2] have similar meanings",
    "antonym": "[w1] is the opposite of [w2]",
    "distinctfrom": "it cannot be both [w1] and [w2]",
    "derivedfrom": "the word [w1] is derived from the word [w2]",
    "definedas": "[w1] is defined as [w2]",
    "entails": "if [w1] is happening, [w2] is also happening",
    "mannerof": "[w1] is a specific way of doing [w2]",
    "locatednear": "[w1] is located near [w2]",
    "dbpedia": "[w1] is conceptually related to [w2]",
    "similarto": "[w1] is similar to [w2]",
    "etymologicallyrelatedto": "the word [w1] and the word [w2] have the same origin",
    "etymologicallyderivedfrom": "the word [w1] comes from the word [w2]",
    "causesdesire": "[w1] makes people want [w2]",
    "madeof": "[w1] is made of [w2]",
    "receivesaction": "[w1] can be [w2]",
    "instanceof": "[w1] is an example of [w2]",
    "notdesires": "[w1] does not want [w2]",
    "notusedfor": "[w1] is not used for [w2]",
    "notcapableof": "[w1] is not capable of [w2]",
    "nothasproperty": "[w1] does not have the property of [w2]",
    "notmadeof": "[w1] is not made of [w2]"
}


def build_conceptnet(conceptnet_dir):
    """
    Download ConceptNet and build it locally.
    First run:
    !wget https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.7.0.csv.gz \
        -O ~/resources/conceptnet-assertions-5.6.0.csv.gz
    """
    resource_dir = conceptnet_dir.replace("conceptnet", "")
    concept2index = defaultdict(itertools.count(0).__next__)
    relation2index = defaultdict(itertools.count(0).__next__)

    # concept -> concept -> relation = weight
    edges = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    with gzip.open(os.path.join(resource_dir, 'conceptnet-assertions-5.7.0.csv.gz'), mode='rt') as f_in:
        csvfile = csv.reader(f_in, delimiter='\t', quotechar="'")
        for row in tqdm.tqdm(csvfile):
            """
             Row format:
                The URI of the whole edge
                The relation expressed by the edge
                The node at the start of the edge
                The node at the end of the edge
                A JSON structure of additional information about the edge, such as its weight

            Example:
                b'/a/[/r/Antonym/,/c/en/arise/,/c/en/repose/]
                /r/Antonym
                /c/en/arise
                /c/en/repose
                {"dataset": "/d/verbosity", "license": "cc:by/4.0", 
                "sources": [{"contributor": "/s/resource/verbosity"}], 
                "surfaceEnd": "repose", "surfaceStart": "arise", 
                "surfaceText": "[[arise]] is the opposite of [[repose]]", "weight": 0.15}
            """
            relation, start, end = row[1:4]

            # Keep only English concepts
            if not start.startswith('/c/en') or not end.startswith('/c/en'):
                continue

            relation_label = os.path.basename(relation).lower()
            edge_info = json.loads(row[-1])
            start_label = edge_info.get('surfaceStart', '').lower().strip()
            end_label = edge_info.get('surfaceEnd', '').lower().strip()

            if len(start_label) > 0 and len(end_label) > 0:
                weight = edge_info['weight']
                start_index, end_index = concept2index[start_label], concept2index[end_label]
                relation_index = relation2index[relation_label]
                edges[start_index][end_index][relation_index] = weight

        conceptnet_dir = os.path.join(resource_dir, 'conceptnet')
        if not os.path.exists(conceptnet_dir):
            os.mkdir(conceptnet_dir)

        index2relation = {index: relation for relation, index in relation2index.items()}
        with open(os.path.join(conceptnet_dir, 'relations.txt'), 'w', encoding='utf-8') as f_out:
            for index in range(len(index2relation)):
                f_out.write(index2relation[index] + '\n')

        index2concept = {index: concept for concept, index in concept2index.items()}
        with open(os.path.join(conceptnet_dir, 'concepts.txt'), 'w', encoding='utf-8') as f_out:
            for index in range(len(index2concept)):
                f_out.write(index2concept[index] + '\n')

        with open(os.path.join(conceptnet_dir, 'edges.txt'), 'w', encoding='utf-8') as f_out:
            for start_index in range(len(index2concept)):
                f_out.write(json.dumps(edges[start_index]) + '\n')

        row_ind, col_ind, cooc_data = zip(*[(c1, c2, 1)
                                            for c1, c1_relations in edges.items()
                                            for c2 in c1_relations.keys()])

        cooc_mat = coo_matrix((cooc_data, (row_ind, col_ind)),
                              shape=(len(concept2index), len(concept2index)))

        np.savez_compressed(os.path.join(conceptnet_dir, 'cooc.npz'),
                            data=cooc_mat.data,
                            row=cooc_mat.row,
                            col=cooc_mat.col,
                            shape=cooc_mat.shape)


def load_conceptnet(conceptnet_dir):
    """
    Load an existing local ConceptNet from this directory
    """
    with open(os.path.join(conceptnet_dir, 'concepts.txt'), 'r', encoding='utf-8') as f_in:
        index2concept = [line.strip() for line in f_in]
        concept2index = {c: i for i, c in enumerate(index2concept)}

    with open(os.path.join(conceptnet_dir, 'relations.txt'), 'r', encoding='utf-8') as f_in:
        index2relation = [line.strip() for line in f_in]
        relation2index = {c: i for i, c in enumerate(index2relation)}

    # concept -> concept -> relation = weight
    edges = {}

    with open(os.path.join(conceptnet_dir, 'edges.txt'), 'r', encoding='utf-8') as f_in:
        for c1, line in enumerate(f_in):
            edges[c1] = json.loads(line.strip())

    edges = {int(c1): {
        int(c2): {int(r): float(score) for r, score in relations.items()}
        for c2, relations in c1_rs.items()}
        for c1, c1_rs in edges.items()}

    with np.load(os.path.join(conceptnet_dir, 'cooc.npz')) as loader:
        cooc_mat = coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])

    return Resource(*(index2concept, concept2index, index2relation,
                      relation2index, edges, cooc_mat))


class NodesOnPathFinder:
    """
    Applies bi-directional search to find the nodes in the shortest paths a pair of terms.
    """

    def __init__(self, resource, include_reverse=True):
        """
        Init the relevant nodes search
        """
        self.adjacency_matrix = resource.cooc_mat
        self.transposed_adjacency_matrix = resource.cooc_mat.T

        # Include reversed relations
        if include_reverse:
            self.adjacency_matrix += self.transposed_adjacency_matrix
            self.transposed_adjacency_matrix = self.adjacency_matrix

    def find_nodes_on_path(self, x, y, max_length=5):
        """
        Finds all nodes in the shortest paths between x and y
        subject to the maximum length.
        :param x -- the index of the first term
        :param y -- the index of the second term
        :param max_length -- the maximum path length
        """
        m = self.adjacency_matrix
        mT = self.transposed_adjacency_matrix

        dim = m.shape[0]
        n_r = create_one_hot_vector(x, dim)
        n_g = create_one_hot_vector(y, dim)

        return find_nodes(m, mT, n_r, n_g, max_length)


def find_nodes(m, mT, n_r, n_g, max_len):
    """
    Finds all nodes in the shortest paths between x and y
    subject to the maximum length.
    :param m -- the adjacency matrix
    :param mT -- the transposed adjacency matrix
    :param n_r -- the one-hot vector representing the root node
    :param n_g -- the one-hot vector representing the goal node
    :param max_len -- the maximum path length
    """
    nodes = set()
    n_x = n_r
    n_y = n_g

    # Stop condition 1 - no paths
    if max_len == 0:
        return nodes

    # Stop condition 2 - the two sides are connected by one edge.
    # Notice that if max_length == 1, then this function will return the two
    # nodes even if they are not connected - this path will be discarded
    # in the second search phase.
    if max_len == 1:
        return set(n_r.nonzero()[1].flatten()).union(set(n_g.nonzero()[1].flatten()))

    # Move one step in each direction until the root and goal meet
    for l in range(max_len + 1):

        # The root and goal met - apply recursively for each half of the path
        if n_r.dot(n_g.T)[0, 0] > 0:
            intersection = n_r.multiply(n_g)
            forward = find_nodes(
                m, mT, n_x, intersection, int(math.ceil((l + 1) / 2.0)))
            backward = find_nodes(
                m, mT, intersection, n_y, int(math.floor((l + 1) / 2.0)))
            return forward.union(backward)

        # Make a step forward
        if l % 2 == 0:
            n_r = n_r.dot(m)
        # Make a step backward
        else:
            n_g = n_g.dot(mT)

    return nodes


def create_one_hot_vector(x, dim):
    """
    Creates the one-hot vector representing this node
    :param x -- the node
    :param dim -- the number of nodes (the adjacency matrix dimension)
    """
    n_x = dok_matrix((1, dim), dtype=np.int16)
    n_x[0, x] = 1
    n_x = n_x.tocsr()
    return n_x


class Graph:
    def __init__(self, edges):
        self.edges = [Edge(*edge) for edge in edges]

    @property
    def nodes(self):
        return set(sum(([edge.start, edge.end] for edge in self.edges), []))

    @property
    def neighbours(self):
        neighbours = {node: set() for node in self.nodes}
        for edge in self.edges:
            neighbours[edge.start].add((edge.end, edge.rel, edge.cost))

        return neighbours

    def bfs(self, start, goal):
        """
        Get the shortest path from source to dest
        """
        queue = [(start, [''], [start], [1.0])]
        min_len_path = np.inf
        paths = list()

        while queue:
            curr_node, edges_on_path, nodes_on_path, weights_on_path = queue.pop(0)
            for next_node, rel, weight in self.neighbours.get(curr_node, set()):
                if next_node in set(nodes_on_path):
                    continue

                if next_node == goal:
                    curr_path = list(zip(edges_on_path, nodes_on_path, weights_on_path)) + [
                        (rel, next_node, weight)]

                    if len(curr_path) <= min_len_path:
                        min_len_path = len(curr_path)
                        path_weight = reduce(mul, weights_on_path, 1)
                        paths.append((curr_path, path_weight))

                    # Already found shorter paths
                    else:
                        return paths

                else:
                    queue.append((next_node,
                                  edges_on_path + [rel],
                                  nodes_on_path + [next_node],
                                  weights_on_path + [weight]))

        return paths


def shortest_paths(resource, c1, c2, max_length=10, exclude_relations=None):
    """
    Return the shortest paths from c1 to c2, up to max_length edges,
    optionally excluding some relations.
    """
    nodes_finder = NodesOnPathFinder(resource, include_reverse=True)
    c1_index = resource.concept2index.get(c1, None)
    c2_index = resource.concept2index.get(c2, None)

    if c1_index is None or c2_index is None:
        logger.warning('{} not found'.format(c1 if c1_index is None else c2))
        return [([], 0)]

    # Find the nodes on the path
    nodes = nodes_finder.find_nodes_on_path(c1_index, c2_index, max_length=max_length)

    # Get all the edges between these nodes in the original graph
    # Get the maximum weight for each start and end
    curr_edges = {resource.index2concept[start]: {} for start in nodes}

    for start, end in itertools.permutations(nodes, 2):
        start_label = resource.index2concept[start]
        end_label = resource.index2concept[end]
        for relation, weight in resource.edges.get(start, {}).get(end, {}).items():
            relation_label = resource.index2relation[relation]
            if exclude_relations is None or relation_label not in exclude_relations:
                if end_label not in curr_edges[start_label] or \
                        curr_edges[start_label][end_label][1] < weight:
                    curr_edges[start_label][end_label] = (relation_label, weight)
                if start_label not in curr_edges[end_label] or \
                        curr_edges[end_label][start_label][1] < weight:
                    curr_edges[end_label][start_label] = (relation_label + '-1', weight)

    # Create the subgraph and use Dijkstra to find the shortest weighted path
    edge_list = [(start, end, rel, 1.0 / weight)
                 for start, start_rels in curr_edges.items()
                 for end, (rel, weight) in start_rels.items()]
    graph = Graph(edge_list)
    result = graph.bfs(c1, c2)
    return result


def pretty_print(path):
    """
    Print a path in a readable format
    param path: a list of (edge_label, node)
    """
    path_str = ''
    if len(path) > 0:
        path_str += path[0][1]

    for rel, node, _ in path[1:]:
        if rel.endswith('-1'):
            path_str += f' <--{rel[:-2]}-- {node}'
        else:
            path_str += f' --{rel}--> {node}'

    return path_str


def to_natural_language(path):
    """
    Print a path in a readable format
    param path: a list of (edge_label, node)
    """
    props = []
    for (_, node1, _), (rel, node2, _) in zip(path, path[1:]):
        w1, w2, rel = (node2, node1, rel.replace("-1", "")) if rel.endswith('-1') else (node1, node2, rel)
        props.append(REL_TO_TEMPLATE[rel].replace("[w1]", w1).replace("[w2]", w2))

    return ". ".join([p[0].upper() + p[1:] for p in props])
