"""
A WordNet class to represent the WordNet synsets and the hyperonymy relation between them, and to support a variety of queries for extracting the information encoded in the hierarchical structure of WordNet.
by Jinghua Xu
"""

import math
from itertools import chain


class WordNet:
    """API for querying WordNet information"""

    def __init__(self, synsets_file, hypernyms_file):
        """
        Constructor for the WordNet class. Build WordNet based on synsets file and hypernyms file.
        Parameters
        ----------
        synsets_file : string
            The file path of synset file.
        hypernyms_file : string
            The file path of hypernyms file.
        """
        # dictionary id: synset
        print("2")
        id2synset = dict()

        with open(synsets_file, 'r', encoding='utf-8') as f_synsets:
            lines = f_synsets.readlines()
            for line in lines:
                # line = line.rstrip('\n')
                data = line.split(',')
                id = data[0]
                # list of lemmas(string)
                lemmas = data[1].split(' ')
                # list of lemmas(object of Lemma)
                new_lemmas = []
                for lemma in lemmas:
                    l = Lemma(lemma)
                    new_lemmas.append(l)

                gloss = data[2]

                synset = Synset(id, new_lemmas, gloss)

                id2synset[id] = synset

        self._verticesDict = id2synset

        # dictionary oringin id : list of relations
        origin2relation = dict()
        with open(hypernyms_file, 'r', encoding='utf-8') as f_hypernyms:
            lines = f_hypernyms.readlines()
            for line in lines:

                line = line.rstrip('\n')
                data = line.split(',')

                origin_id = data[0]
                origin = id2synset[origin_id]

                # list of ids that are hyper
                hypers = data[1:]

                relations = []

                for hyper in hypers:
                    destination_id = hyper
                    # for better readability
                    origin = id2synset[origin_id]
                    destination = id2synset[destination_id]
                    relation = Relation(origin, destination)
                    relations.append(relation)

                origin2relation[origin_id] = relations

        self._edgesDict = origin2relation

        # dictionary lemma : list of synsets where this lemma appears
        lemma2synset = dict()
        for synset in id2synset.values():
            for lemma in synset.lemma:

                if lemma in lemma2synset.keys():
                    lemma2synset[lemma].append(synset)
                else:
                    lemma2synset[lemma] = [synset]

        self._lemmasDict = lemma2synset

        # self._root = self._verticesDict['37987']

    @property
    def edgesdict(self):
        return self._edgesDict

    @property
    def verticesDict(self):
        return self._verticesDict

    @property
    def lemmasDict(self):
        return self._lemmasDict

    def get_synsets(self, noun):
        """
        Returns the list of synsets where noun appears as a lemma. An empty list should be returned if the noun is not part of any WordNet synsets.
        Parameter
        ---------
        noun : string
            a lemma(noun)
        Return
        ------
        synsets : list
            a list of synsets, each synset is an object of Synset
        """
        synsets = []
        lemma = Lemma(noun)
        lemmas_dict = self._lemmasDict
        for synset in lemmas_dict[lemma]:
            synsets.append(synset)
        return synsets

    def bfs(self, synset):
        """
        Returns a dictionary containing all the hypernym synsets on the paths from the current synset to the root node by runing a bfs traversal.
        Parameter
        ---------
        synset : Synset
            Current seynset from which the bfs will start to the root node.
        Return
        ------
        discovered : dictionary
            A dictionary containing all the hypernym synsets on the paths from the current synset to the root node.
            The keys of the dictionary should be Synset objects representing the hypernyms of synset,
            while the values should be tuples of the form (relation, distance). relation is the Relation edge used to discover that hypernym and distance is the integer distance,
            measured in number of edges, from the synset given as parameter to the current key.
        """
        # level number
        distance = 1
        # return dict
        discovered = dict()
        # list of vertices in each level, first level contains only synset
        level = [synset]
        while len(level) > 0:
            next_level = []
            # for each vertex in this level(the origins)
            for u in level:
                # hitting the root node, no outgoing edges from u
                if u.id not in self._edgesDict:
                    break
                # incident relations whose origin is u
                incident_relations = self._edgesDict[u.id]
                for relation in incident_relations:
                    # the hyper synset
                    v = relation.destination
                    if v not in discovered:
                        discovered[v] = (relation, distance)
                        next_level.append(v)
            level = next_level
            distance += 1

        return discovered

    def print_paths_to_root(self, current_synset, path, split_idx):
        """
        A private helper function prints all paths(represented as relations) from synset to root node based on DFS.
        Parameters
        ----------
        current_synset : Synset
            current synset, a synset from which the paths are constructed in the first call.
        path : list
            a list of relations of all paths
        split_idx : list
            a list of split indexes to split path list into individaul paths
        """

        # hitting the root node: a node(synset) does not have any outgoing edge(relation)
        if current_synset.id not in self._edgesDict:
            split_idx.append(len(path))
            print('---------------------------------------------------------')

        # not hitting the root node, keep digging by recursive call
        else:
            for relation in self._edgesDict[current_synset.id]:
                path.append(relation)
                # print path: print a relation in path immmediately when a new relation append to path
                print(relation)
                hyper_to_current = relation.destination
                self.print_paths_to_root(hyper_to_current, path, split_idx)

    def paths_to_root(self, synset):
        """
        A function to print all the different paths from a particular synset to the root node. The function returns a list of Path objects, wihch are paths from synset to root.
        Parameter
        ---------
        synset : Synset
            The seynset vertice where the returned paths are from.
        Return
        ------
        paths_to_root : list
            A list of objects of Path, which are paths from synset to root.
        """
        paths_to_root = []
        # parameters passed to function's initial call
        tmp_path = []
        split_idx = []
        self.print_paths_to_root(synset, tmp_path, split_idx)
        split_idx.pop()

        # split path
        temp = zip(chain([0], split_idx), chain(split_idx, [None]))
        paths = list(tmp_path[i: j] for i, j in temp)

        for path in paths:
            p = Path(path)
            # or print path here(vertices repr in this case)
            # print(p)
            paths_to_root.append(p)
        return paths_to_root

    def lowest_common_hypernyms(self, synset1, synset2):
        """
        A function to compute the lowest common hypernyms between two synsets.
        (A common hypernym is a hypernym that is on the path to root starting from both synset1 and synset2. The lowest common hypernym is the first hypernym that is common to both synsets.)
        Parameters
        ----------
        synset1 : Synset
            One of the two synsets to compute the lowest common synset.
        synset2 : Synset
            The other synset to compute the lowest common synset.
        Return
        ------
        synsets : set
            The set of the lowest common hypernyms between synset1 and synset2.
        """
        synsets1 = set()
        synsets2 = set()
        paths1 = self.paths_to_root(synset1)
        paths2 = self.paths_to_root(synset2)
        for p1 in paths1:
            vertices_p1 = p1.vertices
            tmp = dict()
            for p2 in paths2:
                vertices_p2 = p2.vertices
                for v1 in vertices_p1:
                    if v1 in vertices_p2:
                        tmp[v1] = vertices_p1.index(v1)
                        break
            synsets1.add(min(tmp, key=tmp.get))

        for p1 in paths2:
            vertices_p1 = p1.vertices
            tmp = dict()
            for p2 in paths1:
                vertices_p2 = p2.vertices
                for v1 in vertices_p1:
                    if v1 in vertices_p2:
                        tmp[v1] = vertices_p1.index(v1)
                        break
            synsets2.add(min(tmp, key=tmp.get))

        synsets = synsets1.intersection(synsets2)

        return synsets

    def distance(self, synset1, synset2):
        """
        A function to compute the distance between two synsets. Returns the length (number of edges) of the shortest path between the two synsets.
        Parameters
        ----------
        synset1 : Synset
            One of the two synsets to compute the distance.
        synset2 : Synset
            The other synset to compute the distance.
        Return
        ------
        dist : int
            The distance between synset1 and synset2.
        """
        dist = 0
        dists = []
        lchs = self.lowest_common_hypernyms(synset1, synset2)
        discovered1 = self.bfs(synset1)
        discovered2 = self.bfs(synset2)
        for lch in lchs:
            dist1 = discovered1[lch][1]
            dist2 = discovered2[lch][1]
            add = dist1 + dist2
            dists.append(add)

            dist = min(dists)
            return dist

    def depth_wordnet(self):
        """
        A helper function of lch_similarity computing the overall depth of word net.
        This function calculates the longest possible path from each possible leaf node to root node.
        Return
        ------
        depth : int
            The overall depth of word net.
        """
        depth = 0
        distances_to_root = []
        for vertice in self._verticesDict.values():
            discovered = self.bfs(vertice)
            for relation_distance_tuple in discovered.values():
                distances_to_root.append(relation_distance_tuple[1])
        # depth is distance plus one
        depth = max(distances_to_root) + 1
        return depth

    def lch_similarity(self, synset1, synset2):
        """
        A method to compute the Leacock-Chodorow distance between two synsets. The Leacock-Chodorow distance expresses the similarity between two synsets in terms of the distance between them.
        Parameters
        ----------
        synset1 : Synset
            One synset of the 2 synsets to compute lch similarity.
        synset2 : Synset
            The other synset involving the lch computing.
        Return
        ------
        lc_dist : int
            The Leacock-Chodorow distance between synset1 and synset2.
        """
        lc_dist = 0
        depth = self.depth_wordnet()
        if depth == 0:
            raise Exception(
                "The overall depth of the hierachy is 0, this will lead to a division by 0.")
        else:
            lc_dist = - \
                math.log((1 + self.distance(synset1, synset2))/(depth*2))
        return lc_dist

    def noun_lowest_common_hypernyms(self, noun1, noun2):
        """
        A function to compute the lowest common hypernyms between two nouns. The function returns a set of lowest common hypernyms.
        Parameters
        ----------
        noun1 : string
            One of the 2 nouns to compute lch.
        noun2 : string
            The other noun to compute lch.
        Return
        ------
        synsets : set
            A set of lowest common hypernyms between the synsets of noun1 and noun2.
        """
        synsets = set()

        # possible synsets of nouns
        synsets1 = self.get_synsets(noun1)
        synsets2 = self.get_synsets(noun2)

        # a tmp dictionary stores key(distance of each lch to root node : value(corresponding lch)
        dist2lch = dict()
        # according to the vague description of this exercise and the given example: return the lch(s) that is(are) furthest from root node, since node [48395] is ruled out while [61107] is accepted given the diffence being that the accepted one is further from the root node.

        # each pair of possible synsets of given pair of nouns
        for s1 in synsets1:
            for s2 in synsets2:
                lchs = self.lowest_common_hypernyms(s1, s2)
                for lch in lchs:
                    for path_to_root in self.paths_to_root(lch):
                        dist_lch_to_root = len(path_to_root)
                        if dist_lch_to_root not in dist2lch:
                            dist2lch[dist_lch_to_root] = [lch]
                        else:
                            dist2lch[dist_lch_to_root].append(lch)
        # go with the lch(s) which is farthest from the root node(deepest)
        max_dist = max(dist2lch.keys())
        for lch in dist2lch[max_dist]:
            synsets.add(lch)

        return synsets

    def __iter__(self):
        yield from self._verticesDict.values()

    def __len__(self):
        return len(self._verticesDict)

    def __str__(self):
        """
        The string representation of an object of WordNet class. The representation contains the following information: number of synsets, number of relations, overall depth.

        Return 
        ------
        repr : str
            The meaningful string representaiton of an object of WordNet class.
        """
        repr = ''
        num_synsets = len(self._verticesDict)
        num_edges = 0
        for edges in self._edgesDict.values():
            for _ in edges:
                num_edges += 1
        overall_depth = self.depth_wordnet()

        repr = 'This object of WordNet consists of {} synsets, {} relations, and its overall depth is {}.'
        repr = repr.format(str(num_synsets), str(
            num_edges), str(overall_depth))
        return repr


class Synset:
    """ The node(vertex) class Synset of the Graph class WordNet"""

    def __init__(self, id, lemma, gloss):
        """The constructor of this Synset class.
        Parameters
        ----------
        id : string
            The id if this Synset.
        lemma : list
            List lemmas(objects of Lemma class) of this synset.
        gloss : string
            The gloss of this sysnset.
        """
        self._id = id
        self._lemma = lemma
        self._gloss = gloss

        # id represented as int
        self._index = int(id)

        name = []
        for lm in lemma:
            name.append(lm.lemma)
        # list of lemmas represented as string of this synset.
        self._name = name

    @property
    def id(self):
        """id getter: Return id associated with this Synset(vertex)
        Return
        ------
        self._id : string
            id associated with the synset
        """
        return self._id

    @property
    def lemma(self):
        """lemma getter: Return a list of lemmas associated with this Sysnset(vertex)
        Return
        ------
        self._lemma : list
            the list of lemmas(objects of Lemma class) associated with this Synset(vertex)
        """
        return self._lemma

    @property
    def gloss(self):
        """gloss getter: return the gloss associated with this Sysnset(vertex)
        Return
        ------
        self._gloss : str
            the gloss associated with this synset"""
        return self._gloss

    @property
    def name(self):
        """name getter: Returns a list of lemmas represented as strings.
        Return 
        ------
        self._name : list
            List of lemmas represented as strings.
        """
        return self._name

    @property
    def index(self):
        """index getter: Returns the id of this synset represented as int.
        Return 
        ------
        self._index : int
            id represented as int.
        """
        return self._index

    def __hash__(self):
        """hash function of Synset returns the hash code of id which is the key(identifier) of a unique synset."""
        return hash(self._id)

    def __eq__(self, othr):
        if isinstance(othr, type(self)):
            return ((self._id, self._lemma, self._gloss) == (othr._id, othr._lemma, othr._gloss))

        return NotImplemented

    def __iter__(self):
        """provide an iteration over its lemmas"""
        yield from self._lemma

    def __str__(self):
        """
        The meaningful string representation of an object of Synset class, contaning the following information: id, lemmas and glosses of this synset.
        Return
        ------
        repr : str
            The string representation of a synset.
        """
        repr = 'SYNSET id : {}; list of lemmas: {}; gloss: {}'
        names = ''
        for name in self.name:
            names += name + ', '
        names = names[:-2]
        repr = repr.format(self.id, names, self.gloss)
        return repr


class Relation:
    """ The Relation class stores the origin and the destination of the relation.(functions as the Edge class of a Graph class)"""

    def __init__(self, origin, destination):
        """
        The constructor of the edge class Realtion of Graph Synset
        Parameters
        ----------
        origin : Synset
            origin vertice(synset)
        destination : Synset
            destination vertice(synset)
        """
        self._origin = origin
        self._destination = destination

    @property
    def origin(self):
        """origin(synset) getter"""
        return self._origin

    @property
    def destination(self):
        """destination/hypernym(synset) getter"""
        return self._destination

    def endpoints(self):
        """
        Return (origin, destination) tuple for vertices origin and destination.
        Return
        ------
        (self._origin, self._destination) : tuple
            an edge(relation) representation as tuple(origin, destination)
        """
        return (self._origin, self._destination)

    def __hash__(self):
        """
        Will allow edge to be a map/set key.
        Return
        ------
        hash((self._origin, self._destination)) : int
            a hashcode that is then used to insert objects into hashtables aka dictionaries"""
        return hash((self._origin, self._destination))

    def __str__(self):
        """
        The meaningful string representation of an object of Relation class, containg the following information: origin synset of this relation and destination synset(hypernym) of this relation.
        Return 
        ------
        repr : string
            String representation of riginin synset and destination seynset of this relation.
        """
        repr = ''
        repr = 'RELATION {}\nORIGIN {}DESTINATION(HYPERNYM) {}'
        id2id = '({}, {})'
        id2id = id2id.format(self.origin.id, self.destination.id)
        repr = repr.format(id2id, str(self.origin), str(self.destination))
        return repr


class Lemma:
    """The Lemma class stores lemma."""

    def __init__(self, lemma):
        """
        The constructor of the Lemma class
        Parameters
        ----------
        lemma : string
            a lemma in WordNet
        """
        self._lemma = lemma

    @property
    def lemma(self):
        """
        lemma getter: return the lemma of this lemma
        Return
        ------
        self._lemma : string
            the lemma of this lemma
        """
        return self._lemma

    def __str__(self):
        """
        __str__ function of lemma class, to make sure meaningful representations when displayed via the print() method
        Returns
        -------
        self._lemma : string
            the string lemma stored in this lemma
        """
        return self._lemma

    def __hash__(self):
        return hash(self._lemma)

    def __eq__(self, othr):
        if isinstance(othr, type(self)):
            return (self._lemma == othr._lemma)
        return NotImplemented


class Path:
    """Path class represents paths from one synset(vertex) to another in wordnet(graph)."""

    def __init__(self, relations):
        """
        The constructor of path class, creating a path based on a list of Relation objects.
        Parameter
        ---------
        realtions : list
            A list of Relation objects which the path is based on.
        """
        edges = []
        verts = []

        for relation in relations:
            edges.append(relation)
            origin = relation.origin
            destination = relation.destination
            # no duplicate vertices in verts
            if origin not in verts:
                verts.append(origin)
            if destination not in verts:
                verts.append(destination)

        self._relations = relations
        self._edges = edges
        self._vertices = verts

    @property
    def edges(self):
        """edges getter: Returns the list of edges on this path, which is an object of Path."""
        return self._edges

    @property
    def vertices(self):
        """vertices getter: Returns the list of vertices on this path, which is an object of Path """
        return self._vertices

    def __len__(self):
        """The length of a path is represented as the number of edges on this path."""
        return len(self._edges)

    def __str__(self):
        """
        The meaningful string representation of an object of Path class, which contains the following information: ids of synsets on this path in one line plus all synsets in order line by line(each synset per line).
        Return
        ------
        repr : string
            The meaningfule string representation of a path, represented as a sequence of synsets.
        """
        repr = ''
        mark = 'PATH AS VERTICES '
        # first line of ids
        id_seq = ''
        # lines of synsets on this path
        synset_seq = ''
        for synset in self.vertices:
            id_seq += synset.id + ', '
            synset_seq += str(synset)
        id_seq = id_seq[:-2]
        repr = mark + id_seq + '\n' + synset_seq
        return repr


#"""main method used to visualize print result of __str__ of all classes"""
#"""
def main():
    wn = WordNet("data/synsets.txt", "data/hypernyms.txt")   

    # test print wordnet
    print(wn)

    # test print synset: 38045,eon aeon,the longest division of geological time
    origin = wn._verticesDict['38045']
    print(origin)

    # test print relation: 38045,43088
    destination = wn._verticesDict['43088']
    relation = Relation(origin, destination)
    print(relation)

    # test print lemma: eon
    lemma = Lemma('eon')
    print(lemma)

    # test print path
    dog_synsets = wn.get_synsets("dog")
    domestic_dog = next(
        syn for syn in dog_synsets if "domestic_dog" in syn.name)
    paths = wn.paths_to_root(domestic_dog)
    print(paths[0])
    

if __name__ == "__main__":
    main()


#"""
