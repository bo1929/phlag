# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import time
from collections import defaultdict
from functools import partial

import dendropy
import jax.numpy as jnp
import treeswift as ts
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cpython.dict cimport PyDict_Contains

cimport cython

cdef class QQS:
    cdef object species_tree, lbl_to_nd

    def __init__(self, species_tree):
        self.species_tree = species_tree
        self.lbl_to_nd = self.species_tree.label_to_node(selection="all")

    cdef double f(self, w, x, y, z, gene_tree):
        cdef int c11, c12, c13, c14, c21, c22, c23, c24, c31, c32, c33, c34
        cdef int iw, ix, iy, iz
        cdef object label
        cdef int w_size = len(w)
        cdef int x_size = len(x)
        cdef int y_size = len(y)
        cdef int z_size = len(z)
        cdef double r = 0.0, q
        cdef list s = []
        for u in gene_tree.traverse_postorder():
            if u.is_leaf():
                label = u.label
                iw = PyDict_Contains(w, label)
                ix = PyDict_Contains(x, label)
                iy = PyDict_Contains(y, label)
                iz = PyDict_Contains(z, label)
            else:
                c11, c12, c13, c14 = s.pop()
                c21, c22, c23, c24 = s.pop()
                iw, ix, iy, iz = (c11 + c21, c12 + c22, c13 + c23, c14 + c24)
                c31, c32, c33, c34 = w_size - iw, x_size - ix, y_size - iy, z_size - iz
                q = (
                    c11 * c22 * c33 * c34 +
                    c12 * c21 * c33 * c34 +
                    c13 * c24 * c31 * c32 +
                    c14 * c23 * c31 * c32 +
                    c31 * c22 * c13 * c14 +
                    c32 * c21 * c13 * c14 +
                    c33 * c24 * c11 * c12 +
                    c34 * c23 * c11 * c12 +
                    c11 * c32 * c23 * c24 +
                    c12 * c31 * c23 * c24 +
                    c13 * c34 * c21 * c22 +
                    c14 * c33 * c21 * c22
                )
                r += q
            s.append((iw, ix, iy, iz))
        return r / 2.0

    def get_qp(self, label):
        nd = self.lbl_to_nd[label]
        assert not nd.is_root() and not nd.is_leaf()
        assert(len(list(nd.child_nodes())) == 2)
        cdef object x, y, w, z
        x, y = (self.species_tree.extract_subtree(child).traverse_postorder(internal=False) for child in nd.child_nodes())
        w = self.species_tree.extract_subtree(nd.parent.child_nodes()[1 if nd.parent.child_nodes()[0] == nd else 0]).traverse_postorder(internal=False)
        # if nd.parent.parent is not None:
        #     w = self.species_tree.extract_subtree(nd.parent.child_nodes()[1 if nd.parent.child_nodes()[0] == nd else 0]).traverse_postorder(internal=False)
        # else:
        #     sister = nd.parent.child_nodes()[1 if nd.parent.child_nodes()[0] == nd else 0]
        #     w = self.species_tree.extract_subtree(sister.child_nodes()[0]).traverse_postorder(internal=False)
        z = self.species_tree.traverse_postorder(internal=False)
        get_labels = lambda s: set(node.label for node in s)
        x, y, w, z = get_labels(x), get_labels(y), get_labels(w), get_labels(z)
        z = z - x - y - w
        make_dict = lambda s: {label : True for label in s}
        # xstr = str(",".join(list(x)))
        # ystr = str(",".join(list(y)))
        # wstr = str(",".join(list(w)))
        # zstr = str(",".join(list(z)))
        # qstr = label + "\t(" + xstr + ")+(" + ystr + ") | (" + wstr + ")+(" + zstr + ")"
        # print(qstr)
        cdef dict w_dict = make_dict(w)
        cdef dict x_dict = make_dict(x)
        cdef dict y_dict = make_dict(y)
        cdef dict z_dict = make_dict(z)
        result = (x_dict, y_dict, w_dict, z_dict)
        return result

    def freq(self, x, y, w, z, gene_tree, normalize=True):
        f1, f2, f3 = (self.f(x, y, w, z, gene_tree),
                      self.f(x, w, y, z, gene_tree),
                      self.f(x, z, w, y, gene_tree))
        if not normalize:
            return (f1, f2, f3)
        else:
            m = f1 + f2 + f3
            return (f1 / m, f2 / m, f3 / m)


cdef class MSC:
    """Computing QQS values under the multispecies coalescent model."""
    cdef object qqs, species_tree, gene_trees
    cdef int num_gene_trees

    def __init__(self, species_tree, gene_trees):
        self.species_tree = species_tree
        self.gene_trees = gene_trees
        self.qqs = QQS(species_tree)
        self.num_gene_trees = len(self.gene_trees)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_qqs_freqs(self):
        cdef object label_to_freqs = defaultdict(partial(jnp.zeros, (self.num_gene_trees, 3)))
        cdef list internal_nodes = [
            nd for nd in self.species_tree.traverse_postorder(leaves=False)
            if not nd.is_root() and not nd.is_leaf()
        ]
        cdef object nd, gene_tree, x, y, w, z
        cdef double f1, f2, f3
        cdef str label
        cdef int i

        for nd in internal_nodes:
            label = nd.label
            x, y, w, z = self.qqs.get_qp(nd.label)
            if min(len(x), len(y), len(w), len(z)) >= 1:
                for i in range(self.num_gene_trees):
                    f1, f2, f3 = self.qqs.freq(x, y, w, z, self.gene_trees[i])
                    label_to_freqs[label] = label_to_freqs[label].at[i, :].set((f1, f2, f3))
        return label_to_freqs

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def simulate_qqs_freqs(self, clade_labels, num_reps):
        cdef object tns = dendropy.TaxonNamespace()
        cdef object dendropy_tree = dendropy.Tree.get(
            data=self.species_tree.newick(), schema="newick", taxon_namespace=tns
        )
        cdef object gene_to_species_map = dendropy.TaxonNamespaceMapping.create_contained_taxon_mapping(
            containing_taxon_namespace=tns, num_contained=1
        )
        cdef object simulated_freqs = jnp.zeros((num_reps, len(clade_labels), 3))
        cdef object simulated_tree
        cdef dict x, y, w, z
        cdef int i, j
        cdef object nd
        cdef str label
        cdef double f1, f2, f3

        for i in range(int(num_reps)):
            simulated_tree = ts.read_tree_newick(
                dendropy.simulate.treesim.contained_coalescent_tree(
                    dendropy_tree, gene_to_species_map
                ).as_string(schema="newick")
            )
            for nd in simulated_tree.traverse_leaves():
                parts = nd.label.split("_")
                nd.label = "_".join(parts[:len(parts)-1])
            j = 0
            for clade in clade_labels:
                x, y, w, z = self.qqs.get_qp(clade)
                assert min(len(x), len(y), len(w), len(z)) >= 1
                f1, f2, f3 = self.qqs.freq(x, y, w, z, simulated_tree)
                simulated_freqs = simulated_freqs.at[i, j, :].set((f1, f2, f3))
                j += 1

        return simulated_freqs
