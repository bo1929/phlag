import time
import argparse

import dendropy

from functools import wraps

# from fastroot.MinVar import MV00_Tree

# def partition_tree(tree):
#     for nd in tree.traverse_postorder(leaves=True, internal=True):
#         nd.set_edge_length(1.0)
#     mv_tree = MV00_Tree(ddpTree=tree)
#     mv_tree.Reroot()
#     root = mv_tree.get_root()
#     subtrees = [mv_tree.ddpTree.extract_subtree(nd) for nd in root.child_nodes()]
#     return subtrees


def get_canonical_taxon_namespace(path, schema="newick"):
    tree = dendropy.Tree.get(path=path, schema=schema, preserve_underscores=True)
    return dendropy.TaxonNamespace(sorted(tax.label for tax in tree.taxon_namespace))


def leaf_set(nd_0, nd_1):
    leaves = set()
    stack = [(nd_0, nd_1)]

    while stack:
        nd_j, parent = stack.pop()

        if nd_j.is_leaf():
            leaves.add(nd_j.taxon.label)
            continue

        for nd_i in nd_j.adjacent_nodes():
            if nd_i is not parent:
                stack.append((nd_i, nd_j))
    return leaves


def map_label_to_node(tree):
    lbl_to_nd = {}
    for nd in tree.preorder_node_iter():
        if nd.is_leaf():
            if nd.taxon is not None:
                lbl_to_nd[nd.taxon.label] = nd
        else:
            if nd.label is not None:
                lbl_to_nd[nd.label] = nd
    return lbl_to_nd


def is_terminal_edge(edge):
    return edge.tail_node.is_leaf() or edge.head_node.is_leaf()


def focal_edge_from_label(tree, lbl, lbl_to_nd=None):
    if lbl_to_nd is None:
        lbl_to_nd = map_label_to_node(tree)
    nd = lbl_to_nd.get(lbl, None)
    if nd is None:
        raise ValueError(f"The species tree does not have a node with label {lbl}")

    # Rooted case: return parent edge
    if tree.is_rooted:
        if nd.parent_node is None:
            return None  # nd is the root
        return nd.edge

    # Unrooted case: balanced internal edge
    n_taxa = len(tree.taxon_namespace)
    edge = None
    s = -1
    for edge_i in nd.incident_edges():
        if (edge_i.bipartition is None) or (is_terminal_edge(edge_i)):
            continue
        s_i = bipartition_balance(edge_i, n_taxa)
        if s_i > s:
            s = s_i
            edge = edge_i
    return edge


def get_incident_edges(tree, edge, lbl_to_nd=None, filter_terminal=True):
    head_node = edge.head_node
    tail_node = edge.tail_node
    incident_edges = set()
    if filter_terminal:
        if head_node != tree.seed_node:
            incident_edges.update(
                edge
                for edge in head_node.incident_edges()
                if not is_terminal_edge(edge)
            )
        if tail_node != tree.seed_node:
            incident_edges.update(
                edge
                for edge in tail_node.incident_edges()
                if not is_terminal_edge(edge)
            )
    else:
        if head_node != tree.seed_node:
            incident_edges.update(head_node.incident_edges())
        if tail_node != tree.seed_node:
            incident_edges.update(tail_node.incident_edges())
    return incident_edges


def bipartition_balance(edge, n_taxa):
    """
    min(|A|, |B|)
    """
    bp = edge.bipartition
    k = bin(bp.leafset_bitmask).count("1")
    return min(k, n_taxa - k)


def canonical_quadripartition(Q):
    """
    a canonical (order-independent) of a quadripartition.
    """
    return tuple(sorted((frozenset(p) for p in Q), key=lambda x: tuple(sorted(x))))


def count_lines(filepath):
    try:
        with open(filepath, "r") as f:
            line_count = sum(1 for line in f)
        return line_count
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e


def integer_pair(arg_str):
    try:
        parts = arg_str.split(",")
        if len(parts) != 2:
            raise ValueError(
                "Argument must contain exactly two integers separated by a comma."
            )
        int1 = int(parts[0].strip())
        int2 = int(parts[1].strip())
        return [int1, int2]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid integer pair format: {arg_str}. {e}")


def is_float(val):
    try:
        return float(val) == float(val)
    except (ValueError, TypeError):
        return False


def limited_float(min_val, max_val):
    def check_range(value):
        fvalue = float(value)
        if not (min_val <= fvalue <= max_val):
            raise ValueError(
                f"Argument must be between {min_val} and {max_val} (inclusive)."
            )
        return fvalue

    return check_range


def limited_int(min_val, max_val):
    def check_range(value):
        fvalue = int(value)
        if not (min_val <= fvalue <= max_val) or not (isinstance(value, int)):
            raise ValueError(
                f"Argument must be an integer between {min_val} and {max_val} (inclusive)."
            )
        return fvalue

    return check_range


def timeit(func):
    """
    A decorator that measures the execution time of a function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds.")
        return result

    return wrapper
