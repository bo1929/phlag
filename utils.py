import time
from collections import defaultdict
from functools import wraps

from fastroot.MinVar import MV00_Tree


def is_float(val):
    try:
        return float(val) == float(val)
    except (ValueError, TypeError):
        return False


def label_tree(tree):
    is_labeled = True
    i = 0
    labels = set()
    for node in tree.traverse_postorder():
        if node.is_leaf():
            continue
        if not node.label or (node.label in labels) or is_float(node.label):
            is_labeled = False
            node.set_label("I" + str(i))
            i += 1
        labels.add(node.label)
    return tree


def count_lines(filepath):
    try:
        with open(filepath, "r") as f:
            line_count = sum(1 for line in f)
        return line_count
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e


def partition_tree(tree):
    for nd in tree.traverse_postorder(leaves=True, internal=True):
        nd.set_edge_length(1.0)
    mv_tree = MV00_Tree(ddpTree=tree)
    mv_tree.Reroot()
    root = mv_tree.get_root()
    subtrees = [mv_tree.ddpTree.extract_subtree(nd) for nd in root.child_nodes()]
    return subtrees


def limited_float(min_val, max_val):
    def check_range(value):
        fvalue = float(value)
        if not (min_val <= fvalue <= max_val):
            raise ValueError(f"Argument must be between {min_val} and {max_val} (inclusive).")
        return fvalue

    return check_range


def limited_int(min_val, max_val):
    def check_range(value):
        fvalue = int(value)
        if not (min_val <= fvalue <= max_val):
            raise ValueError(f"Argument must be between {min_val} and {max_val} (inclusive).")
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
