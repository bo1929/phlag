import sys
import argparse
import copy
import pathlib

import jax
import jax.numpy as jnp
import jax.random as jr
import treeswift as ts
import dendropy

from dynamax.hidden_markov_model import BernoulliHMM, CategoricalHMM, GaussianHMM
from scipy.stats import entropy, differential_entropy
from skbio.stats.composition import ilr, multi_replace

import utils

from qqs import QQS, MSC
from utils import timeit


EPS = 1e-5
DELTA = 1e3
BRANCH_LENGTH_LAMBDA = 0.5
MIN_BRANCH_LENGTH = 1e-6


class BCB:
    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.bins, self.bin_centers = self.create_bins()

    def create_bins(self):
        bins = []
        bin_centers = []
        for i in range(self.n_bins):
            for j in range(self.n_bins - i):
                x1, y1, z1 = (
                    i / self.n_bins,
                    j / self.n_bins,
                    (self.n_bins - i - j) / self.n_bins,
                )
                x2, y2, z2 = (
                    (i + 1) / self.n_bins,
                    j / self.n_bins,
                    (self.n_bins - i - 1 - j) / self.n_bins,
                )
                x3, y3, z3 = (
                    i / self.n_bins,
                    (j + 1) / self.n_bins,
                    (self.n_bins - i - j - 1) / self.n_bins,
                )

                bins.append([(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)])
                bin_centers.append(
                    ((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3)
                )

                if i + j < self.n_bins - 1:
                    x1, y1, z1 = (
                        (i + 1) / self.n_bins,
                        j / self.n_bins,
                        (self.n_bins - i - 1 - j) / self.n_bins,
                    )
                    x2, y2, z2 = (
                        i / self.n_bins,
                        (j + 1) / self.n_bins,
                        (self.n_bins - i - j - 1) / self.n_bins,
                    )
                    x3, y3, z3 = (
                        (i + 1) / self.n_bins,
                        (j + 1) / self.n_bins,
                        (self.n_bins - i - j - 2) / self.n_bins,
                    )

                    bins.append([(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)])
                    bin_centers.append(
                        ((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3)
                    )
        return bins, jnp.array(bin_centers)

    def assign_points(self, points):
        d = lambda x, y: jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1))
        return jnp.argmin(jax.vmap(d, (None, 0), 1)(points, self.bin_centers), axis=1)


class ADHMM:
    @timeit
    def __init__(self, args):
        self.args = args
        self.output_file = self.args.output_file

        # Read the species tree and the gene trees
        self.species_tree = utils.label_tree(
            ts.read_tree_newick(self.args.species_tree)
        )
        self.label_to_node = self.species_tree.label_to_node(selection="all")
        with open(self.args.gene_trees) as f:
            self.gene_trees = [ts.read_tree_newick(line.strip()) for line in f]
        self.gc = len(self.gene_trees)
        self.mask = jnp.ones(self.gc, dtype=bool)

        self.output_str = f"# {' '.join(sys.argv)}"
        self.output_str += "\n# " + self.species_tree.newick()
        self.perm_to_int = {
            (0, 1, 2): 0,
            (0, 2, 1): 1,
            (1, 0, 2): 2,
            (1, 2, 0): 3,
            (2, 0, 1): 4,
            (2, 1, 0): 5,
        }
        self.is_valid = (
            lambda node: node is not None
            and not node.is_leaf()
            and node.get_label() is not None
        )

    @timeit
    def select_branches(self, num_branches):
        partitions = [copy.deepcopy(self.species_tree)]
        branches = []
        while len(partitions) < num_branches:
            lp, ix_lp = partitions[-1], len(partitions)
            for ix, p in enumerate(partitions):
                if p.num_nodes(leaves=True) >= lp.num_nodes(leaves=True):
                    lp = p
                    ix_lp = ix
            partitions.pop(ix_lp)
            for p in utils.partition_tree(lp):
                partitions.append(p)
            # partitions = [subtree for partition in partitions for subtree in utils.partition_tree(partition)]
            partitions = [
                tree
                for tree in partitions
                if tree.num_nodes(leaves=True, internal=False) > 2
            ]
        branches = [self.select_best_branch(tree) for tree in partitions]
        branches = [label for label in branches if label is not None]
        return branches

    @timeit
    def select_best_branch(self, tree):
        nd_l, pscore_l = [], []
        for nd, _ in tree.distances_from_parent(internal=True, leaves=False):
            if (
                (not (self.label_to_node.get(nd.get_label(), "")))
                or nd.is_root()
                or (nd.get_parent() is None)
            ):
                continue
            s = self.get_branch_enf(nd)
            if utils.is_float(s):
                nd_l.append(nd)
                pscore_l.append(s)
        if not nd_l:
            best_branch = None
        else:
            best_idx = jnp.argmax(jnp.array(pscore_l))
            best_branch = nd_l[best_idx].get_label()
        return best_branch

    @timeit
    def detect_anomalies(self):
        params, props = self.hmm.initialize()
        em_params, log_probs = self.hmm.fit_em(params, props, self.obs, num_iters=500)
        most_likely_states = self.hmm.most_likely_states(em_params, self.obs)
        self.output_str += "\n" + ",".join(
            map(lambda x: str(x), most_likely_states.astype(int).tolist())
        )
        self.output_str += "\n" + ",".join(
            map(lambda x: str(x), self.obs[:, :].sum(axis=1).tolist())
        )
        with open(self.output_file, "w") as f:
            f.write(self.output_str)

    def set_target_branches(self):
        # Select branches across the tree to use for emissions
        if self.args.branches:
            self.branches = []
            for label in self.args.branches:
                if self.args.expand_branches:
                    node = self.label_to_node[label]
                    parent = node.get_parent()
                    if self.is_valid(parent):
                        self.branches.append(parent.get_label())
                    for child in parent.child_nodes():
                        if self.is_valid(child):
                            self.branches.append(child.get_label())
                    for child in node.child_nodes():
                        if self.is_valid(child):
                            self.branches.append(child.get_label())
                else:
                    self.branches.append(label)
        else:
            self.branches = self.select_branches(self.args.num_branches)
        self.num_branches = len(self.branches)


class QQSHMM(ADHMM):
    @timeit
    def __init__(self, args):
        super().__init__(args)
        self.msc = MSC(self.species_tree, self.gene_trees)
        if not self.args.read_qqs_freqs:
            self.label_to_freqs = self.msc.compute_qqs_freqs()
            if self.args.write_qqs_freqs:
                self.write_qqs_freqs(self.args.write_qqs_freqs)
        else:
            self.label_to_freqs = {}
            self.read_qqs_freqs(self.args.read_qqs_freqs)
        self.compute_branch_lengths()
        self.is_valid = (
            lambda node: node is not None
            and not node.is_leaf()
            and node.get_label() is not None
            and node.get_label in self.label_to_freqs.keys()
        )

        self.label_to_support = {}
        supports_arr = jnp.empty(len(self.label_to_freqs))
        for i, (lbl, freqs) in enumerate(self.label_to_freqs.items()):
            self.label_to_support[lbl] = jnp.mean(freqs[:, 0])
            supports_arr = supports_arr.at[i].set(self.label_to_support[lbl])
        self.median_support = jnp.nanmedian(supports_arr)
        self.fquartile_support = jnp.nanquantile(supports_arr, 0.25)
        self.lquartile_support = jnp.nanquantile(supports_arr, 0.75)

    def compute_branch_lengths(self):
        for nd in self.species_tree.traverse_postorder():
            if nd.is_leaf():
                nd.set_edge_length(0)
                continue
            freqs = self.label_to_freqs.get(nd.label, None)
            if freqs is None:
                nd.set_edge_length(MIN_BRANCH_LENGTH)
                continue
            mfreqs = freqs[self.mask, 0]
            mmask = ~jnp.isnan(mfreqs)
            z1 = jnp.sum(mfreqs[mmask])
            n = jnp.sum(self.gc) + BRANCH_LENGTH_LAMBDA * 2
            if z1 >= (n / 3):
                bl = -jnp.log(3 * (1 - z1 / n) / 2)
            else:
                bl = MIN_BRANCH_LENGTH
            final_bl = max(bl, MIN_BRANCH_LENGTH)
            nd.set_edge_length(final_bl)

    def read_qqs_freqs(self, path):
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                values = line.strip().split("\t")
                label = values[0]
                topology = int(values[1]) - 1
                freq_y = jnp.array([float(x) for x in values[2:]])
                qqs_freq = self.label_to_freqs.get(
                    label, jnp.zeros((freq_y.shape[0], 3))
                )
                self.label_to_freqs[label] = qqs_freq.at[:, topology].set(freq_y)

    def write_qqs_freqs(self, path):
        with open(path, "w") as f:
            header = "branch\ty\t" + "\t".join(f"g_{i}" for i in range(1, self.gc + 1))
            f.write(header)
            for label, freqs in self.label_to_freqs.items():
                if freqs.shape[0] == 0:
                    continue
                for topology in range(3):
                    buffer = StringIO()
                    np.savetxt(
                        buffer,
                        freqs[:, topology],
                        delimiter="",
                        newline="\t",
                        fmt="%.4f",
                    )
                    f.write(f"\n{label}\t{topology + 1}\t{buffer.getvalue()[:-1]}")


# Gaussian Quartet Supports HMM
class GQSHMM(QQSHMM):
    def __init__(self, args):
        super().__init__(args)
        self.set_target_branches()

        observed_freqs = []
        for branch in self.branches:
            observed_freqs.append(self.label_to_freqs[branch])
            self.mask = self.mask & ~jnp.isnan(observed_freqs[-1]).any(axis=-1)
        self.observed_freqs = jnp.stack(observed_freqs, axis=1)[self.mask, :, :]

        if self.args.apply_ilr:
            obs = []
            for i in range(self.observed_freqs.shape[1]):
                obs.append(ilr(multi_replace(self.observed_freqs[:, i, :], delta=1e-5)))
            self.obs = jnp.stack(obs, axis=1)
            self.obs = self.obs.reshape((self.obs.shape[0], -1))
        else:
            self.obs = self.observed_freqs[:, :, :2].reshape(
                (self.observed_freqs.shape[0], -1)
            )

        self.hmm = GaussianHMM(
            2,
            self.obs.shape[1],
            initial_probs_concentration=1.1,
            transition_matrix_concentration=1.1,
            transition_matrix_stickiness=0.0,
            emission_prior_concentration=1.1,
        )
        self.detect_anomalies()

    def get_branch_enf(self, nd):
        obs = self.label_to_freqs[nd.get_label()]
        return -(jnp.mean(obs[:, 0]) - self.lquartile_support)


# Barycentric Coordinate Bins HMM
class BCBHMM(QQSHMM):
    def __init__(self, args):
        super().__init__(args)
        self.n_bins = args.n_bins
        self.bcb = BCB(self.n_bins)
        self.set_target_branches()

        observed_freqs = []
        obs = []
        for branch in self.branches:
            observed_freqs.append(self.label_to_freqs[branch])
            obs.append(self.get_categories(observed_freqs[-1]))
            self.mask = self.mask & ~jnp.isnan(observed_freqs[-1]).any(axis=-1)
        self.observed_freqs = jnp.stack(observed_freqs, axis=1)[self.mask, :, :]
        self.obs = jnp.stack(obs, axis=1)[self.mask, :]
        self.num_classes = self.n_bins**2

        self.hmm = CategoricalHMM(
            2,
            self.obs.shape[1],
            initial_probs_concentration=1.1,
            transition_matrix_concentration=1.1,
            transition_matrix_stickiness=0.0,
            emission_prior_concentration=1.1,
            num_classes=self.num_classes,
        )
        self.detect_anomalies()

    def get_categories(self, obs):
        return self.bcb.assign_points(obs)

    def get_branch_enf(self, nd):
        obs = self.label_to_freqs[nd.get_label()]
        # Total variance-to-mean ratio, i.e., index of dipersion
        # s = jnp.sum(jnp.var(obs, axis=0)/jnp.mean(obs, axis=0))
        t, c = jnp.unique(self.get_categories(obs), return_counts=True)
        s = jnp.mean(obs[:, 0])
        return entropy(c / c.sum(), base=2) + ((s >= self.median_support) * DELTA)


# Dominant Topology Ordering HMM
class DTOHMM(QQSHMM):
    def __init__(self, args):
        super().__init__(args)
        self.set_target_branches()

        observed_freqs = []
        obs = []
        for branch in self.branches:
            observed_freqs.append(self.label_to_freqs[branch])
            obs.append(self.get_categories(observed_freqs[-1]))
            self.mask = self.mask & ~jnp.isnan(observed_freqs[-1]).any(axis=-1)
        self.observed_freqs = jnp.stack(observed_freqs, axis=1)[self.mask, :, :]
        self.obs = jnp.stack(obs, axis=1)[self.mask, :]
        self.num_classes = len(self.perm_to_int)

        self.hmm = CategoricalHMM(
            2,
            self.obs.shape[1],
            initial_probs_concentration=1.1,
            transition_matrix_concentration=1.1,
            transition_matrix_stickiness=0.0,
            emission_prior_concentration=1.1,
            num_classes=self.num_classes,
        )
        self.detect_anomalies()

    def get_categories(self, obs):
        return jnp.array(
            list(
                map(
                    lambda x: self.perm_to_int[tuple(x.tolist())],
                    jnp.argsort(obs, axis=-1),
                )
            )
        )

    def get_branch_enf(self, nd):
        obs = self.label_to_freqs[nd.get_label()]
        # Total variance-to-mean ratio, i.e., index of dipersion
        # s = jnp.sum(jnp.var(obs, axis=0)/jnp.mean(obs, axis=0))
        t, c = jnp.unique(self.get_categories(obs), return_counts=True)
        s = jnp.mean(obs[:, 0])
        return entropy(c / c.sum(), base=2) + ((s >= self.median_support) * DELTA)


# Most Likely Topology HMM
class MLTHMM(QQSHMM):
    def __init__(self, args):
        super().__init__(args)
        self.set_target_branches()

        obs = []
        observed_freqs = []
        for branch in self.branches:
            observed_freqs.append(self.label_to_freqs[branch])
            obs.append(self.get_categories(observed_freqs[-1]))
            self.mask = self.mask & ~jnp.isnan(observed_freqs[-1]).any(axis=-1)
        self.observed_freqs = jnp.stack(observed_freqs, axis=1)[self.mask, :, :]
        self.obs = jnp.stack(obs, axis=1)[self.mask, :]
        self.num_classes = self.observed_freqs.shape[-1]

        self.hmm = CategoricalHMM(
            2,
            self.obs.shape[1],
            initial_probs_concentration=1.1,
            transition_matrix_concentration=1.1,
            transition_matrix_stickiness=0.0,
            emission_prior_concentration=1.1,
            num_classes=self.num_classes,
        )
        self.detect_anomalies()

    def get_categories(self, obs):
        return jnp.argmax(obs, axis=-1)

    def get_branch_enf(self, nd):
        obs = self.label_to_freqs[nd.get_label()]
        # Total variance-to-mean ratio, i.e., index of dipersion
        # s = jnp.sum(jnp.var(obs, axis=0)/jnp.mean(obs, axis=0))
        t, c = jnp.unique(self.get_categories(obs), return_counts=True)
        s = jnp.mean(obs[:, 0])
        return entropy(c / c.sum(), base=2) + ((s >= self.median_support) * DELTA)


# Present Bipartition HMM
class PBPHMM(ADHMM):
    def __init__(self, args):
        super().__init__(args)
        self.compute_bipartitions()
        self.set_target_branches()

        obs = []
        for branch in self.branches:
            obs.append(self.get_pbp(branch))
            self.mask = self.mask & ~jnp.isnan(obs[-1]).any(axis=-1)
        self.obs = jnp.stack(obs, axis=1)[self.mask, :]
        self.num_classes = 2

        self.hmm = BernoulliHMM(
            2,
            self.obs.shape[1],
            initial_probs_concentration=1.1,
            transition_matrix_concentration=1.1,
            transition_matrix_stickiness=0.0,
            emission_prior_concentration0=1.1,
            emission_prior_concentration1=1.1,
        )
        self.detect_anomalies()

    def compute_bipartitions(self):
        self.tns = dendropy.TaxonNamespace()
        self.st = dendropy.Tree.get(
            path=self.args.species_tree,
            schema="newick",
            preserve_underscores=True,
            taxon_namespace=self.tns,
        )
        self.st.encode_bipartitions()
        with open(self.args.gene_trees) as f:
            self.gt = [
                dendropy.Tree.get(
                    data=line.strip(),
                    schema="newick",
                    preserve_underscores=True,
                    taxon_namespace=self.tns,
                )
                for line in f
            ]

    def get_pbp(self, branch):
        edge = self.st.find_node_with_label(branch).edge
        obs = []
        for ix in range(self.gc):
            obs.append(self.gt[ix].is_compatible_with_bipartition(edge.bipartition))
        return jnp.array(obs)

    def get_branch_enf(self, nd):
        obs = self.get_pbp(nd.get_label())
        p = jnp.sum(obs) / obs.shape[0]
        s = entropy([p, 1 - p], base=2)
        return s


def add_shared_arguments(parser):
    parser.add_argument(
        "-s",
        "--species-tree",
        type=pathlib.Path,
        required=True,
        help="Path to species tree in Newick format",
    )
    parser.add_argument(
        "-g",
        "--gene-trees",
        type=pathlib.Path,
        required=True,
        help="Path to file for ordered gene trees (one Newick tree per line)",
    )
    parser.add_argument(
        "-b",
        "--branches",
        nargs="*",
        type=str,
        help="Specific branch labels to analyze (auto-selected if not provided)",
    )
    parser.add_argument(
        "--num-branches",
        type=int,
        default=5,
        help="Minimum number of branches to automatically select",
    )
    parser.add_argument(
        "--expand-branches",
        action="store_true",
        help="Incorporate the neighboring branches.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=pathlib.Path,
        required=True,
        help="Path to output file",
    )
    io_group = parser.add_argument_group("I/O options")
    io_group.add_argument(
        "--write-qqs-freqs",
        type=pathlib.Path,
        help="Write quartet frequencies to the given filepath",
    )
    io_group.add_argument(
        "--read-qqs-freqs",
        type=pathlib.Path,
        help="Read quartet frequencies from the given filepath",
    )
    return parser


def parse_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        description="Anomaly detection using basic HMMs and tree statistics.",
    )
    pbphmm_parser = subparsers.add_parser("pbp-hmm", help="Consistent Bipartition HMM")
    cbphmm_parser = subparsers.add_parser("cbp-hmm", help="Consistent Bipartition HMM")
    mlthmm_parser = subparsers.add_parser("mlt-hmm", help="Most Likely Topology HMM")
    dtohmm_parser = subparsers.add_parser(
        "dto-hmm", help="Dominant Topology Ordering HMM"
    )
    gqshmm_parser = subparsers.add_parser("gqs-hmm", help="Gaussian Quartet Scores HMM")
    bcbhmm_parser = subparsers.add_parser(
        "bcb-hmm", help="Barycentric Coordinate Bins HMM"
    )
    pbphmm_parser = add_shared_arguments(pbphmm_parser)
    cbphmm_parser = add_shared_arguments(cbphmm_parser)
    mlthmm_parser = add_shared_arguments(mlthmm_parser)
    dtohmm_parser = add_shared_arguments(dtohmm_parser)
    gqshmm_parser = add_shared_arguments(gqshmm_parser)
    bcbhmm_parser = add_shared_arguments(bcbhmm_parser)
    gqshmm_parser.add_argument(
        "--apply-ilr",
        action="store_true",
        help="Apply isometric log-ratio transformation on QQS values",
    )
    bcbhmm_parser.add_argument(
        "--n-bins", type=int, default=6, help="The number of bins at each axis"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.command == "pbp-hmm":
        PBPHMM(args)
    elif args.command == "mlt-hmm":
        MLTHMM(args)
    elif args.command == "dto-hmm":
        DTOHMM(args)
    elif args.command == "bcb-hmm":
        BCBHMM(args)
    elif args.command == "gqs-hmm":
        GQSHMM(args)
    else:
        raise ValueError("The given method is not recognized!")


if __name__ == "__main__":
    print(" ".join(sys.argv))
    main()
