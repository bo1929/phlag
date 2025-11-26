import sys
import argparse
import copy
import pathlib

import jax
import jax.numpy as jnp
import jax.random as jr
import treeswift as ts

from dynamax.hidden_markov_model import BernoulliHMM, CategoricalHMM, GaussianHMM
from scipy.stats import entropy, differential_entropy
from skbio.stats.composition import ilr, multi_replace

import utils

from qqs import QQS, MSC


class ADHMM:
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
        self.clade_labels = self.args.clade_labels
        self.output_str += "\n# " + self.species_tree.newick()
        self.perm_to_int = {
            (0, 1, 2): 0,
            (0, 2, 1): 1,
            (1, 0, 2): 2,
            (1, 2, 0): 3,
            (2, 0, 1): 4,
            (2, 1, 0): 5,
        }

    def select_clades(self, num_clades):
        partitions = [copy.deepcopy(self.species_tree)]
        selected_clades = []
        while len(partitions) < num_clades:
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
        selected_clades = [self.select_best_clade(tree) for tree in partitions]
        selected_clades = [label for label in selected_clades if label is not None]
        return selected_clades

    # Entropy based branch selection with three topology labels
    def select_best_clade(self, tree):
        nd_l, entropy_l = [], []
        for nd, _ in tree.distances_from_parent(internal=True, leaves=False):
            if (
                (not (self.label_to_node.get(nd.get_label(), "")))
                or nd.is_root()
                or (nd.get_parent() is None)
            ):
                continue
            s = self.get_branch_pscore(nd)
            if utils.is_float(s):
                nd_l.append(nd)
                entropy_l.append(s)
        if not nd_l:
            best_clade = None
        else:
            best_idx = jnp.argmax(jnp.array(entropy_l))
            best_clade = nd_l[best_idx].get_label()
        return best_clade

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

    def set_selected_clades(self):
        if self.clade_labels:
            self.selected_clades = self.clade_labels
        else:
            self.selected_clades = self.select_clades(self.args.num_clades)


class GQSHMM(ADHMM):
    def __init__(self, args):
        super().__init__(args)
        # Compute all quadripartition quartet scores across branches
        self.msc = MSC(self.species_tree, self.gene_trees)
        # Extract the QQS frequencies (bivariate) and remove the missing data
        self.label_to_freqs = self.msc.compute_qqs_freqs()
        self.set_selected_clades()

        observed_freqs = []
        for clade in self.selected_clades:
            observed_freqs.append(self.label_to_freqs[clade])
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

    def get_branch_pscore(self, nd):
        obs = self.label_to_freqs[nd.get_label()]
        # Entropy based on the QQS of the most dominant category
        s = differential_entropy(obs[:, 0])
        return s


class DTOHMM(ADHMM):
    def __init__(self, args):
        super().__init__(args)
        # Compute all quadripartition quartet scores across branches
        self.msc = MSC(self.species_tree, self.gene_trees)
        # Extract the QQS frequencies (bivariate) and remove the missing data
        self.label_to_freqs = self.msc.compute_qqs_freqs()
        self.set_selected_clades()

        observed_freqs = []
        obs = []
        for clade in self.selected_clades:
            observed_freqs.append(self.label_to_freqs[clade])
            obs.append(
                jnp.array(
                    list(
                        map(
                            lambda x: self.perm_to_int[tuple(x.tolist())],
                            jnp.argsort(observed_freqs[-1], axis=-1),
                        )
                    )
                )
            )
            self.mask = self.mask & ~jnp.isnan(observed_freqs[-1]).any(axis=-1)
        self.observed_freqs = jnp.stack(observed_freqs, axis=1)[self.mask, :, :]
        self.obs = jnp.stack(obs, axis=1)[self.mask, :]
        self.num_classes = len(self.perm_to_int)
        if self.args.dth > 0.33:
            self.obs = (
                self.obs
                + (self.observed_freqs[:, :, 0] > self.args.dth) * self.num_classes
            )
            self.num_classes += self.num_classes

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

    def get_branch_pscore(self, nd):
        obs = self.label_to_freqs[nd.get_label()]
        # Total variance-to-mean ratio, i.e., index of dipersion
        # s = jnp.sum(jnp.var(obs, axis=0)/jnp.mean(obs, axis=0))

        # Entropy based on the QQS of the most dominant category
        s = differential_entropy(obs[:, 0])

        # Entropy of the dominant-topology categories
        # t, c = jnp.unique(jnp.argmax(obs, axis=-1), return_counts=True)
        # p = c / c.sum()
        # s = entropy(p, base=2)
        return s


class MLTHMM(ADHMM):
    def __init__(self, args):
        super().__init__(args)
        # Compute all quadripartition quartet scores across branches
        self.msc = MSC(self.species_tree, self.gene_trees)
        self.label_to_freqs = self.msc.compute_qqs_freqs()
        # Extract the QQS frequencies (bivariate) and remove the missing data
        self.set_selected_clades()

        observed_freqs = []
        for clade in self.selected_clades:
            observed_freqs.append(self.label_to_freqs[clade])
            self.mask = self.mask & ~jnp.isnan(observed_freqs[-1]).any(axis=-1)
        self.observed_freqs = jnp.stack(observed_freqs, axis=1)[self.mask, :, :]

        self.obs = jnp.argmax(self.observed_freqs, axis=-1)
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

    def get_branch_pscore(self, nd):
        obs = self.label_to_freqs[nd.get_label()]
        # Total variance-to-mean ratio, i.e., index of dipersion
        # s = jnp.sum(jnp.var(obs, axis=0)/jnp.mean(obs, axis=0))

        # Entropy based on the QQS of the most dominant category
        s = differential_entropy(obs[:, 0])

        # Entropy of the dominant-topology categories
        # t, c = jnp.unique(jnp.argmax(obs, axis=-1), return_counts=True)
        # p = c / c.sum()
        # s = entropy(p, base=2)
        return s


# Consistent bipartition based anomaly detection
class CBPHMM(ADHMM):
    def __init__(self, args):
        super().__init__(args)
        self.set_selected_clades()

        obs = []
        for clade_labels in self.selected_clades:
            obs.append(self.get_cbp(clade_labels))
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

    def get_cbp(self, clade_label):
        obs = []
        clbl_s = set(
            [nd.label for nd in self.label_to_node[clade_label].traverse_leaves()]
        )
        p_nd = self.label_to_node[clade_label].get_parent()

        for nd in p_nd.child_nodes():
            if nd.label != clade_label:
                tlbl = nd.label

        for ix in range(self.gc):
            glbl_s = [
                nd.label for nd in self.gene_trees[ix].mrca(clbl_s).traverse_leaves()
            ]
            if tlbl in glbl_s:
                obs.append(1)
            else:
                obs.append(0)
        return jnp.array(obs)

    # The binary bipartition consistency
    def get_branch_pscore(self, nd):
        obs = self.get_cbp(nd.get_label())
        p = jnp.sum(obs) / obs.shape[0]
        s = entropy(p, base=2)
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
        "-c",
        "--clade-labels",
        nargs="*",
        type=str,
        help="Specific clade labels to analyze (auto-selected if not provided)",
    )
    parser.add_argument(
        "--num-clades",
        type=int,
        default=5,
        help="Minimum number of clades to automatically select (default: 3)",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=pathlib.Path,
        required=True,
        help="Path to output file",
    )
    return parser


def parse_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        description="Anomaly detection using basic HMMs and tree statistics.",
    )
    cbphmm_parser = subparsers.add_parser("cbp-hmm", help="Consistent Bipartition HMM")
    mlthmm_parser = subparsers.add_parser("mlt-hmm", help="Most Likely Topology HMM")
    dtohmm_parser = subparsers.add_parser(
        "dto-hmm", help="Dominant Topology Ordering HMM"
    )
    gqshmm_parser = subparsers.add_parser("gqs-hmm", help="Gaussian Quartet Scores HMM")
    cbphmm_parser = add_shared_arguments(cbphmm_parser)
    mlthmm_parser = add_shared_arguments(mlthmm_parser)
    dtohmm_parser = add_shared_arguments(dtohmm_parser)
    dtohmm_parser.add_argument(
        "--dth",
        type=float,
        default=0,
        help="An extra treshold for the most dominant topology to create new categories.",
    )
    gqshmm_parser = add_shared_arguments(gqshmm_parser)
    gqshmm_parser.add_argument(
        "--apply-ilr",
        action="store_true",
        help="Apply isometric log-ratio transformation on QQS values",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.command == "cbp-hmm":
        CBPHMM(args)
    elif args.command == "mlt-hmm":
        MLTHMM(args)
    elif args.command == "dto-hmm":
        DTOHMM(args)
    elif args.command == "gqs-hmm":
        GQSHMM(args)
    else:
        raise ValueError("The given method is not recognized!")


if __name__ == "__main__":
    print(" ".join(sys.argv))
    main()
