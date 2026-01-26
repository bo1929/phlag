import sys
import argparse
import copy
import pathlib

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import treeswift as ts

from io import StringIO
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from skbio.stats.composition import ilr, multi_replace
import tensorflow_probability.substrates.jax.distributions as tfd

from jaxtyping import Array, Float, Int
from typing import Union

import hmm
import discretizer
import utils

from qqs import QQS, MSC
from utils import timeit

PRNGKeyT = Array
Scalar = Union[float, Float[Array, ""]]
IntScalar = Union[int, Int[Array, ""]]

NUM_STATES = 2
INITIAL_PROBS = jnp.array([1.0000, 0.0000], dtype=jnp.float32)
BLEN_LAMBDA = 0.5
BLEN_MIN = 1e-6

E_STEP_EPS = 0.0001
PSI_EPS = 0.001


# TODO: Disc and args!!!


class Phlag:
    @ignore_warnings(category=ConvergenceWarning)
    def __init__(self, args):
        self.args = args
        self.output_file = self.args.output_file

        self.read_trees()
        self.compute_qqs()

        self.num_classes_min = max(self.args.num_classes_min, 2)
        self.num_classes_max = min(self.args.num_classes_max, self.n_gt // 2)
        self.ilr_transform = self.args.ilr_transform

        self.set_target_clades()

        # Initialize branch lengths
        self.update_branch_lengths(jnp.ones(self.n_gt))

        self.extract_observed_qqs()

        self.output_str = f"# {' '.join(sys.argv)}"
        self.output_str += "\n# " + self.st.newick()

        # Initialize the discretization function and select the best number of classes for categorical emissions
        # self.discretizer = discretizer.KMeansDiscretization(
        #     self.observed_qqs.shape[1],
        #     self.observed_qqs.shape[2],
        #     self.num_classes_max,
        # )
        # self.discretizer.choose_num_classes(
        #     self.observed_qqs,
        #     range_classes=(self.num_classes_min, self.num_classes_max),
        # )
        self.discretizer = discretizer.DTO()
        self.num_classes = self.discretizer.get_num_classes()
        # self.discretizer.fit_discretization(self.observed_qqs)
        self.observed_emissions = self.discretizer.discretize_qqs(self.observed_qqs)

        # Obtain the initial simulated emissions from the species tree
        self.num_reps = self.args.num_reps
        simulated_qqs = self.msc.simulate_qqs(self.clade_labels, self.num_reps)
        self.simulated_emission_prob = self.discretizer.compute_emission_prob(
            self.transform_qqs(simulated_qqs)
        )

        self.initialize_hmm()

    def read_trees(self):
        self.st = utils.label_tree(ts.read_tree_newick(self.args.species_tree))
        # self.st.suppress_unifurcations()
        self.lbl_to_nd = self.st.label_to_node(selection="all")
        with open(self.args.gene_trees) as f:
            self.gt_l = [ts.read_tree_newick(line.strip()) for line in f]
        self.n_gt = len(self.gt_l)
        self.mask = jnp.ones(self.n_gt, dtype=bool)

    def compute_qqs(self):
        self.msc = MSC(self.st, self.gt_l)
        # TODO: Compute only for the target clades
        if not self.args.read_qqs_qqs:
            self.lbl_to_qqs = self.msc.compute_qqs()
            if self.args.write_qqs:
                self.write_qqs(self.args.write_qqs)
        else:
            self.lbl_to_qqs = {}
            self.read_qqs(self.args.read_qqs_qqs)

    def set_target_clades(self):
        self.is_valid = (
            lambda nd: nd is not None
            and not nd.is_leaf()
            and nd.get_label() is not None
            and nd.get_label() in self.lbl_to_qqs.keys()
        )
        if self.args.clade_labels:
            self.clade_labels = []
            for label in self.args.clade_labels:
                if self.args.expand_branches:
                    nd = self.lbl_to_nd[label]
                    parent = nd.get_parent()
                    if self.is_valid(parent):
                        self.clade_labels.append(parent.get_label())
                    for child in parent.child_nodes():
                        if self.is_valid(child):
                            self.clade_labels.append(child.get_label())
                    for child in nd.child_nodes():
                        if self.is_valid(child):
                            self.clade_labels.append(child.get_label())
                else:
                    self.clade_labels.append(label)
        else:
            self.clade_labels = self.select_clades(self.args.num_clades)
        self.num_clades = len(self.clade_labels)

    def select_clades(self, num_clades):
        partitions = [copy.deepcopy(self.st)]
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

    def extract_observed_qqs(self):
        observed_qqs = []
        for clade in self.clade_labels:
            observed_qqs.append(self.lbl_to_qqs[clade])
            self.mask = self.mask & ~jnp.isnan(observed_qqs[-1]).any(axis=1)
        self.observed_qqs = self.transform_qqs(
            jnp.stack(observed_qqs, axis=1)[self.mask, :, :]
        )

    def select_best_clade(self, tree):
        nd_l, pscore_l = [], []
        for nd, _ in tree.distances_from_parent(internal=True, leaves=False):
            if (
                (not (self.lbl_to_nd.get(nd.get_label(), "")))
                or nd.is_root()
                or (nd.get_parent() is None)
            ):
                continue
            s = self.get_branch_enf(nd)
            if utils.is_float(s):
                nd_l.append(nd)
                pscore_l.append(s)
        if not nd_l:
            best_clade = None
        else:
            best_idx = jnp.argmax(jnp.array(pscore_l))
            best_clade = nd_l[best_idx].get_label()
        return best_clade

    @ignore_warnings(category=ConvergenceWarning)
    def get_branch_enf(self, nd):
        qqs = self.lbl_to_qqs[nd.get_label()]
        qqs_masked = self.transform_qqs(qqs[~jnp.isnan(qqs[:, :]).any(axis=1), None, :])
        discr_curr = discretizer.KMeansDiscretization(
            qqs_masked.shape[1], qqs_masked.shape[2], self.num_classes_max
        )
        # TODO: Perhaps do this?
        # discr_curr.choose_num_classes(qqs_masked, range_classes=(self.num_classes_min, self.num_classes_max))
        discr_curr.fit_discretization(qqs_masked)
        return entropy(discr_curr.compute_emission_prob(qqs_masked)[0, :], base=2)

    def read_qqs(self, path):
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                values = line.strip().split("\t")
                label = values[0]
                y = int(values[1]) - 1
                freq_y = jnp.array([float(x) for x in values[2:]])
                qqs_freq = self.lbl_to_qqs.get(label, jnp.zeros((freq_y.shape[0], 3)))
                self.lbl_to_qqs[label] = qqs_freq.at[:, y].set(freq_y)

    def write_qqs(self, path):
        with open(path, "w") as f:
            header = "clade\ty\t" + "\t".join(f"g_{i}" for i in range(1, self.n_gt + 1))
            f.write(header)
            for label, qqs in self.lbl_to_qqs.items():
                if qqs.shape[0] == 0:
                    continue
                for y in range(3):
                    buffer = StringIO()
                    np.savetxt(
                        buffer, qqs[:, y], delimiter="", newline="\t", fmt="%.4f"
                    )
                    f.write(f"\n{label}\t{y + 1}\t{buffer.getvalue()[:-1]}")

    def update_branch_lengths(self, ps):
        for nd in self.st.traverse_postorder():
            if nd.is_leaf():
                nd.set_edge_length(0)
                continue
            qqs = self.lbl_to_qqs.get(nd.label, None)
            if qqs is None:
                nd.set_edge_length(BLEN_MIN)
                continue
            mqqs = qqs[self.mask, 0]
            mmask = ~jnp.isnan(mqqs)
            z1 = jnp.sum(mqqs[mmask] * ps[mmask])
            n = jnp.sum(ps[mmask]) + BLEN_LAMBDA * 2  # TODO: Make sure this is valid
            if z1 >= (n / 3):
                bl = -jnp.log(3 * (1 - z1 / n) / 2)
            else:
                bl = BLEN_MIN
            final_bl = max(bl, BLEN_MIN)
            nd.set_edge_length(final_bl)

    def transform_qqs(self, qqs):
        if self.ilr_transform:
            x = []
            for i in range(qqs.shape[1]):
                x.append(ilr(multi_replace(qqs[:, i, :], delta=1e-7)))
            return jnp.stack(x, axis=1)
        else:
            return qqs[:, :, :]

    def initialize_hmm(self):
        # Initialize the HMM, adjusting given hyperparameters based on the sequence length (the number of gene trees)
        self.alpha_0 = (
            self.n_gt * (1 - self.args.expected_anamoly_proportion)
            - self.args.expected_num_anomalies
        )
        self.alpha_1 = (
            self.n_gt * (self.args.expected_anamoly_proportion)
            - self.args.expected_num_anomalies
        )
        self.beta_0 = self.args.expected_num_anomalies
        self.beta_1 = self.args.expected_num_anomalies
        self.delta = self.args.emission_similarity_penalty
        self.gamma = self.args.emission_prior_concentration
        self.nu = self.args.initial_probs_concentration
        self.psi = jnp.ones((NUM_STATES, NUM_STATES))
        self.psi = self.psi.at[0, 0].set(self.alpha_0)
        self.psi = self.psi.at[0, 1].set(self.beta_0)
        self.psi = self.psi.at[-1, -1].set(self.alpha_1)
        self.psi = self.psi.at[-1, 0].set(self.beta_1)
        # self.psi = self.psi.at[0, 0].set(1.1)
        # self.psi = self.psi.at[0, 1].set(1.1)
        # self.psi = self.psi.at[-1, -1].set(1.1)
        # self.psi = self.psi.at[-1, 0].set(1.1)
        C = jnp.array(self.discretizer.get_cost_matrices(self.num_clades))

        self.prior_concentration = self.gamma * jnp.ones(self.num_classes)
        prior = tfd.Dirichlet(self.prior_concentration)
        alt_emission_probs = prior.sample(
            seed=jr.PRNGKey(0), sample_shape=(self.num_clades)
        )
        # alt_emission_probs = self.simulated_emission_prob
        self.hmm = hmm.PhlagHMM(
            NUM_STATES,
            self.num_clades,
            self.num_classes,
            emission_similarity_penalty=self.delta,
            emission_prior_concentration=self.gamma,
            emission_transfer_cost=C,
            initial_probs_concentration=self.nu,
            transition_matrix_concentration=self.psi + PSI_EPS,
        )
        self.params, self.props = self.hmm.initialize(
            initial_probs=INITIAL_PROBS,
            emission_probs=jnp.stack(
                [self.simulated_emission_prob, alt_emission_probs], axis=0
            ),
        )
        self.props.emissions.probs.trainable = True
        self.props.transitions.transition_matrix.trainable = True
        self.props.initial.probs.trainable = True
        self.hmm.initialize_m_step_state(
            self.params, self.props, emissions_m_step_state=self.simulated_emission_prob
        )
        self.num_iters = self.args.num_iters

    def propose_simulated_emissions(self):
        ix = lambda x, y: x[y]
        # ps = self.hmm.get_posterior().smoothed_probs[:, 0] + E_STEP_EPS
        ps = (
            self.hmm.smoother(self.params, self.observed_emissions).smoothed_probs[:, 0]
            + E_STEP_EPS
        )
        self.update_branch_lengths(ps)
        simulated_qqs = self.msc.simulate_qqs(self.clade_labels, self.num_reps)
        simulated_emission_prob = self.discretizer.compute_emission_prob(
            self.transform_qqs(simulated_qqs)
        )

        pt_prev = jax.vmap(ix, (0, 1), 0)(
            self.simulated_emission_prob, self.observed_emissions
        )
        pt_next = jax.vmap(ix, (0, 1), 0)(
            simulated_emission_prob, self.observed_emissions
        )

        s_prev = jnp.sum(jnp.log(pt_prev) * ps)
        s_next = jnp.sum(jnp.log(pt_next) * ps)
        if s_next > s_prev:
            self.simulated_emission_prob = simulated_emission_prob

    def save_output(self):
        branch_lengths = ", ".join(
            [
                label + ": " + str(self.lbl_to_nd[label].get_edge_length())
                for label in self.clade_labels
            ]
        )
        kldiv_str = ", ".join(
            list(
                map(
                    lambda x: str(x),
                    self.hmm.emission_dissimilarity(self.params).tolist(),
                )
            )
        )
        most_likely_states = np.full(self.mask.shape, -1)
        most_likely_states[self.mask] = self.hmm.most_likely_states(
            self.params, self.observed_emissions
        )
        ps = np.full(self.mask.shape, np.nan)
        ps[self.mask] = self.hmm.smoother(
            self.params, self.observed_emissions
        ).smoothed_probs[:, 1]
        self.output_str += "\n# " + self.st.newick()
        self.output_str += "\n# " + branch_lengths
        self.output_str += "\n# " + kldiv_str
        self.output_str += "\n" + ",".join(
            map(
                lambda x: str(x) if x >= 0 else "nan",
                most_likely_states.astype(int).tolist(),
            )
        )
        self.output_str += "\n" + ",".join(
            map(lambda x: str(x), jnp.round(ps, decimals=3).tolist())
        )
        with open(self.output_file, "w") as f:
            f.write(self.output_str)

    def run(self):
        # self.num_classes = discretizer.choose_num_classes(self.observed_qqs, (self.args.num_classes_min, self.args.num_classes_max+1))
        # bcb = discretizer.BCB(self.num_classes)
        # self.num_classes = self.num_classes ** 2
        # self.observed_emissions = bcb.assign_points(self.observed_qqs)

        # s = jnp.argsort(self.observed_qqs, axis=1)
        # self.observed_emissions = self.observed_emissions * 2 + (s[:, 0] > s[:, 1]).astype(int)
        # self.num_classes *= 2

        # hmm = CategoricalHMM(
        #     2,
        #     self.observed_emissions.shape[1],
        #     initial_probs_concentration=1.1,
        #     transition_matrix_concentration=1.1,
        #     transition_matrix_stickiness=0.0,
        #     emission_prior_concentration=1.1,
        #     num_classes= self.num_classes
        # )
        # params, props = hmm.initialize()
        # em_params, log_probs = hmm.fit_em(params, props, self.observed_emissions, num_iters=500)
        # most_likely_states = hmm.most_likely_states(em_params, self.observed_emissions)

        # self.output_str = f"# {' '.join(sys.argv)}"
        # self.clade_labels = self.args.clade_labels
        # self.output_str += "\n# " + self.st.newick()
        # self.output_str += "\n" + ",".join(
        #     map(lambda x: str(x), most_likely_states.astype(int).tolist())
        # )
        # self.output_str += "\n" + ",".join(
        #     map(lambda x: str(x), self.observed_emissions[:, :].sum(axis=1).tolist())
        # )
        # with open(self.output_file, "w") as f:
        #     f.write(self.output_str)

        for i in tqdm(range(self.num_iters)):
            num_iters = (i + 1) * 25
            self.params, log_probs = self.hmm.fit_em(
                self.params,
                self.props,
                self.observed_emissions,
                num_iters=num_iters,
                verbose=False,
            )
            self.propose_simulated_emissions()
            self.hmm.initialize_m_step_state(
                self.params,
                self.props,
                emissions_m_step_state=self.simulated_emission_prob,
            )
        self.save_output()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Phlag: Phylogenetic Anomaly Detection Across the Genome"
    )

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
        default=4,
        help="Minimum number of clades to automatically select (default: 3)",
    )
    parser.add_argument(
        "--num-reps",
        type=int,
        default=2000,
        help="Number of simulation replicates (default: 2000)",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=5,
        help="Number of (outer) iterations (default: 5)",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=pathlib.Path,
        required=True,
        help="Path to output file",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Verbosity level: 0=quiet, 1=normal, 2=verbose, 3=debug (default: 1)",
    )
    parser.add_argument(
        "--expand-branches",
        action="store_true",
        help="Incorporate the neighboring branches.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read the data and compute/read/write the QQS values, and do nothing else.",
    )

    hmm_group = parser.add_argument_group("HMM parameters")
    hmm_group.add_argument(
        "--expected-anamoly-proportion",
        type=float,
        default=0.1,
        help="Hyperparameter to control transition matrix and portion of the anomalies (default 0.1)",
    )
    hmm_group.add_argument(
        "--expected-num-anomalies",
        type=int,
        default=4,
        help="Hyperparameter to control transition matrix and contiguity of the anomalies (default 10)",
    )
    hmm_group.add_argument(
        "--emission-similarity-penalty",
        type=float,
        default=1,
        help="Hyperparameter to control deviation of anomalies from MSC (default: 0.01)",
    )
    hmm_group.add_argument(
        "--initial-probs-concentration",
        type=float,
        default=1.1,
        help="Initial probabilities concentration (default: 1.1)",
    )
    hmm_group.add_argument(
        "--emission-prior-concentration",
        type=float,
        default=1.1,
        help="Emission prior concentration (default: 1.1)",
    )

    discr_group = parser.add_argument_group("Discretization and emission parameters")
    discr_group.add_argument(
        "--num-classes-min",
        type=int,
        default=8,
        help="Minimum number of classes (default: 8)",
    )
    discr_group.add_argument(
        "--num-classes-max",
        type=int,
        default=64,
        help="Maximum number of classes (default: 64)",
    )
    discr_group.add_argument(
        "--ilr-transform",
        action="store_true",
        help="Apply isometric log-ratio transformation on QQS values",
    )

    io_group = parser.add_argument_group("I/O options")
    io_group.add_argument(
        "--write-qqs-qqs",
        type=pathlib.Path,
        help="Write quartet frequencies to the given filepath",
    )
    io_group.add_argument(
        "--read-qqs-qqs",
        type=pathlib.Path,
        help="Read quartet frequencies from the given filepath",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    phlag = Phlag(args)
    if not (args.dry_run):
        phlag.run()
    else:
        sys.exit("Exiting: --dry-run is given.")


if __name__ == "__main__":
    main()
