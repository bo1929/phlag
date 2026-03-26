import sys
import copy
import pathlib
import argparse

import jax
import numpy as np
import jax.numpy as jnp
import treeswift as ts

# import tensorflow_probability.substrates.jax.distributions as tfd

from io import StringIO
from tqdm import tqdm
from scipy.stats import entropy
from skbio.stats.composition import ilr, multi_replace

from typing import Union
from jaxtyping import Array, Float, Int

import hmm
import utils
import discretizer

from qqs import QQS, MSC

PRNGKeyT = Array
IntScalar = Union[int, Int[Array, ""]]
Scalar = Union[float, Float[Array, ""]]

NUM_STATES = 2
MIN_BRANCH_LENGTH = 1e-6
BRANCH_LENGTH_LAMBDA = 0.5
INITIAL_PROBS = jnp.array([1.0000, 0.0000], dtype=jnp.float32)

PSI_EPS = 0.01
E_STEP_EPS = 0.0001


class Phlag:
    def __init__(self, args):
        self.args = args
        self.output_file = self.args.output_file
        self.read_trees()
        self.compute_qqs()

        # Select or expand given target branches
        self.set_target_branches()

        # Initialize branch lengths
        self.update_branch_lengths(jnp.ones(self.gc))

        # Set the observed emissions
        self.extract_observed_freqs()
        self.set_discretizer()
        self.observed_emissions = self.discretizer.discretize_freqs(
            self.transform_freqs(self.observed_freqs)
        )

        # Obtain the initial simulated emissions from the species tree
        self.num_samples = self.args.num_samples
        simulated_freqs = self.msc.simulate_qqs_freqs(self.branches, self.num_samples)
        self.simulated_emission_prob = self.discretizer.compute_emission_prob(
            self.transform_freqs(simulated_freqs)
        )
        self.initialize_hmm()
        self.init_output()

    def read_trees(self):
        """Read the species tree and the gene trees"""
        self.species_tree = utils.label_tree(
            ts.read_tree_newick(self.args.species_tree)
        )
        self.species_tree.suppress_unifurcations()
        self.label_to_node = self.species_tree.label_to_node(selection="all")
        with open(self.args.gene_trees) as f:
            self.gene_trees = [ts.read_tree_newick(line.strip()) for line in f]
        self.gc = len(self.gene_trees)
        self.mask = jnp.ones(self.gc, dtype=bool)

    def compute_qqs(self):
        """Compute all quadripartition quartet scores across branches"""
        self.msc = MSC(self.species_tree, self.gene_trees)
        # TODO: Compute only for the target branches
        if not self.args.read_qqs_freqs:
            self.label_to_freqs = self.msc.compute_qqs_freqs()
            if self.args.write_qqs_freqs:
                self.write_qqs_freqs(self.args.write_qqs_freqs)
        else:
            self.label_to_freqs = {}
            self.read_qqs_freqs(self.args.read_qqs_freqs)

    def set_discretizer(self):
        if self.args.kmeans is not None:
            self.num_classes_min = max(self.args.num_classes_min, 2)
            self.num_classes_max = min(self.args.num_classes_max, self.gc // 2)
            self.ilr_transform = self.args.ilr_transform

            self.num_classes = KMeansDiscretization.choose_num_classes(
                self.observed_freqs,
                range_classes=(self.num_classes_min, self.num_classes_max),
            )
            self.discretizer = discretizer.KMeansDiscretization(
                self.observed_freqs.shape[1],
                self.observed_freqs.shape[2],
                self.num_classes,
            )
            self.discretizer.fit_discretization(self.ilr_transform(self.observed_freqs))
        elif self.args.bcb:
            self.num_bins_min = max(self.args.num_bins_min, 2)
            self.num_bins_max = min(self.args.num_bins_max, self.gc // 2)

            self.num_bins = BCB.choose_num_classes(
                self.observed_freqs,
                range_classes=(self.num_bins_min, self.num_bins_max),
            )
            self.discretizer = discretizer.BCB(self.num_bins)
            self.num_classes = self.discretizer.get_num_classes()
        else:
            self.discretizer = discretizer.DTO()
            self.num_classes = self.discretizer.get_num_classes()

    def set_target_branches(self):
        # Select branches across the tree to use for emissions
        is_valid = (
            lambda node: node is not None
            and not node.is_leaf()
            and node.get_label() is not None
        )
        if self.args.branches:
            self.branches = []
            for label in self.args.branches:
                if self.args.expand_branches:
                    node = self.label_to_node[label]
                    parent = node.get_parent()
                    if is_valid(parent):
                        self.branches.append(parent.get_label())
                    for child in parent.child_nodes():
                        if is_valid(child):
                            self.branches.append(child.get_label())
                    for child in node.child_nodes():
                        if is_valid(child):
                            self.branches.append(child.get_label())
                else:
                    self.branches.append(label)
        else:
            self.branches = self.select_branches(self.args.num_branches)
        self.num_branches = len(self.branches)

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

    def extract_observed_freqs(self):
        # Extract the QQS frequencies (bivariate) and remove the missing data
        observed_freqs = []
        for label in self.branches:
            observed_freqs.append(self.label_to_freqs[label])
            self.mask = self.mask & ~jnp.isnan(observed_freqs[-1]).any(axis=1)
        self.observed_freqs = self.transform_freqs(
            jnp.stack(observed_freqs, axis=1)[self.mask, :, :]
        )

    def select_best_branch(self, tree):
        nodes, enfs = [], []
        for node, _ in tree.distances_from_parent(internal=True, leaves=False):
            if (
                (not (self.label_to_node.get(node.get_label(), "")))
                or node.is_root()
                or (node.get_parent() is None)
            ):
                continue
            s = self.get_branch_enf(node)
            if utils.is_float(s):
                nodes.append(node)
                enfs.append(s)
        if not nodes:
            best_branch = None
        else:
            best_idx = jnp.argmax(jnp.array(enfs))
            best_branch = nodes[best_idx].get_label()
        return best_branch

    def get_branch_enf(self, node):
        freqs = self.label_to_freqs[node.get_label()]
        freqs_masked = freqs[~jnp.isnan(freqs[:, :]).any(axis=1), None, :]
        return entropy(
            discretizer.DTO().compute_emission_prob(freqs_masked).reshape(-1), base=2
        )

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

    def update_branch_lengths(self, ps):
        for node in self.species_tree.traverse_postorder():
            if node.is_leaf():
                node.set_edge_length(0)
                continue
            freqs = self.label_to_freqs.get(node.label, None)
            if freqs is None:
                node.set_edge_length(MIN_BRANCH_LENGTH)
                continue
            mfreqs = freqs[self.mask, 0]
            mmask = ~jnp.isnan(mfreqs)
            z1 = jnp.sum(mfreqs[mmask] * ps[mmask])
            n = (
                jnp.sum(ps[mmask]) + BRANCH_LENGTH_LAMBDA * 2
            )  # TODO: Make sure this is valid
            if z1 >= (n / 3):
                bl = -jnp.log(3 * (1 - z1 / n) / 2)
            else:
                bl = MIN_BRANCH_LENGTH
            final_bl = max(bl, MIN_BRANCH_LENGTH)
            node.set_edge_length(final_bl)

    def transform_freqs(self, freqs):
        if self.ilr_transform:
            x = []
            for i in range(freqs.shape[1]):
                x.append(ilr(multi_replace(freqs[:, i, :], delta=1e-7)))
            return jnp.stack(x, axis=1)
        else:
            return freqs[:, :, :]

    def initialize_hmm(self):
        # Initialize the HMM, adjusting given hyperparameters based on the sequence length (the number of gene trees)
        self.alpha_0 = (
            self.gc * (1 - self.args.expected_anamoly_proportion)
            - self.args.expected_num_anomalies
        )
        self.alpha_1 = (
            self.gc * (self.args.expected_anamoly_proportion)
            - self.args.expected_num_anomalies
        )
        self.beta_0 = self.args.expected_num_anomalies
        self.beta_1 = self.args.expected_num_anomalies
        self.delta = self.gc * self.args.emission_anomaly_strength
        self.gamma = self.args.emission_prior_concentration
        self.nu = self.args.initial_probs_concentration
        self.psi = jnp.ones((NUM_STATES, NUM_STATES))
        self.psi = self.psi.at[0, 0].set(self.alpha_0)
        self.psi = self.psi.at[0, 1].set(self.beta_0)
        self.psi = self.psi.at[-1, -1].set(self.alpha_1)
        self.psi = self.psi.at[-1, 0].set(self.beta_1)

        # self.prior_concentration = self.gamma * jnp.ones(self.num_classes)
        # prior = tfd.Dirichlet(self.prior_concentration)
        self.emission_cost_matrices = self.discretizer.get_cost_matrices(
            self.num_branches
        )
        alt_emission_probs = self.simulated_emission_prob

        self.hmm = hmm.PhlagHMM(
            NUM_STATES,
            self.num_branches,
            self.num_classes,
            emission_anomaly_strength=self.delta,
            emission_prior_concentration=self.gamma,
            emission_cost_matrices=self.emission_cost_matrices,
            initial_probs_concentration=self.nu,
            transition_matrix_concentration=self.psi + PSI_EPS,
        )
        self.params, self.props = self.hmm.initialize(
            initial_probs=INITIAL_PROBS,
            emission_probs=jnp.stack(
                [self.simulated_emission_prob, alt_emission_probs], axis=0
            ),
        )
        self.props.initial.probs.trainable = True
        self.props.emissions.probs.trainable = True
        self.props.transitions.transition_matrix.trainable = True
        self.hmm.initialize_m_step_state(
            self.params, self.props, emissions_m_step_state=self.simulated_emission_prob
        )
        self.num_iterations = self.args.num_iterations

    def propose_simulated_emissions(self):
        ix = lambda x, y: x[y]
        # ps = self.hmm.get_posterior().smoothed_probs[:, 0] + E_STEP_EPS
        ps = (
            self.hmm.smoother(self.params, self.observed_emissions).smoothed_probs[:, 0]
            + E_STEP_EPS
        )
        self.update_branch_lengths(ps)
        simulated_freqs = self.msc.simulate_qqs_freqs(self.branches, self.num_samples)
        simulated_emission_prob = self.discretizer.compute_emission_prob(
            self.transform_freqs(simulated_freqs)
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

    def init_output(self):
        self.output_str = f"# {' '.join(sys.argv)}"
        self.output_str += "\n# " + self.species_tree.newick()

    def save_output(self):
        branch_lengths = ", ".join(
            [
                label + ": " + str(self.label_to_node[label].get_edge_length())
                for label in self.branches
            ]
        )
        emission_distances_str = ", ".join(
            list(
                map(
                    lambda x: str(x),
                    self.hmm.compute_emission_distance(self.params).tolist(),
                )
            )
        )
        most_likely_states = self.hmm.most_likely_states(
            self.params, self.observed_emissions
        )
        smoother = self.hmm.smoother(self.params, self.observed_emissions)
        ps = smoother.smoothed_probs[:, 1]
        self.output_str += "\n# " + self.species_tree.newick()
        self.output_str += "\n# " + branch_lengths
        self.output_str += "\n# " + emission_distances_str
        self.output_str += "\n" + ",".join(
            map(lambda x: str(x), most_likely_states.astype(int).tolist())
        )
        self.output_str += "\n" + ",".join(
            map(lambda x: str(x), jnp.round(ps, decimals=3).tolist())
        )
        with open(self.output_file, "w") as f:
            f.write(self.output_str)

    def run(self):
        for i in tqdm(range(self.num_iterations)):
            num_iterations = (i + 1) * 10  # Generalized EM iteration regime
            self.params, log_probs = self.hmm.fit_em(
                self.params,
                self.props,
                self.observed_emissions,
                num_iterations=num_iterations,
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
        "-b",
        "--branches",
        nargs="*",
        type=str,
        help="Specific branch label(s) to analyze (auto-selected if not provided)",
    )
    parser.add_argument(
        "--num-branches",
        type=int,
        default=3,
        help="Number of branches to automatically select (default: 3)",
    )
    parser.add_argument(
        "--expand-branches",
        action="store_true",
        help="Incorporate the neighboring branches.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Number of simulation replicates (default: 2000)",
    )
    parser.add_argument(
        "--num-iterations",
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
        "--dry-run",
        action="store_true",
        help="Read the data and compute/read/write the QQS values, and do nothing else.",
    )

    hmm_group = parser.add_argument_group("HMM parameters")
    hmm_group.add_argument(
        "--expected-anomaly-proportion",
        type=float,
        default=0.1,
        help="Hyperparameter to control transition matrix and portion of the anomalies (default 0.1)",
    )
    hmm_group.add_argument(
        "--expected-num-anomalies",
        type=int,
        default=4,
        help="Hyperparameter to control transition matrix and contiguity of the anomalies (default 4)",
    )
    hmm_group.add_argument(
        "--emission-anomaly-strength",
        type=float,
        default=0.01,
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

    discretizer_group = parser.add_argument_group(
        "Discretization options and parameters"
    )
    exclusive_group = discretizer_group.add_mutually_exclusive_group(required=False)
    exclusive_group.add_argument("--dto", help="Dominant topology order (default)")
    exclusive_group.add_argument(
        "--bcb",
        help="Barycentric simplex bins",
        required="--num-bins-min" in sys.argv or "--num-bins-max" in sys.argv,
    )
    exclusive_group.add_argument(
        "--kmeans",
        required="--ilr-transform" in sys.argv
        or "--num-classes-min" in sys.argv
        or "--num-classes-max" in sys.argv,
        help="K-means discretization",
    )

    discretizer_group.add_argument(
        "--num-bins-min",
        type=int,
        default=3,
        help="Minimum number of classes for bins (on one axis) discretization (default: 3)",
        required="--bcb" in sys.argv,
    )
    discretizer_group.add_argument(
        "--num-bins-max",
        type=int,
        default=12,
        help="Maximum number of bins for bins (on one axis) discretization (default: 12)",
        required="--bcb" in sys.argv,
    )
    discretizer_group.add_argument(
        "--num-classes-min",
        type=int,
        default=6,
        help="Minimum number of classes for K-means (default: 6)",
    )
    discretizer_group.add_argument(
        "--num-classes-max",
        type=int,
        default=64,
        help="Maximum number of classes for K-means (default: 64)",
    )
    discretizer_group.add_argument(
        "--ilr-transform",
        action="store_true",
        help="Apply isometric log-ratio transformation on QQS values before K-means",
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
