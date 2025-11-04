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

from jaxtyping import Array, Float, Int
from typing import Union

import hmm
import discr
import utils

from qqs import QQS, MSC
from utils import timeit

PRNGKeyT = Array
Scalar = Union[float, Float[Array, ""]]
IntScalar = Union[int, Int[Array, ""]]

NUM_STATES = 2
INITIAL_PROBS = jnp.array([0.9999, 0.0001], dtype=jnp.float32)
BRANCH_LENGTH_LAMBDA = 0.5
MIN_BRANCH_LENGTH = 1e-3

E_STEP_EPS = 0.001
PSI_EPS = 0.1


# TODO: Disc and args!!!


class Phlag:
    @ignore_warnings(category=ConvergenceWarning)
    def __init__(self, args):
        self.args = args
        self.output_file = self.args.output_file

        # Read the species tree and the gene trees
        self.species_tree = utils.label_tree(ts.read_tree_newick(self.args.species_tree))
        self.label_to_node = self.species_tree.label_to_node(selection="all")
        with open(self.args.gene_trees) as f:
            self.gene_trees = [ts.read_tree_newick(line.strip()) for line in f]
        self.gc = len(self.gene_trees)
        self.mask = jnp.ones(self.gc, dtype=bool)

        self.output_str = f"# {' '.join(sys.argv)}"

        # Compute all quadripartition quartet scores across branches
        self.msc = MSC(self.species_tree, self.gene_trees)
        if not self.args.read_qqs_freqs:
            self.label_to_freqs = self.msc.compute_qqs_freqs()
            if self.args.write_qqs_freqs:
                self.write_qqs_freqs(self.args.write_qqs_freqs)
        else:
            self.label_to_freqs = {}
            self.read_qqs_freqs(self.args.read_qqs_freqs)

        self.num_classes_min = max(self.args.num_classes_min, 2)
        self.num_classes_max = min(self.args.num_classes_max, self.gc // 2)
        self.ilr_transform = self.args.ilr_transform

        # Select clades across the tree to use for emissions
        if self.args.clade_labels:
            self.clade_labels = self.args.clade_labels
        else:
            self.clade_labels = self.select_clades(self.args.num_clades)
        self.num_clades = len(self.clade_labels)

        # Initialize branch lengths
        self.update_branch_lengths(jnp.ones(self.gc))
        self.output_str += "\n# " + self.species_tree.newick()

        # Extract the QQS frequencies (bivariate) and remove the missing data
        observed_freqs = []
        for clade in self.clade_labels:
            observed_freqs.append(self.label_to_freqs[clade])
            self.mask = self.mask & ~jnp.isnan(observed_freqs[-1]).any(axis=1)
        self.observed_freqs = self.transform_freqs(jnp.stack(observed_freqs, axis=1)[self.mask, :, :])

        # Initialize the discretization function and select the best number of classes for categorical emissions
        self.discr = discr.KMeansDiscretization(self.observed_freqs.shape[1], self.observed_freqs.shape[2], self.num_classes_max)
        self.discr.choose_num_classes(self.observed_freqs, range_classes=(self.num_classes_min, self.num_classes_max))
        self.num_classes = self.discr.get_num_classes()
        self.discr.fit_discretization(self.observed_freqs)
        self.observed_emissions = self.discr.discretize_obs(self.observed_freqs)

        # Obtain the initial simulated emissions from the species tree
        self.num_reps = self.args.num_reps
        simulated_freqs = self.msc.simulate_qqs_freqs(self.clade_labels, self.num_reps)
        self.simulated_emission_prob = self.discr.compute_null_emission_prob(self.transform_freqs(simulated_freqs))

        # Initialize the HMM, adjusting given hyperparameters based on the sequence length (the number of gene trees)
        self.alpha_0 = self.gc * (1 - self.args.expected_anamoly_proportion) - self.args.expected_num_anamolies
        self.alpha_1 = self.gc * (self.args.expected_anamoly_proportion) - self.args.expected_num_anamolies
        self.beta_0 = self.args.expected_num_anamolies
        self.beta_1 = self.args.expected_num_anamolies
        self.lambd = self.gc * self.args.emission_simlarity_penalty
        self.gamma = self.args.emission_prior_concentration
        self.nu = self.args.initial_probs_concentration
        self.psi = jnp.ones((NUM_STATES, NUM_STATES))
        self.psi = self.psi.at[0, 0].set(self.alpha_0)
        self.psi = self.psi.at[0, 1].set(self.beta_0)
        self.psi = self.psi.at[-1, -1].set(self.alpha_1)
        self.psi = self.psi.at[-1, 0].set(self.beta_1)
        pwdist = jnp.array(self.discr.pairwise_cluster_distances())
        self.hmm = hmm.PhlagHMM(
            NUM_STATES,
            self.num_clades,
            self.num_classes,
            emission_similarity_penalty=self.lambd,
            emission_prior_concentration=self.gamma,
            emission_transfer_cost=pwdist,
            initial_probs_concentration=self.nu,
            transition_matrix_concentration=self.psi + PSI_EPS,
        )
        self.params, self.props = self.hmm.initialize(initial_probs=INITIAL_PROBS)
        self.props.emissions.probs.trainable = True
        self.props.transitions.transition_matrix.trainable = True
        self.props.initial.probs.trainable = True
        self.hmm.initialize_m_step_state(self.params, self.props, emissions_m_step_state=self.simulated_emission_prob)

        self.num_iters = self.args.num_iters

    @ignore_warnings(category=ConvergenceWarning)
    def compute_discretized_entropy(self, freqs):
        freqs_masked = self.transform_freqs(freqs[~jnp.isnan(freqs[:, :]).any(axis=1), None, :])
        discr_curr = discr.KMeansDiscretization(freqs_masked.shape[1], freqs_masked.shape[2], self.num_classes_max)
        # TODO: Perhaps do this?
        # discr_curr.choose_num_classes(freqs_masked, range_classes=(self.num_classes_min, self.num_classes_max))
        discr_curr.fit_discretization(freqs_masked)
        return entropy(discr_curr.compute_null_emission_prob(freqs_masked)[0, :], base=2)

    # @timeit
    # def select_best_clade(self, tree):
    #     nd_l, entropy_l = [], []
    #     for nd, _ in tree.distances_from_parent(internal=True, leaves=False):
    #         if (nd.label not in self.label_to_freqs) or nd.is_root() or (nd.get_parent() is None):
    #             continue

    #         ent = self.compute_discretized_entropy(self.label_to_freqs[nd.label])
    #         if utils.is_float(ent):
    #             nd_l.append(nd)
    #             entropy_l.append(ent)
    #     if not nd_l:
    #         best_clade = None
    #     else:
    #         best_idx = jnp.argmax(jnp.array(entropy_l))
    #         best_clade = nd_l[best_idx].label
    #     return best_clade

    @timeit
    def select_clades(self, num_clades):
        partitions = [copy.deepcopy(self.species_tree)]
        selected_clades = []
        while len(selected_clades) < num_clades:
            # TODO: Consider doing this one by one, choosing the largest partitioning every time
            partitions = [subtree for partition in partitions for subtree in utils.partition_tree(partition)]
            selected_clades = [self.select_best_clade(tree) for tree in partitions if tree.num_nodes(leaves=True, internal=False) > 1]
            selected_clades = [label for label in selected_clades if label is not None]
        return selected_clades

    def get_bp(self, clade_label):
        obs = []
        clbl_s = set([nd.label for nd in self.label_to_node[clade_label].traverse_leaves()])
        p_nd = self.label_to_node[clade_label].get_parent()

        for nd in p_nd.child_nodes():
            if nd.label != clade_label:
                tlbl = nd.label

        for ix in range(self.gc):
            glbl_s = [nd.label for nd in self.gene_trees[ix].mrca(clbl_s).traverse_leaves()]
            if tlbl in glbl_s:
                obs.append(1)
            else:
                obs.append(0)
        return jnp.array(obs)

    def select_best_clade(self, tree):
        nd_l, entropy_l = [], []
        for nd, _ in tree.distances_from_parent(internal=True, leaves=False):
            if (not (self.label_to_node.get(nd.label, ""))) or nd.is_root() or (nd.get_parent() is None):
                continue
            obs = self.get_bp(nd.label)
            p = sum(obs) / len(obs)
            ent = entropy([p, 1 - p], base=2)
            if utils.is_float(ent):
                nd_l.append(nd)
                entropy_l.append(ent)
        if not nd_l:
            best_clade = None
        else:
            best_idx = jnp.argmax(jnp.array(entropy_l))
            best_clade = nd_l[best_idx].label
        return best_clade

    def read_qqs_freqs(self, path):
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                values = line.strip().split("\t")
                label = values[0]
                y = int(values[1]) - 1
                freq_y = jnp.array([float(x) for x in values[2:]])
                qqs_freq = self.label_to_freqs.get(label, jnp.zeros((freq_y.shape[0], 3)))
                self.label_to_freqs[label] = qqs_freq.at[:, y].set(freq_y)

    def write_qqs_freqs(self, path):
        with open(path, "w") as f:
            header = "clade\ty\t" + "\t".join(f"g_{i}" for i in range(1, self.gc + 1))
            f.write(header)
            for label, freqs in self.label_to_freqs.items():
                if freqs.shape[0] == 0:
                    continue
                for y in range(3):
                    buffer = StringIO()
                    np.savetxt(buffer, freqs[:, y], delimiter="", newline="\t", fmt="%.4f")
                    f.write(f"\n{label}\t{y + 1}\t{buffer.getvalue()[:-1]}")

    @timeit
    def update_branch_lengths(self, ps):
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
            z1 = jnp.sum(mfreqs[mmask] * ps[mmask])
            n = jnp.sum(ps[mmask]) + BRANCH_LENGTH_LAMBDA * 2  # TODO: Make sure this is valid
            if z1 >= (n / 3):
                bl = -jnp.log(3 * (1 - z1 / n) / 2)
            else:
                bl = MIN_BRANCH_LENGTH
            final_bl = max(bl, MIN_BRANCH_LENGTH)
            nd.set_edge_length(final_bl)

    def transform_freqs(self, freqs):
        if self.ilr_transform:
            x = []
            for i in range(freqs.shape[1]):
                x.append(ilr(multi_replace(freqs[:, i, :], delta=1e-5)))
            return jnp.stack(x, axis=1)
        else:
            return freqs[:, :, :2]

    @timeit
    def propose_simulated_emissions(self):
        ix = lambda x, y: x[y]
        # ps = self.hmm.get_posterior().smoothed_probs[:, 0] + E_STEP_EPS
        ps = self.hmm.smoother(self.params, self.observed_emissions).smoothed_probs[:, 0] + E_STEP_EPS
        self.update_branch_lengths(ps)
        simulated_freqs = self.msc.simulate_qqs_freqs(self.clade_labels, self.num_reps)
        simulated_emission_prob = self.discr.compute_null_emission_prob(self.transform_freqs(simulated_freqs))

        pt_prev = jax.vmap(ix, (0, 1), 0)(self.simulated_emission_prob, self.observed_emissions)
        pt_next = jax.vmap(ix, (0, 1), 0)(simulated_emission_prob, self.observed_emissions)

        s_prev = jnp.sum(jnp.log(pt_prev) * ps)
        s_next = jnp.sum(jnp.log(pt_next) * ps)
        if s_next > s_prev:
            self.simulated_emission_prob = simulated_emission_prob

    def save_output(self):
        branch_lengths = ", ".join([label + ": " + str(self.label_to_node[label].get_edge_length()) for label in self.clade_labels])
        kldiv_str = ", ".join(list(map(lambda x: str(x), self.hmm.emission_dissimilarity(self.params).tolist())))
        most_likely_states = self.hmm.most_likely_states(self.params, self.observed_emissions)
        ps = self.hmm.smoother(self.params, self.observed_emissions).smoothed_probs[:, 1]
        self.output_str += "\n# " + self.species_tree.newick()
        self.output_str += "\n# " + branch_lengths
        self.output_str += "\n# " + kldiv_str
        self.output_str += "\n" + ",".join(map(lambda x: str(x), most_likely_states.astype(int).tolist()))
        self.output_str += "\n" + ",".join(map(lambda x: str(x), jnp.round(ps, decimals=3).tolist()))
        with open(self.output_file, "w") as f:
            f.write(self.output_str)

    @timeit
    def run(self):
        for i in tqdm(range(self.num_iters)):
            num_iters = 50  # (i + 1) * 10
            self.params, log_probs = self.hmm.fit_em(self.params, self.props, self.observed_emissions, num_iters=num_iters, verbose=False)
            self.propose_simulated_emissions()
            self.hmm.initialize_m_step_state(self.params, self.props, emissions_m_step_state=self.simulated_emission_prob)
        self.save_output()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Phlag: Phylogenetic Anomaly Detection Across the Genome")

    parser.add_argument("-s", "--species-tree", type=pathlib.Path, required=True, help="Path to species tree in Newick format")
    parser.add_argument("-g", "--gene-trees", type=pathlib.Path, required=True, help="Path to file for ordered gene trees (one Newick tree per line)")
    parser.add_argument("-c", "--clade-labels", nargs="*", type=str, help="Specific clade labels to analyze (auto-selected if not provided)")
    parser.add_argument("--num-clades", type=int, default=3, help="Minimum number of clades to automatically select (default: 3)")
    parser.add_argument("--num-reps", type=int, default=2000, help="Number of simulation replicates (default: 2000)")
    parser.add_argument("--num-iters", type=int, default=10, help="Number of EM iterations (default: 100)")
    parser.add_argument("-o", "--output-file", type=pathlib.Path, required=True, help="Path to output file")
    parser.add_argument("-v", "--verbosity", type=int, default=1, choices=[0, 1, 2, 3], help="Verbosity level: 0=quiet, 1=normal, 2=verbose, 3=debug (default: 1)")

    hmm_group = parser.add_argument_group("HMM parameters")
    hmm_group.add_argument("--expected-anamoly-proportion", type=float, default=0.1, help="Hyperparameter to control transition matrix and portion of the anomalies (default 0.1)")
    hmm_group.add_argument("--expected-num-anamolies", type=float, default=2, help="Hyperparameter to control transition matrix and contiguity of the anomalies (default 10)")
    hmm_group.add_argument("--emission-simlarity-penalty", type=float, default=0.05, help="Hyperparameter to control deviation of anomalies from MSC (default: 0.05)")
    hmm_group.add_argument("--initial-probs-concentration", type=float, default=1.1, help="Initial probabilities concentration (default: 1.1)")
    hmm_group.add_argument("--emission-prior-concentration", type=float, default=1.1, help="Emission prior concentration (default: 1.1)")

    discr_group = parser.add_argument_group("Discretization and emission parameters")
    discr_group.add_argument("--num-classes-min", type=int, default=8, help="Minimum number of classes (default: 8)")
    discr_group.add_argument("--num-classes-max", type=int, default=64, help="Maximum number of classes (default: 64)")
    discr_group.add_argument("--ilr-transform", action="store_true", help="Apply isometric log-ratio transformation on QQS values")

    io_group = parser.add_argument_group("I/O options")
    io_group.add_argument("--write-qqs-freqs", type=pathlib.Path, help="Write quartet frequencies to the given filepath")
    io_group.add_argument("--read-qqs-freqs", type=pathlib.Path, help="Read quartet frequencies from the given filepath")

    return parser.parse_args()


def main():
    args = parse_arguments()
    phlag = Phlag(args)
    phlag.run()


if __name__ == "__main__":
    main()
