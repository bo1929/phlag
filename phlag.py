import sys
import pathlib
import argparse

import jax
import dendropy
import numpy as np
import jax.numpy as jnp
import jax.random as jrand
import tensorflow_probability.substrates.jax.distributions as tfd

from io import StringIO
from tqdm import tqdm
from skbio.stats.composition import ilr, multi_replace

import hmm
import utils
import discretizer

from qqs import MSC

BLEN_LAMBDA = 0.5
BLEN_MIN = 1e-6
E_STEP_EPS = 0.0001
PSI_EPS = 0.001
NUM_STATES = 2
MAX_INCIDENT_LENGTH = 2.0
INITIAL_PROBS = jnp.array([1.0000, 0.0000], dtype=jnp.float32)


class Phlag:
    def __init__(self, args):
        self.args = args

        self.read_trees()
        self.validate_parameters()
        self.determine_focal_edges()
        self.st.deroot()
        self.st.encode_bipartitions()
        self.compute_qqs()
        self.update_edge_lengths(jnp.ones(self.n_gt))
        self.configure_emissions()
        self.compute_emissions()
        self.initialize_hmm()
        self.initialize_output()

    def read_trees(self):
        self.taxa = utils.get_canonical_taxon_namespace(self.args.species_tree)
        taxa_list = list(self.taxa)
        self.st = dendropy.Tree.get(
            path=self.args.species_tree,
            schema="newick",
            preserve_underscores=True,
            taxon_namespace=self.taxa,
        )
        self.st.suppress_unifurcations()
        self.st.collapse_basal_bifurcation()
        self.st.encode_bipartitions(
            collapse_unrooted_basal_bifurcation=True, suppress_unifurcations=True
        )
        self.gt_l = dendropy.TreeList.get(
            path=self.args.gene_trees,
            schema="newick",
            preserve_underscores=True,
            taxon_namespace=self.taxa,
        )
        self.n_gt = len(self.gt_l)
        self.mask = jnp.ones(self.n_gt, dtype=bool)
        self.lbl_to_nd = utils.map_label_to_node(self.st)

    def validate_parameters(self):
        pass

    def determine_focal_edges(self):
        if self.args.focal_edges:
            self.focal_edges = []
            for lbl in self.args.focal_edges:
                edge = utils.focal_edge_from_label(self.st, lbl, self.lbl_to_nd)
                if self.args.expand_edges:
                    self.focal_edges.extend(
                        incident
                        for incident in utils.get_incident_edges(self.st, edge, self.lbl_to_nd)
                        if incident.length < MAX_INCIDENT_LENGTH or incident == edge
                    )
                else:
                    self.focal_edges.append(edge)
        self.num_edges = len(self.focal_edges)

    def transform_qqs(self, qqs):
        if self.ilr_transform:
            x = []
            for i in range(qqs.shape[1]):
                x.append(ilr(multi_replace(qqs[:, i, :], delta=1e-7)))
            return jnp.stack(x, axis=1)
        else:
            return qqs[:, :, :]

    def configure_emissions(self):
        # self.num_classes_min = max(self.args.num_classes_min, 2)
        # self.num_classes_max = min(self.args.num_classes_max, self.n_gt // 2)
        self.ilr_transform = self.args.ilr_transform

    def compute_emissions(self):
        # Initialize the discretization function and select the best number of classes for categorical emissions
        # self.discretizer = discretizer.KMeansDiscretization(
        #     self.Y.shape[1],
        #     self.Y.shape[2],
        #     self.num_classes_max,
        # )
        # self.discretizer.choose_num_classes(
        #     self.Y,
        #     range_classes=(self.num_classes_min, self.num_classes_max),
        # )
        Y = []
        # self.discretizer.fit_discretization(Y)
        for edge in self.focal_edges:
            Y.append(self.edge_to_qqs[edge])
            self.mask = self.mask & ~jnp.isnan(Y[-1]).any(axis=1)
        Y = self.transform_qqs(jnp.stack(Y, axis=1)[self.mask, :, :])
        self.discretizer = discretizer.DTO()
        self.num_classes = self.discretizer.get_num_classes()
        self.Y = self.discretizer.discretize_qqs(Y)

    def focal_edge_lengths(self):
        return [edge.head_node.label + ": " + str(edge.length) for edge in self.focal_edges]

    def initialize_output(self):
        self.output_file = self.args.output_file
        self.output_str = f"# {' '.join(sys.argv)}"
        self.output_str += "\n# Initial tree: " + self.st.as_string(schema="newick")
        self.output_str += "# Initial focal edge lengths: " + ", ".join(self.focal_edge_lengths())

    def compute_output(self):
        emission_divergence_str = ", ".join(
            list(map(lambda x: str(x), self.hmm.state_emission_divergence(self.params).tolist()))
        )
        most_likely_states = np.full(self.mask.shape, -1)
        most_likely_states[self.mask] = self.hmm.most_likely_states(self.params, self.Y)
        ps = np.full(self.mask.shape, np.nan)
        ps[self.mask] = self.hmm.smoother(self.params, self.Y).smoothed_probs[:, 1]
        self.output_str += "\n# Final tree: " + self.st.as_string(schema="newick")
        self.output_str += "# Final focal edge lengths: " + ", ".join(self.focal_edge_lengths())
        self.output_str += "\n# State divergence: " + emission_divergence_str
        self.output_str += "\n" + ",".join(
            map(lambda x: str(x) if x >= 0 else "nan", most_likely_states.astype(int).tolist())
        )
        self.output_str += "\n" + ",".join(
            map(lambda x: str(x), jnp.round(ps, decimals=3).tolist())
        )

    def simulate_emission_prob(self):
        # Obtain the simulated emissions from the species tree
        simulated_qqs = self.msc.simulate_qqs(self.focal_edges, self.args.n_replicates)
        return self.discretizer.compute_emission_prob(self.transform_qqs(simulated_qqs))

    def compute_qqs(self):
        self.msc = MSC(self.st, self.gt_l)
        if not self.args.read_qqs_path:
            self.edge_to_qqs = self.msc.compute_qqs()
            if self.args.write_qqs_path:
                self.write_qqs(self.args.write_qqs_path)
        else:
            self.edge_to_qqs = {}
            self.read_qqs(self.args.read_qqs_path)

    def read_qqs(self, path):
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                values = line.strip().split("\t")
                lbl = values[0]
                edge = utils.focal_edge_from_label(self.st, lbl, self.lbl_to_nd)
                y = int(values[1]) - 1
                freq_y = jnp.array([float(x) for x in values[2:]])
                qqs_freq = self.edge_to_qqs.get(edge, jnp.zeros((freq_y.shape[0], 3)))
                self.edge_to_qqs[edge] = qqs_freq.at[:, y].set(freq_y)

    def write_qqs(self, path):
        with open(path, "w") as f:
            header = "label\ty\t" + "\t".join(f"g_{i}" for i in range(1, self.n_gt + 1))
            f.write(header)
            for edge, qqs in self.edge_to_qqs.items():
                lbl = edge.head_node.label
                if qqs.shape[0] == 0:
                    continue
                for y in range(3):
                    buffer = StringIO()
                    np.savetxt(buffer, qqs[:, y], delimiter="", newline="\t", fmt="%.4f")
                    f.write(f"\n{lbl}\t{y + 1}\t{buffer.getvalue()[:-1]}")

    def update_edge_lengths(self, ps):
        for leaf in self.st.leaf_nodes():
            terminal_edge = leaf.edge
            terminal_edge.length = None

        for edge in self.st.internal_edges(exclude_seed_edge=False):
            qqs = self.edge_to_qqs.get(edge, None)
            if qqs is None:
                edge.length = BLEN_MIN
                continue
            mqqs = qqs[self.mask, 0]
            mmask = ~jnp.isnan(mqqs)
            z1 = jnp.sum(mqqs[mmask] * ps[mmask])
            n = jnp.sum(ps[mmask]) + BLEN_LAMBDA * 2
            if z1 >= (n / 3):
                bl = -jnp.log(3 * (1 - z1 / n) / 2)
            else:
                bl = BLEN_MIN
            edge.length = max(bl, BLEN_MIN)

    def initialize_emission_prob(self):
        def inverse_dirichlet(gamma=1.0, A=1.0, eps=1e-8):
            p = self.simulated_emission_prob
            # alpha = (p + eps) ** (-gamma)
            # alpha = alpha / alpha.sum() * A
            # prior = tfd.Dirichlet(alpha)
            prior = tfd.Dirichlet(A * (1 - p) / (p.shape[-1] - 1))
            return prior.sample(seed=jrand.PRNGKey(0))

        def random_dirichlet(gamma=1.0):
            prior = tfd.Dirichlet(self.gamma * jnp.ones(self.num_classes))
            return prior.sample(seed=jrand.PRNGKey(0), sample_shape=(self.num_edges))

        self.simulated_emission_prob = self.simulate_emission_prob()
        if self.args.emission_initialization == "random":
            alt_emission_probs = random_dirichlet(self.gamma)
        elif self.args.emission_initialization == "simulation":
            alt_emission_probs = self.simulated_emission_prob
        elif self.args.emission_initialization == "inverse":
            alt_emission_probs = inverse_dirichlet()
        else:
            raise NotImplementedError(
                f"Initial emission probabilities via {self.args.emission_initialization} is not implemented"
            )
        return jnp.stack([self.simulated_emission_prob, alt_emission_probs], axis=0)

    def initialize_hmm(self):
        # Initialize the HMM
        # Adjusting hyperparameters based on the sequence length (the number of gene trees)
        self.alpha_0 = self.n_gt * (self.args.rho) - self.args.beta
        self.alpha_1 = self.n_gt * (1 - self.args.rho) - self.args.beta
        self.beta_0 = self.args.beta
        self.beta_1 = self.args.beta
        self.emission_lambda = self.args.emission_lambda
        self.gamma = self.args.emission_concentration
        self.nu = self.args.initial_probs_concentration
        self.psi = jnp.ones((NUM_STATES, NUM_STATES)) + PSI_EPS
        self.psi = self.psi.at[0, 0].set(self.alpha_0)
        self.psi = self.psi.at[0, 1].set(self.beta_0)
        self.psi = self.psi.at[-1, -1].set(self.alpha_1)
        self.psi = self.psi.at[-1, 0].set(self.beta_1)
        if np.all(self.psi < 0).any():
            raise ValueError(
                "Invalid transition matrix; either beta<0 or beta/rho > # of gene trees"
            )
        self.occupancy_bias = jnp.zeros(NUM_STATES)
        self.occupancy_bias = self.occupancy_bias.at[-1].set(
            -jnp.log((1 - self.args.eta) / (self.args.eta))
        )
        self.emission_parameterization = (
            hmm.EmissionParam(self.args.emission_parameterization),
        ) + tuple(hmm.EmissionParam("free") for _ in range(NUM_STATES - 1))
        self.hmm = hmm.PhlagHMM(
            NUM_STATES,
            self.num_edges,
            self.num_classes,
            emission_lambda=self.emission_lambda,
            emission_concetration=self.gamma,
            emission_parameterization=self.emission_parameterization,
            initial_probs_concetration=self.nu,
            transition_concentration=self.psi,
            occupancy_bias=self.occupancy_bias,
        )
        self.params, self.props = self.hmm.initialize(
            initial_probs=INITIAL_PROBS, emission_probs=self.initialize_emission_prob()
        )
        self.props.transitions.transition_matrix.trainable = True
        self.props.emissions.probs.trainable = True
        self.props.initial.probs.trainable = True
        self.hmm.initialize_m_step_state(
            self.params, self.props, emissions_m_step_state=self.simulated_emission_prob
        )
        self.n_iters = self.args.n_iters
        self.increment_steps = self.args.increment_steps

    def propose_simulated_emission_prob(self):
        ix = lambda x, y: x[y]
        ps = self.hmm.smoother(self.params, self.Y).smoothed_probs[:, 0]
        ps += +E_STEP_EPS
        self.update_edge_lengths(ps)
        simulated_emission_prob = self.simulate_emission_prob()
        pt_prev = jax.vmap(ix, (0, 1), 0)(self.simulated_emission_prob, self.Y)
        pt_next = jax.vmap(ix, (0, 1), 0)(simulated_emission_prob, self.Y)
        s_prev = jnp.sum(jnp.log(pt_prev) * ps)
        s_next = jnp.sum(jnp.log(pt_next) * ps)
        if s_next > s_prev:
            self.simulated_emission_prob = simulated_emission_prob

    def run(self):
        # jnp.set_printoptions(threshold=sys.maxsize)
        for i in tqdm(range(self.n_iters)):
            self.params, log_probs = self.hmm.fit_em(
                self.params,
                self.props,
                self.Y,
                num_iters=(i + 1) * self.increment_steps + 1,
                verbose=False,
            )
            self.propose_simulated_emission_prob()
            self.hmm.initialize_m_step_state(
                self.params, self.props, emissions_m_step_state=self.simulated_emission_prob
            )
        self.compute_output()

    def save_output(self):
        with open(self.output_file, "w") as f:
            f.write(self.output_str)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Phlag: Detection of genomic regions with unexplained phylogenetic heterogeneity"
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
        "-o", "--output-file", type=pathlib.Path, required=True, help="Path to save the output"
    )
    parser.add_argument(
        "--n-replicates",
        type=int,
        default=2000,
        help="Number of simulation replicates for prior updates (default: 2000)",
    )
    parser.add_argument(
        "-L", "--n-iters", type=int, default=5, help="Number of (outer) iterations (default: 5)"
    )
    parser.add_argument(
        "-l",
        "--increment-steps",
        type=int,
        default=50,
        help="Increment for inner EM iterations (default: 50)",
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
    parser.add_argument(
        "-e",
        "--focal-edges",
        nargs="+",
        type=str,
        required=True,
        help="""Focal edge(s) to focus, specified by inner node label(s).
                    Rooted: edge connecting the given node and its parent.
                    Unrooted: edge giving the most balanced bipartition.""",
    )
    parser.add_argument(
        "-ee",
        "--expand-edges",
        action="store_true",
        help="Incorporate the signal from neighboring/incident edges.",
    )

    hmm_group = parser.add_argument_group("HMM parameters")
    hmm_group.add_argument(
        "--rho",
        type=float,
        default=0.9,
        help="Hyperparameter to control sensitivity, reduce to increase sensitivity (default 0.9)",
    )
    hmm_group.add_argument(
        "--beta",
        type=int,
        default=5,
        help="Hyperparameter to control contiguity of intervals, reduce to increase contiguity (default 5)",
    )
    hmm_group.add_argument(
        "--emission-lambda",
        "--lambda",
        type=float,
        default=1.0,
        help="Hyperparameter to control deviation of anomalies from MSC (default: 1.0)",
    )
    hmm_group.add_argument(
        "--initial-probs-concentration",
        type=float,
        default=1.1,
        help="Initial probabilities concentration (default: 1.1)",
    )
    hmm_group.add_argument(
        "--emission-concentration",
        type=float,
        default=1.1,
        help="Emission prior concentration (default: 1.1)",
    )
    hmm_group.add_argument(
        "--eta",
        "--occupancy-bias",
        type=float,
        default=0.5,
        help="""A global occupancy penalty on the marginal log-likelihood.
                Per gene tree, penalty is -log((1-eta)/eta) (default: 0.5)""",
    )
    hmm_group.add_argument(
        "--emission-initialization",
        type=str.lower,
        default="inverse",
        choices=["random", "inverse", "simulation"],
        help="""The default state is initialized based on MSC-based simulations.
                    The alternative state is initialized from inverse, random, or simulation.""",
    )
    hmm_group.add_argument(
        "--emission-parameterization",
        type=str.lower,
        default="attraction",
        choices=["free", "attraction", "anchor"],
        help="""Parameterization of the emission probabilities of the default state (default: attraction):
                    free (no prior), attraction, or anchor (MSC-based simulations).""",
    )

    discr_group = parser.add_argument_group("Discretization and emission parameters")
    discr_group.add_argument(
        "--num-classes-min", type=int, default=8, help="Minimum number of classes (default: 8)"
    )
    discr_group.add_argument(
        "--num-classes-max", type=int, default=64, help="Maximum number of classes (default: 64)"
    )
    discr_group.add_argument(
        "--ilr-transform",
        action="store_true",
        help="Apply isometric log-ratio transformation on QQS values",
    )

    io_group = parser.add_argument_group("I/O options")
    io_group.add_argument(
        "--write-qqs-path", type=pathlib.Path, help="Write QQS values to the given filepath"
    )
    io_group.add_argument(
        "--read-qqs-path", type=pathlib.Path, help="Read QQS values from the given filepath"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    phlag = Phlag(args)
    if not (args.dry_run):
        phlag.run()
        phlag.save_output()
    else:
        sys.exit("Exiting: --dry-run is given.")


if __name__ == "__main__":
    main()
