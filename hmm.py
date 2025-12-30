import jax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from jax.scipy.special import kl_div
from jax.scipy.optimize import minimize
from jax.nn import softmax, one_hot
from jax import vmap

from jaxtyping import Array, Float, Int, PyTree
from typing import Any, NamedTuple, Optional, Tuple, Union, cast
from dynamax.utils.utils import pytree_sum
from dynamax.hidden_markov_model.inference import HMMPosterior
from dynamax.hidden_markov_model.inference import hmm_two_filter_smoother
from dynamax.hidden_markov_model.models.abstractions import (
    HMM,
    HMMEmissions,
    HMMTransitions,
    HMMParameterSet,
    HMMPropertySet,
)
from dynamax.hidden_markov_model.models.initial import (
    StandardHMMInitialState,
    ParamsStandardHMMInitialState,
)
from dynamax.hidden_markov_model.models.transitions import ParamsStandardHMMTransitions
from dynamax.parameters import ParameterProperties, ParameterSet, PropertySet

from utils import timeit

PRNGKeyT = Array
Scalar = Union[float, Float[Array, ""]]
IntScalar = Union[int, Int[Array, ""]]


# TODO: Add types when appropriate


# def wasserstein_distance(p, q, C, epsilon=0.11, n_iter=20):
#     K = C.shape[0]

#     f = jnp.zeros(K)
#     g = jnp.zeros(K)

#     def body_fn(_, fg):
#         f, g = fg
#         # update f
#         kf = epsilon * jnp.log(p + 1e-20) - epsilon * jax.scipy.special.logsumexp((g - C.T) / epsilon, axis=1)
#         # update g
#         g = epsilon * jnp.log(q + 1e-20) - epsilon * jax.scipy.special.logsumexp((f - C) / epsilon, axis=1)
#         return f, g

#     f, g = jax.lax.fori_loop(0, n_iter, body_fn, (f, g))

#     # Compute transport plan in log domain
#     T_log = (f[:, None] + g[None, :] - C) / epsilon
#     T = jnp.exp(T_log)

#     # Wasserstein distance
#     W = jnp.sum(T * C)
#     return W  # / jnp.max(C)


def hellinger_distance(p, q):
    return jnp.sqrt(jnp.sum((jnp.sqrt(p) - jnp.sqrt(q)) ** 2)) / jnp.sqrt(2.0)


def kl_divergence(p, q):
    return jnp.sum(kl_div(p + 1e-10, q + 1e-10))


def total_variation_distance(p, q):
    return 0.5 * jnp.sum(jnp.abs(p - q))


def l2(p, q):
    return jnp.sqrt(jnp.sum((p - q) ** 2))


def wasserstein_distance(p, q, C, lam=100.0, lr=0.1, n_iter=50):
    K = C.shape[0]
    T = jnp.ones((K, K)) / K**2

    def lossw(T_pos):
        # Ensure T_pos >= 0 via small epsilon
        T_pos = jnp.clip(T_pos, 1e-8, None)
        cost = jnp.sum(T_pos * C)
        row_marg = jnp.sum(T_pos, axis=1)
        col_marg = jnp.sum(T_pos, axis=0)
        penalty = lam * (jnp.sum((row_marg - p) ** 2) + jnp.sum((col_marg - q) ** 2))
        return cost + penalty

    grad_fn = jax.grad(lossw)

    def body_fn(i, T):
        T_new = T - lr * grad_fn(T)
        T_new = jnp.clip(T_new, 1e-8, None)  # keep nonnegative
        return T_new / jnp.sum(T_new)  # normalize for stability

    T_opt = jax.lax.fori_loop(0, n_iter, body_fn, T)
    W = jnp.sum(T_opt * C)
    return W / jnp.max(C)


def fdist(emission_0, emission_1, C=None):
    # return total_variation_distance(emission_1, emission_0)
    return hellinger_distance(emission_1, emission_0)
    # return wasserstein_distance(emission_1, emission_0, C)


class PhlagHMMTransitions(HMMTransitions):
    """Standard model for HMM transitions.

    We place a Dirichlet prior over the rows of the transition matrix $A$,

    $$A_k \sim \mathrm{Dir}(\psi 1_K)$$ or $$A_k \sim \mathrm{Dir}(\psi[l,\cdot])$$

    where

    * $1_K$ denotes a length-$K$ vector of ones,
    * $e_i$ denotes the one-hot vector with a 1 in the $i$-th position,
    * $\psi \in \mathbb{R}_+$ or $\psi \in \mathcal{M}_{K, K}(\mathbb{R}_+)$ is the concentration.
    """

    def __init__(
        self,
        num_states: Int,
        concentration: Union[Scalar, Float[Array, "num_states num_states"]] = 1.1,
    ):
        """
        Args:
            transition_matrix[j,k]: prob(hidden(t) = k | hidden(t-1)j)
        """
        self.num_states = num_states
        self.concentration = concentration * jnp.ones((num_states, num_states))

    def distribution(
        self, params: ParamsStandardHMMTransitions, state: IntScalar, inputs=None
    ):
        """Return the distribution over the next state given the current state."""
        return tfd.Categorical(probs=params.transition_matrix[state])

    def initialize(
        self,
        key: Optional[Array] = None,
        method="prior",
        transition_matrix: Optional[Float[Array, "num_states num_states"]] = None,
    ) -> Tuple[ParamsStandardHMMTransitions, ParamsStandardHMMTransitions]:
        """Initialize the model parameters and their corresponding properties."""
        if transition_matrix is None:
            if key is None:
                raise ValueError(
                    "A key must be provided if transition_matrix is not provided."
                )
            else:
                # TODO: Reconsider this
                # if method == "prior":
                tm_sample = tfd.Dirichlet(self.concentration).sample(seed=key)
                # elif method == "random":
                #     random_concentration = jnp.ones(self.concentration.shape) * 1.1
                #     tm_sample = tfd.Dirichlet(random_concentration).sample(seed=key)
                # else:
                #     raise ValueError("method must be either 'prior' or 'random'.")
                transition_matrix = cast(
                    Float[Array, "num_states num_states"], tm_sample
                )
        params = ParamsStandardHMMTransitions(transition_matrix=transition_matrix)
        props = ParamsStandardHMMTransitions(
            transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())
        )
        return params, props

    def log_prior(self, params: ParamsStandardHMMTransitions) -> Scalar:
        """Compute the log prior probability of the parameters."""
        return (
            tfd.Dirichlet(self.concentration).log_prob(params.transition_matrix).sum()
        )

    def _compute_transition_matrices(
        self, params: ParamsStandardHMMTransitions, inputs=None
    ) -> Float[Array, "num_states num_states"]:
        """Compute the transition matrices."""
        return params.transition_matrix

    def collect_suff_stats(self, params, posterior: HMMPosterior, inputs=None) -> Union[
        Float[Array, "num_states num_states"],
        Float[Array, "num_timesteps_minus_1 num_states num_states"],
    ]:
        """Collect the sufficient statistics for the model."""
        return posterior.trans_probs

    def initialize_m_step_state(self, params, props):
        """Initialize the state for the M-step."""
        return None

    def m_step(
        self,
        params: ParamsStandardHMMTransitions,
        props: ParamsStandardHMMTransitions,
        batch_stats: Float[Array, "batch num_states num_states"],
        m_step_state: Any,
    ) -> Tuple[ParamsStandardHMMTransitions, Any]:
        """Perform the M-step of the EM algorithm."""
        if props.transition_matrix.trainable:
            if self.num_states == 1:
                transition_matrix = jnp.array([[1.0]])
            else:
                expected_trans_counts = batch_stats.sum(axis=0)
                transition_matrix = tfd.Dirichlet(
                    self.concentration + expected_trans_counts
                ).mode()
            params = params._replace(transition_matrix=transition_matrix)
        return params, m_step_state


class ParamsCategoricalHMMEmissions(NamedTuple):
    """Parameters for the CategoricalHMM emission distribution."""

    probs: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]


class PhlagHMMEmissions(HMMEmissions):
    r"""
    Categorical emissions for a Phlag's hidden Markov model.
    """

    def __init__(
        self,
        num_states: Int,
        emission_dim: Int,
        num_classes: Int,
        emission_similarity_penalty: Float,
        emission_transfer_cost: Union[
            Scalar, Float[Array, "emission_dim num_classes num_classes"]
        ] = 1,
        emission_prior_concentration: Union[Scalar, Float[Array, "num_classes"]] = 1.1,
    ):
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.num_classes = num_classes
        self.similarity_penalty = emission_similarity_penalty
        self.transfer_cost = emission_transfer_cost * jnp.ones(
            (self.emission_dim, self.num_classes, self.num_classes)
        )
        self.prior_concentration = emission_prior_concentration * jnp.ones(
            self.num_classes
        )

    def set_emission_prior_concentration(self, emission_prior_concentration):
        self.prior_concentration = emission_prior_concentration

    @property
    def emission_shape(self) -> Tuple[Int]:
        """Shape of the emission distribution."""
        return (self.emission_dim,)

    def distribution(
        self, params: ParamsCategoricalHMMEmissions, state: IntScalar, inputs=None
    ) -> tfd.Distribution:
        """Return the emission distribution for a given state."""
        return tfd.Independent(
            tfd.Categorical(probs=params.probs[state]), reinterpreted_batch_ndims=1
        )

    def log_prior(self, params: ParamsCategoricalHMMEmissions) -> Scalar:
        """Return the log prior probability of the emission parameters."""
        return tfd.Dirichlet(self.prior_concentration).log_prob(params.probs).sum()

    def initialize(
        self,
        key: Optional[Array] = jr.PRNGKey(0),
        method="prior",
        emission_probs: Optional[
            Float[Array, "num_states emission_dim num_classes"]
        ] = None,
    ) -> Tuple[ParamsCategoricalHMMEmissions, ParamsCategoricalHMMEmissions]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to jr.PRNGKey(0).
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            emission_probs (array, optional): manually specified emission probabilities. Defaults to None.

        Returns:
            params: nested dataclasses of arrays containing model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        # Initialize the emission probabilities
        if emission_probs is None:
            if method.lower() == "prior":
                if key is None:
                    raise ValueError("key must not be None when emission_probs is None")
                prior = tfd.Dirichlet(self.prior_concentration)
                emission_probs = prior.sample(
                    seed=key, sample_shape=(self.num_states, self.emission_dim)
                )
            elif method.lower() == "kmeans":
                raise NotImplementedError(
                    "kmeans initialization is not yet implemented!"
                )
            else:
                raise Exception("invalid initialization method: {}".format(method))
        else:
            assert emission_probs.shape == (
                self.num_states,
                self.emission_dim,
                self.num_classes,
            )
            assert jnp.all(emission_probs >= 0)
            # assert jnp.allclose(emission_probs.sum(axis=2), 1.0)

        # Add parameters to the dictionary
        params = ParamsCategoricalHMMEmissions(probs=emission_probs)
        props = ParamsCategoricalHMMEmissions(
            probs=ParameterProperties(constrainer=tfb.SoftmaxCentered())
        )
        return params, props

    def collect_suff_stats(self, params, posterior, emissions, inputs=None):
        """Collect sufficient statistics for the emission distribution."""
        expected_states = posterior.smoothed_probs
        x = one_hot(emissions, self.num_classes)
        return dict(sum_x=jnp.einsum("tk,tdi->kdi", expected_states, x))

    def initialize_m_step_state(self, params, props):
        """Initialize the m-step state."""
        return None

    def update_m_step_state(self, params, props):
        """Initialize the m-step state."""
        return None

    def map_with_arbitraryx(self, emission_0, counts_1, C=None, dr=True):
        """
        Computes the MAP estimate for a categorical distribution by maximizing the posterior with emission similarity penalty.
        This version optimizes unconstrained logits and uses the softmax trick.

        Args:
            emission_0 (jnp.ndarray): Array of probabilities for the known normal emission distribution.
            counts_1 (jnp.ndarray): Array of expected observed counts from the anomalous state.

        Returns:
            jnp.ndarray: The MAP estimate for the unknown probabilities.
        """
        k = self.num_classes

        def neg_log_posterior(logits):
            # Convert logits to a valid probability distribution using softmax
            emission_1 = softmax(logits)
            # Negative log-likelihood (from multinomial distribution).
            neg_log_likelihood = -jnp.sum(counts_1 * jnp.log(emission_1 + 1e-10))
            # Negative log-prior without the normalization factor
            neg_log_prior = self.similarity_penalty * fdist(emission_1, emission_0, C)
            return neg_log_likelihood + neg_log_prior

        initial_logits = emission_0
        result = minimize(
            fun=neg_log_posterior, x0=initial_logits, method="BFGS", tol=1e-4
        )
        # Notice that this is not bounded (hence softmax is needed)
        return softmax(result.x)

    def map_with_arbitrary(self, emission_0, counts_1, C=None):
        """
        Computes the MAP estimate for a categorical distribution by maximizing the posterior with emission similarity penalty.
        This version optimizes unconstrained logits and uses the softmax trick.

        Args:
            emission_0 (jnp.ndarray): Array of probabilities for the known normal emission distribution.
            counts_1 (jnp.ndarray): Array of expected observed counts from the anomalous state.

        Returns:
            jnp.ndarray: The MAP estimate for the unknown probabilities.
        """
        k = self.num_classes

        def neg_log_posterior(logits):
            # Convert logits to a valid probability distribution using softmax
            emission_1 = softmax(logits)
            # Negative log-likelihood (from multinomial distribution).
            neg_log_likelihood = -jnp.sum(counts_1 * jnp.log(emission_1 + 1e-10))
            # Negative log-prior without the normalization factor
            neg_log_prior = self.similarity_penalty * (
                1 - fdist(emission_1, emission_0, C)
            )
            return neg_log_likelihood + neg_log_prior

        initial_logits = emission_0
        result = minimize(
            fun=neg_log_posterior, x0=initial_logits, method="BFGS", tol=1e-4
        )
        # Notice that this is not bounded (hence softmax is needed)
        return softmax(result.x)

    def map_with_lagrange(self, emission_0, counts_1, tol=1e-8, max_iter=1000):
        """
        Compute the M-step update for the unknown emission distribution q
        under a dot-product repulsion prior.

        Objective (MAP form):
            \argmax_q   \sum_i c_i * log q_i  -  \delta * \sum_i p_i * q_i
            subject to   \sum_i q_i = 1,  q_i > 0

        Closed-form relation:
            q_i = c_i / (\nu + \delta p_i)
            where \nu is chosen so that sum_i q_i = 1.
        """

        # Define the normalization function f(\nu) = \sum c_i / (\nu + \delta p_i) - 1.
        # We want to find \nu^* such that f(\nu^*) = 0.
        def f(nu):
            return jnp.sum(counts_1 / (nu + self.similarity_penalty * emission_0)) - 1.0

        nu_low = 1e-12
        nu_high = counts_1.sum() * 10.0

        def body(val):
            nu_low, nu_high, i = val
            nu_mid = (nu_low + nu_high) / 2
            f_mid = f(nu_mid)
            nu_low = jnp.where(f_mid > 0, nu_mid, nu_low)
            nu_high = jnp.where(f_mid < 0, nu_mid, nu_high)
            return (nu_low, nu_high, i + 1)

        def cond(val):
            nu_low, nu_high, i = val
            return (i < max_iter) & ((nu_high - nu_low) > tol)

        nu_low, nu_high, _ = jax.lax.while_loop(cond, body, (nu_low, nu_high, 0))
        nu_opt = (nu_low + nu_high) / 2
        q = counts_1 / (nu_opt + self.similarity_penalty * emission_0)
        q /= q.sum()
        return q

    @timeit
    def m_step(
        self,
        params,
        props,
        batch_stats,
        m_step_state: Union[Scalar, Float[Array, "emission_dim num_classes"]],
    ):
        """Perform the m-step for the emission distribution."""
        if props.probs.trainable:
            emission_stats = pytree_sum(batch_stats, axis=0)
            probs = params.probs
            probs = tfd.Dirichlet(
                self.prior_concentration + emission_stats["sum_x"]
            ).mode()
            # probs = probs.at[1].set(
            #     jax.vmap(self.map_with_arbitrary, in_axes=(0, 0, 0), out_axes=0)(
            #         m_step_state,
            #         (self.prior_concentration + emission_stats["sum_x"])[1],
            #         self.transfer_cost,
            #     )
            # )
            probs = probs.at[0].set(
                jax.vmap(self.map_with_arbitraryx, in_axes=(0, 0, 0), out_axes=0)(
                    m_step_state,
                    (self.prior_concentration + emission_stats["sum_x"])[0],
                    self.transfer_cost,
                )
            )
            # probs = probs.at[0].set(m_step_state)
            params = params._replace(probs=probs)
        return params, m_step_state

    def probs_dissimilarity(self, params):
        probs = params.probs
        return jax.vmap(wasserstein_distance, in_axes=(0, 0, 0), out_axes=0)(
            probs[0], probs[1], self.transfer_cost
        )


class ParamsPhlagHMM(NamedTuple):
    """Parameters for the CategoricalHMM model."""

    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsCategoricalHMMEmissions


class PhlagHMM(HMM):
    r"""An HMM with conditionally independent categorical emissions to detect local phylogenetic anomalies.
    All parameters (transition matrix, emission probabilities, initial state probabilities) are from a Dirichlet distributions with corresponding hyperparameters.
    :param num_states: number of discrete states $K$, state 0 is reserved for the null model
    :param emission_dim: number of conditionally independent emissions $N$
    :param num_classes: number of multinomial classes $C$
    :param emission similarity penalty: $\delta$
    :param emission_prior_concentration: $\gamma$
    :param initial_probs_concentration: $\nu$
    :param transition_matrix_concentration: $\psi$
    """

    def __init__(
        self,
        num_states: Int = 2,
        emission_dim: Int = 1,
        num_classes: Int = 2,
        emission_similarity_penalty: Scalar = 0.001,
        emission_prior_concentration: Union[Scalar, Float[Array, "num_classes"]] = 1.1,
        emission_transfer_cost: Union[
            Float[Array, "emission_dim num_classes num_classes"]
        ] = 1,
        initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]] = 1.1,
        transition_matrix_concentration: Union[
            Scalar, Float[Array, "num_states num_states"]
        ] = 1.1,
    ):
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.num_classes = num_classes
        self.emission_similarity_penalty = emission_similarity_penalty
        self.emission_prior_concentration = emission_prior_concentration
        self.emission_transfer_cost = emission_transfer_cost
        self.initial_probs_concentration = initial_probs_concentration
        self.transition_matrix_concentration = transition_matrix_concentration
        self.initial_m_step_state = None
        self.transitions_m_step_state = None
        self.emissions_m_step_state = None

        self.initial_component = StandardHMMInitialState(
            num_states=self.num_states,
            initial_probs_concentration=self.initial_probs_concentration,
        )
        self.transition_component = PhlagHMMTransitions(
            num_states=self.num_states,
            concentration=self.transition_matrix_concentration,
        )
        self.emission_component = PhlagHMMEmissions(
            self.num_states,
            self.emission_dim,
            self.num_classes,
            emission_similarity_penalty=self.emission_similarity_penalty,
            emission_prior_concentration=self.emission_prior_concentration,
            emission_transfer_cost=self.emission_transfer_cost,
        )
        super().__init__(
            num_states=self.num_states,
            initial_component=self.initial_component,
            transition_component=self.transition_component,
            emission_component=self.emission_component,
        )

    def initialize(
        self,
        key: Array = jr.PRNGKey(0),
        method: str = "prior",
        emission_probs: Optional[
            Float[Array, "num_states emission_dim num_classes"]
        ] = None,
        initial_probs: Optional[Float[Array, "num_states"]] = None,
        transition_matrix: Optional[Float[Array, "num_states num_states"]] = None,
    ) -> Tuple[ParameterSet, PropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`), or set to the manually specified values.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to None.
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            emission_probs (array, optional): manually specified emission probabilities. Defaults to None.
            initial_probs (array, optional): manually specified initial state probabilities. Defaults to None.
            transition_matrix (array, optional): manually specified transition matrix. Defaults to None.

        Returns:
            Model parameters and their properties.
        """
        key1, key2, key3 = jr.split(key, 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(
            key1, method=method, initial_probs=initial_probs
        )
        (
            params["transitions"],
            props["transitions"],
        ) = self.transition_component.initialize(
            key2, method=method, transition_matrix=transition_matrix
        )
        params["emissions"], props["emissions"] = self.emission_component.initialize(
            key3, method=method, emission_probs=emission_probs
        )
        return ParamsPhlagHMM(**params), ParamsPhlagHMM(**props)

    # @timeit
    def initialize_m_step_state(
        self,
        params: HMMParameterSet,
        props: HMMPropertySet,
        initial_m_step_state=None,
        transitions_m_step_state=None,
        emissions_m_step_state=None,
    ):
        """Initialize any required state for the M step.
        For example, this might include the optimizer state for Adam.
        """
        if initial_m_step_state is not None:
            self.initial_m_step_state = initial_m_step_state
        if transitions_m_step_state is not None:
            self.transitions_m_step_state = transitions_m_step_state
        if emissions_m_step_state is not None:
            self.emissions_m_step_state = emissions_m_step_state

        if self.initial_m_step_state is None:
            self.initial_m_step_state = self.initial_component.initialize_m_step_state(
                params.initial, props.initial
            )
        if self.transitions_m_step_state is None:
            self.transitions_m_step_state = (
                self.transition_component.initialize_m_step_state(
                    params.transitions, props.transitions
                )
            )
        if self.emissions_m_step_state is None:
            self.emissions_m_step_state = (
                self.emission_component.initialize_m_step_state(
                    params.emissions, props.emissions
                )
            )
        return (
            self.initial_m_step_state,
            self.transitions_m_step_state,
            self.emissions_m_step_state,
        )

    # def get_posterior(self, params: HMMParameterSet = None, emissions: Array = None):
    #     if self.posterior is None:
    #         if params is None or emissions is None:
    #             raise ValueError("params and emissions must be provided if posterior is not already computed")
    #         args = self._inference_args(params, emissions, None)
    #         self.posterior = hmm_two_filter_smoother(*args)
    #     return self.posterior

    # @timeit
    def e_step(
        self,
        params: HMMParameterSet,
        emissions: Array,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    ) -> Tuple[PyTree, Scalar]:
        """The E-step computes expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """
        args = self._inference_args(params, emissions, inputs)
        posterior = hmm_two_filter_smoother(*args)

        initial_stats = self.initial_component.collect_suff_stats(
            params.initial, posterior, inputs
        )
        transition_stats = self.transition_component.collect_suff_stats(
            params.transitions, posterior, inputs
        )
        emission_stats = self.emission_component.collect_suff_stats(
            params.emissions, posterior, emissions, inputs
        )
        return (
            initial_stats,
            transition_stats,
            emission_stats,
        ), posterior.marginal_loglik

    def emission_dissimilarity(self, params):
        return self.emission_component.probs_dissimilarity(params.emissions)

    def m_step(
        self,
        params: HMMParameterSet,
        props: HMMPropertySet,
        batch_stats: PyTree,
        m_step_state: Any,
    ) -> Tuple[HMMParameterSet, Any]:
        """
        Perform an M-step on the model parameters.
        """
        batch_initial_stats, batch_transition_stats, batch_emission_stats = batch_stats
        (
            initial_m_step_state,
            transitions_m_step_state,
            emissions_m_step_state,
        ) = m_step_state

        initial_params, initial_m_step_state = self.initial_component.m_step(
            params.initial, props.initial, batch_initial_stats, initial_m_step_state
        )
        transition_params, transitions_m_step_state = self.transition_component.m_step(
            params.transitions,
            props.transitions,
            batch_transition_stats,
            transitions_m_step_state,
        )
        emission_params, emissions_m_step_state = self.emission_component.m_step(
            params.emissions,
            props.emissions,
            batch_emission_stats,
            emissions_m_step_state,
        )
        params = params._replace(
            initial=initial_params,
            transitions=transition_params,
            emissions=emission_params,
        )
        m_step_state = (
            initial_m_step_state,
            transitions_m_step_state,
            emissions_m_step_state,
        )
        return params, m_step_state
