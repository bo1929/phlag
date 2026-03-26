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


PRNGKeyT = Array
Scalar = Union[float, Float[Array, ""]]
IntScalar = Union[int, Int[Array, ""]]


def wasserstein_distance(p, q, C, lamb=100.0, lr=0.1, n_iter=50):
    raise NotImplemented("Wassertein distance is not implemented!")


def hellinger_distance(p, q):
    return jnp.sqrt(jnp.sum((jnp.sqrt(p) - jnp.sqrt(q)) ** 2)) / jnp.sqrt(2.0)


def kl_divergence(p, q):
    return jnp.sum(kl_div(p + 1e-10, q + 1e-10))


def total_variation_distance(p, q):
    return 0.5 * jnp.sum(jnp.abs(p - q))


def l2(p, q):
    return jnp.sqrt(jnp.sum((p - q) ** 2))


class PhlagHMMTransitions(HMMTransitions):
    def __init__(
        self,
        num_states: Int,
        concentration: Union[Scalar, Float[Array, "num_states num_states"]] = 1.1,
    ):
        self.num_states = num_states
        self.concentration = concentration * jnp.ones((num_states, num_states))

    def distribution(
        self, params: ParamsStandardHMMTransitions, state: IntScalar, inputs=None
    ):
        return tfd.Categorical(probs=params.transition_matrix[state])

    def initialize(
        self,
        key: Optional[Array] = None,
        method="prior",
        transition_matrix: Optional[Float[Array, "num_states num_states"]] = None,
    ) -> Tuple[ParamsStandardHMMTransitions, ParamsStandardHMMTransitions]:
        if transition_matrix is None:
            if key is None:
                raise ValueError(
                    "A key must be provided if transition_matrix is not provided."
                )
            else:
                tm_sample = tfd.Dirichlet(self.concentration).sample(seed=key)
                transition_matrix = cast(
                    Float[Array, "num_states num_states"], tm_sample
                )
        params = ParamsStandardHMMTransitions(transition_matrix=transition_matrix)
        props = ParamsStandardHMMTransitions(
            transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())
        )
        return params, props

    def log_prior(self, params: ParamsStandardHMMTransitions) -> Scalar:
        return (
            tfd.Dirichlet(self.concentration).log_prob(params.transition_matrix).sum()
        )

    def _compute_transition_matrices(
        self, params: ParamsStandardHMMTransitions, inputs=None
    ) -> Float[Array, "num_states num_states"]:
        return params.transition_matrix

    def collect_suff_stats(self, params, posterior: HMMPosterior, inputs=None) -> Union[
        Float[Array, "num_states num_states"],
        Float[Array, "num_timesteps_minus_1 num_states num_states"],
    ]:
        return posterior.trans_probs

    def initialize_m_step_state(self, params, props):
        return None

    def m_step(
        self,
        params: ParamsStandardHMMTransitions,
        props: ParamsStandardHMMTransitions,
        batch_stats: Float[Array, "batch num_states num_states"],
        m_step_state: Any,
    ) -> Tuple[ParamsStandardHMMTransitions, Any]:
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
    probs: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]


class PhlagHMMEmissions(HMMEmissions):
    def __init__(
        self,
        num_states: Int,
        emission_dim: Int,
        num_classes: Int,
        emission_anomaly_strength: Float,
        emission_cost_matrices: Union[
            Scalar, Float[Array, "emission_dim num_classes num_classes"]
        ] = 1,
        emission_prior_concentration: Union[Scalar, Float[Array, "num_classes"]] = 1.1,
    ):
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.num_classes = num_classes
        self.similarity_penalty = emission_anomaly_strength
        self.transfer_cost = emission_cost_matrices * jnp.ones(
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
        expected_states = posterior.smoothed_probs
        x = one_hot(emissions, self.num_classes)
        return dict(sum_x=jnp.einsum("tk,tdi->kdi", expected_states, x))

    def initialize_m_step_state(self, params, props):
        return None

    def update_m_step_state(self, params, props):
        return None

    def compute_prob_distance(self, emission_0, emission_1):
        # return total_variation_distance(emission_1, emission_0)
        # return l2(emission_1, emission_0)
        return hellinger_distance(emission_1, emission_0)
        # return wasserstein_distance(emission_1, emission_0, self.transfer_cost)

    def map_with_strength(self, emission_0, counts_1, se=None, dr=True):
        k = self.num_classes

        def neg_log_posterior(logits):
            # Convert logits to a valid probability distribution using softmax
            emission_1 = softmax(logits)
            # Negative log-likelihood (from multinomial distribution).
            neg_log_likelihood = -jnp.sum(counts_1 * jnp.log(emission_1 + 1e-10))
            # Negative log-prior without the normalization factor
            neg_log_prior = self.similarity_penalty * compute_prob_distance(
                emission_1, emission_0
            )
            return neg_log_likelihood + neg_log_prior

        initial_logits = emission_0
        result = minimize(
            fun=neg_log_posterior, x0=initial_logits, method="BFGS", tol=1e-4
        )
        # Notice that this is not bounded (hence softmax is needed)
        return softmax(result.x)

    def m_step(
        self,
        params,
        props,
        batch_stats,
        m_step_state: Union[Scalar, Float[Array, "emission_dim num_classes"]],
    ):
        if props.probs.trainable:
            emission_stats = pytree_sum(batch_stats, axis=0)
            probs = params.probs
            probs = tfd.Dirichlet(
                self.prior_concentration + emission_stats["sum_x"]
            ).mode()
            probs = probs.at[0].set(
                jax.vmap(self.map_with_strength, in_axes=(0, 0, 0), out_axes=0)(
                    m_step_state,
                    (self.prior_concentration + emission_stats["sum_x"])[0],
                )
            )
            # probs = probs.at[0].set(m_step_state)
            params = params._replace(probs=probs)
        return params, m_step_state

    def compute_emission_distance(self, params):
        probs = params.probs
        return jax.vmap(compute_prob_distance, in_axes=(0, 0, 0), out_axes=0)(
            probs[0], probs[1]
        )


class ParamsPhlagHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsCategoricalHMMEmissions


class PhlagHMM(HMM):
    def __init__(
        self,
        num_states: Int = 2,
        emission_dim: Int = 1,
        num_classes: Int = 2,
        emission_anomaly_strength: Scalar = 0.001,
        emission_prior_concentration: Union[Scalar, Float[Array, "num_classes"]] = 1.1,
        emission_cost_matrices: Union[
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
        self.emission_anomaly_strength = emission_anomaly_strength
        self.emission_prior_concentration = emission_prior_concentration
        self.emission_cost_matrices = emission_cost_matrices
        self.initial_probs_concentration = initial_probs_concentration
        self.transition_matrix_concentration = transition_matrix_concentration
        self.initial_m_step_state = None
        self.transitions_m_step_state = None
        self.emissions_m_step_state = None
        assert self.emission_anomaly_strength > 0

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
            emission_anomaly_strength=self.emission_anomaly_strength,
            emission_prior_concentration=self.emission_prior_concentration,
            emission_cost_matrices=self.emission_cost_matrices,
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
        key1, key2, key3 = jr.split(key, 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(
            key1, method=method, initial_probs=initial_probs
        )
        (params["transitions"], props["transitions"]) = (
            self.transition_component.initialize(
                key2, method=method, transition_matrix=transition_matrix
            )
        )
        params["emissions"], props["emissions"] = self.emission_component.initialize(
            key3, method=method, emission_probs=emission_probs
        )
        return ParamsPhlagHMM(**params), ParamsPhlagHMM(**props)

    def initialize_m_step_state(
        self,
        params: HMMParameterSet,
        props: HMMPropertySet,
        initial_m_step_state=None,
        transitions_m_step_state=None,
        emissions_m_step_state=None,
    ):
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

    def e_step(
        self,
        params: HMMParameterSet,
        emissions: Array,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    ) -> Tuple[PyTree, Scalar]:
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

    def compute_emission_distance(self, params):
        return self.emission_component.compute_emission_distance(params.emissions)

    def m_step(
        self,
        params: HMMParameterSet,
        props: HMMPropertySet,
        batch_stats: PyTree,
        m_step_state: Any,
    ) -> Tuple[HMMParameterSet, Any]:
        batch_initial_stats, batch_transition_stats, batch_emission_stats = batch_stats
        (initial_m_step_state, transitions_m_step_state, emissions_m_step_state) = (
            m_step_state
        )
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
