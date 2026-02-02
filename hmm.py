import jax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from enum import Enum

from jax.scipy.special import kl_div
from jax.scipy.optimize import minimize
from jax.nn import softmax, one_hot
from jax.lax import while_loop
from jaxtyping import Array, Float, Int, PyTree
from typing import Any, NamedTuple, Optional, Tuple, Union, cast
from dynamax.utils.utils import pytree_sum
from dynamax.hidden_markov_model.inference import *
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
from dynamax.parameters import ParameterProperties

PRNGKeyT = Array
Scalar = Union[float, Float[Array, ""]]
IntScalar = Union[int, Int[Array, ""]]


def _normalize(u: Array, axis=0, eps=1e-15):
    u = jnp.where(u == 0, 0, jnp.where(u < eps, eps, u))
    c = u.sum(axis=axis)
    c = jnp.where(c == 0, 1, c)
    return u / c, c


def _condition_on(probs, ll):
    ll_max = ll.max()
    new_probs = probs * jnp.exp(ll - ll_max)
    new_probs, norm = _normalize(new_probs)
    log_norm = jnp.log(norm) + ll_max
    return new_probs, log_norm


def _predict(
    probs: Float[Array, " num states"], A: Float[Array, "num states num states"]
) -> Float[Array, "num states"]:
    return A.T @ probs


@partial(jit, static_argnames=["transition_fn"])
def hmm_backward_filter(
    transition_matrix: Optional[
        Union[
            Float[Array, "num_states num_states"],
            Float[Array, "num_timesteps_minus_1 num_states num_states"],
        ]
    ],
    log_likelihoods: Float[Array, "num_timesteps num_states"],
    transition_fn: Optional[Callable[[int], Float[Array, "num_states num_states"]]] = None,
    occupancy_bias: Union[Scalar, Float[Array, "num_states"]] = 0.0,
) -> Tuple[Scalar, Float[Array, "num_timesteps num_states"]]:
    num_timesteps, num_states = log_likelihoods.shape

    def _step(carry, t):
        """Backward filtering step."""
        log_normalizer, backward_pred_probs = carry
        A = get_trans_mat(transition_matrix, transition_fn, t - 1)
        ll = log_likelihoods[t] + occupancy_bias
        backward_filt_probs, log_norm = _condition_on(backward_pred_probs, ll)
        log_normalizer += log_norm
        next_backward_pred_probs = _predict(backward_filt_probs, A.T)
        return (log_normalizer, next_backward_pred_probs), backward_pred_probs

    (log_normalizer, _), backward_pred_probs = lax.scan(
        _step, (0.0, jnp.ones(num_states)), jnp.arange(num_timesteps), reverse=True
    )
    return log_normalizer, backward_pred_probs


@partial(jit, static_argnames=["transition_fn"])
def hmm_filter(
    initial_distribution: Float[Array, " num_states"],
    transition_matrix: Optional[
        Union[
            Float[Array, "num_states num_states"],
            Float[Array, "num_timesteps_minus_1 num_states num_states"],
        ]
    ],
    log_likelihoods: Float[Array, "num_timesteps num_states"],
    transition_fn: Optional[Callable[[IntScalar], Float[Array, "num_states num_states"]]] = None,
    occupancy_bias: Union[Scalar, Float[Array, "num_states"]] = 0.0,
) -> HMMPosteriorFiltered:
    num_timesteps, num_states = log_likelihoods.shape

    def _step(carry, t):
        log_normalizer, predicted_probs = carry
        A = get_trans_mat(transition_matrix, transition_fn, t)
        ll = log_likelihoods[t] + occupancy_bias
        filtered_probs, log_norm = _condition_on(predicted_probs, ll)
        log_normalizer += log_norm
        predicted_probs_next = _predict(filtered_probs, A)
        return (log_normalizer, predicted_probs_next), (filtered_probs, predicted_probs)

    carry = (0.0, initial_distribution)
    (log_normalizer, _), (filtered_probs, predicted_probs) = lax.scan(
        _step, carry, jnp.arange(num_timesteps)
    )

    post = HMMPosteriorFiltered(
        marginal_loglik=log_normalizer,
        filtered_probs=filtered_probs,
        predicted_probs=predicted_probs,
    )
    return post


@partial(jit, static_argnames=["transition_fn"])
def hmm_two_filter_smoother(
    initial_distribution: Float[Array, " num_states"],
    transition_matrix: Optional[
        Union[
            Float[Array, "num_states num_states"],
            Float[Array, "num_timesteps_minus_1 num_states num_states"],
        ]
    ],
    log_likelihoods: Float[Array, "num_timesteps num_states"],
    transition_fn: Optional[Callable[[IntScalar], Float[Array, "num_states num_states"]]] = None,
    compute_trans_probs: bool = True,
    occupancy_bias: Union[Scalar, Float[Array, "num_states"]] = 0.0,
) -> HMMPosterior:
    post = hmm_filter(
        initial_distribution, transition_matrix, log_likelihoods, transition_fn, occupancy_bias
    )
    ll = post.marginal_loglik
    filtered_probs, predicted_probs = post.filtered_probs, post.predicted_probs

    _, backward_pred_probs = hmm_backward_filter(
        transition_matrix, log_likelihoods, transition_fn, occupancy_bias
    )

    # Compute smoothed probabilities
    smoothed_probs = filtered_probs * backward_pred_probs
    norm = smoothed_probs.sum(axis=1, keepdims=True)
    smoothed_probs /= norm

    posterior = HMMPosterior(
        marginal_loglik=ll,
        filtered_probs=filtered_probs,
        predicted_probs=predicted_probs,
        smoothed_probs=smoothed_probs,
        initial_probs=smoothed_probs[0],
    )

    # Compute the transition probabilities if specified
    if compute_trans_probs:
        trans_probs = compute_transition_probs(transition_matrix, posterior, transition_fn)
        posterior = posterior._replace(trans_probs=trans_probs)

    return posterior


def hellinger2_distance(p: Float[Array, "num_classes"], q: Float[Array, "num_classes"]) -> Float:
    return jnp.sum((jnp.sqrt(p) - jnp.sqrt(q)) ** 2) * 0.5


def kl_divergence(p: Float[Array, "num_classes"], q: Float[Array, "num_classes"]) -> Float:
    d1 = jnp.sum(kl_div(p + 1e-10, q + 1e-10))
    d2 = jnp.sum(kl_div(q + 1e-10, p + 1e-10))
    return (d1 + d2) / 2


def total_variation_distance(
    p: Float[Array, "num_classes"], q: Float[Array, "num_classes"]
) -> Float:
    return 0.5 * jnp.sum(jnp.abs(p - q))


def l2(p: Float[Array, "num_classes"], q: Float[Array, "num_classes"]) -> Float:
    return jnp.sqrt(jnp.sum((p - q) ** 2))


def divergence_e(e_0: Float[Array, "num_classes"], e_1: Float[Array, "num_classes"]) -> Float:
    return hellinger2_distance(e_1, e_0)


class PhlagHMMTransitions(HMMTransitions):
    def __init__(
        self,
        num_states: int,
        concentration: Union[Scalar, Float[Array, "num_states num_states"]] = 1.1,
    ):
        self.num_states = num_states
        self.concentration = concentration * jnp.ones((num_states, num_states))

    def distribution(
        self, params: ParamsStandardHMMTransitions, state: IntScalar, inputs=None
    ) -> tfd.Distribution:
        return tfd.Categorical(probs=params.transition_matrix[state])

    def initialize(
        self,
        key: Optional[PRNGKeyT] = None,
        method: str = "prior",
        transition_matrix: Optional[Float[Array, "num_states num_states"]] = None,
    ) -> Tuple[ParamsStandardHMMTransitions, ParamsStandardHMMTransitions]:
        if transition_matrix is None:
            if method.lower() == "prior":
                if key is None:
                    raise ValueError("A key required if transition matrix not provided")
                tm_sample = tfd.Dirichlet(self.concentration).sample(seed=key)
                transition_matrix = cast(Float[Array, "num_states num_states"], tm_sample)
            else:
                raise Exception("Invalid initialization method: {}".format(method))
        else:
            assert transition_matrix.shape == (self.num_states, self.num_states)
        params = ParamsStandardHMMTransitions(transition_matrix=transition_matrix)
        props = ParamsStandardHMMTransitions(
            transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())
        )
        return params, props

    def log_prior(self, params: ParamsStandardHMMTransitions) -> Scalar:
        return tfd.Dirichlet(self.concentration).log_prob(params.transition_matrix).sum()

    def _compute_transition_matrices(
        self, params: ParamsStandardHMMTransitions, inputs=None
    ) -> Float[Array, "num_states num_states"]:
        return params.transition_matrix

    def collect_suff_stats(
        self, params: ParamsStandardHMMTransitions, posterior: HMMPosterior, inputs=None
    ):
        return posterior.trans_probs

    def initialize_m_step_state(
        self, params: ParamsStandardHMMTransitions, props: ParamsStandardHMMTransitions
    ) -> Any:
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
                transition_matrix = tfd.Dirichlet(self.concentration + expected_trans_counts).mode()
            params = params._replace(transition_matrix=transition_matrix)
        return params, m_step_state


class ParamsCategoricalHMMEmissions(NamedTuple):
    probs: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]


class EmissionParam(Enum):
    ANCHOR = "anchor"
    REPULSION = "repulsion"
    ATTRACTION = "attraction"
    FREE = "free"


class PhlagHMMEmissions(HMMEmissions):
    def __init__(
        self,
        num_states: int,
        emission_dim: int,
        num_classes: int,
        penalty_lambda: float,
        parameterization: Tuple[EmissionParam, ...],
        concentration: Union[Scalar, Float[Array, "num_classes"]] = 1.1,
    ):
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.num_classes = num_classes
        self.penalty_lambda = penalty_lambda
        self.parameterization = parameterization
        self.concentration = concentration * jnp.ones(self.num_classes)

    def set_emission_prior_concentration(
        self, concentration: Union[Scalar, Float[Array, "num_classes"]]
    ):
        self.concentration = concentration

    @property
    def emission_shape(self) -> Tuple[int]:
        return (self.emission_dim,)

    def distribution(
        self, params: ParamsCategoricalHMMEmissions, state: IntScalar, inputs=None
    ) -> tfd.Distribution:
        return tfd.Independent(
            tfd.Categorical(probs=params.probs[state]), reinterpreted_batch_ndims=1
        )

    def log_prior(self, params: ParamsCategoricalHMMEmissions) -> Scalar:
        return tfd.Dirichlet(self.concentration).log_prob(params.probs).sum()

    def initialize(
        self,
        key: Optional[PRNGKeyT] = jr.PRNGKey(0),
        method: str = "prior",
        emission_probs: Optional[Float[Array, "num_states emission_dim num_classes"]] = None,
    ) -> Tuple[ParamsCategoricalHMMEmissions, ParamsCategoricalHMMEmissions]:
        if emission_probs is None:
            if method.lower() == "prior":
                if key is None:
                    raise ValueError("A key must be provided when emissions is not")
                prior = tfd.Dirichlet(self.concentration)
                emission_probs = prior.sample(
                    seed=key, sample_shape=(self.num_states, self.emission_dim)
                )
            else:
                raise Exception("Invalid initialization method: {}".format(method))
        else:
            assert emission_probs.shape == (self.num_states, self.emission_dim, self.num_classes)
            assert jnp.all(emission_probs >= 0)
            assert jnp.allclose(
                jnp.sum(emission_probs, axis=-1), jnp.ones(emission_probs.shape[:-1]), atol=1e-03
            )
            emission_probs = emission_probs / jnp.sum(emission_probs, axis=-1)[:, :, None]

        params = ParamsCategoricalHMMEmissions(probs=emission_probs)
        props = ParamsCategoricalHMMEmissions(
            probs=ParameterProperties(constrainer=tfb.SoftmaxCentered())
        )
        return params, props

    def collect_suff_stats(
        self,
        params: ParamsCategoricalHMMEmissions,
        posterior: HMMPosterior,
        emissions: Array,
        inputs=None,
    ) -> dict:
        return dict(
            sum_x=jnp.einsum(
                "tk,tdi->kdi", posterior.smoothed_probs, one_hot(emissions, self.num_classes)
            )
        )

    def initialize_m_step_state(
        self, params: ParamsCategoricalHMMEmissions, props: ParamsCategoricalHMMEmissions
    ) -> Any:
        return None

    def update_m_step_state(
        self, params: ParamsCategoricalHMMEmissions, props: ParamsCategoricalHMMEmissions
    ) -> Any:
        return None

    def map_estimate_kl(
        self,
        e_est: Float[Array, "num_classes"],
        cstats: Float[Array, "num_classes"],
        reverse: bool = False,
    ) -> Float[Array, "num_classes"]:
        """
        Regularized estimate for a categorical distribution.
        reverse = False:
            Forward KL similarity (Bayesian Dirichlet MAP, closed form)
        reverse = True:
            Forward KL dissimilarity (repulsion), solved numerically
        """
        eps = 1e-12
        max_iter = 100
        if not reverse:
            alpha = self.penalty_lambda * e_est
            v = cstats + alpha - 1.0
            v = jnp.maximum(v, eps)
            e_curr = v / jnp.sum(v)
        else:

            def objective(theta):
                p = jax.nn.softmax(theta)
                ll = jnp.sum(cstats * jnp.log(p + eps))
                kl = jnp.sum(p * (jnp.log(p + eps) - jnp.log(e_est + eps)))
                return -(ll + self.penalty_lambda * kl)

            grad_obj = jax.grad(objective)
            theta = cstats / jnp.sum(cstats)
            lr = 0.1
            for _ in range(max_iter):
                theta = theta - lr * grad_obj(theta)

            e_curr = jax.nn.softmax(theta)
        return e_curr

    def map_estimate_bfgs(
        self,
        e_est: Float[Array, "num_classes"],
        cstats: Float[Array, "num_classes"],
        inverse: bool = False,
    ) -> Float[Array, "num_classes"]:
        def neg_log_posterior(logits: Float[Array, "num_classes"]) -> Float[Array, ""]:
            e_curr = softmax(logits)
            neg_log_likelihood = -jnp.sum(cstats * jnp.log(e_curr + 1e-10))
            if inverse:
                neg_log_prior = self.penalty_lambda * (1 - divergence_e(e_est, e_curr))
            else:
                neg_log_prior = self.penalty_lambda * divergence_e(e_est, e_curr)
            return neg_log_likelihood + neg_log_prior

        e_init = e_est
        result = minimize(fun=neg_log_posterior, x0=e_init, method="BFGS", tol=1e-5)
        return softmax(result.x)

    def map_estimate_lagrange(
        self,
        e_est: Float[Array, "num_classes"],
        cstats: Float[Array, "num_classes"],
        inverse: bool = False,
    ) -> Float[Array, "num_classes"]:
        """
        MAP estimate using Lagrange multiplier method with inverse support using dot product repulsion
        """
        max_iter = 1000
        tol = 1e-6

        def f1(nu):
            return jnp.sum(cstats / (nu + self.penalty_lambda * e_est)) - 1.0

        def f0(nu):
            return jnp.sum(cstats / (nu - self.penalty_lambda * e_est)) - 1.0

        if inverse:
            f = f1
            lower = 0.0
        else:
            f = f0
            lower = self.penalty_lambda * jnp.max(e_est) + 1e-10

        upper = jnp.sum(cstats) * 10.0
        nu_low, nu_high, _ = while_loop(
            lambda val: (val[2] < max_iter) & ((val[1] - val[0]) > tol),
            lambda val: (
                jnp.where(f((val[0] + val[1]) / 2) > 0, (val[0] + val[1]) / 2, val[0]),
                jnp.where(f((val[0] + val[1]) / 2) < 0, (val[0] + val[1]) / 2, val[1]),
                val[2] + 1,
            ),
            (lower, upper, 0),
        )
        nu_opt = (nu_low + nu_high) / 2

        if inverse:
            q = cstats / (nu_opt + self.penalty_lambda * e_est)
        else:
            q = cstats / (nu_opt - self.penalty_lambda * e_est)

        return q / (jnp.sum(q) + 1e-10)

    def m_step(
        self,
        params: ParamsCategoricalHMMEmissions,
        props: ParamsCategoricalHMMEmissions,
        batch_stats: dict,
        m_step_state: Union[Scalar, Float[Array, "emission_dim num_classes"]],
    ) -> Tuple[
        ParamsCategoricalHMMEmissions, Union[Scalar, Float[Array, "emission_dim num_classes"]]
    ]:
        # Tuple[ParamsCategoricalHMMEmissions, Any]:
        emission_stats = pytree_sum(batch_stats, axis=0)
        S = self.concentration + emission_stats["sum_x"]
        probs = tfd.Dirichlet(S).mode()
        if props.probs.trainable:
            emission_stats = pytree_sum(batch_stats, axis=0)
            S = self.concentration + emission_stats["sum_x"]
            probs = tfd.Dirichlet(S).mode()
            if self.parameterization[1] is EmissionParam.FREE:
                pass
            elif self.parameterization[1] is EmissionParam.REPULSION:
                # Not recommended...
                probs = probs.at[1].set(
                    jax.vmap(
                        lambda x, y: self.map_estimate_kl(x, y, True),
                        # lambda x, y: self.map_estimate_bfgs(x, y, True),
                        # lambda x, y: self.map_estimate_lagrange(x, y, True),
                        in_axes=(0, 0),
                        out_axes=0,
                    )(m_step_state, S[1])
                )
            else:
                raise ValueError(f"Invalid emission parameterization for : {self.parameterization}")

            if self.parameterization[0] is EmissionParam.FREE:
                pass
            elif self.parameterization[0] is EmissionParam.ATTRACTION:
                probs = probs.at[0].set(
                    jax.vmap(
                        self.map_estimate_kl,
                        # self.map_estimate_bfgs,
                        # self.map_estimate_lagrange,
                        in_axes=(0, 0),
                        out_axes=0,
                    )(m_step_state, S[0])
                )
            elif self.parameterization[0] is EmissionParam.ANCHOR:
                probs = probs.at[0].set(m_step_state)
            else:
                raise ValueError(f"Invalid emission parameterization: {self.parameterization}")
            params = params._replace(probs=probs)
        return params, m_step_state

    def state_divergence(self, params: ParamsCategoricalHMMEmissions) -> Float:
        probs = params.probs
        total_divergence = 0
        for i in range(probs.shape[0]):
            for j in range(i, probs.shape[0]):
                total_divergence += jax.vmap(divergence_e, in_axes=(0, 0), out_axes=0)(
                    probs[i], probs[j]
                )
        return total_divergence


class ParamsPhlagHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsCategoricalHMMEmissions


class PhlagHMM(HMM):
    def __init__(
        self,
        num_states: int = 2,
        emission_dim: int = 1,
        num_classes: int = 2,
        emission_lambda: Scalar = 1,
        emission_parameterization: Tuple[EmissionParam, ...] = None,
        emission_concetration: Union[Scalar, Float[Array, "num_classes"]] = 1.1,
        initial_probs_concetration: Union[Scalar, Float[Array, "num_states"]] = 1.1,
        transition_concentration: Union[Scalar, Float[Array, "num_states num_states"]] = 1.1,
        occupancy_bias: Union[Scalar, Float[Array, "num_states"]] = 0.0,
        **kwargs,
    ):
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.num_classes = num_classes
        self.emission_lambda = emission_lambda
        if emission_parameterization is not None:
            self.emission_parameterization = emission_parameterization
        else:
            self.emission_parameterization = (EmissionParam.FREE,) * self.num_states

        self.emission_concetration = emission_concetration
        self.initial_probs_concetration = initial_probs_concetration
        self.transition_concentration = transition_concentration

        self.initial_probs_m_step_state = None
        self.emissions_m_step_state = None
        self.transitions_m_step_state = None
        self.occupancy_bias = occupancy_bias

        self.initial_component = StandardHMMInitialState(
            num_states=self.num_states, initial_probs_concentration=self.initial_probs_concetration
        )
        self.transition_component = PhlagHMMTransitions(
            num_states=self.num_states, concentration=self.transition_concentration
        )
        self.emission_component = PhlagHMMEmissions(
            self.num_states,
            self.emission_dim,
            self.num_classes,
            penalty_lambda=self.emission_lambda,
            concentration=self.emission_concetration,
            parameterization=self.emission_parameterization,
        )
        super().__init__(
            num_states=self.num_states,
            initial_component=self.initial_component,
            transition_component=self.transition_component,
            emission_component=self.emission_component,
        )

    def initialize(
        self,
        key: PRNGKeyT = jr.PRNGKey(0),
        method: str = "prior",
        emission_probs: Optional[Float[Array, "num_states emission_dim num_classes"]] = None,
        initial_probs: Optional[Float[Array, "num_states"]] = None,
        transition_matrix: Optional[Float[Array, "num_states num_states"]] = None,
    ) -> Tuple[ParamsPhlagHMM, ParamsPhlagHMM]:
        key1, key2, key3 = jr.split(key, 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(
            key1, method=method, initial_probs=initial_probs
        )
        params["transitions"], props["transitions"] = self.transition_component.initialize(
            key2, method=method, transition_matrix=transition_matrix
        )
        params["emissions"], props["emissions"] = self.emission_component.initialize(
            key3, method=method, emission_probs=emission_probs
        )
        return ParamsPhlagHMM(**params), ParamsPhlagHMM(**props)

    def initialize_m_step_state(
        self,
        params: HMMParameterSet,
        props: HMMPropertySet,
        initial_probs_m_step_state=None,
        transitions_m_step_state=None,
        emissions_m_step_state=None,
    ) -> Tuple[Any, Any, Any]:
        if initial_probs_m_step_state is not None:
            self.initial_probs_m_step_state = initial_probs_m_step_state
        if transitions_m_step_state is not None:
            self.transitions_m_step_state = transitions_m_step_state
        if emissions_m_step_state is not None:
            self.emissions_m_step_state = emissions_m_step_state

        if self.initial_probs_m_step_state is None:
            self.initial_probs_m_step_state = self.initial_component.initialize_m_step_state(
                params.initial, props.initial
            )
        if self.transitions_m_step_state is None:
            self.transitions_m_step_state = self.transition_component.initialize_m_step_state(
                params.transitions, props.transitions
            )
        if self.emissions_m_step_state is None:
            self.emissions_m_step_state = self.emission_component.initialize_m_step_state(
                params.emissions, props.emissions
            )
        return (
            self.initial_probs_m_step_state,
            self.transitions_m_step_state,
            self.emissions_m_step_state,
        )

    def state_emission_divergence(self, params: HMMParameterSet) -> Float[Array, "emission_dim"]:
        return self.emission_component.state_divergence(params.emissions)

    def m_step(
        self, params: HMMParameterSet, props: HMMPropertySet, batch_stats: PyTree, m_step_state: Any
    ) -> Tuple[HMMParameterSet, Any]:
        batch_initial_stats, batch_transition_stats, batch_emission_stats = batch_stats
        initial_probs_m_step_state, transitions_m_step_state, emissions_m_step_state = m_step_state

        initial_params, initial_probs_m_step_state = self.initial_component.m_step(
            params.initial, props.initial, batch_initial_stats, initial_probs_m_step_state
        )
        transition_params, transitions_m_step_state = self.transition_component.m_step(
            params.transitions, props.transitions, batch_transition_stats, transitions_m_step_state
        )
        emission_params, emissions_m_step_state = self.emission_component.m_step(
            params.emissions, props.emissions, batch_emission_stats, emissions_m_step_state
        )
        params = params._replace(
            initial=initial_params, transitions=transition_params, emissions=emission_params
        )
        m_step_state = (
            initial_probs_m_step_state,
            transitions_m_step_state,
            emissions_m_step_state,
        )
        return params, m_step_state

    def e_step(
        self,
        params: HMMParameterSet,
        emissions: Array,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    ) -> Tuple[PyTree, Scalar]:
        args = self._inference_args(params, emissions, inputs)
        posterior = hmm_two_filter_smoother(*args, occupancy_bias=self.occupancy_bias)

        initial_stats = self.initial_component.collect_suff_stats(params.initial, posterior, inputs)
        transition_stats = self.transition_component.collect_suff_stats(
            params.transitions, posterior, inputs
        )
        emission_stats = self.emission_component.collect_suff_stats(
            params.emissions, posterior, emissions, inputs
        )
        return (initial_stats, transition_stats, emission_stats), posterior.marginal_loglik
