import sys
import sklearn.cluster as cluster
import sklearn.mixture as mixture
import jax.numpy as jnp
import jax.lax as lax
import math

from jaxtyping import Array, Float
from sklearn.metrics import silhouette_samples, silhouette_score


PEPS = 1e-3
PCOUNT = 1


class Discretization:
    def __init__(self, emission_dim: int, input_dim: int, num_classes: int):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.num_classes = num_classes

    def get_num_classes(self):
        return self.num_classes


class BinDiscretization(Discretization):
    def __init__(self, emission_dim: int, input_dim: int, num_bins: int):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.num_bins = num_bins
        self.num_classes = (num_bins) ** input_dim
        self.bins = jnp.empty((self.emission_dim, self.input_dim, self.num_bins + 1))

    def discretize_obs(self, obs):
        obsd = jnp.zeros((obs.shape[0], self.emission_dim))
        for i in range(self.emission_dim):
            for j in range(self.input_dim):
                obsd = obsd.at[:, i].set(obsd[:, i] + (jnp.digitize(obs[:, i, j], self.bins[i, j]) - 1).astype(int) * ((self.num_bins) ** (self.input_dim - j - 1)))
        return obsd.astype(int)


class EqualBinDiscretization(BinDiscretization):
    def fit_null_emission(self, null_sample: Float[Array, "emission_dim input_dim _"]):
        assert null_sample.shape[1] == self.emission_dim
        assert null_sample.shape[2] == self.input_dim
        null_emission_prob = jnp.empty((self.emission_dim, self.num_classes))
        eps = 1e-5
        for i in range(self.emission_dim):
            hist, b = jnp.histogramdd(null_sample[:, i, :], bins=[self.num_bins for _ in range(self.input_dim)], range=[(0 - eps, 1 + eps) for _ in range(self.input_dim)])
            hist = (hist / hist.sum()).reshape(-1)
            null_emission_prob = null_emission_prob.at[i, :].set(hist)
            self.bins = self.bins.at[i, :].set(b)
        return null_emission_prob


class FixedBinDiscretization(BinDiscretization):
    def __init__(self, emission_dim: int, input_dim: int, bins: Float[Array, "input_dim _"]):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.num_bins = bins.shape[-1] - 1
        self.num_classes = (self.num_bins) ** input_dim
        self.bins = bins

    def fit_null_emission(self, null_sample: Float[Array, "emission_dim input_dim _"]):
        assert null_sample.shape[1] == self.emission_dim
        assert null_sample.shape[2] == self.input_dim
        null_emission_prob = jnp.empty((self.emission_dim, self.num_classes))
        eps = 1e-5
        for i in range(self.emission_dim):
            hist, b = jnp.histogramdd(null_sample[:, i, :], bins=self.bins[i])
            hist = (hist / hist.sum()).reshape(-1)
            null_emission_prob = null_emission_prob.at[i, :].set(hist)
            self.bins = self.bins.at[i, :].set(b)
            self.bins = self.bins.at[i, :, 0].set(self.bins[i, :, 0] - eps)
            self.bins = self.bins.at[i, :, -1].set(self.bins[i, :, -1] + eps)
        return null_emission_prob


class ClusterDiscretization(Discretization):
    def __init__(self, emission_dim: int, input_dim: int, num_classes: int):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.f = []

    def choose_num_classes(self, data, range_classes=(2, 81)):
        # TODO: Work on this!
        assert len(self.f) > 0
        assert range_classes[0] < range_classes[1]
        assert range_classes[0] > 1
        best_num_classes, best_sill_avg = range_classes[0], -1 + 1e-5
        j = range_classes[0]
        range_max = range_classes[1] + int(math.sqrt(range_classes[1]))
        while j < range_max:
            sill_avg = 0
            for i in range(len(self.f)):
                self.f[i].set_params(n_clusters=j)
                pred = self.f[i].fit_predict(data[:, i, :])
                # if jnp.unique(pred).shape[0] == j:
                if jnp.unique(pred).shape[0] == j:  # TODO: Remove this!
                    sill_avg += silhouette_score(data[:, i, :], pred)
                else:
                    sill_avg += 0  # TODO: or should we return -1?
            # if jnp.unique(pred).shape[0] != j: # TODO: Remove this!
            #     break
            sill_avg /= len(self.f)
            if sill_avg >= best_sill_avg:
                best_sill_avg = sill_avg
                best_num_classes = j
            if ((sill_avg) > 0.99) or (sill_avg < 1e-5):
                break
            print(f"For {j} classes, the avg. silhouette score: {sill_avg}", file=sys.stderr)
            j += int(math.sqrt(j))
        print(f"Best: {best_num_classes} classes, the avg. silhouette score: {best_sill_avg}", file=sys.stderr)
        self.num_classes = best_num_classes
        for i in range(len(self.f)):
            self.f[i].set_params(n_clusters=self.num_classes)

    def pairwise_cluster_distances(self):
        pw_dist = []
        for i in range(self.emission_dim):
            pwd = []
            for j in range(self.num_classes):
                pwd_i = []
                for k in range(self.num_classes):
                    pwd_i.append(math.sqrt(sum((self.f[i].cluster_centers_[j, :] - self.f[i].cluster_centers_[k, :]) ** 2)))
                pwd.append(pwd_i)
            pw_dist.append(pwd)
        return pw_dist

    def discretize_obs(self, obs):
        assert len(self.cl) == self.emission_dim
        obsd = jnp.zeros((obs.shape[0], self.emission_dim))
        for i in range(self.emission_dim):
            obsd = obsd.at[:, i].set(lax.map(lambda x: jnp.argwhere(self.cl[i][0] == x, size=1)[0], self.f[i].predict(obs[:, i, :])).reshape(-1))
            # obsd = obsd.at[:, i].set(self.f[i].predict(obs[:, i, :]))
        return obsd.astype(int)


class KMeansDiscretization(ClusterDiscretization):
    def __init__(self, emission_dim: int, input_dim: int, num_classes: int):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.cl = []
        self.f = [cluster.KMeans(n_clusters=self.num_classes) for _ in range(emission_dim)]

    def fit_discretization(self, observed_freqs: Float[Array, "_ emission_dim input_dim"]):
        assert observed_freqs.shape[1] == self.emission_dim
        assert observed_freqs.shape[2] == self.input_dim
        for i in range(self.emission_dim):
            self.f[i].fit(observed_freqs[:, i, :])
            c = jnp.unique(self.f[i].predict(observed_freqs[:, i, :]), return_counts=True)
            self.cl.append(c)
            curr_num_classes = c[0].shape[0]
            self.num_classes = curr_num_classes if i == 0 else max(self.num_classes, curr_num_classes)

    def compute_null_emission_prob(self, simulated_freqs: Float[Array, "_ emission_dim input_dim"]):
        assert simulated_freqs.shape[1] == self.emission_dim
        assert simulated_freqs.shape[2] == self.input_dim
        null_emission_prob = jnp.zeros((self.emission_dim, self.num_classes))
        simulated_emissions = self.discretize_obs(simulated_freqs)
        for i in range(self.emission_dim):
            k, v = jnp.unique(simulated_emissions[:, i], return_counts=True)
            c = zip(k, v)
            for k, v in c:
                null_emission_prob = null_emission_prob.at[i, k].set(v)
            # null_emission_prob = null_emission_prob.at[i,:].set(null_emission_prob[i, :] + PCOUNT * (null_emission_prob[i, :] < 1e-4))
            null_emission_prob = null_emission_prob.at[i, :].set(null_emission_prob[i, :] / jnp.sum(null_emission_prob[i, :]))
            null_emission_prob = null_emission_prob.at[i, :].set(null_emission_prob[i, :] + PEPS) / jnp.sum(null_emission_prob[i, :] + PEPS)
        return null_emission_prob
