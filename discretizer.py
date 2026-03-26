import sys
import math

import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float

import sklearn.cluster as cluster
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

PEPS = 1e-5
PCOUNT = 1


class Discretizer:
    def __init__(self):
        pass

    def compute_emission_prob(self, freqs):
        emission_prob = jnp.zeros((freqs.shape[1], self.get_num_classes()))
        emissions = self.discretize_freqs(freqs)
        for i in range(freqs.shape[1]):
            k, v = jnp.unique(emissions[:, i], return_counts=True)
            c = zip(k, v)
            for k, v in c:
                emission_prob = emission_prob.at[i, k].set(v)
            emission_prob = emission_prob.at[i, :].set(
                emission_prob[i, :] / jnp.sum(emission_prob[i, :])
            )
            emission_prob = emission_prob.at[i, :].set(emission_prob[i, :] + PEPS) / jnp.sum(
                emission_prob[i, :] + PEPS
            )
        return emission_prob


class DTO:
    def __init__(self):
        self.perm_to_int = {
            (0, 1, 2): 0,
            (0, 2, 1): 1,
            (1, 0, 2): 2,
            (1, 2, 0): 3,
            (2, 0, 1): 4,
            (2, 1, 0): 5,
        }
        self.perm_to_dist = {
            0: jnp.array([0, 1.0 / 3.0, 2.0 / 3.0]),
            1: jnp.array([0, 2.0 / 3.0, 1.0 / 3.0]),
            2: jnp.array([1.0 / 3.0, 0, 2.0 / 3.0]),
            3: jnp.array([1.0 / 3.0, 2.0 / 3.0, 0]),
            4: jnp.array([2.0 / 3.0, 0, 1.0 / 3.0]),
            5: jnp.array([2.0 / 3.0, 2.0 / 3.0, 0]),
        }

    def get_num_classes(self):
        return len(self.perm_to_int)

    def discretize_freqs(self, freqs):
        emissions = []
        for i in range(freqs.shape[1]):
            emissions.append(
                jnp.array(
                    list(
                        map(
                            lambda x: self.perm_to_int[tuple(x.tolist())],
                            jnp.argsort(freqs[:, i, :], axis=-1),
                        )
                    )
                )
            )
        return jnp.stack(emissions, axis=1)

    def get_cost_matrices(self, emission_dim=1):
        C = jnp.zeros((emission_dim, self.get_num_classes(), self.get_num_classes()))
        for i in range(self.get_num_classes()):
            for j in range(self.get_num_classes()):
                p = self.perm_to_dist[i]
                q = self.perm_to_dist[j]
                # Hellinger distance
                C = C.at[:, i, j].set(
                    jnp.sqrt(jnp.sum((jnp.sqrt(p) - jnp.sqrt(q)) ** 2)) / jnp.sqrt(2.0)
                )
        return C


class BCB:
    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.bins, self.bin_centers = self.create_bins()

    def get_num_classes(self):
        return self.n_bins**2

    def create_bins(self):
        bins = []
        bin_centers = []
        for i in range(self.n_bins):
            for j in range(self.n_bins - i):
                x1, y1, z1 = (i / self.n_bins, j / self.n_bins, (self.n_bins - i - j) / self.n_bins)
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
                bin_centers.append(((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3))

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
                    bin_centers.append(((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3))
        return bins, jnp.array(bin_centers)

    def discretize_freqs(self, freqs):
        d = lambda x, y: jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1))
        return jnp.argmin(jax.vmap(d, (None, 0), 1)(freqs, self.bin_centers), axis=1)

    @staticmethod
    def choose_num_bins(freqs, range_bins=(3, 12)):
        assert range_bins[0] < range_bins[1]
        assert range_bins[0] > 1
        best_num_bins, best_sill_avg = range_bins[0], -1 + 1e-5
        j = range_bins[0]
        range_max = range_bins[1] + int(math.sqrt(range_bins[1]))
        while j < range_max:
            sill_avg = 0
            bcb = BCB(j)
            pred = bcb.assign_points(freqs[:, :, :])
            for i in range(freqs.shape[1]):
                if jnp.unique(pred[:, i]).shape[0] == 1:
                    sill_avg += 0
                else:
                    sill_avg += silhouette_score(freqs[:, i, :], pred[:, i])
            sill_avg /= freqs.shape[1]
            if sill_avg > best_sill_avg:
                best_sill_avg = sill_avg
                best_num_bins = j
            if ((sill_avg) > 0.99) or (sill_avg < 1e-5):
                break
            # print(
            #     f"For {j} bins, the avg. silhouette score: {sill_avg}",
            #     file=sys.stderr,
            # )
            j += int(math.sqrt(j))  # Regime for the optimal number of bins search
        # print(
        #     f"Best: {best_num_bins} bins, the avg. silhouette score: {best_sill_avg}",
        #     file=sys.stderr,
        # )
        return best_num_bins

    def get_cost_matrices(self, emission_dim=1):
        C = jnp.zeros((emission_dim, self.get_num_classes(), self.get_num_classes()))
        for j in range(C.shape[2]):
            p = self.bin_centers[i]
            q = self.bin_centers[j]
            # Hellinger distance
            C = C.at[:, i, j].set(
                jnp.sqrt(jnp.sum((jnp.sqrt(p) - jnp.sqrt(q)) ** 2)) / jnp.sqrt(2.0)
            )
        return C


class KMeansDiscretization:
    @ignore_warnings(category=ConvergenceWarning)
    def __init__(self, emission_dim: int, input_dim: int, num_classes: int):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.cl = []
        self.f = [cluster.KMeans(n_clusters=self.num_classes) for _ in range(emission_dim)]

    def get_num_classes(self):
        return self.num_classes

    @staticmethod
    def choose_num_classes(freqs, range_classes=(2, 81)):
        assert range_classes[0] < range_classes[1]
        assert range_classes[0] > 1
        best_num_classes, best_sill_avg = range_classes[0], -1 + 1e-5
        j = range_classes[0]
        range_max = range_classes[1] + int(math.sqrt(range_classes[1]))
        while j < range_max:
            f = cluster.KMeans(n_clusters=j)
            sill_avg = 0
            for i in range(freqs.shape[1]):
                pred = f.fit_predict(freqs[:, i, :])
                # if jnp.unique(pred).shape[0] != j: # TODO: Consider this!
                if jnp.unique(pred).shape[0] == 1:
                    sill_avg += 0
                else:
                    sill_avg += silhouette_score(freqs[:, i, :], pred)
            sill_avg /= freqs.shape[1]
            if sill_avg >= best_sill_avg:
                best_sill_avg = sill_avg
                best_num_classes = j
            if ((sill_avg) > 0.99) or (sill_avg < 1e-5):
                break
            # print(
            #     f"For {j} classes, the avg. silhouette score: {sill_avg}",
            #     file=sys.stderr,
            # )
            j += int(math.sqrt(j))
        # print(
        #     f"Best: {best_num_classes} classes, the avg. silhouette score: {best_sill_avg}",
        #     file=sys.stderr,
        # )
        return best_num_classes

    def get_cost_matrices(self, emission_dim=1):
        C = jnp.zeros((emission_dim, self.get_num_classes(), self.get_num_classes()))
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                for k in range(C.shape[2]):
                    d = self.f[i].cluster_centers_[j, :] - self.f[i].cluster_centers_[k, :]
                    C = C.at[i, j, k].set(jnp.sqrt(jnp.sum(d**2)))
        return C

    def discretize_freqs(self, freqs):
        assert freqs.shape[2] == self.input_dim
        assert freqs.shape[1] == self.emission_dim
        emissions = jnp.zeros((freqs.shape[0], freqs.shape[1]))
        for i in range(freqs.shape[1]):
            emissions = emissions.at[:, i].set(self.f[i].predict(freqs[:, i, :]))
        return emissions.astype(int)

    @ignore_warnings(category=ConvergenceWarning)
    def fit_discretization(self, freqs):
        for i in range(freqs.shape[1]):
            self.f[i].fit(freqs[:, i, :])
            c = jnp.unique(self.f[i].predict(freqs[:, i, :]), return_counts=True)
            self.cl.append(c)
