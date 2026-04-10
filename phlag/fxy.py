import jax
import jax.numpy as jnp

from jaxtyping import Array, Float


PEPS = 1e-5


class DTO:
    def __init__(self):
        self.perm_to_dist = jnp.array([
            [0, 1.0 / 3.0, 2.0 / 3.0],
            [0, 2.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 0, 2.0 / 3.0],
            [1.0 / 3.0, 2.0 / 3.0, 0],
            [2.0 / 3.0, 0, 1.0 / 3.0],
            [2.0 / 3.0, 1.0 / 3.0, 0],
        ])
        # Lookup: encode permutation (a,b,c) as a*9+b*3+c -> class index
        self._perm_lookup = jnp.full(27, -1, dtype=jnp.int32)
        for cls, (a, b, c) in enumerate([(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]):
            self._perm_lookup = self._perm_lookup.at[a * 9 + b * 3 + c].set(cls)

    def discretize_qqs(self, qqs):
        # qqs: (n_gt, emission_dim, 3)
        sorted_idx = jnp.argsort(qqs, axis=-1)  # (n_gt, emission_dim, 3)
        encoded = sorted_idx[..., 0] * 9 + sorted_idx[..., 1] * 3 + sorted_idx[..., 2]
        return self._perm_lookup[encoded]  # (n_gt, emission_dim)

    def get_num_classes(self):
        return self.perm_to_dist.shape[0]

    def compute_emission_prob(self, qqs: Float[Array, "_ emission_dim input_dim"]):
        emission_prob = jnp.zeros((qqs.shape[1], self.get_num_classes()))
        emissions = self.discretize_qqs(qqs)
        for i in range(qqs.shape[1]):
            k, v = jnp.unique(emissions[:, i], return_counts=True)
            c = zip(k, v)
            for k, v in c:
                emission_prob = emission_prob.at[i, k].set(v)
            emission_prob = emission_prob.at[i, :].set(
                emission_prob[i, :] / jnp.sum(emission_prob[i, :])
            )
            row = emission_prob[i, :] + PEPS
            emission_prob = emission_prob.at[i, :].set(row / jnp.sum(row))
        return emission_prob

    def get_cost_matrices(self, emission_dim=1):
        p = self.perm_to_dist  # (6, 3)
        # Hellinger distance between all pairs
        diff = jnp.sqrt(p[:, None, :]) - jnp.sqrt(p[None, :, :])  # (6, 6, 3)
        D = jnp.sqrt(jnp.sum(diff ** 2, axis=-1)) / jnp.sqrt(2.0)  # (6, 6)
        return jnp.broadcast_to(D, (emission_dim, *D.shape))


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

    def discretize_qqs(self, qqs):
        d = lambda x, y: jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1))
        return jnp.argmin(jax.vmap(d, (None, 0), 1)(qqs, self.bin_centers), axis=1)


class BinDiscretization:
    def __init__(self, emission_dim: int, input_dim: int, bins: Float[Array, "input_dim _"]):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.num_bins = bins.shape[-1] - 1
        self.num_classes = (self.num_bins) ** input_dim
        self.bins = bins

    def get_num_classes(self):
        return self.num_classes

    def discretize_qqs(self, obs):
        obsd = jnp.zeros((obs.shape[0], self.emission_dim))
        for i in range(self.emission_dim):
            for j in range(self.input_dim):
                obsd = obsd.at[:, i].set(
                    obsd[:, i]
                    + (jnp.digitize(obs[:, i, j], self.bins[i, j]) - 1).astype(int)
                    * ((self.num_bins) ** (self.input_dim - j - 1))
                )
        return obsd.astype(int)

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
