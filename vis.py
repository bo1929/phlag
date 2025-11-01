import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from dynamax.utils.plotting import CMAP, COLORS, white_to_color_cmap

mpl.rcParams.update({"font.size": 22})
import jax.numpy as jnp


def plot_emissions(emissions, title="Emissions", xlabel="Position"):
    plt.figure(figsize=(30, 5))
    assert len(emissions.shape) == 3
    for i in range(emissions.shape[1]):
        for j in range(emissions.shape[2]):
            pass

    plt.plot(emissions, "-k")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.show()


def plot_hmm_wseq(
    hmm, params, emissions, x, states, xlim=None, title="Emissions", xlabel="Position"
):
    num_timesteps = emissions.shape[0]
    emission_dim = hmm.emission_dim
    try:
        means = params.emissions.means[states]
    except:
        None
    ulim = 1.05 * (emissions).max()
    llim = 1.05 * (emissions).min()
    fig, axs = plt.subplots(emissions.shape[-1], 1, sharex=True, figsize=(30, 5))
    for d in range(emissions.shape[-1]):
        axs[d].imshow(
            states[None, :],
            aspect="auto",
            interpolation="none",
            cmap=CMAP,
            vmin=0,
            vmax=len(COLORS) - 1,
            alpha=0.25,
            extent=(
                0,
                num_timesteps,
                1.05 * emissions[:, d].min(),
                1.05 * emissions[:, d].max(),
            ),
        )
        axs[d].plot(emissions[:, d], "-k", linewidth=0.25)
        try:
            axs[d].plot(means[:, d], ":k")
        except:
            None
        axs[d].set_ylabel("$y_{{t,{} }}$".format(d + 1))

    if xlim is None:
        plt.xlim(0, num_timesteps)
    else:
        plt.xlim(xlim)
    plt.xticks(
        ticks=plt.xticks()[0][1:-1],
        labels=((x[np.array(plt.xticks()[0][1:-1], dtype=int)])).astype(int),
    )
    axs[-1].set_xlabel(xlabel)
    axs[0].set_title(title)
    plt.tight_layout()
    plt.show()


def plot_gaussian_hmm(
    hmm, params, emissions, states, title="Emission Distributions", alpha=0.25
):
    num_timesteps = emissions.shape[0]
    emission_dim = hmm.emission_dim
    XX, YY, ZZ = jnp.meshgrid(
        jnp.linspace(emissions[:, 0].min(), emissions[:, 0].max(), 200),
        jnp.linspace(emissions[:, 1].min(), emissions[:, 1].max(), 200),
        jnp.linspace(emissions[:, 2].min(), emissions[:, 2].max(), 200),
    )
    grid = jnp.column_stack((XX.ravel(), YY.ravel(), ZZ.ravel()))

    fig, axs = plt.subplots(1, emission_dim, figsize=(30, 5))
    for k in range(hmm.num_states):
        lls = hmm.emission_distribution(params, k).log_prob(grid)
        axs[0].contour(
            XX[:, :, 0],
            YY[:, :, 0],
            (jnp.exp(lls).reshape(XX.shape)).sum(axis=2),
            cmap=white_to_color_cmap(COLORS[k]),
        )
        # axs[0].plot(emissions[states == k, 0], emissions[states == k, 1], "o", mfc=COLORS[k], mec="none", ms=1, alpha=alpha)
        axs[1].contour(
            XX[:, :, 0],
            ZZ[:, :, 0],
            (jnp.exp(lls).reshape(XX.shape)).sum(axis=1),
            cmap=white_to_color_cmap(COLORS[k]),
        )
        # axs[1].plot(emissions[states == k, 0], emissions[states == k, 2], "o", mfc=COLORS[k], mec="none", ms=1, alpha=alpha)
        axs[2].contour(
            YY[0, :, :],
            ZZ[0, :, :],
            (jnp.exp(lls).reshape(XX.shape)).sum(axis=0),
            cmap=white_to_color_cmap(COLORS[k]),
        )
        # axs[2].plot(emissions[states == k, 1], emissions[states == k, 2], "o", mfc=COLORS[k], mec="none", ms=1, alpha=alpha)

    # plt.xlabel("$y_1$")
    # plt.ylabel("$y_2$")
    # plt.title(title)
    # plt.gca().set_aspect(1.0)
    # plt.tight_layout()
