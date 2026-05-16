# Phlag

**Phlag** detects and flags phylogenetic anomalies across the genome.

Given a species tree and a sequence of gene trees, Phlag detects strong
deviations from the multi-species coalescent (MSC) using quadripartition quartet
scores (QQS) and a hidden Markov model (HMM).

The input is a sequence of gene trees, ideally sampled uniformly along a
chromosome. The output is a set of flagged subsequences corresponding to model
violations (e.g., introgression, recombination suppression, or dramatic changes
in effective population size), plus an updated species tree with branch lengths
in coalescent units (CU), estimated after excluding flagged gene trees.

## Installation

Phlag requires Python 3.9.

```shell
micromamba create -n phlag python=3.9 -y
micromamba activate phlag
git clone https://github.com/bo1929/phlag.git
cd phlag
pip install .
```

<!-- You can simply use pip. -->
<!-- ```shell -->
<!-- pip install phlag -->
<!-- ``` -->

<!-- Alternatively, install Phlag from the source. -->
<!-- ```shell -->
<!-- git clone https://github.com/bo1929/phlag.git -->
<!-- cd phlag -->
<!-- pip install . -->
<!-- ``` -->

## Quickstart with a toy example

The `test/` directory contains a simulated dataset based on a Neoaves species
tree with 191 taxa. We use it to demonstrate how Phlag detects a genomic region
where the MSC model is violated.

We have a species tree (`test/neoaves.nwk`) and 1500 gene trees
(`test/emission.gtrees`) ordered along a chromosome. The sequence is a mixture
of two coalescent-with-recombination (CwR) processes:

- **Background**: Gene trees simulated under the standard MSC with the original species tree parameters.
- **Anomalous**: Gene trees simulated with a 10-fold increase in effective population size on the branch leading to the *Charadriiformes* clade (labeled `N159`).

Out of the 1500 consecutive gene trees, 150 (10%) come from the anomalous process.
They form one contiguous block at indices [913, 1063) (0-indexed).
The goal is to recover that region with Phlag.

### Input

- **Species tree** (`-s`): A Newick species tree with labeled internal nodes.
- **Gene trees** (`-g`): One Newick tree per line, ordered along the genome (internal nodes need not be labeled).
- **Focal edge(s)** (`-e`): Label(s) of internal node(s) defining the edge(s) to target. In this example, the clade under suspicion is `N159`.

### Usage

```shell
phlag \
  -s test/neoaves.nwk \
  -g test/emission.gtrees \
  -e N159 \
  -L 10 \
  --read-qqs-path test/qqs.tsv \
  -o results-neoaves-N159.txt
```

Quadripartition quartet scores for this dataset are precomputed in
`test/qqs.tsv`. Using `--read-qqs-path` avoids recomputing QQS from scratch. If
you omit this flag, QQS values are computed on the fly, which increases runtime.
You can precompute and save QQS for later runs (e.g., different focal edges or
hyperparameters) with `--write-qqs-path`.

For all options, run `phlag --help`.

### Key hyperparameters

- **`--rho`** (default: `0.9`): Controls sensitivity; lower values flag more
regions. Interpreted as the expected fraction of gene trees generated under the
MSC.

- **`--beta-prime`** and **`--beta`**: Control how contiguous flagged intervals
are expected to be. Lower values encourage merging nearby regions into longer
runs.

  By default, Phlag uses **`--beta-prime`** and parameterizes the prior distribution
  for effective contiguity using `beta_prime` times the number of gene trees, so the
  prior scales with input size. The default `beta_prime = 0.0025` gives an effective
  beta of 5 for 2000 gene trees (roughly five contiguous anomalous regions expected
  genome-wide). Valid values for `--beta-prime` are in `(0, 0.5)`.

  Pass **`--beta`** to use a fixed contiguity prior instead (e.g., `--beta 5`
  for the alpha-version behavior, independent of the number of gene trees).

- **`--emission-lambda`** (default: `1.0`): Controls the expected deviation of
anomalies from the MSC.

- **`--eta`** (default: `0.5`): Occupancy bias penalty on the marginal
log-likelihood.

Note that, `--rho`, `--beta-prime`, `--beta`, and `--emission-lambda` parameterize priors;
they do not impose strong expectations on their own.

**Note:** If your data lacks strong MSC deviations, you may see few or no
*flagged trees. Conversely, an incorrect or unreliable species tree can produce
*too many flags. Adjust hyperparameters to match the resolution you need.

A high-level and generic suggestion is to start with the defaults and run
Phlag with varying hyperparameters (in a sensible range), and focusing on
regions that are consistently detected.

You may want to aim for a more granular structure and break up large contiguous
blocks. If so, try lowering `--rho` or increasing `--beta-prime` (or `--beta` if
you set it explicitly). To tune how strongly the model favors the anomalous
state, adjust `--eta`.

### Output

The output file contains:

- **Header lines** (prefixed with `#`):
  - The invoked command.
  - The initial species tree with branch lengths in CU, estimated from the full gene tree sequence.
  - Labels and final branch lengths of the focal edges.
  - The final species tree with branch lengths re-estimated after excluding flagged gene trees.
  - Final focal edge lengths and distances between emission distributions of the two HMM states for each focal edge.
- **State labels**: A comma-separated sequence where `1` = flagged (anomalous), `0` = MSC-compliant, and `nan` = excluded due to missing data.
- **Posterior probabilities**: Smoothed posterior probability of the anomalous state for each gene tree.
