# Phlag

**Phlag** detects and flags phylogenetic anomalies across the genome.

Given a species tree and a sequence of gene trees, Phlag aims to detect strong deviations from the multi-species coalescent (MSC) process using quadripartition quartet scores (QQS) and a hidden Markov model (HMM).

The input is a sequence of gene trees, ideally sampled uniformly across a chromosome.
The expected output is a set of "flagged" subsequences corresponding to model violations (e.g., introgression, recombination suppression, or dramatic changes in effective population size) and an updated species tree with branch lengths in coalescent units (CU), estimated by ignoring the flagged gene trees.

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

The `test/` directory contains a simulated dataset based on a Neoaves species tree with 191 taxa.
We use this to demonstrate how Phlag detects a genomic region where the MSC model is violated.

Suppose, we have a species tree (`test/neoaves.nwk`) and a sequence of 1500 gene trees (`test/emission.gtrees`), ordered along a chromosome.
This gene tree sequence is a mixture of two coalescent with recombination (CwR) processes:
- **Background (default)**: Gene trees simulated under the standard MSC with the original species tree parameters.
- **Anomalous**: Gene trees simulated with a 10-fold increase in the effective population size for the branch leading to the *Charadriiformes* clade (labeled `N159`).

Out of 1500 consecutive gene trees, 150 (10%) come from the anomalous process.
These occupy a single contiguous block at indices [913, 1062] (0-indexed, inclusive).
Our goal is to detect this region using Phlag.

### Input
* **Species tree** (`-s`): A Newick species tree with labeled internal nodes.
* **Gene trees** (`-g`): One Newick tree per line, ordered along the genome (not necessarily with labeled internal nodes).
* **Focal edge(s)** (`-e`): The label of the internal node(s) defining the edge(s) to target. In this example, the clade under suspicion is labeled `N159`.

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

The quadripartition quartet scores for this dataset are precomputed and stored in `test/qqs.tsv`.
Reading them with `--read-qqs-path` avoids recomputing QQS values from scratch.
If you omit this flag, QQS values will be computed on the fly, which adds to the running time.
You can also precompute and save QQS values for later runs (e.g., with different focal edges or hyperparameters) using `--write-qqs-path`.

For the full list of options, run `phlag --help`.

### Key hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--rho` | 0.9 | Controls sensitivity; reduce to flag more regions |
| `--beta` | 5 | Controls contiguity of flagged intervals; reduce to merge nearby flags |
| `--emission-lambda` | 1.0 | Controls expected deviation of anomalies from MSC |
| `--eta` | 0.5 | Occupancy bias penalty on the marginal log-likelihood |

**Note**: If your data lacks strong deviations from the MSC, you may see few or no flagged trees.
Conversely, an incorrect or unreliable species tree may result in too many flagged gene trees.
In such cases, you can adjust the hyperparameters to detect anomalies at the resolution you desire.

A high-level and generic suggestion is running Phlag with varying hyperparameters (in a sensible range tested in the paper), and focusing on regions that are consistently detected. You may want to increase `--expected-num-anomalies` if you are looking for more granular results or if large contigous chunks are flagged. Similarly, `--expected-anamoly-proportion` can be varied between 0 and 0.5 to change to prior on the stationary distribution. 

### Output
The output file contains:
- **Header lines** (prefixed with `#`):
    * The invoked command.
    * The initial species tree with branch lengths in CU, estimated using the entire gene tree sequence.
    * The labels and final branch lengths of the focal edges.
    * The final species tree with branch lengths re-estimated after excluding the flagged gene trees.
    * The final focal edge lengths, and distances between the emission distributions of the two HMM states for each focal edge.
- **State labels**: A comma-separated sequence where `1` = flagged (anomalous) and `0` = MSC-compliant. Gene trees excluded due to missing data are labeled `nan`.
- **Posterior probabilities**: The smoothed posterior probability of being in the anomalous state for each gene tree.
