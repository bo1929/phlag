# Phlag
Phlag flags phlogenetic anomalies across the genome.

Given a species tree, and a sequence of gene trees,
Phlag attempts to detect strong deviations from the multispecies coalescent process using quadripartition quartet scores and a hidden Markov model.
pip install -r requirements.txt
## Installation

For installation, please follow the steps below, you will need Python >= 3.9 environment.
```shell
micromamba create -n phlag python=3.9 -y
micromamba activate phlag
git clone https://github.com/bo1929/phlag.git
cd phlag
pip install -r requirements.txt
./build.sh
```
