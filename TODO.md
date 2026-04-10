# TODO

* Make sure that both labels around the root gives you the same QQS values.

## Code Issues

### Design Improvements

- **`phlag.py` (heavy `__init__`)**: Constructing a `Phlag` object triggers tree parsing, QQS computation, edge length estimation, discretization, HMM initialization, and output header writing. This makes it impossible to test or use any step in isolation. A more conventional pattern would be a lightweight `__init__` followed by explicit method calls.
- **`hmm.py`**: Wildcard import `from dynamax.hidden_markov_model.inference import *` still present. Key symbols (`partial`, `jit`, `lax`, `Callable`) now have explicit imports, but `HMMPosterior`, `HMMPosteriorFiltered`, `get_trans_mat`, `compute_transition_probs` still come from the wildcard.
- **`qqs.py` (performance)**: `MSC.compute_qqs` is O(n_edges * n_gt * n_taxa) in pure Python. The inner loop iterates every gene tree for every edge, calling `QQS.compute()` which traverses the tree. This was previously Cython. For large datasets this will be a major bottleneck. Consider Cython/Numba/JAX acceleration for the inner loop.
