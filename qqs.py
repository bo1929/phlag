from collections import defaultdict
from functools import partial

import dendropy
import jax.numpy as jnp

from tqdm import tqdm

import utils


class QQS:
    def __init__(self):
        pass

    def f(self, w, x, y, z, gt):
        ws = len(w)
        xs = len(x)
        ys = len(y)
        zs = len(z)

        r = 0.0
        stack = []

        leaf_to_part = {}
        for taxon in w:
            leaf_to_part[taxon] = 0
        for taxon in x:
            leaf_to_part[taxon] = 1
        for taxon in y:
            leaf_to_part[taxon] = 2
        for taxon in z:
            leaf_to_part[taxon] = 3

        for u in gt.postorder_node_iter():
            if u.is_leaf():
                lbl = u.taxon.label
                part = leaf_to_part.get(lbl)
                if part == 0:
                    stack.append((1, 0, 0, 0))
                elif part == 1:
                    stack.append((0, 1, 0, 0))
                elif part == 2:
                    stack.append((0, 0, 1, 0))
                elif part == 3:
                    stack.append((0, 0, 0, 1))
                else:
                    stack.append((0, 0, 0, 0))
                    # raise ValueError(f"Taxon {lbl} is not in any partition")
            else:
                c1_w, c1_x, c1_y, c1_z = stack.pop()
                c2_w, c2_x, c2_y, c2_z = stack.pop()

                iw = c1_w + c2_w
                ix = c1_x + c2_x
                iy = c1_y + c2_y
                iz = c1_z + c2_z

                c3_w = ws - iw
                c3_x = xs - ix
                c3_y = ys - iy
                c3_z = zs - iz

                r += (
                    (c1_w * c2_x + c1_x * c2_w) * c3_y * c3_z
                    + (c1_y * c2_z + c1_z * c2_y) * c3_w * c3_x
                    + (c3_w * c2_x + c3_x * c2_w) * c1_y * c1_z
                    + (c3_y * c2_z + c3_z * c2_y) * c1_w * c1_x
                    + (c1_w * c3_x + c1_x * c3_w) * c2_y * c2_z
                    + (c1_y * c3_z + c1_z * c3_y) * c2_w * c2_x
                )

                stack.append((iw, ix, iy, iz))

        return r / 2.0 if stack else 0.0

    def extract_quadripartition(self, edge):
        u = edge.tail_node
        v = edge.head_node

        if u.is_leaf() or v.is_leaf():
            raise ValueError("Terminal edges not allowed for QQS")

        parts = []
        for nd in u.adjacent_nodes():
            if nd is not v:
                parts.append(utils.leaf_set(nd, u))

        for nd in v.adjacent_nodes():
            if nd is not u:
                parts.append(utils.leaf_set(nd, v))

        if len(parts) != 4:
            raise ValueError("Invalid quadripartition")

        for p in parts:
            if len(p) == 0:
                raise ValueError("Empty partition in quadripartition")

        return parts

    def compute(self, x, y, w, z, gt, normalize=True):
        f1 = self.f(x, y, w, z, gt)
        f2 = self.f(x, w, y, z, gt)
        f3 = self.f(x, z, w, y, gt)

        if not normalize:
            return f1, f2, f3

        m = f1 + f2 + f3
        if m == 0.0:
            return 0.0, 0.0, 0.0
        return f1 / m, f2 / m, f3 / m


class MSC:
    def __init__(self, st, gt_l):
        self.qqs = QQS()
        self.st = st
        self.gt_l = gt_l
        self.n_gt = len(gt_l)

    def compute_qqs(self):
        edge_to_qqs = defaultdict(partial(jnp.zeros, (self.n_gt, 3)))

        for edge in tqdm(self.st.internal_edges(exclude_seed_edge=True), desc="Computing QQS..."):
            x, y, w, z = self.qqs.extract_quadripartition(edge)
            if min(len(x), len(y), len(w), len(z)) < 1:
                continue

            for i in range(self.n_gt):
                f1, f2, f3 = self.qqs.compute(x, y, w, z, self.gt_l[i])
                edge_to_qqs[edge] = edge_to_qqs[edge].at[i, :].set((f1, f2, f3))

        return edge_to_qqs

    def simulate_qqs(self, focal_edge_l, n_replicates=100):
        if n_replicates <= 0:
            raise ValueError("Number of replicates must be positive")
        assert len(focal_edge_l) > 0

        tgt_to_tst = dendropy.TaxonNamespaceMapping.create_contained_taxon_mapping(
            containing_taxon_namespace=self.st.taxon_namespace, num_contained=1
        )
        gt_l = [
            dendropy.simulate.treesim.contained_coalescent_tree(self.st, tgt_to_tst)
            for _ in range(n_replicates)
        ]
        sqqs = jnp.zeros((n_replicates, len(focal_edge_l), 3))

        for j, edge in enumerate(focal_edge_l):
            assert edge is not None
            x, y, w, z = self.qqs.extract_quadripartition(edge)

            for i in range(n_replicates):
                gt = gt_l[i]
                for leaf in gt.leaf_node_iter():
                    if leaf.taxon is not None:
                        leaf.taxon.label = leaf.taxon.label.split()[0]

                f1, f2, f3 = self.qqs.compute(x, y, w, z, gt)
                sqqs = sqqs.at[i, j, :].set((f1, f2, f3))

        return sqqs
