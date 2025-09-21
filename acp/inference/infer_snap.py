"""
Inference on SNAP graphs with ACP (GraphSAGE/DGL encoder) and Louvain comparison.
- Read SNAP edgelist (+ optional ground-truth communities)
- Sample K communities -> induce subgraph -> run ACP inference
- Plot colored adjacency matrices and compute AMI/Q
"""

import os
import time
import random
import argparse

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree
import dgl
import networkx as nx
from networkx.algorithms.community.quality import modularity
from community import community_louvain
from sklearn.metrics import adjusted_mutual_info_score

# project imports
from ..encoders.sbm_graphsage_encoder import get_sbm_graph_sage_encoder
from ..encoders.sbm_gatedgcn_dgl_encoder import get_sbm_gated_gcn_dgl_encoder
from ..models.acp_model import ACP_Model
from ..models.acp_sampler import ACP_Sampler
from ..data_generator.utils import remap_labels_by_cluster_size
from ..utils.sbm_utils import plot_colored_adj_matrix_with_prediction
from ..utils.plotting import DEFAULT_COLORS
from ..utils.graph_utils import edge_list_to_adj_matrix

from ..utils.viz_utils import plot_graph_with_communities

# ---------------------------
# Utilities for SNAP handling
# ---------------------------
def read_snap_edgelist_with_remap(path):
    """Read SNAP edgelist; ignore lines starting with '#'.
    Return: N (after remap), edge_index [2,E] (0..N-1), remap dict old->new.
    """
    edges, node_set = [], set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            a, b = s.split()
            u, v = int(a), int(b)
            edges.append((u, v))
            node_set.update((u, v))

    nodes = sorted(node_set)
    N = len(nodes)
    contiguous = (nodes[0] == 0 and nodes[-1] == N - 1)

    if contiguous:
        remap = {i: i for i in nodes}
    else:
        remap = {nid: i for i, nid in enumerate(nodes)}
        edges = [(remap[u], remap[v]) for (u, v) in edges]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)
    return N, edge_index, remap


def read_communities_lines(path):
    """Each line is a community: space/tab-separated node IDs (original IDs)."""
    comms = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            comm = np.array([int(p) for p in parts], dtype=np.int64)
            comms.append(comm)
    return comms


def sample_k_communities(comms, k, rng=None):
    rng = rng or random
    if k > len(comms):
        k = len(comms)
    idxs = rng.sample(range(len(comms)), k)
    return idxs, [comms[i] for i in idxs]

def top_k_communities_from_max(comms, k, max_nodes=None):
    k = min(k, len(comms))
    sizes = [(len(comms[i]), -i, i) for i in range(len(comms))]
    sizes.sort(reverse=True)  # 大到小排
    
    if max_nodes is None:
        idxs = [t[2] for t in sizes[:k]]
        selected = [comms[i] for i in idxs]
        return idxs, selected
    
    # 找到第一个 ≤ max_nodes 的位置
    start = None
    for j, (sz, _, i) in enumerate(sizes):
        if sz <= max_nodes:
            start = j
            break
    
    if start is None:
        # 所有社区都比 max_nodes 大，那就取最大的前 k 个
        start = 0
    
    end = min(start + k, len(sizes))
    idxs = [sizes[j][2] for j in range(start, end)]
    selected = [comms[i] for i in idxs]
    return idxs, selected

def induce_subgraph_from_nodes(edge_index, node_ids, old_to_new=None):
    """Induced subgraph on node_ids.
    Inputs:
      - edge_index: [2,E] (already 0..N-1)
      - node_ids:   ids to keep (same id space as edge_index); if original IDs, pass old_to_new
    Returns:
      - sub_edge_index: [2, E_sub] with new 0..n_sub-1 indexing
      - sub_remap: dict old_id_in_edge_space -> 0..n_sub-1
    """
    if old_to_new is not None:
        node_ids = np.array([old_to_new[n] for n in node_ids if n in old_to_new], dtype=np.int64)

    keep = set(int(n) for n in node_ids)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    mask = np.isin(src, list(keep)) & np.isin(dst, list(keep))
    sub_edges = np.stack([src[mask], dst[mask]], axis=1)

    unique_nodes = sorted(list(keep))
    sub_remap = {nid: i for i, nid in enumerate(unique_nodes)}
    if sub_edges.shape[0] > 0:
        sub_edges_remap = np.array([[sub_remap[u], sub_remap[v]] for u, v in sub_edges], dtype=np.int64)
        sub_edge_index = torch.tensor(sub_edges_remap, dtype=torch.long).t().contiguous()
        sub_edge_index = to_undirected(sub_edge_index)
    else:
        sub_edge_index = torch.empty((2, 0), dtype=torch.long)

    return sub_edge_index, sub_remap


def build_labels_from_comms_for_subset(sub_remap, selected_comms, global_old_to_new=None):
    """labels for subgraph nodes: (n_sub,), unlabeled nodes = -1."""
    n_sub = len(sub_remap)
    labels = np.full((n_sub,), -1, dtype=np.int64)
    for cid, comm in enumerate(selected_comms):
        for old in comm:
            gid = global_old_to_new[old] if global_old_to_new is not None else old
            if gid in sub_remap:
                labels[sub_remap[gid]] = cid
    return labels


def build_features_from_degree(N, edge_index, enc_in_dim, device):
    """First dim = degree, pad zeros to enc_in_dim."""
    deg = degree(edge_index[0], num_nodes=N).unsqueeze(1)  # [N,1]
    X = torch.zeros((N, enc_in_dim), dtype=torch.float32, device=device)
    X[:, :1] = deg
    return X


# -------------
# Main pipeline
# -------------
def main():
    parser = argparse.ArgumentParser(description='Run ACP inference on SNAP graphs.')
    parser.add_argument('--encoder_type', type=str, required=True, choices=['graphsage', 'gatedgcn_dgl'],
                        help="encoder type")
    parser.add_argument('--model_file', type=str, required=True, help="trained ACP checkpoint (.pt)")
    parser.add_argument('--snap_edges', type=str, required=True, help="SNAP edgelist path")
    parser.add_argument('--snap_comms', type=str, default="", help="ground-truth communities file (each line = one community)")
    parser.add_argument('--out_dir', type=str, default="outputs_snap", help="output dir")
    parser.add_argument('--gpu', type=int, default=0, help="gpu id")
    parser.add_argument('--n_graphs', type=int, default=1, help="how many random subgraphs to sample")
    parser.add_argument('--K', type=int, default=10, help="how many communities to sample per subgraph")
    parser.add_argument('--select_topk', action='store_true',
                    help='select top k communities (defalt 0)')
    parser.add_argument('--max_nodes', type=int, default=0,
                    help='upper limit of sub graph node number (0 = no upper limit)')
    parser.add_argument('--S', type=int, default=10, help="ACP: number of parallel samples")
    parser.add_argument('--prob_nZ', type=int, default=1, help="ACP: how many Z to sample when estimating probability")
    parser.add_argument('--prob_nA', type=int, default=10, help="ACP: how many anchors to sample when estimating probability")
    parser.add_argument('--seed', type=int, default=42, help="random seed for community sampling")
    args = parser.parse_args()

    print("\n")
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")

    # Load checkpoint & build encoder/model
    checkpoint = torch.load(args.model_file, map_location=device)
    params = checkpoint['params']
    print("model params:", params)

    if args.encoder_type == "graphsage":
        encoder = get_sbm_graph_sage_encoder(params)
        data_lib = "torch_geom"
    else:
        encoder = get_sbm_gated_gcn_dgl_encoder(params)
        data_lib = "dgl"

    model = ACP_Model(params, encoder)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    SelectedSampler = ACP_Sampler  # keep same interface

    # Prepare output
    fname_prefix = f"snap_{args.encoder_type}"
    fname_postfix = f"_S-{args.S}_pA-{args.prob_nA}_pZ-{args.prob_nZ}"
    out_dir = os.path.join(args.out_dir, os.path.basename(args.model_file).strip(".pt") + fname_postfix)
    fig_dir = os.path.join(out_dir, "figures")
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Read whole SNAP graph once
    N_full, edge_index_full, remap_full = read_snap_edgelist_with_remap(args.snap_edges)
    print(f"Loaded SNAP graph: N={N_full}, E={edge_index_full.size(1)}")

    comms_full = []
    if args.snap_comms:
        comms_full = read_communities_lines(args.snap_comms)
        print(f"Loaded {len(comms_full)} communities from: {args.snap_comms}")

    rng = random.Random(args.seed)
    enc_in_dim = int(params.get('enc_in_dim', 20))

    # Metrics
    all_amis, all_times = [], []

    for i in range(args.n_graphs):
        print(i, end=' ')

        # 1) pick K communities (or fallback: random nodes if no comms file)
        if len(comms_full) > 0:
            _, selected = top_k_communities_from_max(comms_full, args.K, max_nodes=args.max_nodes)
            selected_nodes = np.unique(np.concatenate(selected))
        else:
            # fallback: pick a random subset of nodes as a "pseudo-community union"
            pick = min(max(1000, args.K * 100), N_full)  # heuristic
            selected_nodes = np.array(rng.sample(range(N_full), pick), dtype=np.int64)
            selected = []  # no GT

        # 2) induce subgraph
        sub_edge_index, sub_remap = induce_subgraph_from_nodes(
            edge_index_full, selected_nodes, old_to_new=remap_full
        )
        N = len(sub_remap)
        if N == 0 or sub_edge_index.size(1) == 0:
            print(" [skip empty subgraph]")
            continue

        # 3) labels on subgraph
        if len(selected) > 0:
            labels_np = build_labels_from_comms_for_subset(sub_remap, selected, global_old_to_new=remap_full)
        else:
            labels_np = np.full((N,), -1, dtype=np.int64)

        # 4) features & Data/DGLGraph
        X = build_features_from_degree(N, sub_edge_index, enc_in_dim, device)
        if data_lib == "torch_geom":
            data = Data(x=X, edge_index=sub_edge_index.to(device))
            edge_index = data.edge_index
            features = data.x
        else:
            g = dgl.graph((sub_edge_index[0], sub_edge_index[1]), num_nodes=N, device=device)
            g = dgl.to_simple(g)
            g.ndata['feat'] = X
            data = g
            edge_index = torch.stack(data.all_edges())
            features = data.ndata['feat']

        features_np = features.detach().cpu().numpy()
        adj_matrix_np = edge_list_to_adj_matrix(edge_index, N).cpu().numpy()

        # 5) ACP inference
        t0 = time.time()
        sampler = SelectedSampler(model, data, device=device)
        clusters, probs = sampler.sample(
            S=args.S, sample_Z=False, sample_B=False,
            prob_nZ=args.prob_nZ, prob_nA=args.prob_nA
        )
        predicted = clusters[np.argmax(probs)]
        infer_time = time.time() - t0
        all_times.append(infer_time)

        # 6) metrics
        G = nx.from_numpy_array(adj_matrix_np)
        comms_pred = [set(np.where(predicted == c)[0]) for c in np.unique(predicted)]
        Q_pred = modularity(G, comms_pred)

        mask = labels_np != -1
        ami = adjusted_mutual_info_score(labels_np[mask], np.asarray(predicted)[mask]) if mask.any() else np.nan
        all_amis.append(ami)

        if mask.any():
            comms_gnd = [set(np.where(labels_np == c)[0]) for c in np.unique(labels_np) if c != -1]
            Q_gnd = modularity(G, comms_gnd) if len(comms_gnd) > 0 else np.nan
        else:
            Q_gnd = np.nan

        plot_graph_with_communities(G, comms_gnd, save_path=os.path.join(fig_dir, f"acp_gnd{i}.png"))

        # 7) visualization (ACP)
        labels_sorted = remap_labels_by_cluster_size(
            torch.from_numpy(labels_np if mask.any() else np.zeros(N, dtype=np.int64))
        ).numpy()
        pred_sorted = remap_labels_by_cluster_size(torch.from_numpy(np.asarray(predicted))).numpy()

        fname = f"{fname_prefix}_sub{i}_K{args.K}_N{N}"
        title = (f"ACP (i={i}, N={N}, AMI={ami:.3f}, "
                 f"Q_gnd={(Q_gnd if not np.isnan(Q_gnd) else float('nan')):.3f}, "
                 f"Q_pred={Q_pred:.3f})")
        plot_colored_adj_matrix_with_prediction(
            adj_matrix_np, labels_sorted, pred_sorted, DEFAULT_COLORS,
            title=title, fontsize=16, bg_colors=['white', 'dimgray'],
            save_name=os.path.join(fig_dir, fname + "_ACP.png")
        )

        # 8) Louvain comparison
        part = community_louvain.best_partition(G)
        pred_louvain = np.array([part[n] for n in range(N)], dtype=int)
        comms_lou = [set(np.where(pred_louvain == c)[0]) for c in np.unique(pred_louvain)]
        Q_lou = modularity(G, comms_lou)
        ami_lou = adjusted_mutual_info_score(labels_np[mask], pred_louvain[mask]) if mask.any() else np.nan

        pred_lou_sorted = remap_labels_by_cluster_size(torch.from_numpy(pred_louvain)).numpy()
        title_lou = (f"Louvain (i={i}, N={N}, AMI={ami_lou:.3f}, "
                     f"Q_gnd={(Q_gnd if not np.isnan(Q_gnd) else float('nan')):.3f}, "
                     f"Q_lou={Q_lou:.3f})")
        plot_colored_adj_matrix_with_prediction(
            adj_matrix_np, labels_sorted, pred_lou_sorted, DEFAULT_COLORS,
            title=title_lou, fontsize=16, bg_colors=['white','dimgray'],
            save_name=os.path.join(fig_dir, fname + "_Louvain.png")
        )

        # 9) save npz
        np.savez_compressed(
            os.path.join(data_dir, fname + ".npz"),
            adj_matrix=adj_matrix_np,
            node2vec=features_np,
            labels=labels_np,
            predicted=np.asarray(predicted),
            inference_time=infer_time,
            AMI=ami,
            Q_pred=Q_pred,
            Q_gnd=Q_gnd
        )

    print("\n")
    if len(all_amis) > 0:
        mean_ami = float(np.nanmean(all_amis))
        median_ami = float(np.nanmedian(all_amis))
        print(f"AMI  -- mean: {mean_ami:.4f}, median: {median_ami:.4f}")
    mean_t = float(np.mean(all_times)) if len(all_times) > 0 else float('nan')
    med_t  = float(np.median(all_times)) if len(all_times) > 0 else float('nan')
    print(f"time -- mean: {mean_t:.4f}s, median: {med_t:.4f}s")


if __name__ == "__main__":
    main()
