import argparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def read_snap_edgelist(path):
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            a, b = s.split()
            edges.append((int(a), int(b)))
    return edges

def read_communities(path):
    comms = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            comms.append([int(x) for x in s.split()])
    return comms

def top_k_communities(comms, k, max_nodes=None):
    """按社区大小从大到小选前 K 个；如果加了 max_nodes，会跳过大于 max_nodes 的社区"""
    k = min(k, len(comms))
    sizes = [(len(comms[i]), -i, i) for i in range(len(comms))]
    sizes.sort(reverse=True)
    selected = []
    idxs = []
    for _, _, i in sizes:
        if max_nodes is not None and len(comms[i]) > max_nodes:
            continue
        idxs.append(i)
        selected.append(comms[i])
        if len(selected) >= k:
            break
    return idxs, selected

def build_node_colors(G, comms):
    n = G.number_of_nodes()
    colors = ["lightgrey"] * n

    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab20", len(comms))

    # 给子图的节点一个连续编号映射
    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    for cid, comm in enumerate(comms):
        for node in comm:
            if node in node_to_idx:  # 只给子图里的点上色
                idx = node_to_idx[node]
                colors[idx] = cmap(cid)
    return colors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snap_edges", type=str, required=True, help="SNAP edgelist file")
    parser.add_argument("--snap_comms", type=str, required=True, help="SNAP communities file")
    parser.add_argument("--k", type=int, default=5, help="select top K communities")
    parser.add_argument("--max_nodes", type=int, default=None, help="skip communities larger than max_nodes")
    args = parser.parse_args()

    edges = read_snap_edgelist(args.snap_edges)
    comms = read_communities(args.snap_comms)

    # 选取 top-K 社区
    _, selected = top_k_communities(comms, args.k, max_nodes=args.max_nodes)
    selected_nodes = np.unique(np.concatenate(selected))

    # 用这些节点诱导子图
    G = nx.Graph()
    G.add_edges_from(edges)
    G_sub = G.subgraph(selected_nodes)

    colors = build_node_colors(G_sub, selected)

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G_sub, seed=42)
    nx.draw_networkx_nodes(G_sub, pos, node_color=colors, node_size=20)
    nx.draw_networkx_edges(G_sub, pos, alpha=0.2, width=0.5)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
