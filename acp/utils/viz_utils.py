import matplotlib.pyplot as plt
import matplotlib
import networkx as nx

def plot_graph_with_communities(G, communities, save_path=None, node_size=20):
    """
    using G and communities, plot graph with colored community
    Args:
        G: networkx.Graph
        communities: List[Iterable[int]], each element is a list of node in one community
        save_path: (png)
        node_size
    """
    pos = nx.spring_layout(G, seed=42)
    n_comm = len(communities)

    cmap = matplotlib.colormaps.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(n_comm)]
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    # default color lightgrey
    colors = ["lightgrey"] * len(node_list)
    for cid, comm in enumerate(communities):
        for node in comm:
            if node in node_to_idx:
                colors[node_to_idx[node]] = cmap(cid)

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=colors, node_size=node_size)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
