import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import zipfile
import io
import urllib.request

import warnings
warnings.filterwarnings('ignore')

def create_directed_graph(graph: nx.Graph) -> tuple[nx.DiGraph, dict]:
    """
    Convert the input graph to a directed graph, relabel nodes, and remove isolated nodes.

    Args:
        graph (nx.Graph): The input graph.

    Returns:
        tuple: A tuple containing the directed graph and a dictionary mapping integer node labels to original node labels.
    """
    if not nx.is_directed(graph):
        graph = graph.to_directed()

    n_unique_nodes = len(set(graph.nodes()))
    int_node_dict = dict(zip(set(graph.nodes()), range(n_unique_nodes)))
    node_int_dict = {v: k for k, v in int_node_dict.items()}

    graph = nx.relabel_nodes(graph, int_node_dict)
    graph.remove_nodes_from([node for node in graph.nodes() if len(graph.edges(node)) == 0])

    return graph, node_int_dict

def create_directed_graph_from_matrix(adj_matrix: np.ndarray) -> nx.DiGraph:
    """
    Create a directed graph from an adjacency matrix.

    Args:
        adj_matrix (np.ndarray): The adjacency matrix representation of the graph.

    Returns:
        nx.DiGraph: The directed graph.
    """
    graph = nx.DiGraph(adj_matrix)
    return graph


def build_traditional_pagerank_matrix(graph: nx.DiGraph, damping_factor: float) -> np.ndarray:
    """
    Construct the PageRank matrix for the given graph and damping factor.

    Args:
        graph (nx.DiGraph): The directed graph.
        damping_factor (float): The damping factor for the PageRank algorithm.

    Returns:
        np.ndarray: The PageRank matrix.
    """
    n_nodes = len(graph.nodes())
    adj_matrix = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()))

    transition_matrix = adj_matrix / np.sum(adj_matrix, axis=1, keepdims=True)
    transition_matrix[np.isnan(transition_matrix)] = 0  # Handle division by zero

    absorbing_nodes = np.sum(adj_matrix, axis=0) == 0
    absorbing_node_matrix = np.outer(absorbing_nodes, np.ones(n_nodes)) / n_nodes

    stochastic_matrix = transition_matrix + absorbing_node_matrix
    random_jump_matrix = np.ones((n_nodes, n_nodes)) / n_nodes

    pagerank_matrix = damping_factor * stochastic_matrix + (1 - damping_factor) * random_jump_matrix

    return pagerank_matrix

def build_in_degree_jump_matrix(graph: nx.DiGraph, damping_factor: float) -> np.ndarray:
    """
    Construct the PageRank random jump matrix based on in-degrees.

    Args:
        graph (nx.DiGraph): The directed graph.
        damping_factor (float): The damping factor for the PageRank algorithm.

    Returns:
        np.ndarray: The PageRank random jump matrix based on in-degrees.
    """
    n_nodes = len(graph.nodes())
    in_degrees = np.array([graph.in_degree(node) for node in graph.nodes()])
    in_degree_probabilities = in_degrees / np.sum(in_degrees)

    random_jump_matrix = np.diag(in_degree_probabilities)
    pagerank_matrix = damping_factor * random_jump_matrix + (1 - damping_factor) * np.ones((n_nodes, n_nodes)) / n_nodes

    return pagerank_matrix

def build_out_degree_jump_matrix(graph: nx.DiGraph, damping_factor: float) -> np.ndarray:
    """
    Construct the PageRank random jump matrix based on out-degrees.

    Args:
        graph (nx.DiGraph): The directed graph.
        damping_factor (float): The damping factor for the PageRank algorithm.

    Returns:
        np.ndarray: The PageRank random jump matrix based on out-degrees.
    """
    n_nodes = len(graph.nodes())
    out_degrees = np.array([graph.out_degree(node) for node in graph.nodes()])
    out_degree_probabilities = out_degrees / np.sum(out_degrees)

    random_jump_matrix = np.diag(out_degree_probabilities)
    pagerank_matrix = damping_factor * random_jump_matrix + (1 - damping_factor) * np.ones((n_nodes, n_nodes)) / n_nodes

    return pagerank_matrix

def build_pagerank_matrix(graph: nx.DiGraph, damping_factor: float, jump_matrix_type: str) -> np.ndarray:
    """
    Construct the PageRank matrix for the given graph and damping factor.

    Args:
        graph (nx.DiGraph): The directed graph.
        damping_factor (float): The damping factor for the PageRank algorithm.
        jump_matrix_type (str): The type of jump matrix to use ('uniform', 'in_degree', 'out_degree').

    Returns:
        np.ndarray: The PageRank matrix.
    """
    if jump_matrix_type == 'traditional':
        return build_traditional_pagerank_matrix(graph, damping_factor)
    elif jump_matrix_type == 'in_degree':
        return build_in_degree_jump_matrix(graph, damping_factor)
    elif jump_matrix_type == 'out_degree':
        return build_out_degree_jump_matrix(graph, damping_factor)
    else:
        raise ValueError("Invalid jump matrix type")

def run_pagerank_algorithm(graph: nx.DiGraph, damping_factor: float, max_iterations: int, jump_matrix_type: str) -> np.ndarray:
    """
    Perform the PageRank algorithm using the random walk method.

    Args:
        graph (nx.DiGraph): The directed graph.
        damping_factor (float): The damping factor for the PageRank algorithm.
        max_iterations (int): The maximum number of iterations for the algorithm.
        jump_matrix_type (str): The type of jump matrix to use ('uniform', 'in_degree', 'out_degree').

    Returns:
        np.ndarray: The final PageRank scores for each node.
    """
    n_nodes = len(graph.nodes())
    initial_state = np.ones(n_nodes) / n_nodes
    pagerank_matrix = build_pagerank_matrix(graph, damping_factor, jump_matrix_type)

    new_state = initial_state
    norms = []
    for _ in range(max_iterations):
        final_state = np.dot(pagerank_matrix.T, new_state)

        prev_state = new_state
        new_state = final_state
        l2_norm = np.linalg.norm(new_state - prev_state)
        norms.append(l2_norm)

        if np.allclose(new_state, prev_state):
            print(f"Converged after {len(norms)} iterations.")
            break

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(len(norms)), norms)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('L2 Norm')
    ax.set_title('Convergence Plot')
    plt.savefig(f'convergence_plots_jumps/{jump_matrix_type}.png')  # Save the plot as PNG
    plt.close(fig)  # Close the plot to prevent it from being displayed

    return final_state

def plot_graph(graph: nx.DiGraph, final_probs: np.ndarray, node_int_dict: dict, bool_final_probs: bool = False, node_size: int = 500, jump_matrix_type = 'traditional', ax=None):
    """ Plot the graph with optional node coloring based on PageRank scores.
    Args:
        graph (nx.DiGraph): The directed graph to plot.
        final_probs (np.ndarray): The PageRank scores for each node.
        node_int_dict (dict): A dictionary mapping integer node labels to original node labels.
        bool_final_probs (bool, optional): Whether to color nodes based on PageRank scores. Defaults to False.
        node_size (int, optional): The size of the nodes in the plot. Defaults to 50.
        ax (matplotlib.axes.Axes, optional): The Axes object to plot on. Defaults to None.
    """
    labels = node_int_dict
    try:
        clubs = np.array([graph.nodes[node]['club'] for node in graph.nodes()])
        labels = dict(zip(graph.nodes(), clubs))
    except KeyError:
        pass

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate node positions using spring layout
    pos = nx.spring_layout(graph, seed=42)

    if bool_final_probs:
        nx.draw(graph, with_labels=False, node_color=final_probs, cmap='viridis', node_size=node_size, ax=ax, pos=pos)
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=node_size/40, font_color='maroon', ax=ax)
        plt.colorbar(ax.collections[0], ax=ax, label='Final PageRank Scores')
        plt.savefig(f'graphs_jumps/graph_with_pagerank_{jump_matrix_type}.png')  # Save the plot as PNG
        plt.close()  # Close the plot to prevent it from being displayed
    else:
        nx.draw(graph, with_labels=False, node_size=node_size, ax=ax, pos=pos)
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=node_size/40, font_color='maroon', ax=ax)
        plt.savefig('graphs_jumps/graph.png')  # Save the plot as PNG
        plt.close()  # Close the plot to prevent it from being displayed

def compute_and_plot_pagerank(graph: nx.Graph, damping_factor: float, max_iterations: int, jump_matrix_type: str) -> np.ndarray:
    """
    Run the PageRank algorithm on the given graph and plot the results.

    Args:
        graph (nx.Graph): The input graph.
        damping_factor (float): The damping factor for the PageRank algorithm.
        max_iterations (int): The maximum number of iterations for the algorithm.
        jump_matrix_type (str): The type of jump matrix to use ('uniform', 'in_degree', 'out_degree').

    Returns:
        np.ndarray: The final PageRank scores for each node.
    """
    directed_graph, node_int_dict = create_directed_graph(graph)

    print(f"Number of nodes: {len(directed_graph.nodes())}")
    print(f"Number of edges: {len(directed_graph.edges())}")

    final_probs = run_pagerank_algorithm(directed_graph, damping_factor, max_iterations, jump_matrix_type)

    assert len(final_probs) == len(directed_graph.nodes()), "Pagerank importance lengths don't match"
    # assert np.isclose(np.sum(final_probs), 1), "Pagerank probabilities don't sum to 1"

    # Normalize the final PageRank scores to ensure they sum to 1
    final_probs /= np.sum(final_probs)

    plot_graph(directed_graph, final_probs, node_int_dict, bool_final_probs=True, jump_matrix_type=jump_matrix_type)

    return final_probs

def generate_football_graph():
    url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"

    sock = urllib.request.urlopen(url)  # open URL
    s = io.BytesIO(sock.read())  # read into BytesIO "file"
    sock.close()

    zf = zipfile.ZipFile(s)  # zipfile object
    txt = zf.read("football.txt").decode()  # read info file
    gml = zf.read("football.gml").decode()  # read gml data
    # throw away bogus first line with # from mejn files
    gml = gml.split("\n")[1:]
    G = nx.parse_gml(gml)  # parse gml data
    return G

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(array, array)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(array)
    # Gini coefficient
    gini_coefficient = 0.5 * rmad
    return gini_coefficient

if __name__ == "__main__":
    damping_factor = 0.95
    max_iterations = 1000

    A = np.array([[0, 1, 0, 0], [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 0]])
    A = A.T
    # graph = create_directed_graph_from_matrix(A)
    # graph = nx.karate_club_graph()
    graph = generate_football_graph()
    # graph = nx.random_geometric_graph(200, 0.125, seed=896803)
    damping_factor = 0.85
    jump_matrix_types = ['traditional', 'in_degree', 'out_degree']  # List of jump matrix types to experiment with

    for jump_matrix_type in jump_matrix_types:
        final_probs_power_iteration = compute_and_plot_pagerank(graph, damping_factor, max_iterations, jump_matrix_type)
        gini_coefficient = gini(final_probs_power_iteration)

        print(f"\nPageRank Scores (Power iteration) for jump matrix type {jump_matrix_type}:")
        print(final_probs_power_iteration)
        print(f"Gini Coefficient: {gini_coefficient}\n")