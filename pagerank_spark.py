import numpy as np
import networkx as nx
from pyspark import SparkContext
import matplotlib.pyplot as plt
import zipfile
import io
import urllib.request

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

def mapper(node, graph, damping_factor, num_nodes):
    """
    Mapper function for PageRank algorithm.

    Args:
        node (int): The node ID.
        graph (nx.DiGraph): The directed graph.
        damping_factor (float): The damping factor for the PageRank algorithm.
        num_nodes (int): The total number of nodes in the graph.

    Returns:
        list: A list of (destination_node, (source_node, contribution)) tuples.
    """
    outgoing_edges = graph.edges(node)
    num_outgoing_edges = len(outgoing_edges)
    contribution = (1 - damping_factor) / num_nodes + damping_factor / num_outgoing_edges if num_outgoing_edges > 0 else (1 - damping_factor) / num_nodes
    
    result = []
    for dest_node in outgoing_edges:
        result.append((dest_node[1], (node, contribution)))
    return result

def reducer(node, iterator, damping_factor, num_nodes):
    """
    Reducer function for PageRank algorithm.

    Args:
        node (int): The node ID.
        iterator (iterator): An iterator over (source_node, contribution) tuples.
        damping_factor (float): The damping factor for the PageRank algorithm.
        num_nodes (int): The total number of nodes in the graph.

    Returns:
        tuple: A tuple containing (node, new_pr_score).
    """
    pr_score = (1 - damping_factor) / num_nodes
    incoming_contributions = [contribution for _, contribution in iterator]
    pr_score += damping_factor * sum(incoming_contributions)
    return node, pr_score

def run_pagerank_mapreduce(graph, damping_factor, max_iterations, convergence_threshold=1e-2):
    """
    Run the PageRank algorithm using MapReduce.

    Args:
        graph (nx.Graph): The input graph.
        damping_factor (float): The damping factor for the PageRank algorithm.
        max_iterations (int): The maximum number of iterations for the algorithm.
        convergence_threshold (float, optional): The threshold for convergence. Defaults to 1e-6.

    Returns:
        dict: A dictionary containing the final PageRank scores for each node.
    """
    directed_graph, node_int_dict = create_directed_graph(graph)
    num_nodes = len(directed_graph.nodes())

    sc = SparkContext(appName="PageRank")
    initial_ranks = sc.parallelize(list(directed_graph.nodes()), numSlices=16).map(lambda x: (x, 1.0 / num_nodes))

    differences = []
    for iteration in range(max_iterations):
        contributions = initial_ranks.flatMap(lambda x: mapper(x[0], directed_graph, damping_factor, num_nodes))
        new_ranks = contributions.reduceByKey(lambda x, y: x + y, numPartitions=16).mapValues(lambda x: x[1] * damping_factor + (1 - damping_factor) / num_nodes)

        diff = new_ranks.values().reduce(lambda x, y: abs(x - y))
        differences.append(diff)

        if diff < convergence_threshold:
            print(f"Converged after {iteration + 1} iterations.")
            break

        initial_ranks = new_ranks

    final_ranks = new_ranks.collectAsMap()
    final_ranks = {node_int_dict[node]: score for node, score in final_ranks.items()}

    sc.stop()

    # Plot the convergence
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(differences)), differences)
    plt.xlabel('Iterations')
    plt.ylabel('Absolute Difference')
    plt.title('Convergence Plot')
    plt.show()

    return final_ranks

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

if __name__ == "__main__":
    damping_factor = 0.85
    max_iterations = 100

    graph = generate_football_graph()
    final_ranks = run_pagerank_mapreduce(graph, damping_factor, max_iterations)

    print("\nPageRank importances:")
    for node, score in sorted(final_ranks.items(), key=lambda x: x[1], reverse=True):
        print(f"Node {node}: {score:.6f}")