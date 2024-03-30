# PageRank

This repository contains an implementation of the PageRank algorithm in Python. The code provides various functionalities for analyzing and visualizing the PageRank scores of nodes in a graph.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Features](#features)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

PageRank is a widely used algorithm for ranking the importance of nodes in a graph based on the structure of the graph and the connections between nodes. It was originally developed by Larry Page and Sergey Brin, the founders of Google, to rank web pages in their search engine results.

This implementation focuses on exploring the behavior of the PageRank algorithm under different parameter settings, such as the damping factor and the jump matrix. It also provides functionality for visualizing the graph and the PageRank scores of nodes.

## Installation

To use the code in this repository, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/googlercolin/pagerank.git
   ```

2. Install the required dependencies:
   ```
   pip install numpy networkx matplotlib pyspark
   ```

## Usage

The code consists of three main files:

1. `pagerank.py`: Contains the core implementation of the PageRank algorithm and related functions.
2. `pagerank_jump_matrix.py`: Extends the PageRank implementation to support different types of jump matrices.
3. `pagerank_mapreduce.py`: Implements the PageRank algorithm using the MapReduce programming model with PySpark.

To use the code, import the desired functions from the respective files and call them with the appropriate parameters. Refer to the code documentation and examples for more details.

## Code Structure

The code is structured as follows:

- `pagerank.py`:
  - `create_directed_graph`: Converts an input graph to a directed graph, relabels nodes, and removes isolated nodes.
  - `create_directed_graph_from_matrix`: Creates a directed graph from an adjacency matrix.
  - `build_pagerank_matrix`: Constructs the PageRank matrix for a given graph and damping factor.
  - `run_pagerank_algorithm`: Performs the PageRank algorithm using the random walk method.
  - `plot_graph`: Plots the graph with optional node coloring based on PageRank scores.
  - `compute_and_plot_pagerank`: Runs the PageRank algorithm on a given graph and plots the results.

- `pagerank_jump_matrix.py`:
  - `build_traditional_pagerank_matrix`: Constructs the traditional PageRank matrix.
  - `build_in_degree_jump_matrix`: Constructs the PageRank random jump matrix based on in-degrees.
  - `build_out_degree_jump_matrix`: Constructs the PageRank random jump matrix based on out-degrees.

- `pagerank_mapreduce.py`:
  - `mapper`: Mapper function for the PageRank algorithm using MapReduce.
  - `reducer`: Reducer function for the PageRank algorithm using MapReduce.
  - `run_pagerank_mapreduce`: Runs the PageRank algorithm using MapReduce with PySpark.

## Features
- Computation of PageRank scores using different methods
- Exploration of the impact of the damping factor and jump matrix types
- Visualization of the convergence behavior using convergence plots
- Graph plotting with node coloring based on PageRank scores
- Calculation of the Gini coefficient to measure the inequality in the distribution of PageRank scores

## Examples

The code includes example usage in the `if __name__ == "__main__":` block of each file. These examples demonstrate how to use the functions to compute PageRank scores, plot the graph, and analyze the results.

The code includes examples of small and large graphs, such as the football network, to illustrate the algorithm's performance. You can modify the code to use your own graph data or adjust the parameters as needed.

Feel free to modify the examples or create your own scripts to experiment with different graphs and parameter settings.

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This code is released under the [GNU GPL-3.0 License](LICENSE). Feel free to use, modify, and distribute the code as per the terms of the license.