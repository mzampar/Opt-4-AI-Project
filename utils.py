# --- IMPORTS ---

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import tsplib95
import statistics

# --- FUNCTIONS TO WORK ON GRAPHS ---

# Get the cost of the edge between two nodes
def get_cost(Graph, node1, node2):
    if node2 in Graph[node1]:
        return Graph[node1][node2]['weight']
    else:
        return Graph[node2][node1]['weight']

# Get the neighbours of a node in the graph, returns a dicrionary with the neighbours as keys and the weights as values
def get_neighbours(Graph, node):
    return {node: attributes['weight'] for node, attributes in Graph[node].items()}

# Find the min_path for each problem based only on heuristic information
def min_path_cost(graph):
    random.seed(0)
    np.random.seed(0)
    n = len(graph.nodes)
    min_path = []
    visited = [False] * n
    # Start with a random initial node
    current_node = random.randint(0, n-1)
    min_path.append(current_node)
    visited[current_node] = True
    cost=0

    # Repeat until all nodes are visited
    while len(min_path) < n:
        neighbours = get_neighbours(graph, current_node)
        # Filter unvisited neighbors and their weights
        unvisited_neighbors = [(node, graph[current_node][node]['weight']) for node in neighbours if not visited[node]]
        if not unvisited_neighbors:
            # No valid neighbors, graph might not support a Hamiltonian circuit
            raise ValueError("Hamiltonian circuit cannot be completed.")
        # Find the neighbor with the minimum weight
        min_node = min(unvisited_neighbors, key=lambda x: x[1])[0]
        cost += graph[min_node][current_node]['weight']
        # Update the path and mark the node as visited
        min_path.append(min_node)
        visited[min_node] = True
        current_node = min_node

    # Close the circuit by returning to the first node
    min_path.append(min_path[0])
    cost += graph[min_path[-2]][min_path[-1]]['weight']

    return min_path, cost


# --- PLOT FUNCTIONS ---

# Plot the graph (used when images are displayed in a grid)
def get_graph_image(Graph, ax, title=None):
    pos = nx.get_node_attributes(Graph, 'coord')  # Retrieve node positions
    ax.axis('off')  # Turn off axes
    
    # Draw the graph on the provided axes
    nx.draw_networkx_nodes(Graph, pos, ax=ax, node_size=100)
    nx.draw_networkx_edges(Graph, pos, ax=ax, width=0.1)
    nx.draw_networkx_labels(Graph, pos, ax=ax, font_size=6, font_color="white")

    if title is not None:
        ax.set_title(title)

# Plot the graph 
def plot_graph(Graph, title=None):
    pos = nx.get_node_attributes(Graph, 'coord')  # Retrieve node positions
    # Draw nodes
    nx.draw_networkx_nodes(Graph, pos, node_size=100)
    
    # Draw edges
    nx.draw_networkx_edges(Graph, pos, width=.1)
    
    # Draw labels
    nx.draw_networkx_labels(Graph, pos, font_size=6, font_color="white")
    
    # Draw edge weights
    #edge_labels = nx.get_edge_attributes(self, 'weight')
    #nx.draw_networkx_edge_labels(self, pos, edge_labels=edge_labels, font_size=5)
    
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Plot the graph with the given path highlighted
def plot_path(Graph, path, title=None):
    pos = nx.get_node_attributes(Graph, 'coord')  # Retrieve node positions
    
    # Draw nodes
    nx.draw_networkx_nodes(Graph, pos, node_size=100)
    
    # Draw edges
    nx.draw_networkx_edges(Graph, pos, width=.1)
    
    # Draw labels
    nx.draw_networkx_labels(Graph, pos, font_size=6, font_color="white")
    
    # Highlight the given path in green
    path_edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
    nx.draw_networkx_edges(Graph, pos, edgelist=path_edges, edge_color='green', width=1)
    
    # Draw edge weights
    #edge_labels = nx.get_edge_attributes(self, 'weight')
    #nx.draw_networkx_edge_labels(self, pos, edge_labels=edge_labels, font_size=5)
    
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()


# Plot the history of the best path costs
def plot_history(history, interval_size, ax=None, y_range=None):
    # Compute medians for fixed intervals
    medians = []
    intervals = []
    for i in range(0, len(history), interval_size):
        interval = history[i:i + interval_size]
        if interval:  # Ensure the interval is not empty
            medians.append(statistics.median(interval))
            intervals.append(i + interval_size // 2)  # Center of the interval for plotting

    # Use the provided ax or create a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the history
    ax.plot(history, label="Best Path Cost", linewidth=1, color="blue")
    ax.scatter(intervals, medians, color="red", label="Median", zorder=5)  # Red dots for medians
    ax.set_title("History of Best Path Costs")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Best Path Cost")
    ax.legend()
    ax.grid(True)

    # Set the y-axis range if provided
    if y_range is not None:
        ax.set_ylim(y_range)

    # Show the plot only if no ax is passed
    if ax is None:
        plt.show()


# --- TSP SOLVER ---
        
class TSP:
    def __init__(self, graph, alpha, beta, alpha_rate, beta_rate, rate, rho, max_iter, k, heuristic_cost, update_steps=None, elitism = False, elitism_steps = None, bias = False, seed = 0):
        self.alpha = alpha # weight of the pheromone
        self.beta = beta # weight of the heuristic information
        self.rho = rho # evaporation rate
        self.max_iter = max_iter # number of iterations
        self.num_vertices = len(graph.nodes)
        self.k = k # number of ants
        self.heuristic_cost = heuristic_cost
        self.pheromone = np.ones([self.num_vertices, self.num_vertices]) / heuristic_cost
        self.paths = [] # list of paths
        self.best_path = None
        self.history = [] # history of the best path costs
        self.alpha_rate = alpha_rate # coeff to update of the alpha parameter
        self.beta_rate = beta_rate # coeff to update of the beta parameter
        self.rate = rate # rate at which to update the alpha and beta parameters
        # for how many iterations to update the alpha and beta parameters
        self.update_steps = update_steps if update_steps is not None else max_iter
        self.elitism = elitism # whether or not to use elitism
        self.seed = seed
        self.elitism_steps = elitism_steps # from how many iterations to use elitism
        if not elitism:
            self.elitism_steps = max_iter
        self.bias = bias # whether or not to add the best path to the paths at each iteration
        Path.set_shared_pheromone(self.pheromone) # set the shared pheromone for each path
        Path.set_shared_vars(graph, alpha, beta) # set the shared graph for each path

    def forward(self):
        path = Path()
        path.generate_path()
        self.paths.append(path)

    def update_pheromone(self):
        # Evaporate pheromones on all edges
        self.pheromone *= self.rho
        # If elitism is enabled, only update the pheromones using the best half of the paths
        paths = self.paths
        if self.elitism:
            paths = sorted(self.paths, key=lambda path: path.cost)
            paths = paths[:-len(self.paths)//2]
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                delta = 0
                for path in paths:
                    # check if edge [i,j] is in the path
                    if path.path[i] == j:
                        delta += 1 / path.cost
                self.pheromone[i][j] += delta
        # Avoid pheromones going to 0, this prevents the algorithm from exploding
        self.pheromone = np.maximum(self.pheromone, 1e-10)

    # Main function to solve the TSP
    def solve(self):
        freezed = False
        # For a fair comparison when analysing the impact of the parameters, we set the seed to 0
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        for i in range(self.max_iter):
            best_cost = 100000000
            best_path = None
            self.paths = []
            # Generate paths for all ants
            for _ in range(self.k):  
                self.forward()

            # Find the best path in the current iteration
            for path in self.paths: 
                if path.cost < best_cost:
                    best_cost = path.cost
                    best_path = path
            self.history.append(best_cost)

            if self.best_path is None or best_cost < self.best_path.cost:
                self.best_path = best_path
                iteration = i

            if Path.zero_prob and freezed == False: 
                freezed = True
                print(f"Model freezed at iteration {i} because beta has become too big.")
                # Take back the parameters to the configuration of the best path found
                self.alpha /= (self.alpha_rate) ** (iteration//self.rate)
                self.beta /= (self.beta_rate) ** (iteration//self.rate)

            if not freezed and i < self.update_steps:  
                if i % self.rate == 0:
                    self.alpha *= self.alpha_rate
                    self.beta *= self.beta_rate
                    Path.update_alpha_beta(self.alpha, self.beta)

            self.update_pheromone()
            Path.set_shared_pheromone(self.pheromone)

            # If the model is initialised with self.elitism is False, elitism_steps=max_iter
            if i < self.elitism_steps:
                self.elitism = False
            else: 
                self.elitism = True

            # this is done to add some exploitation, we noted that we get far 
            # from the optimum with the median best costs at each iteration
            if self.bias:
                self.paths.append(self.best_path)

        return self.best_path, iteration

# --- PATH CLASS UTILITY ---

class Path:
    # Class-level graph and pheromone shared among all instances for efficiency and memory saving
    shared_graph = None
    num_vertices = 0
    shared_pheromone = None
    alpha = 0
    beta = 0
    zero_prob = False

    @classmethod
    def set_shared_vars(cls, graph, alpha, beta):
        cls.shared_graph = graph
        cls.num_vertices = len(graph.nodes)
        cls.alpha = alpha
        cls.beta = beta
        cls.zero_prob = False

    @classmethod
    def set_shared_pheromone(cls, pheromone, zero_prob = False):
        cls.shared_pheromone = pheromone
        cls.zero_prob = zero_prob

    @classmethod
    def update_alpha_beta(cls, alpha, beta):
        cls.alpha = alpha
        cls.beta = beta

    # Initialize a path with the class level variables and path-specific variables
    def __init__(self, path=None):
        self.graph = Path.shared_graph
        self.pheromone = Path.shared_pheromone
        self.num_vertices = Path.num_vertices
        self.cost = 0
        self.path = None
        self.alpha = Path.alpha
        self.beta = Path.beta
        self.unvisited = [i for i in range(self.num_vertices)]
        # If a path is provided, initialize the path and visited nodes
        if path is not None:
            self.path = path
            self.cost = sum(get_cost(self.graph, path[i], path[i+1]) for i in range(len(path) - 1))

    # Choose the next node based on the probabilities: pheromone and heuristic information
    def choose_next_node(self):
        current_node = self.path[-1]
        neighbours = get_neighbours(self.graph, current_node)
        # Calculate probabilities based on pheromone and cost
        # To avoid numerical problems to compare pheromone and distance, we compute the 2 separately, 
        # normalise and then combine them
        unvisited_neighbours = [n for n, _ in neighbours.items() if n in self.unvisited]
        pheromone_probs = [self.pheromone[current_node][n] for n in unvisited_neighbours]
        pheromone_probs = [p / sum(pheromone_probs) for p in pheromone_probs]
        distance_probs = [1 / weight for n, weight in neighbours.items() if n in self.unvisited]
        distance_probs = [p / sum(distance_probs) for p in distance_probs]
        probabilities = [(a ** self.alpha) * (b ** self.beta) for a,b in zip(pheromone_probs, distance_probs)]
        # If the beta parameter becomes too big, use the pheromone only
        if sum(probabilities) > 0:
            probabilities = [p / sum(probabilities) for p in probabilities]
        else:
            probabilities = pheromone_probs
        if not sum([b ** self.beta for b in distance_probs]) > 0:
            Path.zero_prob = True

        # Choose the next node and update the visited nodes, we work with 2 lists: visited and unvisited
        # This is why we use the visited 
        next_node = np.random.choice(self.unvisited, p=probabilities)
        self.unvisited.remove(next_node)

        return next_node
    
    def generate_path(self):
        first_node = random.randint(0, self.num_vertices - 1)
        self.path = [first_node]
        self.unvisited.remove(first_node)
        while len(self.unvisited) > 0:
            next_node = self.choose_next_node()
            self.path.append(next_node)

        # Add the first node to create a loop
        self.path.append(first_node)
        self.cost = sum(get_cost(self.graph, self.path[i], self.path[i+1]) for i in range(len(self.path) - 1))

    # Optimize the path using the 2-opt algorithm
    def opt_alg(self):
        improvement = True
        while improvement:
            improvement = False
            for i in range(len(self.path) - 2):
                for j in range(i + 2, len(self.path) - 1):
                    if self._swap_edges(i, j):
                        improvement = True

    def _swap_edges(self, i, j):
        # Check if swapping improves the cost
        a, b, c, d = self.path[i], self.path[i+1], self.path[j], self.path[j+1]
        cost_before = get_cost(self.graph, a, b) + get_cost(self.graph, c, d)
        cost_after = get_cost(self.graph, a, c) + get_cost(self.graph, b, d)
        if cost_after < cost_before:
            # If the cost is better, perform the swap
            self.path[i+1:j+1] = reversed(self.path[i+1:j+1])
            self.cost -= (cost_before - cost_after)
            return True
        return False




        

