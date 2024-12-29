import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import tsplib95
import statistics

def get_cost(Graph, node1, node2):
    if node2 in Graph[node1]:
        return Graph[node1][node2]['weight']
    else:
        return Graph[node2][node1]['weight']

def get_neighbours(Graph, node):
    # return also the distacne between the nodes
    return {node: attributes['weight'] for node, attributes in Graph[node].items()}

# possibly to do: compute the savings matrix

def get_graph_image(Graph, ax):
    pos = nx.get_node_attributes(Graph, 'coord')  # Retrieve node positions
    ax.axis('off')  # Turn off axes
    
    # Draw the graph on the provided axes
    nx.draw_networkx_nodes(Graph, pos, ax=ax, node_size=100)
    nx.draw_networkx_edges(Graph, pos, ax=ax, width=0.1)
    nx.draw_networkx_labels(Graph, pos, ax=ax, font_size=6, font_color="white")

    ax.set_title("Graph Visualization")

def plot_graph(Graph):
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
    
    plt.title("Graph Visualization")
    plt.axis('off')
    plt.show()

def plot_path(Graph, path):
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
    
    plt.title("Graph with Path Highlighted")
    plt.axis('off')
    plt.show()


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



        
class TSP:
    # graph is an instance of class Graph
    def __init__(self, graph, alpha, beta, alpha_rate, beta_rate, rate, rho, max_iter, k, heuristic_cost, norm, opt=False, check_improvement=True, update_steps=200, elitism = False, elitism_steps = None, bias = False):
        Path.set_shared_graph(graph, norm, alpha, beta)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.max_iter = max_iter
        self.norm = norm
        self.num_vertices = len(graph.nodes)
        self.k = k # number of ants
        self.pheromone = np.ones([self.num_vertices, self.num_vertices]) / heuristic_cost
        self.heuristic_cost = heuristic_cost
        Path.set_shared_pheromone(self.pheromone)
        self.paths = [] # list of paths
        self.best_path = None
        self.history = []
        self.opt = opt
        self.alpha_rate = alpha_rate
        self.beta_rate = beta_rate
        self.rate = rate
        self.check = check_improvement # whether or not to check if the cost is decreasing
        self.update_steps = update_steps # for how many iterations to update the alpha and beta parameters
        self.elitism = elitism
        self.elitism_steps = elitism_steps
        if not elitism:
            self.elitism_steps = max_iter
        self.bias = bias

    def forward(self):
        path = Path()
        path.generate_path()
        self.paths.append(path)

    def update_pheromone(self):
        # Evaporate pheromones on all edges
        self.pheromone *= (1 - self.rho)
        n = len(self.paths)
        if self.elitism:
            ranked_paths = sorted(self.paths, key=lambda path: path.cost)
            ranked_paths = ranked_paths[:-n//2]
            n = len(ranked_paths)
            for i in range(self.num_vertices):
                for j in range(self.num_vertices):
                    delta = 0
                    for k in range(n):
                        # check if edge [i,j] is in path k
                        if ranked_paths[k].path[i] == j:
                            delta += 1 / ranked_paths[k].cost
                    self.pheromone[i][j] += delta
        else:
            for i in range(self.num_vertices):
                for j in range(self.num_vertices):
                    delta = 0
                    for k in range(n):
                        # check if edge [i,j] is in path k
                        if self.paths[k].path[i] == j:
                            delta += 1 / self.paths[k].cost
                    self.pheromone[i][j] += delta

    def check_improvement(self):
        # compute the medians of the last 5 intervals of rate size
        medians = []
        for k in reversed(range(5)):
            medians.append(np.median(self.history[-((k+1)*self.rate+1):-(k*self.rate+1)]))
        # check if the cost is increasing or decreasing
        slope = np.polyfit(range(5), medians, 1)[0]
        return slope < 0

    def solve(self):
        for i in range(self.max_iter):
            #print(f"Iteration {i}")
            best_cost = 100000000
            best_path = None
            self.paths = []
            if i % self.rate == 0:
                if i < self.update_steps:
                    self.alpha *= self.alpha_rate
                    self.beta *= self.beta_rate
                    Path.update_alpha_beta(self.alpha, self.beta)
                else:
                    if self.check:
                        if self.check_improvement():
                            self.alpha *= self.alpha_rate
                            self.beta *= self.beta_rate
                            Path.update_alpha_beta(self.alpha, self.beta)
                        else:
                            self.alpha /= self.alpha_rate
                            self.beta /= self.beta_rate
                            Path.update_alpha_beta(self.alpha, self.beta)

            for _ in range(self.k):  # Generate paths for all ants
                try:
                    self.forward()  # May raise an exception
                except ValueError as e:
                    print(f"Error in path generation: {e}. Skipping this ant.")
                    self.pheromone += 1 / self.heuristic_cost
                    continue
            if Path.zero_pheromone:
                print("Zero pheromone")
            for path in self.paths: # find the best path in the current iteration
                if path.cost < best_cost:
                    best_cost = path.cost
                    best_path = path
            self.history.append(best_cost)
            if self.best_path is None or best_cost < self.best_path.cost:
                self.best_path = best_path

            if i < self.elitism_steps:
                self.elitism = False
            else: 
                self.elitism = True
            # this is done to add some exploitation, we noted that we get far from the optimum with the median best costs at each iteration
            if self.bias and i > self.update_steps:
                self.paths.append(self.best_path)
            self.update_pheromone()
            Path.set_shared_pheromone(self.pheromone)

        if self.opt:
            self.best_path.opt_alg()
        return self.best_path

class Path:
    # Class-level graph shared among all instances
    shared_graph = None
    num_vertices = 0
    shared_pheromone = None
    norm = 0
    alpha = 0
    beta = 0

    @classmethod
    def set_shared_graph(cls, graph, alpha, beta, norm):
        cls.shared_graph = graph
        cls.num_vertices = len(graph.nodes)
        cls.norm = norm
        cls.alpha = alpha
        cls.beta = beta
        cls.zero_pheromone = False

    @classmethod
    def set_shared_pheromone(cls, pheromone):
        cls.shared_pheromone = pheromone

    @classmethod
    def update_alpha_beta(cls, alpha, beta):
        cls.alpha = alpha
        cls.beta = beta

    def __init__(self, path=None):
        self.graph = Path.shared_graph
        self.pheromone = Path.shared_pheromone
        self.num_vertices = Path.num_vertices
        self.visited = [False] * self.num_vertices
        self.cost = 0
        self.path = None
        self.alpha = Path.alpha
        self.beta = Path.beta
        self.norm = Path.norm
        self.unvisited = [i for i in range(self.num_vertices)]
        if path is not None:
            self.path = path
            self.visited = [True] * self.num_vertices
            self.cost = sum([get_cost(self.graph, path[i], path[i+1]) for i in range(len(path) - 1)])

    def choose_next_node(self):
        # Choose the next node based on the probabilities
        current_node = self.path[-1]
        neighbours = get_neighbours(self.graph, current_node)
        # Calculate probabilities based on pheromone and cost
        probabilities = [self.pheromone[current_node][n] ** self.alpha * (1 / (self.norm * weight)) ** self.beta  for n, weight in neighbours.items() if not self.visited[n]]
        if sum(probabilities) > 0:
            Path.zero_pheromone = False
            probabilities = [p / sum(probabilities) for p in probabilities]
        else:
            Path.zero_pheromone = True
            try:
                probabilities = [(1 / weight) for n, weight in neighbours.items() if not self.visited[n]]
                probabilities = [p / sum(probabilities) for p in probabilities]
            except ZeroDivisionError:
                print("ZeroDivisionError")
        next_node = np.random.choice(self.unvisited, p=probabilities)
        self.visited[next_node] = True
        self.unvisited.remove(next_node)

        cost = neighbours[next_node]
        return next_node, cost
    
    def generate_path(self):
        first_node = random.randint(0, self.num_vertices - 1)
        self.path = [first_node]
        self.visited[first_node] = True
        self.unvisited.remove(first_node)
        while sum(self.visited) < self.num_vertices:
            next_node, cost = self.choose_next_node()
            self.path.append(next_node)
            self.cost += cost
        # add the first node to create a loop
        neighbours = get_neighbours(self.graph, next_node)
        cost = neighbours[first_node]
        self.cost += cost
        self.path.append(first_node)

    def opt_alg(self):
        improvement = True
        while improvement:
            improvement = False
            indices = list(range(len(self.path) - 2))
            random.shuffle(indices)
            for i in indices:
                sub_indices = list(range(i + 2, len(self.path) - 1))
                random.shuffle(sub_indices)
                for j in sub_indices:
                    if self._swap_edges(i, j):
                        improvement = True

    def _swap_edges(self, i, j):
        # Check if swapping improves the cost
        a, b, c, d = self.path[i], self.path[i+1], self.path[j], self.path[j+1]
        cost_before = get_cost(self.graph, a, b) + get_cost(self.graph, c, d)
        cost_after = get_cost(self.graph, a, c) + get_cost(self.graph, b, d)
        if cost_after < cost_before:
            # Perform the swap
            self.path[i+1:j+1] = reversed(self.path[i+1:j+1])
            self.cost -= (cost_before - cost_after)
            return True
        return False


# find the min_path for each problem based only on heuristic information
def min_path_cost(graph):
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

        

