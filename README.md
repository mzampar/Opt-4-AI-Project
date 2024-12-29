# Solving the Traveling Salesman Problem with Ant Colony Optimisation

## Optimisation for Artificial Intelligence - Final Project

## Author: Marco Zampar

## Problem Statement

The Traveling salesman problem, as the name suggests, is a combinatorial optimisation problem in which we seek to find the shortest hamiltonian circuit in a weighted and complete graph, a graph where each node is connected to any other nodes. It is a NP-hard problem.

A hamiltonian circuit is a sequence of nodes in which any node occurs just once, excpet for the begin and end of the sequence, which has to be the same. The shortest path means that the sum of the weights of the edges that compose the path is minimal.

## Benchmarks

Benchmarks for the TSP were taken from https://github.com/mastqe/tsplib/tree/master.

[//]: # (Benchmarks for the VRP were taken from https://github.com/Fedoration/CVRPLIB/tree/master/data https://github.com/PyVRP/VRPLIB/tree/main/tests/data.)

## Ant Colony Optimisation

Ant colony is based on the idea of having many agents that move independently on each other. They move through the graph based on a probability distribution built on 2 different kinds of informations: the heuristics and the knowledge of the previous generations. 

The heuristics is proportional to the inverse of the weight of the graph, while the knowledge of the previous information is based on the pheromone left by the previous ants: an ant leaves an amount of pheromone on the circuit it traveled proportional to the cost of the circuit.

This is a very interesting example of exploration and exploitation: we exploit the heuristic knowledge (the cost of an edge), but we still explore other solutions because we move with a probability distribution: this is done through the pheromone, which compensates the effect of the heuristics.

It is interesting to note that the 2 aspects are both fundamental, and working with just one of them doesn't work.

The most important contribution I implemented is the balancing of the parameters that regulate the weight of the pheromone and heuristics in computing the probability distribution: we start with similar coefficients but then we give more importance to the knowledge based on the pheromone.

## Parallel considerations

It is interesting to note that, since the agents that travel through the graph are various and independent, it is not difficult to parallelise the implementation of the algorithm: each agent is a single process that has to keep in memory the graph (adjacency matrix) and the pheromone matrix and when a circuit is built it sends it to the master process. This approach will be probably beneficial for large graphs, where the creation of the circuit can be expensive.

## Conclusion

In conclusion, we implemented a code to solve the TSP that works and can find sufficiently good solutions in a relative small amount of time, in many cases this can be helpful: we trade-off some cost in the solution for a faster execution. We also implemented a local optimisation algorithm that improves the final solution.

