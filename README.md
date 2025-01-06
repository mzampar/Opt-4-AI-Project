# Solving the Traveling Salesman Problem with Ant Colony Optimisation

## Optimisation for Artificial Intelligence - Final Project

## Author: Marco Zampar

## Problem Statement

The Traveling salesman problem, as the name suggests, is a combinatorial optimisation problem in which we seek to find the shortest hamiltonian circuit in a weighted and complete graph, a graph where each node is connected to any other nodes. It is a NP-hard problem.

A hamiltonian circuit is a sequence of nodes in which any node occurs just once, except for the beginning and end of the sequence, which have to be the same. The shortest path means that the sum of the weights of the edges that compose the path is minimal.

## Benchmarks

Benchmarks for the TSP were taken from https://github.com/mastqe/tsplib/tree/master.

[//]: # (Benchmarks for the VRP were taken from https://github.com/Fedoration/CVRPLIB/tree/master/data https://github.com/PyVRP/VRPLIB/tree/main/tests/data.)


## Ant Colony Optimisation

Ant colony is based on the idea of having many agents that move independently on each other. They move through the graph based on a probability distribution built on 2 different kinds of informations: the heuristics and the knowledge of the previous generations. 

The heuristics is proportional to the inverse of the weight of the graph, while the knowledge of the previous information is based on the pheromone left by the previous ants: an ant leaves an amount of pheromone on the circuit it traveled proportional to the cost of the circuit.

This is a very interesting example of exploration and exploitation: we exploit the heuristic knowledge (the cost of an edge), but we still explore other solutions because we move with a probability distribution: this is done through the pheromone, which compensates the effect of the heuristics.

It is interesting to note that the 2 aspects are both fundamental, and working with just one of them doesn't work.

## Description of the implemented algorithm

I leveraged a class `Graph` of the `networkx` package and I implemented two classes: `TSP`, to regulate the parameters of the algorithm, and `Path` class, to produce solutions at each step, with global attributes, to store and access efficiently the Graph of interest and the pheromone matrix.

If we think about the first iterations of the algorithm, we know that the pheromone matrix is uniformily initialised, so the path will be mostly determined by the information coming from the heuristic, i.e. the inter-nodes distance. So we expect that in the first iterations the cost of the solutions will be close to the cost of a path determined choosing the next node as the closest node to the actual node. That is why to have a "smooth" update of the pheromone matrix we initialise it with 1/`heruistic_cost`.

After having implemented a first version of the algorithm, I noticed it didn't work very well, that's why I decided to try to gradually update of the coefficients `alpha` and `beta`, the first was reduced while the second increased by a factor (0.95, 1.05) after a certain number of steps: in this way the algorithm starts with similar coefficients but then gives more importance to the knowledge based on the pheromone. 

I noticed that this allowed a more efficient and better learning than using fixed alpha and beta.

But it can lead to numerical problems when the `beta` coefficient becomes too big: the probabilites can become NaN. If this happens, we take back alpha and beta to the configuration of the last best path found.

Another numerical problem comes in when computing the probablities to choose the next node of a path: the orders of magnitude of the pheromone matrix and the heuristic information have to be comparable, otherwise the greater of the two will overwhelm the other, loosing all the power of ACO, I will show indeed that in both cases, using only the pheromone either the heuristic information, leads to useless solutions.

To solve this problem, I choose to compute the probabilites of moving to the next node in two separate ways, one for the pheromone and another for the heuristic and then weighting them with alpha and beta.

I also tried to use elitism in the update of the pheromone, i.e. to update it using only the best half of the generated paths, but this didn't bring any improvement.

Also biasing the pheromone with the current best path found doesn't bring a significant improvement.

To have a fair comparison between the different parameters, whenever the `solve` function is called, I set the seed to 0.

I also implemented the 2-opt algorithm and found out that it significantly improves the solutions and combined with my algorithm can, in some cases, find the optimal.


## Parallel considerations

It is interesting to note that, since the agents that travel through the graph are various and independent, it is not difficult to parallelise the implementation of the algorithm: each agent is a single process that needs access to the graph (adjacency matrix) and the pheromone matrix and when it builds a circuit it sends it to the master process. This approach will be probably beneficial for large graphs, where the creation of the circuit can be long and expensive.

## Conclusion

In conclusion, the code I implemented to solve the TSP works and can find sufficiently good solutions in a relative small amount of time, compared to all the possible solutions to this kind of problem, which is `n!`.


