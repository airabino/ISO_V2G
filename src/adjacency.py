'''
Module for computing adjacency for a graph via routing on another graph. An example
would be computing travel times between cities connected by highways or latency between
computers connected via the internet. Another case would be compting network distances
between all points in a subset of a greater network. In any case, the nodes of the former
network must be coincident, or nearly so, with nodes in the latter network.

In this module the graph for which adjacency is being computed will be referred to as the
"graph" while the graph on which the routing occurs will be referred to as the "atlas". In
cases where either could be used "graph" will be used as default.
'''

import numpy as np

from scipy.spatial import KDTree

from .progress_bar import ProgressBar
from .dijkstra import dijkstra

# Routing functions and related objects

def closest_nodes_from_coordinates(graph, x, y):
	'''
	Creates an assignment dictionary mapping between points and closest nodes
	'''

	# Pulling coordinates from graph
	xy_graph = np.array([(n['x'], n['y']) for n in graph._node.values()])
	xy_graph = xy_graph.reshape((-1,2))

	# Creating spatial KDTree for assignment
	kd_tree = KDTree(xy_graph)

	# Shaping input coordinates
	xy_query = np.vstack((x, y)).T

	# Computing assignment
	result = kd_tree.query(xy_query)

	node_assignment = []

	for idx in range(len(x)):

		node = result[1][idx]

		node_assignment.append({
			'id':node,
			'query':xy_query[idx],
			'result':xy_graph[node],
			})

	return node_assignment

def single_source_dijkstra(atlas, source, targets, weights, return_paths = False):
	'''
	Compute lowest cost route(s) from source to target(s) on atlas.
	See .dijkstra.dijkstra for details on inputs
	'''

	if not hasattr(targets, '__iter__'):
		targets=[targets]

	route_weights, routes = dijkstra(
		atlas,
		[source],
		[],
		weights,
		return_paths,
		)

	keys = np.array(list(route_weights.keys()))
	route_information = np.array(list(route_weights.values()))

	select = np.isin(keys, targets)

	keys_select = keys[select]
	route_information_select = route_information[select]

	result = []

	for idx, key in enumerate(keys_select):

		result.append({
			'source': source,
			'target': key,
			**{weight: route_information_select[idx][idx_weight] \
			for idx_weight, weight in enumerate(weights)},
			})

	return result

def multiple_source_dijkstra(atlas, sources, targets, weights, **kwargs):
	'''
	Compute lowest cost route(s) from source to target(s) on atlas.
	See .dijkstra.dijkstra for details on inputs
	'''

	kwargs.setdefault('pb_kwargs', {'disp': True})
	kwargs.setdefault('dijkstra_kwargs', {'return_paths': False})

	if not hasattr(targets, '__iter__'):
		targets=[targets]

	results = []

	for source in ProgressBar(sources, **kwargs['pb_kwargs']):

		result = single_source_dijkstra(
			atlas, source, targets, weights, **kwargs['dijkstra_kwargs'])

		results.extend(result)

	return results

def node_assignment(atlas, graph):
	'''
	Maps closest nodes from atlas to graph and graph to atlas - assumes 2D graph
	'''

	# Pulling coordinates from atlas
	xy_atlas = np.array([(n['x'], n['y']) for n in atlas._node.values()])
	xy_atlas = xy_atlas.reshape((-1,2))

	# Creating spatial KDTree for assignment
	kd_tree = KDTree(xy_atlas)

	# Pulling coordinates from graph
	xy_graph = np.array([(n['x'], n['y']) for n in graph._node.values()])
	xy_graph = xy_graph.reshape((-1,2))

	graph_nodes=list(graph.nodes)

	# Computing assignment
	result = kd_tree.query(xy_graph)

	graph_to_atlas = {}
	atlas_to_graph = {}

	for idx in range(len(xy_graph)):

		graph_to_atlas[graph_nodes[idx]] = result[1][idx]
		atlas_to_graph[result[1][idx]] = graph_nodes[idx]

	return graph_to_atlas, atlas_to_graph

def adjacency(atlas, graph, weights, **kwargs):
	'''
	Computing adjacency for graph by routing along atlas
	'''

	kwargs.setdefault('pb_kwargs', {'disp': True})
	kwargs.setdefault('dijkstra_kwargs', {'return_paths': False})
	kwargs.setdefault('compute_all', False)
	kwargs.setdefault('node_assignment_function', node_assignment)

	# Maps closest nodes from atlas to graph and graph to atlas
	graph_to_atlas, atlas_to_graph = kwargs['node_assignment_function'](atlas, graph)

	# All nodes of graph are assumed to be targets
	targets = [graph_to_atlas[n] for n in list(graph.nodes)]

	# Collecting status of all nodes in graph
	statuses = np.array([n['status'] for n in graph._node.values()])

	# Creating sources based on statuses and compute_all
	if kwargs['compute_all']:

		sources = targets[:]

	else:

		sources = [targets[idx] for idx, status in enumerate(statuses) if status == 0]

	for n in list(graph.nodes):

		graph._node[n]['status'] = 1

	# Computing routes between selected sources and all targets
	results = multiple_source_dijkstra(atlas, sources, targets, weights, **kwargs)

	# Compiling edge information from results into 3-tuple for adding to graph
	edges = []

	for result in results:

		source = atlas_to_graph[result.pop('source')]
		target = atlas_to_graph[result.pop('target')]

		edges.append((source, target, result))

	# Adding edges to graph
	graph.add_edges_from(edges)

	return graph