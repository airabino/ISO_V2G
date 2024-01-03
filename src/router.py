'''
Creates routes for vehicles and depots on a graph
'''

import os
import sys
import time
import numpy as np
import networkx as nx

from copy import deepcopy
from operator import itemgetter
from itertools import product as iter_prod

from .utilities import ProgressBar
from .graph import subgraph
from .clarke_wright  import *
from .simulated_annealing import *

def voronoi_cells(graph, center_nodes, nodes = None, weight = None, **kwargs):
	'''
	assign nodes to reference nodes by proximity
	'''

	voronoi_cells = nx.voronoi_cells(graph, center_nodes, weight = weight)

	for key in voronoi_cells.keys():

		if nodes is not None:

			voronoi_cells[key] = np.intersect1d(list(voronoi_cells[key]), nodes)

		else:

			voronoi_cells[key] = np.array(list(voronoi_cells[key]))

	return voronoi_cells

def assign_depot(graph, depot_nodes, nodes = None, weight = None, **kwargs):
	'''
	Assign nodes to depots using weighted Voronoi cells 
	'''
	kwargs.setdefault('field', 'depot')
	kwargs.setdefault('overwrite_depots', True)

	field = kwargs['field']

	vc = voronoi_cells(graph, depot_nodes, nodes = nodes, weight = weight)

	for depot in depot_nodes:

		for node in vc[depot]:

			if kwargs['overwrite_depots'] or (field not in graph._node[node].keys()):

				graph._node[node][field] = depot

	for node in graph.nodes:

		if field not in graph._node[node].keys():

			graph._node[node][field] = ''

	return graph

def assign_rng(graph, seed = None, field = 'rng', **kwargs):
	'''
	Assign a random number to each node
	'''

	rng  = np.random.default_rng(seed)

	for node in graph.nodes:
		graph._node[node][field] = rng.random()

	return graph

def assign_vehicle(graph, vehicles, field = 'vehicle', **kwargs):
	'''
	Assigns vehicles based on vehicle node_criteria functions
	'''

	for node in graph.nodes:

		graph._node[node][field] = []

	for vehicle, vehicle_information in vehicles.items():

		for key, fun in vehicle_information['node_criteria'].items():

			if type(fun) is str:

				vehicles[vehicle]['node_criteria'][key] = eval(fun)
	
	for vehicle, vehicle_information in vehicles.items():

		for node in graph.nodes:

			meets_criteria = True

			try:

				for key, fun in vehicle_information['node_criteria'].items():

					meets_criteria *= fun(graph._node[node])

			except:

				meets_criteria = False

			if meets_criteria:

				graph._node[node][field].append(vehicle)

	return graph

def produce_subgraphs(graph, categories, **kwargs):

	subgraphs = {}

	combinations = list(iter_prod(*[v for v in categories.values()]))

	# print(combinations)

	for combination in combinations:

		nodelist = []

		for node in graph.nodes:

			include = True

			for idx, field in enumerate(categories.keys()):

				val = graph._node[node][field]

				if hasattr(val, '__iter__'):

					if len(val) == 0:

						include = False

					else:

						include *= combination[idx] in graph._node[node][field]

				else:

					include *= graph._node[node][field] == combination[idx]

			if include:

				nodelist.append(node)

		subgraphs[combination] = subgraph(graph, nodelist)

	return subgraphs

def produce_adjacency(subgraphs, weights = [], **kwargs):

	adjacency = {}

	for key, value in subgraphs.items():

		adjacency[key] = adjacency_matrices(value, weights = weights)

	return adjacency

def adjacency_matrices(graph, nodelist = None, weights = [], **kwargs):
	'''
	Produces list of adjacency matrices for each weight in weights
	'''
	adjacency = []

	for weight in weights:

		adjacency.append(nx.to_numpy_array(
			graph,
			nodelist = nodelist,
			weight = weight,
			nonedge = np.inf,
			))

	return adjacency

def produce_assignments(subgraphs, **kwargs):

	subgraph_assignments = {}

	for key, value in subgraphs.items():

		node_to_idx, idx_to_node = assignments(list(value.nodes))

		subgraph_assignments[key] = {}
		subgraph_assignments[key]['node_to_idx'] = node_to_idx
		subgraph_assignments[key]['idx_to_node'] = idx_to_node

	return subgraph_assignments

def assignments(nodes, **kwargs):

	node_to_idx = {nodes[idx]: idx for idx in range(len(nodes))}
	idx_to_node = {val: key for key, val in node_to_idx.items()}

	return node_to_idx, idx_to_node

def produce_bounds(subgraphs, vehicles, weights, **kwargs):

	route_bounds = {}
	leg_bounds = {}
	stop_weights = {}

	for key, value in subgraphs.items():

		vehicle, _ = key

		route_bounds[key] = []
		leg_bounds[key] = []
		stop_weights[key] = []

		for weight in weights:

			route_bounds[key].append(vehicles[vehicle]["route_bounds"][weight])
			leg_bounds[key].append(vehicles[vehicle]["leg_bounds"][weight])
			stop_weights[key].append(vehicles[vehicle]["stop_weights"][weight])

	return route_bounds, leg_bounds, stop_weights

def produce_information(subgraphs, vehicles, **kwargs):

	information = {}

	for key, value in subgraphs.items():

		vehicle, depot = key

		information[key] = {}

		information[key]['vehicle'] = vehicle
		information[key]['depot'] = depot
		information[key]['fleet_size'] = vehicles[vehicle]['fleet_size']

	return information

def produce_routing_inputs(graph, parameters, **kwargs):

	# Assigning depots by Voronoi cells unless otherwise specified
	depot_nodes = parameters['depot_nodes']
	voronoi_weight = parameters['voronoi_weight']

	graph = assign_depot(graph, depot_nodes, voronoi_weight = voronoi_weight, **kwargs)

	# Assinging random number to each node for selection
	seed = parameters['rng_seed']

	graph = assign_rng(graph, seed, **kwargs)
	
	# Assinging vehicles to nodes
	vehicles = parameters['vehicles']

	graph = assign_vehicle(graph, vehicles, **kwargs)

	# Producing subgraphs for routing
	categories = {
		'vehicle': list(parameters['vehicles'].keys()),
		'depot': parameters['depot_nodes'],
	}

	subgraphs = produce_subgraphs(graph, categories, **kwargs)

	# Producing adjacency matrices for routing
	route_weights = parameters['route_weights']

	adjacency = produce_adjacency(subgraphs, route_weights, **kwargs)

	# Producing assignments for routing
	assignments = produce_assignments(subgraphs, **kwargs)

	# Producing bounds
	route_bounds, leg_bounds, stop_weights = produce_bounds(
		subgraphs, vehicles, route_weights, **kwargs)

	# Producing case information
	information = produce_information(subgraphs, vehicles, **kwargs)

	# Combining
	cases = {}

	for key in subgraphs.keys():

		cases[key]={}

		cases[key]['graph'] = subgraphs[key]
		cases[key]['adjacency'] = adjacency[key]
		cases[key]['assignments'] = assignments[key]
		cases[key]['route_bounds'] = route_bounds[key]
		cases[key]['leg_bounds'] = leg_bounds[key]
		cases[key]['stop_weights'] = stop_weights[key]
		cases[key]['information'] = information[key]
	
	return cases

def router(case, **kwargs):

	kwargs.setdefault('steps_routes', 1000)
	kwargs.setdefault('steps_route', 100)

	depot_index = case['assignments']['node_to_idx'][case['information']['depot']]

	routes, success = clarke_wright(
		case['adjacency'],
		depot_index,
		case['route_bounds'],
		case['leg_bounds'],
		case['stop_weights'],
		)

	# Removing depot - depot - depot route
	try:

		routes.remove([depot_index, depot_index, depot_index])

	except:

		pass
	
	# Annealing between routes
	routes = anneal_routes(
		case['adjacency'],
		routes,
		case['route_bounds'],
		case['leg_bounds'],
		case['stop_weights'],
		steps = kwargs['steps_routes'],
		)
	
	# Annealing within routes
	for route in routes:

		route = anneal_route(
			case['adjacency'],
			route,
			case['route_bounds'],
			case['leg_bounds'],
			case['stop_weights'],
			steps = kwargs['steps_route'],
			)

	# Adding routes to unreached destinations
	reached = []
	for route in routes:
		reached.extend(route)

	possible_destinations = list(range(len(case['adjacency'][0])))
	possible_destinations.remove(depot_index)
	not_reached = np.array(possible_destinations)[~np.isin(possible_destinations, reached)]

	for destination in not_reached:

		routes.append([depot_index, destination, depot_index])

	# Converting routes from indices to nodes
	for idx, route in enumerate(routes):

		routes[idx] = [case['assignments']['idx_to_node'][node_idx] for node_idx in route]

	if not np.isinf(case['information']['fleet_size']):

		route_lenghts = [len(route) for route in routes]
		keep_indices = np.argsort(route_lenghts)[-case['information']['fleet_size']:]
		routes = [routes[idx] for idx in keep_indices]

	return routes, success

def route_information(graph, raw_routes, fields):

	for key, fun in fields.items():

		if isinstance(fun, str):

			fields[key] = eval(fun)

	routes = []

	for raw_route in raw_routes:

		route = {key: [] for key in fields.keys()}
		route['nodes'] = raw_route

		for node in raw_route:

			for key, fun in fields.items():

				route[key].append(fun(graph._node[node]))

		routes.append(route)

	return routes