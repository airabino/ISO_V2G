import time
import numpy as np

from operator import itemgetter

def savings_matrix(adjacency, depot_index = 0, bounds = (0, np.inf)):
	'''
	Computing the savings matrix from an adjacency matrix.

	Savings is the difference between:

	depot -> destination 1 -> depot -> destination 2 -> depot

	and

	depot -> destination 1 -> destination 2 -> depot


	Values for links with negative savings or invalid lnks will be set to zero

	Requires the definition of a depot location
	'''

	cost_from, cost_to = np.meshgrid(adjacency[:, depot_index], adjacency[depot_index])

	savings = cost_from + cost_to - adjacency

	# Negative savings should not be considered
	savings[savings < 0] = 0

	# No self-savings
	savings[np.diag_indices(adjacency.shape[0])] = 0

	#No savings from infeasible edges
	savings[adjacency < bounds[0]] = 0
	savings[adjacency > bounds[1]] = 0

	return savings

def initial_routes(adjacency, depot_index):
	'''
	Prodces list of initial 1-stop routes for the Clarke Wright algorithm where
	each route connects a stop to the nearest depot
	'''

	if type(adjacency) is np.ndarray:
		adjacency = [adjacency]

	#Pulling destination indices
	destination_indices = np.array([idx for idx in range(adjacency[0].shape[0])])

	# Creating initial routes
	routes = []
	route_weights = []

	for destination_index in destination_indices:

		routes.append([
			depot_index,
			destination_index,
			depot_index,
			])

		route_weight=([
			adj[depot_index, destination_index]+
			adj[destination_index, depot_index] \
			for adj in adjacency
		])

		route_weights.append(route_weight)

	return routes, route_weights

def find_routes(routes, node_0, node_1):

	first_route_index = []
	second_route_index = []

	itemget = itemgetter(1)

	result = filter(
		lambda idx: itemget(routes[idx]) == (node_0),
		list(range(len(routes)))
		)

	for res in result:

		first_route_index = res

	itemget = itemgetter(-2)

	result = filter(
		lambda idx: itemget(routes[idx]) == (node_1),
		list(range(len(routes)))
		)

	for res in result:

		second_route_index = res

	return first_route_index, second_route_index

def evaluate_route(adjacency, route, route_bounds, leg_bounds, stop_weights):

	n = len(adjacency)

	if n < 4:

		weights = [0] * n

		validity = [True] * n

		for idx_adj in range(n):

			from_indices = route[:-1]
			to_indices = route[1:]

			for idx_leg in range(len(from_indices)):

				leg_weight = adjacency[idx_adj][from_indices[idx_leg], to_indices[idx_leg]]

				weights[idx_adj] += leg_weight + stop_weights[idx_adj]

				validity[idx_adj] *= leg_weight >= leg_bounds[idx_adj][0]
				validity[idx_adj] *= leg_weight <= leg_bounds[idx_adj][1]

			validity[idx_adj] *= weights[idx_adj] >= route_bounds[idx_adj][0]
			validity[idx_adj] *= weights[idx_adj] <= route_bounds[idx_adj][1]

		return weights, validity

	else:

		weights = [0] * n

		validity = [True] * n

		for idx_adj in range(n):

			from_indices = route[1:-2]
			to_indices = route[2:-1]

			weights[idx_adj] += adjacency[idx_adj][route[0], route[1]]
			weights[idx_adj] += adjacency[idx_adj][route[-2], route[-1]]

			for idx_leg in range(len(from_indices)):

				leg_weight = adjacency[idx_adj][from_indices[idx_leg], to_indices[idx_leg]]

				weights[idx_adj] += leg_weight + stop_weights[idx_adj]

				validity[idx_adj] *= leg_weight >= leg_bounds[idx_adj][0]
				validity[idx_adj] *= leg_weight <= leg_bounds[idx_adj][1]

			validity[idx_adj] *= weights[idx_adj] >= route_bounds[idx_adj][0]
			validity[idx_adj] *= weights[idx_adj] <= route_bounds[idx_adj][1]

		return weights, validity

def clarke_wright(adjacency, depot_index, route_bounds, leg_bounds, stop_weights, **kwargs):
	'''
	Implements Clarke and Wright savings algorith for solving the VRP with flexible
	numbers of vehicles per depot. Vehicles have range and capacity limitations. This
	implementation allows for multiple depots.

	The Clarke and Wright method attempts to implment all savings available in a savings
	matrix by iteratively merging routes. Routes are initialized as 1-stop routes between
	each destination and its closest depot. During iteration, savings are implemented
	by merging the routes which allow for the capture of the greatest savings link
	available.
	'''
	
	kwargs.setdefault('max_iterations', 100000)
	
	#Computing savings matrices for all adjacency matrices and all depots
	savings = []

	for idx, adj in enumerate(adjacency):

		savings_adj = savings_matrix(adj, depot_index, leg_bounds[idx])

		savings.append(savings_adj)

	# Initializing routes - initial assumption is that locations will be served by
	# closest depot. All initial routes are 1-stop (depot -> destination -> depot)
	routes, route_weights = initial_routes(adjacency, depot_index)

	# Removing routes which are inherently non-combinable because they consist of legs
	# which are greater than half of the route bound
	routes = np.array(routes)
	route_weights = np.array(route_weights)

	combinable = np.array([True] * len(routes))

	for idx_adj in range(len(adjacency)):

		for idx_route in range(len(routes)):

			combinable[idx_route] *= (
				route_weights[idx_route, idx_adj] / 2 <= leg_bounds[idx_adj][1]
				)

	savings[0][~combinable, :] = 0
	savings[0][:, ~combinable] = 0

	routes = routes[combinable].tolist()
	route_weights = route_weights[combinable].tolist()

	# Implementing savings
	success = False

	for idx in range(kwargs['max_iterations']):

		# Computing remaining savings
		remaining_savings = savings[0].sum()

		# If all savings incorporated then exit
		if remaining_savings == 0:

			success = True

			break

		# Finding link with highest remaining savings
		best_savings_link = np.unravel_index(np.argmax(savings[0]), savings[0].shape)

		# Finding routes to merge - the routes can only be merged if there are
		# routes which start with and end with the to and from index respectively.

		first_route_index = []
		second_route_index = []

		first_route_index,second_route_index = find_routes(
			routes,
			best_savings_link[0],
			best_savings_link[1],
			)

		# If a valid merge combination is found create a tentative route and evaluate
		if first_route_index and second_route_index:

			# Creating tentative route
			tentative_route = (
				routes[first_route_index][:-1] +
				routes[second_route_index][1:]
				)

			# Finding the best of the tentative routes
			tentative_route_weights = []
			tentative_route_validity = []

			tentative_route_weights, tentative_route_validity = evaluate_route(
				adjacency,
				tentative_route,
				route_bounds,
				leg_bounds,
				stop_weights
				)

			# Checking if the merged route represents savings over the individual routes
			improvement = tentative_route_weights[0] <= (
				route_weights[first_route_index][0] +
				route_weights[second_route_index][0]
				)

			feasible=np.all(tentative_route_validity)

			# If the merged route is an improvement and feasible it is integrated
			if improvement and feasible:

				# Adding the merged route
				routes[first_route_index] = tentative_route
				route_weights[first_route_index] = tentative_route_weights

				# Removing the individual routes
				routes.remove(routes[second_route_index])
				route_weights.remove(route_weights[second_route_index])

		# Removing the savings
		savings[0][best_savings_link[0], best_savings_link[1]] = 0
		savings[0][best_savings_link[1], best_savings_link[0]] = 0

	return routes, success