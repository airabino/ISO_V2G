import numpy as np

from copy import deepcopy

def acceptance_probability(e, e_prime, temperature):

	return min([1, np.exp(-(e_prime - e) / temperature)])

def acceptance(e, e_prime, temperature):

	return acceptance_probability(e, e_prime, temperature) > np.random.rand()

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

def evaluate_routes(adjacency, routes, route_bounds, leg_bounds, stop_weights):

	weights = []

	validity = []

	for route in routes:

		route_weights, route_validity = evaluate_route(
			adjacency,
			route,
			route_bounds,
			leg_bounds,
			stop_weights,
			)

		weights.append(route_weights)
		validity.append(route_validity)

	return weights, validity

def anneal_route(adjacency, route, leg_bounds, route_bounds, stop_weights, **kwargs):

	kwargs.setdefault('steps', 100)
	kwargs.setdefault('initial_temperature', 1)

	# At lease 2 destinations need to be present for annealing
	if len(route) < 4:

		return route

	else:

		# Initializing temperature
		temperature = kwargs['initial_temperature']

		# Getting initial route weights
		route_weights, _ = evaluate_route(
			adjacency,
			route,
			route_bounds,
			leg_bounds,
			stop_weights,
			)

		# Saving initial route and weights
		initial_route = route.copy()
		initial_route_weights = route_weights.copy()

		# Looping while there is still temperature
		while temperature > 0:

			# Randomly selecting swap vertices
			swap_indices = np.random.choice(
				list(range(1, len(route) - 1)),
				size = 2,
				replace = False,
				)

			# Creating tentative route by swapping vertex order
			tentative_route = route.copy()
			tentative_route[swap_indices[0]] = route[swap_indices[1]]
			tentative_route[swap_indices[1]] = route[swap_indices[0]]

			tentative_route_weights, tentative_route_valid = evaluate_route(
				adjacency,
				tentative_route,
				route_bounds,
				leg_bounds,
				stop_weights,
				)

			# Proceeding only for valid routes
			if all(tentative_route_valid):

				# Determining acceptance of tentative route
				accept_tentative_route = acceptance(
					route_weights[0],
					tentative_route_weights[0],
					temperature
					)

				# If accepted, replace route with tentative route
				if accept_tentative_route:
					
					route = tentative_route
					route_weights = tentative_route_weights

			# Reducing temperature
			temperature -= kwargs['initial_temperature'] / kwargs['steps']

		if route_weights[0] <= initial_route_weights[0]:

			return route

		else:

			return initial_route

def anneal_routes(adjacency, routes, leg_bounds, route_bounds, stop_weights, **kwargs):

	kwargs.setdefault('steps', 100)
	kwargs.setdefault('initial_temperature', 1)

	# Initializing temperature
	temperature = kwargs['initial_temperature']

	# Getting initial routes weights
	routes_weights, _ = evaluate_routes(
		adjacency,
		routes,
		route_bounds,
		leg_bounds,
		stop_weights,
		)

	sum_initial = sum([rw[0] for rw in routes_weights])

	# Saving initial routes and weights
	routes = deepcopy(routes)
	initial_routes = deepcopy(routes)
	# initial_routes_weights = deepcopy(routes_weights)

	# Looping while there is still temperature
	while temperature > 0:

		# Randomly selecting swap routes
		swap_route_indices = np.random.choice(
			list(range(1, len(routes) - 1)),
			size = 2,
			replace = False,
			)

		swap_route_0 = routes[swap_route_indices[0]]
		swap_route_1 = routes[swap_route_indices[1]]

		# Randomly selecting swap vertices
		swap_index_0 = np.random.choice(
			list(range(1, len(swap_route_0) - 1)),
			replace = False,
			)

		swap_index_1 = np.random.choice(
			list(range(1, len(swap_route_1) - 1)),
			replace = False,
			)

		# Creating tentative route by swapping vertex from route 0 to route 1
		tentative_route_0 = swap_route_0.copy()
		tentative_route_1 = swap_route_1.copy()

		tentative_route_1 = (
			tentative_route_1[:swap_index_1] +
			[tentative_route_0[swap_index_0]] +
			tentative_route_1[swap_index_1:]
			)

		tentative_route_0 = (
			tentative_route_0[:swap_index_0] +
			tentative_route_0[swap_index_0 + 1:]
			)

		tentative_routes = [tentative_route_0, tentative_route_1]

		tentative_routes_weights, tentative_routes_valid = evaluate_routes(
			adjacency,
			tentative_routes,
			route_bounds,
			leg_bounds,
			stop_weights,
			)

		trv = [item for row in tentative_routes_valid for item in row]

		# Proceeding only for valid routes
		if all(trv):

			# Determining acceptance of tentative routes
			accept_tentative_routes = acceptance(
				sum([rw[0] for rw in routes_weights]),
				sum([trw[0] for trw in tentative_routes_weights]),
				temperature
				)

			# If accepted, replace route with tentative route
			if accept_tentative_routes:
				
				routes[swap_route_indices[0]] = tentative_routes[0]
				routes[swap_route_indices[1]] = tentative_routes[1]

				routes_weights[swap_route_indices[0]] = tentative_routes_weights[0]
				routes_weights[swap_route_indices[1]] = tentative_routes_weights[1]

		# Reducing temperature
		temperature -= kwargs['initial_temperature'] / kwargs['steps']

	routes_weights, _ = evaluate_routes(
		adjacency,
		routes,
		route_bounds,
		leg_bounds,
		stop_weights,
		)

	sum_new = sum([rw[0] for rw in routes_weights])

	# print(sum_new, sum_initial)

	if sum_new <= sum_initial:

		return routes

	else:

		return initial_routes
