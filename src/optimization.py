import sys
import time
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import pyomo.environ as pyomo

from copy import copy,deepcopy
from itertools import chain
from scipy.stats import binom

from .utilities import IsIterable,FullFact

class Node():

	def variables(self, model):

		return model

	def objective(self, model, step):

		return 0

	def energy(self, model, step):

		return 0

	def constraints(self, model, step):

		return model

	def transmission(self, model, step):

		return model

class Bus(Node):

	def __init__(self, handle, **kwargs):

		self.handle = handle
		self.bounds = kwargs.get('bounds', (-np.inf, np.inf))

	def variables(self, model):

		generation = pyomo.Var(
			model.time,
			domain = pyomo.NonNegativeReals,
			bounds = self.bounds,
			)

		setattr(model, self.handle, generation)

		return model

	def transmission(self, model, step):

		return getattr(model, self.handle)[step]

class Generation(Node):

	def __init__(self, handle, **kwargs):

		self.handle = handle
		self.cost = kwargs.get('cost', 0)
		self.bounds = kwargs.get('bounds', (0, np.inf))

	def variables(self, model):

		var = pyomo.Var(
			model.time,
			domain = pyomo.NonNegativeReals,
			bounds = self.bounds,
			)

		setattr(model, self.handle, var)

		return model

	def energy(self, model, step):

		return getattr(model, self.handle)[step]

	def objective(self, model, step):

		return self.cost * getattr(model, self.handle)[step]

	def transmission(self, model, step):

		return self.energy(model, step)

class Load(Node):

	def __init__(self, handle, **kwargs):

		self.handle = handle
		self.value = kwargs.get('value', 0)

	def variables(self, model):

		if not hasattr(self.value, '__iter__'):

			self.value = [self.value for idx in model.time]

		param = pyomo.Param(model.time, initialize = self.value)

		setattr(model, self.handle, param)

		return model

	def energy(self, model, step):

		return getattr(model, self.handle)[step]

	def transmission(self, model, step):

		return self.energy(model, step)

class Storage(Node):

	def __init__(self, handle, **kwargs):

		self.handle = handle
		self.capacity = kwargs.get('capacity', 0)
		self.initial = kwargs.get('initial', .5)
		self.final = kwargs.get('final', .5)
		self.bounds = kwargs.get('bounds', (0, 1))

	def variables(self, model):

		var = pyomo.Var(
			model.time,
			domain = pyomo.NonNegativeReals,
			bounds = self.bounds,
			)

		setattr(model, self.handle, var)

		return model

	def energy(self, model, step):

		soc = getattr(model, self.handle)

		if step == model.time.first():

			delta_soc = soc[step] - self.initial

		else:

			delta_soc = soc[step] - soc[step - 1]

		return delta_soc * self.capacity

	def constraints(self, model, step):

		soc = getattr(model, self.handle)[step]
		# print(self.initial, self.final)

		if (step == model.time.first()) & (self.initial is not None):

			constraint = pyomo.Constraint(expr = soc == self.initial)

			setattr(model, self.handle + '_initial', constraint)

		elif (step == model.time.last()) & (self.final is not None):

			constraint = pyomo.Constraint(expr = soc == self.final)

			setattr(model, self.handle + '_final', constraint)

		return model

	def transmission(self, model, step):

		return self.energy(model, step)

class Dissipation(Node):

	def __init__(self, handle, time, links, **kwargs):

		self.handle = handle
		self.time = time
		self.links = links
		self.bounds = kwargs.get('bounds', (-np.inf, 0))

	def variables(self, model):

		var = pyomo.Var(
			self.time,
			domain = pyomo.NonNegativeReals,
			bounds = self.bounds,
			)

		setattr(model, self.handle, var)

		return model

	def energy(self, model, step):

		return getattr(model, self.handle)[step]

	def transmission(self, model, step):

		return self.energy(model, step)

class Link():

	def __init__(self, source, target, **kwargs):

		self.source = source
		self.target = target

class Line(Link):

	def __init__(self, source, target, **kwargs):

		self.source = source
		self.target = target
		self.efficiency = kwargs.get('efficiency', 1) # From source to target
		self.susceptance = kwargs.get('susceptance', 1)
		self.limit = kwargs.get('limit', np.inf)

	def bounds(self, model, step):

		if hasattr(self.limit, '__iter__'):

			return -self.limit[step], self.limit[step]

		else:

			return -self.limit, self.limit

class Feeder(Link):

	def __init__(self, source, target, **kwargs):

		self.source = source
		self.target = target
		self.efficiency = kwargs.get('efficiency', 1) # From source to target
		self.limit = kwargs.get('limit', np.inf)

	def bounds(self, model, step):

		if hasattr(self.limit, '__iter__'):

			return -self.limit[step], self.limit[step]

		else:

			return -self.limit, self.limit

class DC_OPF():

	def __init__(self, graph, time, **kwargs):

		self.graph = graph
		self.time = time

		self.build()

	def solve(self, **kwargs):

		#Generating the solver object
		solver = pyomo.SolverFactory(**kwargs)

		# Building and solving as a linear problem
		self.result = solver.solve(self.model)

		# Making solution dictionary
		self.solution_dictionary()

	def build(self, **kwargs):

		self.build_objects()

		self.build_model()

	def build_objects(self, **kwargs):

		for handle, node in self.graph._node.items():

			if node['type'] == 'bus':

				node['object'] = Bus(handle, **node)

			elif node['type'] == 'generation':

				node['object'] = Generation(handle, **node)

			elif node['type'] == 'load':

				node['object'] = Load(handle, **node)

			elif node['type'] == 'storage':

				node['object'] = Storage(handle, **node)

			elif node['type'] == 'dissipation':

				node['object'] = Dissipator(handle, **node)

		for source, adjacency in self.graph._adj.items():

			for target, link in adjacency.items():

				if link['type'] == 'line':

					link['object'] = Line(source, target, **link)

				elif link['type'] == 'feeder':

					link['object'] = Feeder(source, target, **link)

	def build_model(self, **kwargs):

		#Initializing the model as a concrete model
		self.model = pyomo.ConcreteModel()

		self.model.dual = pyomo.Suffix(direction = pyomo.Suffix.IMPORT_EXPORT)

		# Adding time
		self.model.time = pyomo.Set(initialize = self.time)

		# Adding variables
		self.variables()

		# Adding objective
		self.objective()

		# Adding constraints
		self.constraints()

	def variables(self, **kwargs):

		for node in self.graph._node.values():

			self.model = node['object'].variables(self.model)

	def objective(self, **kwargs):

		cost = 0

		for node in self.graph._node.values():

			for step in self.model.time:

				cost += node['object'].objective(self.model, step)

		self.model.objective = pyomo.Objective(expr = cost)

	def constraints(self, **kwargs):

		self.local_constraints()

		self.balancing_constraints()

	def local_constraints(self, **kwargs):

		for node in self.graph._node.values():

			for step in self.model.time:

				self.model = node['object'].constraints(self.model, step)

	def balancing_constraints(self, **kwargs):

		self.model.balancing_constraints = pyomo.ConstraintList()
		self.model.line_flow_constraints = pyomo.ConstraintList()

		sources = list(self.graph._adj.keys())

		direction = {}

		for step in self.model.time:

			for source, adjacency in self.graph._adj.items():

				direction[source] = {}

				for target, link in adjacency.items():

					direction[source][target] = 1

			for source, adjacency in self.graph._adj.items():

				source_node = self.graph._node[source]

				if source_node['type'] == 'bus':

					energy = 0

					for target, link in adjacency.items():

						target_node = self.graph._node[target]

						if target_node['type'] == 'bus':

							source_transmission = (
								source_node['object'].transmission(
									self.model, step)
								)

							target_transmission = (
								target_node['object'].transmission(
									self.model, step)
								)

							line_flow = target_transmission - source_transmission

							line_limits = link['object'].bounds(self.model, step)

							if line_limits[0] == line_limits[1]:

								line_flow = 0
							
							self.model.line_flow_constraints.add(
								(
									line_limits[0],
									line_flow,
									line_limits[1]
									)
								)

							energy += line_flow * link['object'].susceptance
							energy += (
								(line_flow * (1 - link['object'].efficiency)) / 2 *
								direction[source][target]
								)

							direction[target][source] *= -1

						else:

							energy += (
								target_node['object'].transmission(self.model, step) *
								link['object'].efficiency
								)

					self.model.balancing_constraints.add(
						expr = energy == 0,
						)

	def solution_dictionary(self):
		'''
		From StackOverflow
		https://stackoverflow.com/questions/67491499/
		how-to-extract-indexed-variable-information-in-pyomo-model-and-build-pandas-data
		'''
		model_vars = self.model.component_map(ctype = pyomo.Var)

		data_frames=[]   # collection to hold the converted "serieses"
		for key, var in model_vars.items():   # this is a map of {name:pyo.Var}

			# make a pd.Series from each    
			series = pd.Series(var.extract_values(), index = var.extract_values().keys())

			data_frame = pd.DataFrame(series, columns = [key])

			data_frames.append(data_frame)

		self.solution = {}

		self.solution['data'] = pd.concat(data_frames, axis = 1)

		#Location Marginal Price
		keys = list(self.model.dual.keys())
		keys = [key for key in keys if 'balancing_constraints' in key.getname()]

		buses = [key for key, node in self.graph._node.items() if node['type'] == 'bus']

		location_marginal_price=np.zeros((len(self.time),len(buses)))

		idx = 0

		for t in range(len(self.time)):

			for n in range(len(buses)):

				location_marginal_price[t, n] = self.model.dual[keys[idx]]

				idx += 1

		for idx, bus in enumerate(buses):

			self.solution['data'][bus + '_lmp'] = location_marginal_price[:, idx]

		objective_lower_bound = self.result['Problem']._list[0]['Lower bound']
		objective_upper_bound = self.result['Problem']._list[0]['Upper bound']

		if objective_lower_bound == objective_upper_bound:

			self.solution['objective'] = (objective_lower_bound + objective_upper_bound) / 2

		else:

			self.solution['objective'] = objective_lower_bound + objective_upper_bound