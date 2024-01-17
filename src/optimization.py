import sys
import time
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import pyomo.environ as pyomo
import networkx as nx

from copy import copy,deepcopy
from itertools import chain
from scipy.stats import binom

from .utilities import IsIterable,FullFact

class Bus():

	def __init__(self, handle, **kwargs):

		self.handle = handle
		self.bounds = kwargs.get('bounds', (-np.inf, np.inf))
		self.dependents = kwargs.get('dependents', [])

		self.build_objects()

	def build_objects(self):

		self.objects = []

		for dependent in self.dependents:

			if dependent['type'] == 'generation':

				self.objects.append(Generation(dependent['id'], **dependent))

			elif dependent['type'] == 'load':

				self.objects.append(Load(dependent['id'], **dependent))

			elif dependent['type'] == 'storage':

				self.objects.append(Storage(dependent['id'], **dependent))

			elif dependent['type'] == 'dissipation':

				self.objects.append(Dissipation(dependent['id'], **dependent))

	def variables(self, model):

		transmission = pyomo.Var(
			model.time,
			domain = pyomo.NonNegativeReals,
			bounds = self.bounds,
			)

		setattr(model, self.handle + '_transmission', transmission)

		for obj in self.objects:

			model = obj.variables(model)

		return model

	def constraints(self, model, step):

		for obj in self.objects:

			model = obj.constraints(model, step)

		return model

	def objective(self, model, step):

		objective = 0

		for obj in self.objects:

			objective += obj.objective(model, step)

		return objective

	def energy(self, model, step):

		energy = 0

		for obj in self.objects:

			energy += obj.energy(model, step)

		return energy

	def transmission(self, model, step):

		return getattr(model, self.handle + '_transmission')[step]

	def results(self, model, results):

		transmission = np.array(
			list(getattr(model, self.handle + '_transmission').extract_values().values())
			)

		results[self.handle] = {
			'transmission': transmission,
			'objects': {},
		}

		for obj in self.objects:

			results[self.handle]['objects'] = obj.results(
				model, results[self.handle]['objects'])

		return results

class Object():

	def variables(self, model):

		return model

	def objective(self, model, step):

		return 0

	def energy(self, model, step):

		return 0

	def constraints(self, model, step):

		return model

	def results(self, model, results):

		return results

class Generation(Object):

	def __init__(self, handle, **kwargs):

		self.handle = handle
		self.cost = kwargs.get('cost', 0)
		self.bounds = kwargs.get('bounds', (0, np.inf))
		self.bus = kwargs.get('bus', None)

	def variables(self, model):

		generation = pyomo.Var(
			model.time,
			domain = pyomo.NonNegativeReals,
			bounds = self.bounds,
			)

		setattr(model, self.handle, generation)

		return model

	def energy(self, model, step):

		return getattr(model, self.handle)[step]

	def objective(self, model, step):

		return self.cost * getattr(model, self.handle)[step]

	def results(self, model, results):

		generation = np.array(
			list(getattr(model, self.handle).extract_values().values())
			)

		results[self.handle] = {
			'type': 'generation',
			'generation': generation,
			'cost': generation * self.cost,
			}

		return results

class Load(Object):

	def __init__(self, handle, **kwargs):

		self.handle = handle
		self.value = kwargs.get('value', 0)
		self.bus = kwargs.get('bus', None)

	def variables(self, model):

		if not hasattr(self.value, '__iter__'):

			self.value = [self.value for idx in model.time]

		param = pyomo.Param(model.time, initialize = self.value)

		setattr(model, self.handle, param)

		return model

	def energy(self, model, step):

		return getattr(model, self.handle)[step]

	def results(self, model, results):

		load = np.array(
			list(getattr(model, self.handle).extract_values().values())
			)

		results[self.handle] = {
			'type': 'load',
			'load': load,
			}

		return results

class Storage(Object):

	def __init__(self, handle, **kwargs):

		self.handle = handle
		self.capacity = kwargs.get('capacity', 0)
		self.initial = kwargs.get('initial', .5)
		self.final = kwargs.get('final', .5)
		self.bounds = kwargs.get('bounds', (0, 1))
		self.limit = kwargs.get('limit', 1)
		self.efficiency = kwargs.get('efficiency', 1)
		self.bus = kwargs.get('bus', None)

	def variables(self, model):

		charge = pyomo.Var(
			model.time,
			domain = pyomo.Reals,
			bounds = (0, self.limit),
			)

		setattr(model, self.handle + '_charge', charge)

		discharge = pyomo.Var(
			model.time,
			domain = pyomo.Reals,
			bounds = (0, self.limit),
			)

		setattr(model, self.handle + '_discharge', discharge)

		return model

	def energy(self, model, step):

		charge_energy = (
			getattr(model, self.handle + '_charge')[step] * self.capacity / self.efficiency
			)

		discharge_energy = (
			getattr(model, self.handle + '_discharge')[step] *
			self.capacity * self.efficiency
			)

		return -charge_energy + discharge_energy 

	def constraints(self, model, step):

		soc = self.initial

		for idx in range(step + 1):

			delta_soc = (
				getattr(model, self.handle + '_charge')[idx] - 
				getattr(model, self.handle + '_discharge')[idx]
				)

			soc += delta_soc

		if step == model.time.last():

			constraint = pyomo.Constraint(expr = soc == self.final)

			setattr(model, self.handle + '_final', constraint)

		else:

			constraint = pyomo.Constraint(
				rule = (self.bounds[0], soc, self.bounds[1])
				)

			setattr(model, self.handle + f'_bounds_{step}', constraint)

		return model

	def results(self, model, results):

		charge = np.array(
			list(getattr(model, self.handle + '_charge').extract_values().values())
			)

		discharge = np.array(
			list(getattr(model, self.handle + '_discharge').extract_values().values())
			)

		state_of_charge = np.cumsum(charge) - np.cumsum(discharge) + self.initial

		results[self.handle] = {
			'type': 'storage',
			'discharge': discharge * self.capacity,
			'charge': charge * self.capacity,
			'state_of_charge': state_of_charge,
			}

		return results

class Dissipation(Object):

	def __init__(self, handle, **kwargs):

		self.handle = handle
		self.bounds = kwargs.get('bounds', (-np.inf, 0))
		self.bus = kwargs.get('bus', None)

	def variables(self, model):

		var = pyomo.Var(
			model.time,
			domain = pyomo.NegativeReals,
			bounds = self.bounds,
			)

		setattr(model, self.handle, var)

		return model

	def energy(self, model, step):

		return getattr(model, self.handle)[step]

	def results(self, model, results):

		dissipation = np.array(
			list(getattr(model, self.handle).extract_values().values())
			)

		results[self.handle] = {
			'type': 'dissipation',
			'dissipation': dissipation,
			}

		return results

class Link():

	def __init__(self, source, target, **kwargs):

		self.source = source
		self.target = target
		self.handle = f'{source}_{target}'
		self.efficiency = kwargs.get('efficiency', 1)
		self.susceptance = kwargs.get('susceptance', 1)
		self.limit = kwargs.get('limit', (-np.inf, np.inf))
		self.direction = kwargs.get('direction', 1)

	def variables(self, model):

		return model

	def energy(self, model, step):

		return getattr(model, self.handle)[step]

	def bounds(self, model, step):

		if hasattr(self.limit[0], '__iter__'):

			return self.limit[step][0], self.limit[step][1]

		else:

			return self.limit[0], self.limit[1]

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
		self.collect_results()

	def collect_results(self, **kwargs):

		self.results = {}

		for node in self.graph._node.values():

			self.results = node['object'].results(self.model, self.results)

	def build(self, **kwargs):

		self.build_objects()

		self.build_model()

	def build_objects(self, **kwargs):

		for handle, node in self.graph._node.items():

			node['object'] = Bus(handle, **node)

		for source, adjacency in self.graph._adj.items():

			for target, link in adjacency.items():

				link['object'] = Link(source, target, **link)

	def build_model(self, **kwargs):

		#Initializing the model as a concrete model
		self.model = pyomo.ConcreteModel()

		self.model.dual = pyomo.Suffix(direction = pyomo.Suffix.IMPORT_EXPORT)

		# Adding time
		self.model.time = pyomo.Set(initialize = list(range(len(self.time))))

		# Adding variables
		self.variables()

		# Adding objective
		self.objective()

		# Adding constraints
		self.constraints()

	def variables(self, **kwargs):

		for node in self.graph._node.values():

			self.model = node['object'].variables(self.model)

		for source, adjacency in self.graph._adj.items():

			for target, link in adjacency.items():

				self.model = link['object'].variables(self.model)

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

		# energy = {source: 0 for source in list(self.graph.nodes)}
		energy = {}

		# Energy at nodes
		for step in self.model.time:

			for source, adjacency in self.graph._adj.items():

				source_node = self.graph._node[source]

				energy[(source, step)] = source_node['object'].energy(self.model, step)

		# Energy between nodes
		self.model.transmission_constraints = pyomo.ConstraintList()
		for step in self.model.time:

			for source, adjacency in self.graph._adj.items():

				source_node = self.graph._node[source]

				for target, link in adjacency.items():

					target_node = self.graph._node[target]

					source_transmission = source_node['object'].transmission(self.model, step)
					target_transmission = target_node['object'].transmission(self.model, step)

					transmission_energy = target_transmission - source_transmission

					link_limits = link['object'].bounds(self.model, step)

					# Only consider active links
					if link_limits[0] != link_limits[1]:

						self.model.transmission_constraints.add(
							(link_limits[0], transmission_energy, link_limits[1])
							)

						energy[(source, step)] -= (
							transmission_energy /
							link['object'].efficiency *
							link['object'].susceptance
							)

						energy[(target, step)] += (
							transmission_energy *
							link['object'].efficiency *
							link['object'].susceptance
							)

		# Adding balancing constraints
		self.model.balancing_constraints = pyomo.ConstraintList()

		for step in self.model.time:

			for source, adjacency in self.graph._adj.items():

				self.model.balancing_constraints.add(
					expr = energy[(source, step)] == 0,
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

		buses = [key for key, node in self.graph._node.items()]

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