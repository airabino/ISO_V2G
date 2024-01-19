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

	def constraints(self, model, step, dt):

		for obj in self.objects:

			model = obj.constraints(model, step, dt)

		return model

	def objective(self, model, step, dt):

		objective = 0

		for obj in self.objects:

			objective += obj.objective(model, step, dt)

		return objective

	def energy(self, model, step, dt):

		energy = 0

		for obj in self.objects:

			energy += obj.energy(model, step, dt)

		return energy

	def transmission(self, model, step, dt):

		return getattr(model, self.handle + '_transmission')[step] * dt

	def results(self, model, results, dt):

		transmission = np.array(
			list(getattr(model, self.handle + '_transmission').extract_values().values())
			)

		results[self.handle] = {
			'transmission': transmission,
			'objects': {},
		}

		for obj in self.objects:

			results[self.handle]['objects'] = obj.results(
				model, results[self.handle]['objects'], dt)

		return results

class Object():

	def variables(self, model):

		return model

	def objective(self, model, step, dt):

		return 0

	def energy(self, model, step, dt):

		return 0

	def constraints(self, model, step, dt):

		return model

	def results(self, model, results, dt):

		return results

class Generation(Object):

	def __init__(self, handle, **kwargs):

		self.handle = handle
		self.cost = kwargs.get('cost', 0)
		self.bounds = kwargs.get('bounds', (0, np.inf))
		self.bus = kwargs.get('bus', None)
		self.slew = kwargs.get('slew', .1)

	def variables(self, model):

		generation = pyomo.Var(
			model.time,
			domain = pyomo.NonNegativeReals,
			bounds = self.bounds,
			)

		setattr(model, self.handle, generation)

		return model

	def energy(self, model, step, dt):

		return getattr(model, self.handle)[step] * dt

	def objective(self, model, step, dt):

		return self.cost * getattr(model, self.handle)[step] * dt

	def constraints(self, model, step, dt):

		if step != model.time.first():

			current_generation = getattr(model, self.handle)[step]
			previous_generation = getattr(model, self.handle)[step-1]

			constraint = pyomo.Constraint(
				expr = current_generation >= previous_generation * (1 - .1)
				)

			setattr(model, self.handle + f'_{step}_lb', constraint)

			constraint = pyomo.Constraint(
				expr = current_generation <= previous_generation * (1 + .1)
				)

			setattr(model, self.handle + f'_{step}_ub', constraint)

		return model

	def results(self, model, results, dt):

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

	def energy(self, model, step, dt):

		return getattr(model, self.handle)[step] * dt

	def results(self, model, results, dt):

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
		self.charge_limit = kwargs.get('charge_limit', 1)
		self.discharge_limit = kwargs.get('discharge_limit', 1)
		self.efficiency = kwargs.get('efficiency', 1)
		self.bus = kwargs.get('bus', None)

	def variables(self, model):

		charge = pyomo.Var(
			model.time,
			domain = pyomo.Reals,
			bounds = (0, self.charge_limit),
			)

		setattr(model, self.handle + '_charge', charge)

		discharge = pyomo.Var(
			model.time,
			domain = pyomo.Reals,
			bounds = (0, self.discharge_limit),
			)

		setattr(model, self.handle + '_discharge', discharge)

		return model

	def energy(self, model, step, dt):

		charge_energy = (
			getattr(model, self.handle + '_charge')[step] /
			self.efficiency * dt
			)

		discharge_energy = (
			getattr(model, self.handle + '_discharge')[step] *
			self.efficiency * dt
			)

		return -charge_energy + discharge_energy 

	def constraints(self, model, step, dt):

		soc = self.initial

		for idx in range(step + 1):

			delta_soc = (
				getattr(model, self.handle + '_charge')[idx] / self.capacity * dt - 
				getattr(model, self.handle + '_discharge')[idx] / self.capacity * dt
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

	def results(self, model, results, dt):

		charge = np.array(
			list(getattr(model, self.handle + '_charge').extract_values().values())
			) / self.capacity * dt

		discharge = np.array(
			list(getattr(model, self.handle + '_discharge').extract_values().values())
			) / self.capacity * dt

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

	def energy(self, model, step, dt):

		return getattr(model, self.handle)[step] * dt

	def results(self, model, results, dt):

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

	def bounds(self, model, step):

		if hasattr(self.limit[0], '__iter__'):

			return self.limit[step][0], self.limit[step][1]

		else:

			return self.limit[0], self.limit[1]

class DC_OPF():

	def __init__(self, graph, time, **kwargs):

		self.graph = graph
		self.time = time
		self.dt = time[1] - time[0]

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

			self.results = node['object'].results(self.model, self.results, self.dt)

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

				cost += node['object'].objective(self.model, step, self.dt)

		self.model.objective = pyomo.Objective(expr = cost)

	def constraints(self, **kwargs):

		self.local_constraints()

		self.balancing_constraints()

	def local_constraints(self, **kwargs):

		for node in self.graph._node.values():

			for step in self.model.time:

				self.model = node['object'].constraints(self.model, step, self.dt)

	def balancing_constraints(self, **kwargs):

		# energy = {source: 0 for source in list(self.graph.nodes)}
		energy = {}

		# Energy at nodes
		for step in self.model.time:

			for source, adjacency in self.graph._adj.items():

				source_node = self.graph._node[source]

				energy[(source, step)] = source_node['object'].energy(
					self.model, step, self.dt)

		# Energy between nodes
		self.model.transmission_constraints = pyomo.ConstraintList()
		for step in self.model.time:

			for source, adjacency in self.graph._adj.items():

				source_node = self.graph._node[source]

				for target, link in adjacency.items():

					target_node = self.graph._node[target]

					source_transmission = source_node['object'].transmission(
						self.model, step, self.dt)
					target_transmission = target_node['object'].transmission(
						self.model, step, self.dt)

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