import os
import sys
import time
import matplotlib
import numpy as np
import numpy.random as rand
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

from .utilities import IsIterable

#Named color schemes
default_prop_cycle = mpl.rcParamsDefault['axes.prop_cycle'].by_key()['color'].copy()

colors={
    'day_night': ["#e6df44", "#f0810f", "#063852", "#011a27"],
    'beach_house': ["#d5c9b1", "#e05858", "#bfdccf", "#5f968e"],
    'autumn': ["#db9501", "#c05805", "#6e6702", "#2e2300"],
    'ocean': ["#003b46", "#07575b", "#66a5ad", "#c4dfe6"],
    'forest': ["#7d4427", "#a2c523", "#486b00", "#2e4600"],
    'aqua': ["#004d47", "#128277", "#52958b", "#b9c4c9"],
    'field': ["#5a5f37", "#fffae1", "#524a3a", "#919636"],
    'misty': ["#04202c", "#304040", "#5b7065", "#c9d1c8"],
    'greens': ["#265c00", "#68a225", "#b3de81", "#fdffff"],
    'citroen': ["#b38540", "#563e20", "#7e7b15", "#ebdf00"],
    'blues': ["#1e1f26", "#283655",  "#4d648d", "#d0e1f9"],
    'dusk': ["#363237", "#2d4262", "#73605b", "#d09683"],
    'ice': ["#1995ad", "#a1d6e2", "#bcbabe", "#f1f1f2"],
    'csu': ["#1e4d2b", "#c8c372"],
    'ucd': ['#022851', '#ffbf00'],
    'incose': ["#f2606b", "#ffdf79", "#c6e2b1", "#509bcf"],
    'sae': ["#01a0e9", "#005195", "#cacac8", "#9a9b9d", "#616265"],
    'trb': ["#82212a", "#999999", "#181818"],
    'default_prop_cycle': default_prop_cycle,
}

def ReturnColorMap(colors):

    if type(colors) == str:

        cmap = matplotlib.cm.get_cmap(colors)

    else:

        cmap = LinearSegmentedColormap.from_list('custom', colors, N = 256)

    return cmap

def PlotDashboard(graph, results, time, ax = None, **kwargs):

    plt.rcParams.update(
        {
            'axes.titlesize': kwargs.get('titlesize', 11),
            'xtick.labelsize': kwargs.get('labelsize', 5),
            'ytick.labelsize': kwargs.get('labelsize', 5),
            'legend.fontsize': kwargs.get('legendsize', 5),
            'axes.prop_cycle': mpl.cycler(
                color = kwargs.get('prop_cycle', colors['default_prop_cycle']))
        }
    )

    return_fig = False

    if ax is None:

        fig, ax = plt.subplots(2, 3, **kwargs.get('figure', {}))
        return_fig = True

    # Generation
    for bus, bus_data in results.items():

        for obj, obj_data in bus_data['objects'].items():

            if obj_data['type'] == 'generation':
                
                ax[0, 0].plot(time, obj_data['generation'], label = f'{obj} ({bus})',
                    **kwargs.get('plot', {}))

    ax[0, 0].set(title = 'Generation', xlabel = 'Time Period', ylabel = 'kWh')

    # Generation cost
    for bus, bus_data in results.items():

        for obj, obj_data in bus_data['objects'].items():

            if obj_data['type'] == 'generation':
                
                ax[0, 1].plot(time, np.cumsum(obj_data['cost']), label = f'{obj} ({bus})',
                    **kwargs.get('plot', {}))

    ax[0, 1].set(title = 'Generation Cost', xlabel = 'Time Period', ylabel = 'USD')

    # Net Transmission
    for source, link in graph._adj.items():

        net_transmission = np.array([0.] * len(time))

        for target in graph._adj.keys():

            source_transmission = results[source]['transmission']
            target_transmission = results[target]['transmission']

            for idx in range(len(net_transmission)):

                if (
                    (source_transmission[idx] is not None) and
                    (target_transmission[idx] is not None)
                    ):

                    net_transmission[idx] += (
                        target_transmission[idx]- 
                        source_transmission[idx]
                        )

        ax[0, 2].plot(time, net_transmission, label = f'{source}',
                    **kwargs.get('plot', {}))

    ax[0, 2].set(title = 'Net Transmission', xlabel = 'Time Period', ylabel = 'kWh')

    # Loads
    for bus, bus_data in results.items():
        for obj, obj_data in bus_data['objects'].items():
            if obj_data['type'] == 'load':
                
                ax[1, 0].plot(time, obj_data['load'], label = f'{obj} ({bus})',
                    **kwargs.get('plot', {}))

    ax[1, 0].set(title = 'Loads', xlabel = 'Time Period', ylabel = 'kWh')

    # Storage
    for bus, bus_data in results.items():
        for obj, obj_data in bus_data['objects'].items():
            if obj_data['type'] == 'storage':
                
                ax[1, 1].plot(time, obj_data['state_of_charge'], label = f'{obj} ({bus})',
                    **kwargs.get('plot', {}))

    ax[1, 1].set(title = 'Storage SOC', xlabel = 'Time Period', ylabel = '[-]')

    # Dissipation
    for bus, bus_data in results.items():
        for obj, obj_data in bus_data['objects'].items():
            if obj_data['type'] == 'dissipation':
                
                ax[1, 2].plot(time, obj_data['dissipation'], label = f'{obj} ({bus})',
                    **kwargs.get('plot', {}))

    ax[1, 2].set(title = 'Dissipation', xlabel = 'Time Period', ylabel = 'kWh')

    _ = [ax.legend(**kwargs.get('legend', {})) for ax in ax for ax in ax]
    _ = [ax.set(**kwargs.get('axes', {})) for ax in ax for ax in ax]


    if return_fig:

        return fig

def PlotGraph(graph, ax = None, cmap = ReturnColorMap('viridis'), field = None, **kwargs):
    
    return_fig = False

    if ax is None:

        fig, ax = plt.subplots(**kwargs.get('figure', {}))
        return_fig = True

    coords = np.array([[v['x'], v['y']] for v in graph._node.values()])

    if field is None:

        values = None

    else:

        values = np.array([v[field] for v in graph._node.values()])

    ax.scatter(
        coords[:, 0], coords[:, 1], c = values, cmap = cmap, **kwargs.get('scatter', {}))

    if line_kwargs:

        for v in graph._adj.values():

            for e in v.values():

                ax.plot(e['x'], e['y'], zorder = 0, **kwargs.get('plot', {}))

    ax.set(**kwargs.get('axes', {}))

    if return_fig:

        return fig