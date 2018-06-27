#!/usr/bin/env python2

import os
import sys
import glob
import cPickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import ipywidgets as widgets
from IPython.display import clear_output, display, Javascript, HTML


def in_jupyter():
    ''' whether the current script is running in IPython/Jupyter '''
    try:
        __IPYTHON__
    except NameError:
        return False
    return True

if in_jupyter():
    print('Setting up Jupyter')
    # hide scroll bars that sometimes appear (apparently by chance) because the
    # image fills the entire sub-area.
    style = HTML('''
<style>
.widget-readout {
    width: 100px;
}
</style>''')
    display(style)

def set_notebook_width(width):
    """set the width of the central element of the notebook

    Args:
        width (str): a css string to set the width to (eg "123px" or "90%" etc)
    """
    display(Javascript('document.getElementById("notebook-container").style.width = "{}"'.format(width)))

class Experiment:
    def __init__(self, filename):
        self.filename = filename
        with open(filename, 'rb') as f:
            self.states = pkl.load(f)

    def get_key(self, key):
        """
        Returns: vals, dims
        """
        if key == 'time':
            return range(len(self.states)), 1
        else:
            vals = [s[key] for s in self.states]
            return [np.append(v, np.linalg.norm(v)) for v in vals], 3

    def get_key_dims(self, key):
        return 1 if key == 'time' else 3

    def get_range(self, key):
        if key == 'time':
            return 0, len(self.states)
        else:
            mins = [min(s[key].tolist() + [np.linalg.norm(s[key])]) for s in self.states]
            maxs = [max(s[key].tolist() + [np.linalg.norm(s[key])]) for s in self.states]
            return min(mins), max(maxs)
        

    def plot(self, x_key, y_key, x_lim, y_lim):
        xs, x_dims = self.get_key(x_key)
        ys, y_dims = self.get_key(y_key)

        fig, axs = plt.subplots(ncols=4, figsize=(16, 5))
        fig.suptitle(os.path.basename(self.filename))

        for i, dim in enumerate(['x', 'y', 'z', 'mag']):
            ax = axs[i]
            ax.set_title(dim)
            #TODO: this is a hack, make this nicer
            plot_xs = [x[i] for x in xs] if x_dims == 3 else xs
            plot_ys = [y[i] for y in ys] if y_dims == 3 else ys
            ax.plot(plot_xs, plot_ys)
            if y_key == 'ee_point' and x_key == 'time':
                plot_extra_ys, _ = self.get_key('req_traj')
                plot_extra_ys = [y[i] for y in plot_extra_ys]
                ax.plot(plot_xs, plot_extra_ys, label='desired')
                ax.legend()
            ax.set_xlabel('time' if x_key == 'time' else '{} {}'.format(x_key, dim))
            ax.set_ylabel('time' if y_key == 'time' else '{} {}'.format(y_key, dim))
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
        fig.tight_layout()
        plt.show(fig)



class Experiments:
    def __init__(self, filenames_glob, keys, default_keys):
        """
        Args:
            filenames_glob: the glob pattern for filenames to load (eg /some/dir/something_*.pkl)
            default_keys: keys for (y, x) for the drop down to start on
        """
        self.keys = keys
        self.default_keys = default_keys
        self.experiments = []
        for filename in sorted(glob.glob(filenames_glob)):
            self.experiments.append(Experiment(filename))
        if not self.experiments:
            print('no matches for glob: "{}"'.format(filenames_glob))
            sys.exit(1)

        print('{} experiments loaded:'.format(len(self.experiments)))
        raw_keys = self.experiments[0].states[0].keys()
        for i, e in enumerate(self.experiments):
            for s in e.states:
                assert s.keys() == raw_keys, \
                        'experiments has mismatch in keys: {}!={}'.format(s.keys(), raw_keys)
            print('{}: "{}"'.format(i, os.path.basename(e.filename)))
        print('\nexperiment keys: {}'.format(raw_keys))

    def plot_experiments(self, x, y, x_lim, y_lim):
        assert x != y, 'x axis == y axis'
        for i, e in enumerate(self.experiments):
            e.plot(x, y, x_lim, y_lim)

    def get_range(self, key):
        ranges = [e.get_range(key) for e in self.experiments]
        return min([r[0] for r in ranges]), max([r[1] for r in ranges])

    def _update_lim_slider(self, lim, key):
        krange = self.get_range(key)
        def expand(r, frac):
            span = r[1] - r[0]
            return [r[0] - span * frac, r[1] + span * frac]

        lims = expand(krange, 0.3)
        lim.min = lims[0]
        lim.max = lims[1]
        lim.step = (lim.max-lim.min) / 100
        lim.value = expand(krange, 0.1)

    def plot_interactive(self, index=None):
        """
        Args:
            index: plot only the experiment with this index
        """
        y_key = widgets.Dropdown(options=self.keys, value=self.default_keys[0], description='y: ')
        x_key = widgets.Dropdown(options=self.keys, value=self.default_keys[1], description='x: ')

        x_lim = widgets.FloatRangeSlider(
            description='X range:',
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        x_lim.layout.width = '500px'
        y_lim = widgets.FloatRangeSlider(
            description='Y range:',
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        y_lim.layout.width = '500px'

        def selection_changed(change):
            self._update_lim_slider(x_lim, x_key.value)
            self._update_lim_slider(y_lim, y_key.value)
        selection_changed(None)


        controls = widgets.interactive(self.plot_experiments, y=y_key, x=x_key, x_lim=x_lim, y_lim=y_lim)

        output = controls.children[-1]
        controls.children = (widgets.HBox([y_key, y_lim]), widgets.HBox([x_key, x_lim]), output)

        x_key.observe(selection_changed, names='value')
        y_key.observe(selection_changed, names='value')

        display(controls)


