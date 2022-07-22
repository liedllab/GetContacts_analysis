from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import _kw_handler, mask_gen, max_extent

class MatrixPlot(ABC):
    def __init__(self, data, fig=None, ax=None, **kwargs):
        if data.empty:
            raise pd.errors.EmptyDataError("The DataFrame is empty")
        self.data = data
        
        self.fig = fig
        self.ax = ax
        
        self.label_data = None
        self.is_split = False
        self._mesh_data = None

    @abstractmethod
    def plot(self, cmap, **kwargs):
        pass
    
    def recolor_matrix(self, cmaps, masker):
        mesh = self.ax.collections[0]
        
        if self._mesh_data is None:
            self._mesh_data = mesh.get_array().data

        cmaps = [plt.cm.get_cmap(name, 256) for name in cmaps]
        fc = mesh.get_facecolors()
        for n, (mask, value) in enumerate(zip(masker, self._mesh_data)):
            fc[n] = cmaps[mask](value)

        mesh.set_array(None)
        mesh.set_facecolor(fc)

    def split_labels(self):
        if not self.label_data:
            self._labeler()
        if not self.is_split:
            labels, _, _ = zip(*[v.values() for v in self.label_data.values()])
            ticks = list(self.label_data.keys())

            self.ax.set_yticks(ticks, labels)

            self.is_split = True

    def align_labels(self, *, factor=None):
        if factor is not None:
            self.factor = factor

        _, _, shifts = zip(*[v.values() for v in self.label_data.values()])
        extent = max_extent(self.ax)
        
        for label, shift in zip(self.ax.get_yticklabels(), shifts):
            pos = label.get_position()
            label.set_position(
                    (self.label_position + self.factor*shift*extent, pos[1])
                    )
        return self.ax

    def color_labels(self, colors, alpha = 0.5):
        if not self.is_split:
            self.split_labels()

        _, systems, _ = zip(*[v.values() for v in self.label_data.values()])

        if isinstance(colors, str):
            colors = [colors]
        # there are always 2 systems due to - or / being system 0
        while len(colors) < len(set(systems))-1:
            # "none" equals no color, regular None colors blue somehow
            colors.append("none")

        for label, system in zip(self.ax.get_yticklabels(), systems):
            if system:
                label.set_bbox(
                        dict(
                            facecolor=colors[system - 1],
                            alpha= alpha,
                            edgecolor="none",
                            )
                        )

    def _norm_fig_size(self, norm=100):
        bb = self.ax.get_window_extent()
        scale = self.data.columns.shape[0] * norm / bb.width
        self.fig.set_size_inches(self.fig.get_size_inches()*scale)
    
    def _labeler(self):
        n_systems_left = max(
                [len(set(label.get_text().split('-')[::2])) for label in self.ax.get_yticklabels()]
                )
        label_data = dict()
        for label in self.ax.get_yticklabels():
            _, height = label.get_position()
            split = label.get_text().split('-')

            left = split[::2]
            right = split[1::2]
            
            if len(set(left)) == 1:
                left = set(left)
            if len(set(right)) == 1:
                right = set(right)
            
            for system, (pos, l) in enumerate(enumerate(left), start=1):
                if system > 1:
                    label_data[height+1e-10*pos+1e-11] = {
                        'text' : ' / ',
                        'system' : 0,
                        'pos' : pos - 0.2,
                    }
                label_data[height + 1e-10*pos] = {
                    'text' : l,
                    'system' : system,
                    'pos' : pos,
                }
            pos = n_systems_left
            label_data[height+1e-10*(pos+0.5)] = {
                    'text' : ' - ',
                    'system' : 0,
                    'pos' : pos-0.1,
                }
            
            for system, (pos, r) in enumerate(enumerate(right, start=pos), start=1):
                pos += 0.2
                if system > 1:
                    label_data[height+1e-10*pos+1e-11] = {
                        'text' : ' / ',
                        'system' : 0,
                        'pos' : pos - 0.2,
                    }
                label_data[height + 1e-10*pos] = {
                    'text' : r,
                    'system' : system,
                    'pos' : pos,
                }

        self.label_data = label_data                       
        return label_data


class Heatmapper(MatrixPlot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = -0.5
        self.label_position = 0

    def plot(self, cmap: str = "Greens", **kwargs):
        kwargs = _kw_handler({
            'cbar' : True,
            'vmin' : 0,
            'vmax' : 1,
            'annot' : False,
            'linewidths' : 0.5,
            'linecolor' : 'black',
            }, kwargs, error=False)

        if not self.fig:
            self.fig = plt.figure()
        if not self.ax:
            self.ax = self.fig.add_subplot()

        sns.heatmap(
                self.data,
                yticklabels = 1,
                ax = self.ax,
                cmap = cmap,
                square = True,
                **kwargs,
                )
        
        self.ax.yaxis.set_label_text(None)
        self.ax.set_xticklabels(self.ax.get_xticklabels(), rotation=90)
        self.ax.set_yticklabels(self.ax.get_yticklabels(), rotation=0)


class Fingerprinter(MatrixPlot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = 1
        self.label_position = 1

    def plot(self, cmap:str = "Greens", cbar:bool=False, **kwargs):
        kwargs = _kw_handler({
            'cbar_pos' : (1.5, 0.05, 0.1, 0.38),
            'vmin' : 0,
            'vmax' : 1,
            'annot' : False,
            'linewidths' : 0.5,
            'linecolor' : 'black',
            'dendrogram_ratio' : (0.2, 0),
            'col_cluster' : False,
            }, kwargs, error=False)

        finger = sns.clustermap(
                self.data,
                yticklabels = 1,
                cmap = cmap,
                **kwargs,
                )

        self.fig = finger.fig
        self.ax = finger.ax_heatmap

        self.ax.set_aspect("equal", anchor=(0,0))
        self.draw_cbar(cbar)
        
        self.ax.yaxis.set_label_text(None)
        plt.setp(self.ax.yaxis.get_majorticklabels(), rotation=0)
        plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=90)

    def set_title(self, text:str, pad: int = 10):
        self.ax.set_title(text, pad=pad)

    def draw_cbar(self, /, draw):
        self.fig.axes[3].set_visible(draw)
    

def fingerprint(data, *args, **kwargs):
    finger = Fingerprinter(data)
    finger.plot(*args, **kwargs)
    finger.split_labels()
    finger._norm_fig_size()
    finger.align_labels(factor=2)
    return finger

def heatmap(data, *args, **kwargs):
    heat = Heatmapper(data)
    heat.plot(*args, **kwargs)
    heat.split_labels()
    heat.align_labels()
    heat._norm_fig_size()

    return heat

if __name__ == "__main__":
    data = np.array(
        [[3.80e+01, 3.09e+02, 3.21e-01, 0.00e+00],
        [8.90e+01, 3.17e+02, 7.05e-01, 0.00e+00],
        [9.00e+01, 3.17e+02, 0.00e+00, 8.15e-01],
        [9.20e+01, 3.16e+02, 4.73e-01, 0.00e+00],
        [9.30e+01, 2.67e+02, 7.81e-01, 0.00e+00],
        [9.70e+01, 3.13e+02, 0.00e+00, 9.70e-01],
        [1.18e+02, 3.48e+02, 3.25e-01, 0.00e+00],
        [2.54e+02, 3.80e+01, 9.50e-01, 0.00e+00],
        [2.54e+02, 3.90e+01, 0.00e+00,6.60e-01],
        [3.19e+02, 3.60e+01, 6.35e-01, 0.00e+00],
        [3.19e+02, 3.70e+01, 0.00e+00, 7.12e-01],
        [3.20e+02, 5.50e+01, 4.12e-01, 0.00e+00]]
        )
    df = pd.DataFrame(data, columns=["number 1", "number 2", "wildtype", "mashup"])
    df["number 1"] = df["number 1"].astype(np.int16)
    df["number 2"] = df["number 2"].astype(np.int16)
    df = df.set_index(["number 1", "number 2"])

#     hm = Heatmap(df)
#     hm.plot()

    p = Fingerprinter(df)
    p.plot()
    p.recolor_matrix(["Greens", "Blues"], mask_gen(1,1))
    plt.show()
