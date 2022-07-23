"""Heatmaps and Clustermaps with added functionalities such as recoloring, and
highlighting of labels."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import _kw_handler, max_extent


class MatrixPlot(ABC):
    def __init__(self, data: pd.DataFrame,
            fig: plt.Figure=None,
            ax: plt.Axes=None):
        if data.empty:
            raise pd.errors.EmptyDataError("The DataFrame is empty")
        self.data = data

        self.fig = fig
        self.ax = ax

        self.label_position = None
        self.label_data = None
        self.is_split = False
        self._mesh_data = None

    @abstractmethod
    def plot(self,*, cmap: str='Greens', **kwargs) -> MatrixPlot:
        pass

    def recolor_matrix(self, cmaps: list[str], masker: Iterable) -> MatrixPlot:
        mesh = self.ax.collections[0]
        if self._mesh_data is None:
            self._mesh_data = mesh.get_array().data
        cmaps = [plt.cm.get_cmap(name, 256) for name in cmaps]
        fc = mesh.get_facecolors()
        for n, (mask, value) in enumerate(zip(masker, self._mesh_data)):
            fc[n] = cmaps[mask](value)
        mesh.set_array(None)
        mesh.set_facecolor(fc)
        return self

    def split_labels(self) -> MatrixPlot:
        if not self.label_data:
            self._labeler()
        if not self.is_split:
            labels, _, _ = zip(*[v.values() for v in self.label_data.values()])
            ticks = list(self.label_data.keys())
            self.ax.set_yticks(ticks, labels)
            self.is_split = True
        return self

    def align_labels(self, *, factor: float) -> MatrixPlot:
        _, _, shifts = zip(*[v.values() for v in self.label_data.values()])
        extent = max_extent(self.ax)

        for label, shift in zip(self.ax.get_yticklabels(), shifts):
            pos = label.get_position()
            label.set_position(
                    (self.label_position + factor*shift*extent, pos[1])
                    )
        return self

    def color_labels(self, colors: list[str], alpha: float=0.5) -> MatrixPlot:
        if not self.is_split:
            self.split_labels()
        _, systems, _ = zip(*[v.values() for v in self.label_data.values()])
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
        return self

    def norm_fig_size(self, norm: float=100) -> MatrixPlot:
        bb = self.ax.get_window_extent()
        scale = self.data.columns.shape[0] * norm / bb.width
        self.fig.set_size_inches(self.fig.get_size_inches()*scale)
        return self

    def _labeler(self) -> MatrixPlot:
        n_systems_left = max(
                [len(label.get_text().split('-')[::2]) for label in self.ax.get_yticklabels()])
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
        return self

    def _repr_png_(self):
        return display(self.fig)


class Heatmapper(MatrixPlot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_position = 0

    def plot(self, *, cmap: str="Greens", **kwargs) -> Heatmapper:
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
        return self


class Fingerprinter(MatrixPlot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_position = 1

    def plot(self, *, cmap: str="Greens", **kwargs) -> Fingerprinter:
        kwargs = _kw_handler({
            'cbar' : False,
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
        self.draw_cbar(kwargs['cbar'])

        self.ax.yaxis.set_label_text(None)
        plt.setp(self.ax.yaxis.get_majorticklabels(), rotation=0)
        plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=90)
        return self

    def set_title(self, text:str, pad: int = 10) -> Fingerprinter:
        self.ax.set_title(text, pad=pad)
        return self

    def draw_cbar(self, /, draw) -> Fingerprinter:
        self.fig.axes[3].set_visible(draw)
        return self


def fingerprint(data: pd.DataFrame, *args, **kwargs) -> Fingerprinter:
    """ Wrapper around the Fingerprinter class.

The plotting is done using `seaborn.clustermap`. For further more
thorough documentation check their website. Below some possible
keywords are elaborated. All keywords that are compatible with
`seaborn.clustermap` can be used.

:param data: pandas.DataFrame
        data of merged DataFrame to be plotted.
:param cmap: str
        colormap to use for the connecting lines. Standard
        matplotlib colormaps are supported.
    :default: "Greens"
        colormap from white to green.
:param cbar_pos: tuple[float, float, float, float]
        tuple of (left, bottom, width, heigth), controls the
        position of the colorbar in the figure
    :default: (1.5, 0.05, 0.1, 0.38)
:param vmin: float
        low point for mapping the colors of :param cmap: to the
        cells values.
    :default: 0
:param vmax: float
        high point for mapping the colors of :param cmap: to the
        cells values.
    :default: 1
:param annot: bool
        to display the value inside the cells.
    :default: False
        disables the display of the cell value.
:param linewidths: float
        thickness of the lines inbetween individual cells.
    :default: 0.5
:param linecolor: str
        color of the lines inbetween individual cells.
    :default: "black"
:param dendrogram_ratio: float|tuple[float, float]
        proportion of the figure size devoted to the dendrogram
        lines
    :default: (0.2, 0)
:param col_cluster: bool
        controls the clustering of the columns.
    :default: False
        disables the clustering of the columns.

:returns: Fingerprinter

:raises: pandas.errors.EmptyDataError
    """
    finger = Fingerprinter(data)
    finger.plot(*args, **kwargs)
    finger.split_labels()
    finger.align_labels(factor=2)
    finger.norm_fig_size()
    return finger


def heatmap(data: pd.DataFrame, *args, **kwargs) -> Heatmapper:
    """ Wrapper around the Heatmapper class.

The plotting is done using `seaborn.heatmap`. For further more
thorough documentation check their website. Below some possible
keywords are elaborated. All keywords that are compatible with
`seaborn.heatmap` can be used.

:param data: pandas.DataFrame
        data of merged DataFrame to be plotted.
:param cmap: str
        colormap to use for the connecting lines. Standard
        matplotlib colormaps
        are supported.
    :default: "Greens"
        colormap from white to green.
:param cbar: bool
        to display the colorbar belonging to the figure.
    :default: True
:param vmin: float
        low point for mapping the colors of :param cmap: to the
        cells values.
    :default: 0
:param vmax: float
        high point for mapping the colors of :param cmap: to the
        cells values.
    :default: 1
:param annot: bool
        to display the value inside the cells.
    :default: False
        disables the display of the cell value.
:param linewidths: float
        thickness of the lines inbetween individual cells.
    :default: 0.5
:param linecolor: str
        color of the lines inbetween individual cells.
    :default: "black"

:returns: matplotlib.figure
        containing the heatmap

:raises: pandas.errors.EmptyDataError
        when the merged DataFrame is empty.
    """
    heat = Heatmapper(data)
    heat.plot(*args, **kwargs)
    heat.split_labels()
    heat.align_labels(factor=-0.5)
    heat.norm_fig_size()

    return heat


if __name__ == "__main__":
    sys.exit()
