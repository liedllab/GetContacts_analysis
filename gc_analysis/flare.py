"""Plotting relational data using Flareplots."""
from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import chain
from typing import NamedTuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .coordinates import bezier, Polar
from .utils import closest, Mark, Label

class MarkupStrategy(ABC):
    def __init__(self, gap_scale: float=0.3):
        self._mark = list()
        self._colors = list()
        self._labels = list()
        self._data = None
        self.gap_scale = gap_scale

    def marks(self, positions, marking, on='number'):
        positions = positions.reset_index().loc[:,[on, "tick"]]
        positions = positions.set_index(on).squeeze()
        for label, color, selection in marking:
            self.selector(positions, label, color, selection)
        return self.data

    def gap(self, width):
        gap = width * self.gap_scale
        side = width / 2 - gap

        return np.array([-side, side])

    @abstractmethod
    def selector(self, positions, label, color, selection) -> None:
        pass

    @abstractmethod
    def as_df(self) -> None:
        pass

    @property
    def data(self):
        df = self.as_df()
        df["color"] = self._colors
        df["label"] = self._labels
        self._data = df
        return self._data


class RangeMarkup(MarkupStrategy):
    def selector(self, positions, label, color, selection) -> None:
        values = positions.loc[
                [closest(min(selection), positions.index, lower=True),
                    closest(max(selection), positions.index, lower=False)]].values
        self._mark.append(values + self.gap(positions.values[1]))
        self._colors.append(color)
        self._labels.append(label)

    def as_df(self) -> None:
        return pd.DataFrame(self._mark)


class SelectionMarkup(MarkupStrategy):
    def selector(self, positions, label, color, selection) -> None:
        values = positions.loc[selection].values
        self._mark.append(values.reshape(-1,1) + self.gap(positions.values[1]))
        for _ in values:
            self._colors.append(color)
            label = None if label in self._labels else label
            self._labels.append(label)

    def as_df(self) -> None:
        return pd.DataFrame(np.concatenate(self._mark))


class IndividualMarkup(MarkupStrategy):
    def selector(self, positions, label, color, selection) -> None:
        values = positions.loc[min(selection): max(selection)].values
        self._mark.append(values.reshape(-1,1) + self.gap(positions.values[1]))
        for _ in values:
            self._colors.append(color)
            label = None if label in self._labels else label
            self._labels.append(label)

    def as_df(self) -> None:
        return pd.DataFrame(np.concatenate(self._mark))


class Flareplotter:
    def __init__(self, data, fig=None, ax=None):
        self.data = data

        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(polar=True)
        else:
            fig = plt.gcf()

        self.fig = fig
        self.ax = ax
        self.highlighted = list()
        self.labels = None
        self.ticks = None


    def plot(self, *, cmap: str = "Greys", cbar: bool=False,
            vmin: float=0, vmax: float=1, linewidth: float=1.5,
            scale_thickness: bool=True, **line_kwargs) -> Fingerprinter:
        """
:param cmap: str
        colormap to use for the connecting lines. Standard matplotlib colormaps
        are supported.
    :default: "Greys"
        which is a colormap from white to black.
:param cbar: bool
        wether a cbar should be added to the plot or not.
    :default: False
        which plots a colorbar
:param vmin: float
        low point for mapping the colors of :param cmap: to the
        cells values.
    :default: 0
:param vmax: float
        high point for mapping the colors of :param cmap: to the
        cells values.
    :default: 1
:param linewidth: int|float
        base thickness of the connecting lines.
    :default: 1.5
:param scale_thickness: bool
        wether the thickness of the connecting lines scale with the value.
    :default: True

:returns: Fingerprinter
        """
        self.ax.set_theta_zero_location("E", offset=0)
        self.ax.grid(False)

        info = self.label_info()
        self.ax.set_xticks(self.ticks, self.labels)
        self.ax.set_rlim(top=1)
        self.ax.set_yticklabels([])

        sm_cmap = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(vmin=vmin, vmax=vmax),
                )

        if cbar:
            cax = self.fig.colorbar(sm_cmap, pad=0.2, ax=self.ax)
            cax.set_label("Frequency")

        for (left, right), value in self.data.iteritems():
            curve = bezier(
                    Polar(theta=info[left].tick, r=1),
                    Polar(theta=info[right].tick, r=1),
                    )
            curve_points = np.array([point.convert().values for point in curve])
            lw = (linewidth + value)**2 if scale_thickness else linewidth
            self.ax.plot(
                    *curve_points.T,
                    color=sm_cmap.get_cmap()(value),
                    lw=lw,
                    zorder=10-10*(1-value),
                    **line_kwargs
                    )
            self.ax.spines['polar'].set_visible(True)

        return self

    def mark(self, marking: dict[str, Mark], *, strategy_name: str = "range", **strategy_kwargs) -> None:
        self.reset_marking()
        strategies = {
                "range": RangeMarkup,
                "individual": IndividualMarkup,
                "selection": SelectionMarkup,
                }
        strat = strategies[strategy_name](**strategy_kwargs)
        strat.marks(self.positions, marking)

        for _, row in strat.data.iterrows():
            thetas = np.arange(row[0], row[1]+0.01, 0.01)
            fill = self.ax.fill_between(
                    thetas,
                    np.ones_like(thetas),
                    1.1,
                    color=row["color"],
                    label=row["label"],
                    zorder=11
                    )
            self.highlighted.append(fill)

        self.ax.set_rlim(top=1.1)
        self.ax.spines['polar'].set_visible(False)

        return self.ax

    def reset_marking(self) -> Flareplotter:
        for n in range(len(self.highlighted)):
            self.highlighted[n].remove()
        self.highlighted = []

        try:
            self.ax.get_legend().remove()
        except AttributeError:
            # None if no legend
            pass

        for _ in range(len(self.fig.legends)):
            self.fig.legends[0].remove()

        self.ax.set_rlim(top=1)
        self.ax.spines['polar'].set_visible(True)
        return self

    def rotate_labels(self, shift: float=-0.1) -> Flareplotter:
        self.reset_labels()
        self.fig.canvas.draw()

        for label in self.ax.get_xticklabels():
            text = label.get_text()
            if text.isnumeric():
                text = int(text)
            deg = np.rad2deg(self.label_info()[text].tick)

            if 90 <= deg <= 270:
                deg -= 180

            lab = self.ax.text(*label.get_position()+np.array([0, shift]),
                    text,
                    transform=label.get_transform(),
                    ha=label.get_ha(),
                    va=label.get_va(),
                    )
            lab.set_rotation(deg)
        self.ax.set_xticklabels([])
        return self

    def reset_labels(self):
        for _ in range(len(self.ax.texts)):
            self.ax.texts[0].remove()

        labels, ticks = zip(*[(text, label.tick) for text, label in self.label_info().items()])
        self.ax.set_xticks(ticks, labels)

    def label_info(self):
        unique_labels = set(chain(*self.data.index))
        numbers = map(
                lambda lab: int(''.join(filter(str.isdigit, str(lab)))),
                unique_labels,
                )
        ticks = np.linspace(0, 2*np.pi, endpoint=False, num=len(unique_labels))

        try:
            info = dict()
            for (number, label), tick in zip(
                    sorted(zip(numbers, unique_labels)),
                    ticks,
                    ):
                info[label] = Label(number, tick)
        except ValueError:
            info = dict()
            for number, (label, tick) in enumerate(zip(unique_labels, ticks)):
                info[label] = Label(number, tick)

        self.labels, self.ticks = zip(*[(text, label.tick) for text, label in info.items()])
        return info

    @property
    def positions(self) -> pd.Series:
        df = pd.DataFrame(self.label_info()).T
        df[0] = df[0].astype(int)
        df = (df.set_index(0, append=True)
                .swaplevel())
        df.index.names = ["number", "label"]
        df.columns = ["tick"]

        return df

    def _repr_png_(self):
        return display(self.fig)


def flareplot(data, *, cmap="Greys", ax=None, **line_kwargs):
    """Wrapper around Flareplotter class.

:param data: pandas.DataFrame
        data of merged DataFrame to be plotted.
:param cmap: str
        colormap to use for the connecting lines. Standard matplotlib colormaps
        are supported.
    :default: "Greys"
        which is a colormap from white to black.
:param cbar: bool
        wether a cbar should be added to the plot or not.
    :default: False
        which plots a colorbar
:param vmin: float
        low point for mapping the colors of :param cmap: to the
        cells values.
    :default: 0
:param vmax: float
        high point for mapping the colors of :param cmap: to the
        cells values.
    :default: 1
:param linewidth: int|float
        base thickness of the connecting lines.
    :default: 1.5
:param scale_thickness: bool
        wether the thickness of the connecting lines scale with the value.
    :default: True

:returns: Fingerprinter
        """

    flare = Flareplotter(data, ax=ax)
    flare.plot(cmap=cmap, **line_kwargs)
    flare.rotate_labels()
    return flare


if __name__ == "__main__":
    sys.exit()
