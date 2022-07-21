from abc import ABC, abstractmethod
from itertools import chain
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .coordinates import bezier, Polar, Cartesian
from .utils import closest

class MarkupStrategy(ABC):
    def __init__(self, gap_scale=0.3):
        self._mark = list()
        self._colors = list()
        self._labels = list()
        self.data = None
        self.gap_scale = gap_scale

    def marks(self, positions, marking, on='number'):
        positions = positions.reset_index().loc[:,[on, "tick"]]
        positions = positions.set_index(on).squeeze()
        for label, (color, selection) in marking.items():
            self.selector(positions, list(selection), color, label)
        self.as_df()
        return self.data

    def gap(self, width):
        gap = width * self.gap_scale
        side = width / 2 - gap

        return np.array([-side, side])

    @abstractmethod
    def selector(self, positions, selection, color, label) -> None:
        pass

    @abstractmethod
    def as_df(self) -> None:
        pass


class RangeMarkup(MarkupStrategy):
    def selector(self, positions, selection, color, label) -> None:
        values = positions.loc[
                [closest(min(selection), positions.index, lower=True),
                    closest(max(selection), positions.index, lower=False)]].values
        self._mark.append(values + self.gap(positions.values[1]))
        self._colors.append(color)
        self._labels.append(label)

    def as_df(self) -> None:
        df = pd.DataFrame(self._mark)
        df["color"] = self._colors
        df["label"] = self._labels
        self.data = df


class SelectionMarkup(MarkupStrategy):
    def selector(self, positions, selection, color, label) -> None:
        values = positions.loc[selection].values
        self._mark.append(values.reshape(-1,1) + self.gap(positions.values[1]))
        for _ in values:
            self._colors.append(color)
            label = None if label in self._labels else label
            self._labels.append(label)

    def as_df(self) -> None:
        df = pd.DataFrame(np.concatenate(self._mark))
        df["color"] = self._colors
        df["label"] = self._labels
        self.data = df
       

class IndividualMarkup(MarkupStrategy):
    def selector(self, positions, selection, color, label) -> None:
        values = positions.loc[min(selection): max(selection)].values
        self._mark.append(values.reshape(-1,1) + self.gap(positions.values[1]))
        for _ in values:
            self._colors.append(color)
            label = None if label in self._labels else label
            self._labels.append(label)

    def as_df(self) -> None:
        df = pd.DataFrame(np.concatenate(self._mark))
        df["color"] = self._colors
        df["label"] = self._labels
        self.data = df


class Label(NamedTuple):
    n : int
    tick: float

class Flareplotter:
    def __init__(self, data, fig=None, ax=None, **kwargs):
        self.data = data
        
        if not fig:
            fig = plt.figure()
        if not ax:
            ax = fig.add_subplot(polar=True)

        self.fig = fig
        self.ax = ax
        self.highlighted = list()

    
    def plot(self, cmap: str = "Greys", vmin=0, vmax=1, cbar=True):
        self.ax.set_theta_zero_location("E", offset=0)
        self.ax.grid(False)
        
        info = self.label_info()
        labels, ticks = zip(*[(text, label.tick) for text, label in info.items()])
        self.ax.set_xticks(ticks, labels)
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

            self.ax.plot(
                    *curve_points.T,
                    color=sm_cmap.get_cmap()(value),
                    lw=(1.5+value)**2,
                    zorder=10-10*(1-value),
                    )
            self.ax.spines['polar'].set_visible(True)

        return self

    def mark(self, marking: dict, *, strategy_name: str = "range", **kwargs) -> None:
        strategies = {
                "range": RangeMarkup,
                "individual": IndividualMarkup,
                "selection": SelectionMarkup,
                }
        strat = strategies[strategy_name](**kwargs)
        strat.marks(self.positions, marking)

        self.reset_marking()

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
    
    def reset_marking(self) -> None:
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

    def rotate_labels(self, new_labels=None, shift=-0.1):
        self.reset_labels()
        self.fig.canvas.draw()

        for label in self.ax.get_xticklabels():
            text = label.get_text()
            if text.isnumeric():
                text = int(text)
            deg = np.rad2deg(self.label_info()[text].tick)

            if deg > 90 and deg < 270:
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

    def __repr__(self):
        return self.fig

def flareplot(data):
    flare = Flareplotter(data)
    flare.plot()
    flare.rotate_labels()
    return flare


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
    
    flare = Flareplotter(df.loc[:, "wildtype"])
    flare.plot()
    flare.rotate_labels()
    flare.mark(marking = {
        "light chain" : ("steelblue", [36, 118]),
        "heavy chain" : ("lightgray", [254, 348]),
        },
        strategy_name = "individual",
        gap_scale=0.1,
        )

    flare.ax.set_title("wildtype", pad=40)
#    flare.fig.legend(loc="upper left")

    plt.show()


