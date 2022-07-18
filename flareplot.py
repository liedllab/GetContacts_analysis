from abc import ABC, abstractmethod
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from coordinates import bezier, Polar, Cartesian

class MarkupStrategy(ABC):
    def __init__(self, gap_scale=0.3):
        self._mark = list()
        self._colors = list()
        self._labels = list()
        self.data = None
        self.gap_scale = gap_scale

    def marks(self, positions, marking):
        for label, (color, selection) in marking.items():
            self.selector(positions, selection, color, label)
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
        values = positions.loc[[min(selection), max(selection)]].values
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
            self._labels.append(label)

    def as_df(self) -> None:
        df = pd.DataFrame(np.concatenate(self._mark))
        df["color"] = self._colors
        df["label"] = self._labels
        self.data = df


class Flareplot:
    def __init__(self, data, fig=None, ax=None, **kwargs):
        self.data = data
        
        if not fig:
            fig = plt.figure()
        if not ax:
            ax = fig.add_subplot(polar=True)

        self.fig = fig
        self.ax = ax
        self._unique_labels = sorted(set(chain(*self.data.index)))
    
    def plot(self, cmap: str = "Greys", vmin=0, vmax=1, cbar=True):
        self.ax.set_theta_zero_location("E", offset=0)
        self.ax.grid(False)
        self.ax.set_xticks(self.ticks, self.unique_labels)
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
                    Polar(theta=self.positions[left], r=1), 
                    Polar(theta=self.positions[right], r=1),
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

        for _, row in strat.data.iterrows():
            thetas = np.arange(row[0], row[1]+0.01, 0.01)
            self.ax.fill_between(
                    thetas, 
                    np.ones_like(thetas), 
                    1.1, 
                    color=row["color"],
                    label=row["label"],
                    zorder=11
                    )

        self.ax.set_rlim(top=1.1)
        self.ax.spines['polar'].set_visible(False)

        return self.ax

    def reset_labels(self):
        for _ in range(len(self.ax.texts)):
            self.ax.texts[0].remove()

        self.ax.set_xticks(self.ticks, self.unique_labels)

    def rotate_labels(self, new_labels=None, shift=-0.1):
        self.reset_labels()
        self.fig.canvas.draw()

        for label in self.ax.get_xticklabels():
            text = label.get_text()
            if text.isnumeric():
                text = int(text)
            deg = np.rad2deg(self.positions.loc[text])

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
    
    @property
    def ticks(self) -> np.array:
        return np.linspace(0, 2*np.pi, endpoint=False, num=len(self.unique_labels))
   
    @property
    def positions(self) -> pd.Series:
        positions = {label: pos for label, pos in zip(self.unique_labels, self.ticks)}
        return pd.Series(positions)

    @property
    def unique_labels(self) -> set:
        return self._unique_labels

    @unique_labels.setter
    def unique_labels(self, values):
        if not len(self._unique_labels) == len(values):
            raise ValueError("lengths do not match!")
        self._unique_labels = values

    def __repr__(self):
        return self.fig

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
    
    flare = Flareplot(df.loc[:, "wildtype"])
    flare.plot()
    flare.rotate_labels()
    flare.mark(marking = {
        "light chain" : ("steelblue", [36, 118]),
        "heavy chain" : ("lightgray", [254, 348]),
        },
        strategy_name = "range",
        gap_scale=0.1,
        )

    flare.ax.set_title("wildtype", pad=40)
    flare.fig.legend(loc="upper left")

    plt.show()


