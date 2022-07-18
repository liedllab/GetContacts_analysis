from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class MatrixPlot(ABC):
    def __init__(self, data, fig=None, ax=None, **kwargs):
        if data.empty:
            raise pd.errors.EmptyDataError("The DataFrame is empty")
        self.data = data
        
        self.fig = fig
        self.ax = ax

    @abstractmethod
    def plot(self, cmap, **kwargs):
        pass
    
    def recolor(self, cmaps, masker):
        mesh = self.ax.collections[0]
        data = mesh.get_array().data

        cmaps = [plt.cm.get_cmap(name, 256) for name in cmaps]
        fc = mesh.get_facecolors()
        for n, (mask, value) in enumerate(zip(masker, data)):
            fc[n] = cmaps[mask](value)

        mesh.set_array(None)
        mesh.set_facecolor(fc)
    

class Heatmap(MatrixPlot):
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


class Fingerprint(MatrixPlot):
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
    


def mask_gen(*args: int):
    """Returns an infinite sequence that enumerates the elements provided and repeats the enumerated value as many time as specified.
    
    Example:
    >>> for i in mask_gen(1,2):
            print(i)
    0
    1
    1
    0
    1
    ...
    """
    
    while True:
        for n, a in enumerate(args):
            for _ in range(a):
                yield n

def _kw_handler(defaults: dict, kwargs:dict, error: bool = False):
    """updates a default dictionary with the kwargs dictionary.

    :param defaults: dict, containing the default variable names and values.
    :param kwargs: dict, dictionary containing the variable names and values from the function call

    :returns: dict, updated defaults dictionary

    :raises: KeyError, if `kwargs` contains a key that `defaults` does not.
    """
    defaults = defaults.copy()
    if error:
        for key in kwargs:
            if not key in defaults:
                raise KeyError(f"{key} not defined")
    defaults.update(kwargs)
    return defaults

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

    p = Fingerprint(df)
    p.plot()
    p.recolor(["Greens", "Blues"], mask_gen(1,1))
    plt.show()
