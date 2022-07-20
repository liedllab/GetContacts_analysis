from functools import reduce
import mdtraj as md
import numpy as np
import pandas as pd


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

def max_extent(ax):
    widths = list()
    for label in ax.get_yticklabels():
        bb = label.get_window_extent()
        width = bb.transformed(ax.transData.inverted()).width
        widths.append(width)
    return max(widths)

def sort_index(df, *, inplace=False):
    if not inplace:
        df = df.copy()

    df.index = pd.MultiIndex.from_arrays(
            np.array([sorted(pair) for pair in df.index]).T
            )
    return df


def sequencer(*paths, start:int = 1) -> pd.DataFrame:
    dfs = list()
    for path in paths:
        top = md.load_topology(path)
        sequence, _ = top.to_dataframe()
        dfs.append(
                (sequence.loc[top.select_atom_indices("alpha"), ["resSeq", "resName"]]
                    .reset_index(drop=True))
                )
    sequence = reduce(lambda left, right: pd.merge(left, right, on=["resSeq"], how="outer"), dfs)
    sequence = sequence.set_index("resSeq")
    if start:
        sequence.index += start - sequence.index[0]
    return sequence
