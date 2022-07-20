from functools import reduce, partial
from collection import deque
from typing import Iterable
import mdtraj as md
import numpy as np
import pandas as pd


def read_tsv(path):
    df = pd.read_csv(path, sep=r'\t|:', skiprows=2, header=None, engine='python')
    df = df.drop(labels=[0,3], axis=1)
    df.columns = ["res 1", "number 1", "res 2", "number 2", "frequency"]
    return df

def select(df: pd.DataFrame, 
            selection: Iterable, 
            exclusive: bool =True, 
            kind: str = "range",
            ) -> pd.DataFrame:

    sel_function = {
            "range" : lambda x: range(*x),
            "selection" : lambda x: x,
            }
    selection = deque(map(sel_function[kind], selection))
    selection.appendleft(df)

    par_selector = partial(_selector, exclusive)
    return reduce(par_selector, selection)

def merge_on_number(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    prepared_dfs = [(df.drop(labels=["res 1", "res 2"], axis=1)
                        .set_index(keys=["number 1", "number 2"])) for df in dfs]
    concat = pd.concat(
            prepared_dfs,
            keys=range(len(prepared_dfs)),
            )
    merged = (concat.unstack(level=-3, fill_value=0)
                    .droplevel(0, axis=1)
                    .sort_index())
    return merged

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

def sort_index(df, *, inplace=False):
    if not inplace:
        df = df.copy()

    df.index = pd.MultiIndex.from_arrays(
            np.array([sorted(pair) for pair in df.index]).T
            )
    return df

def max_extent(ax):
    widths = list()
    for label in ax.get_yticklabels():
        bb = label.get_window_extent()
        width = bb.transformed(ax.transData.inverted()).width
        widths.append(width)
    return max(widths)

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

def _selector(exclusive: bool, df: pd.DataFrame, selection: Iterable) -> pd.DataFrame:

    bool_df = (df.select_dtypes(int)
                .isin(selection))
    if exclusive:
        return df[bool_df.nunique(axis=1) == 2]
    else:
        return df[bool_df.any(axis=1)]
    

