"""Utility functions for handling the GetContacts Frequency Datafiles and helper
functions for the plotting functions."""

from __future__ import annotations

from functools import reduce, partial
from collections import deque
import sys
from typing import Iterable, NamedTuple
import mdtraj as md
import numpy as np
import matplotlib
import pandas as pd

class Mark(NamedTuple):
    label: str
    color: str
    selection: Iterable


class Label(NamedTuple):
    n : int
    tick: float


def read_tsv(path: str, drop_chain_info: bool=True) -> pd.DataFrame:
    """Loads a Frequency tsv-file as pandas DataFrame.

    :param path: str
            Path to the file to be read in.
    :param drop_chain_info: bool
            Wether the chain information should be discarded.
        :default: True

    :returns: pandas.DataFrame
    """

    df = pd.read_csv(path, sep=r'\t|:', skiprows=2, header=None, engine='python')
    df.columns = ["chain 1", "res 1", "number 1", "chain 2",  "res 2", "number 2", "frequency"]
    if drop_chain_info:
        df = df.drop(labels=["chain 1", "chain 2"], axis=1)

    return df


def select(df: pd.DataFrame,
            selection: Iterable[Iterable], *,
            kind: str="range",
            include_end=True,
            exclusive: bool=True,
            ) -> pd.DataFrame:
    """Select residues from frequency DataFrame.

    :param df: pd.DataFrame
            DataFrame containing the Frequeny information
    :param selection: Iterable[Iterable]
            Iterable that contains Iterables that define the
            selection.
    :param kind: str
           "range" transform selection iterables into ranges from
           min to max.
           "selection" takes exactly the values in the iterables.
        :default: "range"
    :param include_ends: bool
            When kind="range" if the max value of the iterable should
            be included or not.
        :default: True
    :param exclusive: bool
            Wether only one of the interacting partners may be in
            the same selection.
        :default: True

    :returns: pd.DataFrame
    """

    sel_function = {
            "range" : lambda x: range(x[0], x[-1]+include_end),
            "selection" : lambda x: x,
            }
    func = sel_function[kind]
    selection = deque(func(sel) for sel in selection)
#    selection = deque(map(sel_function[kind], selection))
    selection.appendleft(df)

    par_selector = partial(_selector, exclusive)
    return reduce(par_selector, selection)


def _selector(exclusive: bool, df: pd.DataFrame, selection: Iterable) -> pd.DataFrame:
    """Helper function for select that handles the exclusive option."""

    bool_df = (df.select_dtypes(int)
                .isin(selection))
    if exclusive:
        return df[bool_df.nunique(axis=1) == 2]
    return df[bool_df.any(axis=1)]


def merge_on_number(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge Frequency DataFrame on numbers.

    :param dfs: list[pd.DataFrame]
            list that contains the DataFrames to be merged.

    :returns: pd.DataFrame
    """

    prepared_dfs = list()
    for df in dfs:
        _df = (df.drop(
                labels=["res 1", "res 2", "chain 1", "chain 2"],
                axis=1,
                errors="ignore",
                )
                .set_index(keys=["number 1", "number 2"]))
        prepared_dfs.append(_df)

    concat = pd.concat(
            prepared_dfs,
            keys=range(len(prepared_dfs)),
            )
    merged = (concat.unstack(level=-3, fill_value=0)
                    .droplevel(0, axis=1)
                    .sort_index())
    return merged


def sequencer(paths: list[str], *, start: int=1) -> pd.DataFrame:
    """Reads in a list of paths to topologies to return a DataFrame
    containing the sequence. This DataFrame then can be used to
    correctly label the merged Frequency DataFrame.

    :param paths: list[str]
            list of paths to the topologies
    :param start: int
            start residue number for the index
        :default: 1

    :returns: pd.DataFrame
            containing the sequence information of the topologies
    """

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


def label_merged(df: pd.DataFrame, sequences: pd.DataFrame, inplace: bool=False) -> pd.DataFrame:
    """Labels the merged Frequency DataFrame with sequence information.

    :param df: pd.DataFrame
            DataFrame with residue numbers of interacting residues as
            MultiIndex
    :param sequences: pd.DataFrame
            DataFrame with the sequence information, e.g. as obtained
            from `sequencer`
    :param inplace: bool
            change the merged DataFrame inplace or not.
        :default: True

    :returns: pd.DataFrame
            containing the labeled DataFrame
    """
    if not inplace:
        df = df.copy()

    df.index = pd.MultiIndex.from_arrays(
            np.array([ind_pair * sequences.shape[1] for ind_pair in df.index.to_list()]).T,
            names = np.repeat(sequences.columns, sequences.ndim),
            )

    for n, (_, col) in enumerate(sequences.items()):
        df = (df.rename(col.to_dict(), level=n*2)
                .rename(col.to_dict(), level=1+n*2))

    return df


def select_system(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Select slice from selected DataFrame with same name for column and indexes.
    Always returns a copy.

    :param df: pd.DataFrame
            containing multiple systems with identical index and column names
    :param name: str
            name to be selected

    :returns: pd.DataFrame
            containing the index and the column with selected name.
    """
    df = df.copy()
    df = df.droplevel(
            list(np.argwhere(~np.array(
                [ind_name == name for ind_name in df.index.names]
                )).reshape(-1))
            )
    df = df.loc[:, name]

    return df


def mask_gen(*args: int):
    """Returns an infinite sequence that enumerates the elements
    provided and repeats the enumerated value as many time as
    specified.

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
        for n, arg in enumerate(args):
            for _ in range(arg):
                yield n

def sort_index(df: pd.DataFrame, *,
               inplace: bool=False,
               merge_duplicate: bool=False) -> pd.DataFrame:
    """Sorts the index of DataFrame.

    :param df: pd.DataFrame
            which contains the index to be sorted
    :param inplace: bool
            change the merged DataFrame inplace or not.
        :default: True
    :param merge_duplicate: bool
            merge duplicate labels after the sorting of the indices.

    :returns: pd.DataFrame
            with sorted index
    """

    if not inplace:
        df = df.copy()

    df.index = pd.MultiIndex.from_arrays(
            np.array([sorted(pair) for pair in df.index]).T
            )
    if merge_duplicate:
        df = (df.reset_index()
                .groupby(['level_0', 'level_1'])[list(df.columns)]
                .sum())

    return df


def max_extent(ax: matplotlib.axes.Axes) -> float:
    """Get the max extent of all yticklabels.

    :param ax:
            Axis that contains the yticklabels

    :returns: float
            maximum extent of the yticklabels
    """
    widths = list()
    for label in ax.get_yticklabels():
        bb = label.get_window_extent()
        width = bb.transformed(ax.transData.inverted()).width
        widths.append(width)
    return max(widths)


def closest(val: float, arr: np.array,*,  lower: bool=True) -> float:
    """Calculate the next closest value in array.

    :param val: float
            value
    :param arr: np.array
            Array in which the next closes element to value is searched.
    :param lower: bool
            Wether to get the next lower or the next higher value
        :default: True

    :returns: float
            the next closest value
    """

    if lower:
        return arr[arr-val >= 0][0]

    return arr[arr-val <= 0][-1]


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

three2one = {
        'CYS' : 'C',
        'CYX' : 'C',

        'ASP' : 'D',
        'ASH' : 'D+',
        'ASN' : 'N',

        'GLN' : 'Q',
        'GLH' : 'E+',
        'GLU' : 'E',

        'LYS' : 'K',
        'LYN' : 'K+',

        'HIS' : 'H',
        'HIE' : 'HIE',
        'HID' : 'HID',
        'HIP' : 'HIP',

        'SER' : 'S',
        'ILE' : 'I',
        'PRO' : 'P',
        'THR' : 'T',
        'PHE' : 'F',
        'GLY' : 'G',
        'LEU' : 'L',
        'ARG' : 'R',
        'TRP' : 'W',
        'ALA' : 'A',
        'VAL' : 'V',
        'TYR' : 'Y',
        'MET' : 'M'
    }

if __name__ == '__main__':
    sys.exit()
