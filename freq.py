#! /usr/bin/env python3

"""Main module containing the :class:`Frequency` which is a container for handeling the \
        tsv-datafile.

:class:`Frequency` currently depends on `coordinates.py` to utilize the :method:`Frequency.flare`.

The two plotting functions :func:`fingerprint` and :func:`heatmap` are available to
visualize the data and do not depend on `coordinates.py`.
"""

from typing import Optional, Tuple
import warnings

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#for flare
import coordinates as coord


class Frequency:
    """Container for the data loaded from a Frequency-file calculated with getcontacts.

    The data from the tsv-file is stored as a pandas.DataFrame. Further some more information
    can be provided which can be useful when multiple instances exist.
    A instance should be initialized through :classmethod:`Frequency.from_tsv`.

    :attribute df: pandas.DataFrame, containing the data from the tsv-file.
    :attribute filename: str, the path to the tsv-file containing the data.
    :attribute interaction: Optional[str], short description of the kind of interaction the
            Frequencies were calculated for.
        :default: "all"
    :attribute name: Optional[str], individual name, that can be chosen. If not provided, the value
            for :instance_attribute:`interaction` is used.
        :default: None
    :attribute selection: Optional[tuple[tuple[int, int], tuple[int, int]]],
            The selection that was applied on the original data.
        :default: None, The usual workflow is to calculate the interactions for all residues and
            then later choose the selection with :method:`select`.

    :classmethod from_tsv:  usually used to initialize a new Frequency object.

    :method select: select one or two ranges of residues.
    :method flare: plot the data as a circular flareplot.
    :method info: prints a short description of the system.
    :method compatible: check wether its sensible to merge the data of two :instance:`Frequency`'s.
    """

    def __init__(self, df: pd.DataFrame, filename:str,
                 interaction: Optional[str] = "all",
                 name: Optional[str] = None,
                 selection: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None):

        if not isinstance(df, pd.DataFrame):
            raise ValueError("DataFrame expected.")
        self._df = df
        self._filename = filename
        self._interaction = interaction
        self.name = name if name else interaction
        # if not name:
        #     self.name = interaction
        # else:
        #     self.name = name
        self._selection = selection

    @classmethod
    def from_tsv(cls, filename: str, interaction: Optional[str] = "all",
                 name: Optional[str] = None):
        """Classmethod to initialize a new Frequency object.

        :param filename: str, path to the file containing the data.
        :param interaction: Optional[str], short description of the kind of interactoin the
                Frequencies were calculated for.
            :default: "all"
        :param name: Optional[str], name for the Frequency Object, e.g. its pdb code. If not
                provided, the value of :param interaction: is used.
            :default: None

        :returns: :instance:`Frequency`

        :raises: pandas.errors.EmptyDataError as a warning.
        """
        try:
            df = pd.read_csv(filename, sep=r'\t|:', skiprows=2, header=None, engine='python')
        except pd.errors.EmptyDataError:
            # error is caught because for some interactions the tsv-file might be empty
            warnings.warn(f"{filename} is empty.")
            df = pd.DataFrame(columns=range(7))

        df = df.drop(labels=[0,3], axis=1)
        df.columns = ['res 1', 'number 1', 'res 2', 'number 2', 'contact_frequency']

        return cls(df, filename, interaction, name)

    def select(self, residues:Tuple[int, int], more: Optional[Tuple[int, int]]=None):
        """Method for the selection of ranges of residues.

        :param residues: tuple[int, int], range of residues that is selected.
        :param more: Optional[tuple[int, int]], range of furter residues to be selected.

        :returns: :instance:`Frequency`, a new instance with the chosen seleciton is returned.
        """
        if more:
            df = self.df[self.df['number 1'].between(*residues) & \
                    self.df['number 2'].between(*more) | \
                    self.df['number 2'].between(*residues) & \
                    self.df['number 1'].between(*more)]
        else:
            df = self.df[self.df['number 1'].between(*residues) | \
                    self.df['number 2'].between(*residues)]

        return self.__class__(df, self.filename, self.interaction, self.name, (residues, more))

    def flare(self, *, cutoff = 0.6, cmap = 'Greys', cbar = True, one_letter = True,
             linewidth = 6, fontsize = 10, shift = 0, label_offset = 0.2,
             tick_length = 3, tick_width = 1):
        """Create a flareplot using the systems data.

        :param cutoff: float, datapoints with a frequency below the cutoff threshold are not used
                for plotting.
        :param cmap: str, colormap to use for the connecting lines. Standard matplotlib colormaps
                are supported.
            :default: "Greys", which is a colormap from white to black.
        :param cbar: bool, wether a cbar should be added to the plot or not.
            :default: True, which plots a colorbar
        :param one_letter: bool, wether the default threeletter code labels should be converted to
            oneletter code.
        :param linewidth: int|float, base thickness of the connecting lines. The thickness gets also
            scaled with the frequency.
        :param fontsize: int|float, size of the labels font.
        :param shift: int|float, is added to the angles, in degrees.
        :param label_offset: int|float, amount the labels are offset from their baseposition. Should
                be chosen in a way, so the labels don't overlap with the plot.
        :param tick_length: int|float, length of the ticks.
        :param tick_width: int|float, width of the ticks.
        """
        # apply cutoff to df
        cut = self.df[self.df['contact_frequency'] > cutoff].reset_index(drop=True)

        # get all unique labels for the labels on the flare plot
        all_labels = cut[['number 1', 'res 1', 'number 2', 'res 2']].values.reshape(
                 cut.shape[0]*2, 2).tolist()
        all_unique_labels = sorted(set(zip(*zip(*all_labels))))
        # create a point for each label between which the connecting bezier
        # curve is drawn
        points = {num: coord.Polar(r=1, theta= angle)
                  for (num,_), angle in zip(all_unique_labels,
                                            np.linspace(0, 2*np.pi,
                                                len(all_unique_labels), endpoint=False)
                                            )}

        fig = plt.figure()
        ax = fig.add_subplot(polar=True)
        # set zero location to be at 0°
        ax.set_theta_zero_location('E', offset=shift)

        cmap = mpl.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=cutoff, vmax=1)
        if cbar:
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), pad=0.2, ax=ax)

        # plot all connecting lines
        # for more information about the coordinates function bezier and the
        # coordinate classes Polar and Cartesian check out the coordinates.py
        # file.
        for i in cut.index:
            ax.plot(
                *np.array(
                    [b.convert().values for b in coord.bezier(
                        points[cut.loc[i, 'number 1']],
                        points[cut.loc[i, 'number 2']]
                    )]).T,
                # decreasing visibility for less frequent contacts
                alpha = cut.loc[i, 'contact_frequency']**2,
                # color according to cmap and value
                color=cmap(norm(cut.loc[i, 'contact_frequency'])),
                # thicker lines for more frequent contacts
                lw=linewidth * cut.loc[i, 'contact_frequency']**2,
                # less frequent contact lines are rendered earlier than more
                # frequent contact lines. This makes sure the more important
                # connecting lines are always on top.
                zorder = cut.loc[i, 'contact_frequency']*100
            )

        ax.set_xlabel(ax.get_xlabel())
        ax.set_thetagrids(np.linspace(0,360,len(all_unique_labels), endpoint=False),
                          [f"{three2one[res] if one_letter else res} {num:03}"
                              for num, res in all_unique_labels])

        ax.grid(False)
        ax.set_rlim([0,1])
        ax.set_rticks([])

        # the following code is needed to have the labels at the edge of the
        # circle at the right angle

        plt.gcf().canvas.draw()
        angles = np.linspace(0,360, len(all_unique_labels), endpoint=False)

        angles[np.cos(np.deg2rad(angles+shift)) < 0] += 180
        labels = []
        for label, angle in zip(ax.get_xticklabels(), angles):
            x,y = label.get_position()
            lab = ax.text(x, y-label_offset,
                          label.get_text(), transform=label.get_transform(), fontsize=fontsize,
                          ha = label.get_ha(), va=label.get_va())
            lab.set_rotation(angle+shift)
            labels.append(lab)
        ax.set_xticklabels([])

        #ticks for the edge of the circle needs to be plotted extra too
        if tick_length:
            for t in np.deg2rad(np.linspace(0, 360, len(all_unique_labels), endpoint=False)):
                ax.plot([t, t], [1,1-tick_length/100], lw=tick_width, color="k")

        return fig

    @property
    def df(self):
        """:instance_attribute: containing the underlying data.

        :returns: pandas.DataFrame
        """
        return self._df

    @property
    def interaction(self):
        """What kind of interaction the data is about.

        :returns: str
        """
        return self._interaction

    @property
    def filename(self):
        """The filename from which the data was read in.

        :returns: str
        """
        return self._filename

    @property
    def selection(self):
        """The selection applied to the original data.

        :returns: tuple[tuple[int, int], tuple[int,int]]
        """
        return self._selection

    def info(self):
        """Short description of the `Frequency`:instante:."""
        print(f"System:\t\t\t{self.name}\n\
                Interactiontype:\t{self.interaction}\n\
                Frequencies from:\t{self.filename}")

    def compatible(self, other):
        """small check for compatibility
        :param other: Frequency, `Frequency`:instance: to compare the current object to.

        :returns: bool, True if the `Frequency.interaction`:instance_attribute: and the
            `Frequency.selection`:instance_attribute: are the same else False."""

        return self.interaction == other.interaction and self.selection == other.selection

    def __eq__(self, other):
        """check for equality.

        :param other: Frequency, `Frequency`:instance: to compare the current object to.

        :returns: bool, True if the `Frequency.interaction`:instance_attribute:, the
            `Frequency.selection`:instance_attribute: and `Frequency.df.values`:instance_attribute:
            are the same else False."""

        return self.interaction == other.interaction and \
                self.selection == other.selection and \
                np.equal(self.df.values, other.df.values)
                #self.df.values is other.df.values

    def __repr__(self):
        "returns the `pandas.DataFrame.__repr__()` of the `Frequency.df`:instance_attribute:."
        return f"{self.df.__repr__()}"


def heatmap(data: pd.DataFrame,
            title: Optional[str] = None,
            cmap: str = 'Greens',
            y_size: float = 4,
            linewidths: float = 0.5,
            linecolor: str = 'black',
            annot: Optional[bool] = False,
            vmin:float = 0,
            vmax:float = 1,
            cbar: bool = True):
    """Function to conveniently plot the heatmap of interactions.
    The plotting is done using seaborn.heatmap.
    The arguments of the function are exactly the same as for
    seaborn.heatmap. except title."""

    if data.empty:
        raise pd.errors. EmptyDataError("The DataFrame is empty")

    fig = plt.figure()
    ax = fig.add_subplot()

    sns.heatmap(data,
                annot = annot,
                linewidths = linewidths,
                linecolor = linecolor,
                cmap = cmap, vmin = vmin, vmax = vmax, cbar = cbar,
                yticklabels = 1,
                ax = ax)

    # get square fields
    fig.set_size_inches(
        _resize(data.T.shape, y_size)
    )
    #fig.axes[0].set_yticklabels([f"{a}-{b}" for a, b in data.index])

    if title:
        fig.suptitle(title.upper())

    ax.yaxis.set_label_text(None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    return fig

def fingerprint(data: pd.DataFrame,
                title: Optional[str] = None,
                cmap: str = 'Greens',
                y_size: float = 4,
                linewidths: float = 0.5,
                linecolor: str = 'black',
                vmin: float = 0,
                vmax: float = 1,
                col_cluster: bool = False,
                annot: Optional[bool] = False,
                dendrogram_ratio: float = 0.2,
                cbar_pos: Optional[Tuple[float, float, float, float]] = (1.5, 0.05, 0.1, 0.38)):
    """Function to conveniently plot the fingerprint of interactions.
    The plotting is done using seaborn.clustermap.
    The arguments of the function are exactly the same as for
    seaborn.clustermap."""

    if data.empty:
        raise pd.errors.EmptyDataError("The DataFrame is empty")

    finger = sns.clustermap(
        data,
        annot = annot,
        linewidths = linewidths,
        linecolor = linecolor,
        col_cluster = col_cluster,
        dendrogram_ratio = dendrogram_ratio,
        yticklabels = 1,
        cmap = cmap, vmin = vmin, vmax = vmax, cbar_pos = cbar_pos
    )

    # square fields
    finger.fig.set_size_inches(
        _resize(data.T.shape, y_size)
    )

    #finger.fig.axes[2].set_yticklabels([f"{a}-{b}" for a, b in data.index])
    finger.ax_heatmap.yaxis.set_label_text(None)

    if title:
        finger.fig.suptitle(title.upper())
    #finger.fig.supylabel('')

    plt.setp(finger.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.setp(finger.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)

    return finger


def _resize(sequence, target=5):
    """Resizes all elements of sequence, so that the target value is the lowest.

    This function is mainly used for resizing figsizes.

    :param sequence: list[float|int], list that contains the elements to be resized.
    :param target: int|float, lowest value in the output.

    :returns: list[int|float], resized in a way that the lowest value equals the target value.
        """
    smallest = min(sequence)
    return [i/smallest*target for i in sequence]


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
