#! /usr/bin/env python3

"""Main module containing the :class:`Frequency` which is a container for handeling the \
tsv-datafile.

:class:`Frequency` currently depends on `coordinates.py` to utilize the :method:`Frequency.flare`.

The two plotting functions :func:`fingerprint` and :func:`heatmap` are available to \
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
from . import coordinates as coord

class Frequency:
    """Container for the data loaded from a Frequency-file calculated with GetContacts.

    The data from the tsv-file is stored as a pandas.DataFrame. Furthermore, extra identifying
    information can be added which can be useful when multiple instances exist.
    A instance should be initialized through :classmethod:`Frequency.from_tsv`.

    :attribute df: pandas.DataFrame
            containing the data from the tsv-file.
    :attribute filename: str
            the path to the tsv-file containing the data.
    :attribute selection: Optional[tuple[tuple[int, int], tuple[int, int]]]
            The selection that was applied on the original data.
        :default: None
            The usual workflow is to calculate the interactions for all residues and
            then later choose the selection with :method:`select`.
    :attribute interaction: Optional[str]
            short description of the kind of interaction the Frequencies were calculated for.
        :default: "all"
    :attribute name: Optional[str]
            individual name, that can be chosen. If not provided, the value for
            :instance_attribute:`interaction` is used.
        :default: None

    :classmethod from_tsv:
            usually used to initialize a new Frequency object.

    :method select:
            select one or two ranges of residues.
    :method flare:
            plot the data as a circular flareplot.
    :method info:
            prints a short description of the system.
    :method compatible:
            check wether its sensible to merge the data of two :instance:`Frequency`'s.
    """

    def __init__(self, df: pd.DataFrame, filename:str, *,
                 selection: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                 **kwargs):
#                  interaction: Optional[str] = "all",
#                  name: Optional[str] = None,
#                  ):

        if not isinstance(df, pd.DataFrame):
            raise ValueError("DataFrame expected.")

        defaults = _kw_handler({
            'interaction' : 'all',
            'name': None,
            }, kwargs)

        self._df = df
        self._filename = filename
        self._selection = selection
        self._interaction = defaults['interaction']
        self.name = defaults['name'] if defaults['name'] else defaults['interaction']
        # if not name:
        #     self.name = interaction
        # else:
        #     self.name = name

    @classmethod
    def from_tsv(cls, filename: str, **kwargs):
        """Classmethod to initialize a new Frequency object.

        :param filename: str
                path to the file containing the data.
        :param interaction: Optional[str]
                short description of the kind of interactoin the Frequencies were calculated for.
            :default: "all"
        :param name: Optional[str]
                name for the Frequency Object, e.g. its pdb code. If not provided,
                the value of :param interaction: is used.
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

        return cls(df, filename, **kwargs)

    def select(self, residues:Tuple[int, int], more: Optional[Tuple[int, int]]=None):
        """Method for the selection of ranges of residues.

        :param residues: tuple[int, int]
                range of residues that is selected.
        :param more: Optional[tuple[int, int]]
                range of further residues to be selected.

        :returns: :instance:`Frequency`
                a new instance with the chosen selection is returned.
        """
        if more:
            df = self.df[self.df['number 1'].between(*residues) & \
                    self.df['number 2'].between(*more) | \
                    self.df['number 2'].between(*residues) & \
                    self.df['number 1'].between(*more)]
        else:
            df = self.df[self.df['number 1'].between(*residues) | \
                    self.df['number 2'].between(*residues)]

        return self.__class__(df, self.filename, selection = (residues, more),
                interaction = self.interaction, name = self.name,
                )

    def flare(self, *, cutoff = 0.6, cmap = 'Greys', cbar = True, one_letter = True,
             **kwargs):
#              linewidth = 6, fontsize = 10, shift = 0, label_offset = 0.2,
#              tick_length = 3, tick_width = 1):
        """Create a flareplot using the systems data.

        :param cutoff: float
                datapoints with a frequency below the cutoff threshold are not used
                for plotting.
        :param cmap: str
                colormap to use for the connecting lines. Standard matplotlib colormaps
                are supported.
            :default: "Greys"
                which is a colormap from white to black.
        :param cbar: bool
                wether a cbar should be added to the plot or not.
            :default: True
                which plots a colorbar
        :param one_letter: bool
                wether the default threeletter code labels should be converted to
                oneletter code.
            :default: True
                threeletter code is converted to oneletter code.
        :param linewidth: int|float
                base thickness of the connecting lines. The thickness gets also
                scaled with the frequency.
            :default: 6
        :param fontsize: int|float
                size of the labels font.
            :default: 10
        :param shift: int|float
                is added to the angles, in degrees.
            :default: 0
        :param label_offset: int|float
                amount the labels are offset from their baseposition. Should
                be chosen in a way, so the labels don't overlap with the plot.
            :default: 0.2
        :param tick_length: int|float
                length of the ticks.
            :default: 3
        :param tick_width: int|float
                width of the ticks.
            :default: 1
        """

        defaults = _kw_handler( {
            'linewidth' : 6,
            'fontsize' : 10,
            'shift' : 0,
            'label_offset' : 0.2,
            'tick_length' : 3,
            'tick_width' : 1
            }, kwargs)


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
        ax.set_theta_zero_location('E', offset=defaults['shift'])

        cmap = mpl.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=cutoff, vmax=1)
        if cbar:
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), pad=0.2, ax=ax)

        # plot all connecting lines
        # for more information about the coordinates function bezier and the
        # coordinate classes Polar and Cartesian check out the coordinates.py file.
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
                lw=defaults['linewidth'] * cut.loc[i, 'contact_frequency']**2,
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
        angles[np.cos(np.deg2rad(angles+defaults['shift'])) < 0] += 180
        labels = []
        for label, angle in zip(ax.get_xticklabels(), angles):
            x,y = label.get_position()
            lab = ax.text(x, y-defaults['label_offset'],
                          label.get_text(),
                          transform=label.get_transform(),
                          fontsize=defaults['fontsize'],
                          ha = label.get_ha(), va=label.get_va(),
                          )
            lab.set_rotation(angle+defaults['shift'])
            labels.append(lab)
        ax.set_xticklabels([])

        #ticks for the edge of the circle needs to be plotted extra too
        if defaults['tick_length']:
            for t in np.deg2rad(np.linspace(0, 360, len(all_unique_labels), endpoint=False)):
                ax.plot([t, t], [1,1-defaults['tick_length']/100],
                        lw=defaults['tick_width'], color='k')

        return fig

    @property
    def df(self):
        """:instance_attribute: containing the underlying data.

        :returns: pandas.DataFrame
        """
        return self._df

    @property
    def interaction(self):
        """:instance_attribtue: interaction the data is about.

        :returns: str
        """
        return self._interaction

    @property
    def filename(self):
        """:instance_attribute: path to the file containing the data.

        :returns: str
        """
        return self._filename

    @property
    def selection(self):
        """:instance_attribute: selection applied to the original data.

        :returns: tuple[tuple[int, int], tuple[int,int]]
        """
        return self._selection

    def info(self):
        """Short description of the `Frequency`:instance:."""
        print(f"System:\t\t\t{self.name}\n\
                Interactiontype:\t{self.interaction}\n\
                Frequencies from:\t{self.filename}")

    def compatible(self, other):
        """small check for compatibility
        :param other: Frequency
                `Frequency`:instance: to compare the current object to.

        :returns: bool
                True if the `Frequency.interaction`:instance_attribute: and the
                `Frequency.selection`:instance_attribute: are the same else False.
        """

        return self.interaction == other.interaction and self.selection == other.selection

    def __eq__(self, other):
        """check for equality.

        :param other: Frequency
                `Frequency`:instance: to compare the current object to.

        :returns: bool
                True if the `Frequency.interaction`:instance_attribute:, the
                `Frequency.selection`:instance_attribute: and
                `Frequency.df.values`:instance_attribute: are the same else False.
        """

        return self.interaction == other.interaction and \
                self.selection == other.selection and \
                np.equal(self.df.values, other.df.values)
                #self.df.values is other.df.values

    def __repr__(self):
        "returns the `pandas.DataFrame.__repr__()` of the `Frequency.df`:instance_attribute:."
        return f"{self.df.__repr__()}"


def heatmap(data: pd.DataFrame, *,
            title: Optional[str] = None,
            y_size: float = 4, cmap: str = 'Greens', **kwargs):
    """Function to plot the heatmap of interactions.

    The plotting is done using `seaborn.heatmap`. For further more thorough documentation check \
their website. Below some possible keywords are elaborated. All keywords that are compatible \
with `seaborn.heatmap` can be used.

    :param data: pandas.DataFrame
            data of merged DataFrame to be plotted.
    :param title: Optional[str]
            title of the figure.
        :default: None
    :param y_size: int|float
            height of a single cell.
        :default: 4
    :param cmap: str
            colormap to use for the connecting lines. Standard matplotlib colormaps
            are supported.
        :default: "Greens"
            colormap from white to green.
    :param cbar: bool
            to display the colorbar belonging to the figure.
        :default: True
    :param vmin: float
            low point for mapping the colors of :param cmap: to the cells values.
        :default: 0
    :param vmax: float
            high point for mapping the colors of :param cmap: to the cells values.
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

    kwargs = _kw_handler({
        'cbar' : True,
        'vmin' : 0,
        'vmax' : 1,
        'annot' : False,
        'linewidths' : 0.5,
        'linecolor' : 'black',
        }, kwargs, error=False)

    if data.empty:
        raise pd.errors.EmptyDataError("The DataFrame is empty")

    fig = plt.figure()
    ax = fig.add_subplot()

    sns.heatmap(data,
                yticklabels = 1,
                ax = ax,
                cmap = cmap,
                **kwargs)

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

def fingerprint(data: pd.DataFrame,*,
                title: Optional[str] = None,
                y_size: float = 4, cmap: str = 'Greens', **kwargs):
    """Function to plot the fingerprint of interactions.

    The plotting is done using `seaborn.clustermap`. For further more thorough documentation check \
their website. Below some possible keywords are elaborated. All keywords that are compatible \
with `seaborn.clustermap` can be used.

    :param data: pandas.DataFrame
            data of merged DataFrame to be plotted.
    :param title: Optional[str]
            title of the figure.
        :default: None
    :param y_size: int|float
            height of a single cell.
        :default: 4
    :param cmap: str
            colormap to use for the connecting lines. Standard matplotlib colormaps
            are supported.
        :default: "Greens"
            colormap from white to green.
    :param cbar_pos: tuple[float, float, float, float]
            tuple of (left, bottom, width, heigth), controls the position of the
            colorbar in the figure
        :default: (1.5, 0.05, 0.1, 0.38)
    :param vmin: float
            low point for mapping the colors of :param cmap: to the cells values.
        :default: 0
    :param vmax: float
            high point for mapping the colors of :param cmap: to the cells values.
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
            proportion of the figure size devoted to the dendrogram lines
        :default: (0.2, 0)
    :param col_cluster: bool
            controls the clustering of the columns.
        :default: False
            disables the clustering of the columns.

    :returns: matplotlib.figure
            containing the fingerprint

    :raises: pandas.errors.EmptyDataError
            when the merged DataFrame is empty.
    """
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

    if data.empty:
        raise pd.errors.EmptyDataError("The DataFrame is empty")

    finger = sns.clustermap(
            data,
            yticklabels = 1,
            cmap = cmap,
            **kwargs)

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
    import sys
    sys.exit()
