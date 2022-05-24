#! /usr/bin/env python3

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, Tuple, List
import warnings

#for flare
import coordinates as coord


class Frequency:
    """Class to contain information about a loaded frequency tsv-file. The class
    also contains helpful methods to select residues.

    The class should be initialized with Frequency.from_tsv().

    """

    def __init__(self, df: pd.DataFrame, filename:str,  interaction: str, 
                 name: Optional[str] = None, 
                 selection: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None):
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("DataFrame expected.")
        self._df = df
        
        self._interaction = interaction
        self._filename = filename
        
        self._selection = selection
        
        if not name:
            self.name = interaction
        else:
            self.name = name
            
    @classmethod
    def from_tsv(cls, filename: str, interaction: str, 
                 name: Optional[str] = None):
        """Classmethod to initialize a new Frequency object.

    Expects:
        filename (str) -> path to the tsv-file generated from getcontacts.
       
        interaction (str) -> name of the calculated interaction. This is not
            used for any calculation, so any name can be provided. It is mainly
            used for displaying information.

        name (str) -> Optional name of the Frequency Object, for example the
            pdb code. Just additional information, if no name is provided, the
            provided interaction is used for name.
    
     Returns:
        Frequency-Object
        """
        try:
            df = pd.read_csv(filename, sep=r'\t|:', skiprows=2, header=None, engine='python')
        except pd.errors.EmptyDataError:
            warnings.warn(f"{filename} is empty.")
            df = pd.DataFrame(columns=range(7))
           
        df = df.drop(labels=[0,3], axis=1)
        df.columns = ['res 1', 'number 1', 'res 2', 'number 2', 'contact_frequency']

        return cls(df, filename, interaction, name)
        
    def select(self, residues:Tuple[int, int], more: Optional[Tuple[int, int]]=None):
        """Method for the selection of specific residues.
        
        Expects:
            residues (Int, Int) -> range of residues that is selected.

            more Optional[(Int, Int)] -> range of further residues to be
            selected.

        Returns:
            New Frequency-Object with selected residues in the df attribute.
        """
        if more:
            df = self.df[self.df['number 1'].between(*residues) & self.df['number 2'].between(*more) | \
                         self.df['number 2'].between(*residues) & self.df['number 1'].between(*more)]
        else:
            df = self.df[self.df['number 1'].between(*residues) | self.df['number 2'].between(*residues)]
            
        return self.__class__(df, self.filename, self.interaction, self.name, (residues, more))
    
    def flare(self, cutoff = 0.6, cmap = 'Greys', cbar = False, linewidth = 6,
            shift = 0, label_offset = 0.2, one_letter = True,
              fontsize = 10, tick_length = 3, tick_width = 1):
        """Experimental Flare Plot"""

        # apply cutoff to df
        cut = self.df[self.df['contact_frequency'] > cutoff].reset_index(drop=True)
        
        # get all unique labels for the labels on the flare plot
        l = cut[['number 1', 'res 1', 'number 2', 'res 2']].values.reshape(cut.shape[0]*2, 2).tolist()
        l = sorted(set(zip(*zip(*l))))
        # create a point for each label between which the connecting bezier
        # curve is drawn
        points = {num: coord.Polar(r=1, theta= angle) 
                  for (num,_), angle in zip(l, 
                                            np.linspace(0, 2*np.pi, len(l), endpoint=False))}

        fig = plt.figure()
        ax = fig.add_subplot(polar=True, )
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
        ax.set_thetagrids(np.linspace(0,360,len(l), endpoint=False), 
                          [f"{three2one[res] if one_letter else res} {num:03}" for num, res in l])

        ax.grid(False)
        ax.set_rlim([0,1])
        ax.set_rticks([])
        
        # the following code is needed to have the labels at the edge of the
        # circle at the right angle

        plt.gcf().canvas.draw()
        angles = np.linspace(0,360, len(l), endpoint=False)

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
            for t in np.deg2rad(np.linspace(0, 360, len(l), endpoint=False)): 
                ax.plot([t, t], [1,1-tick_length/100], lw=tick_width, color="k")
                
        return fig

    @property
    def df(self):
        return self._df
    
    @property
    def interaction(self):
        return self._interaction
    
    @property
    def filename(self):
        return self._filename
    
    @property
    def selection(self):
        return self._selection
    
    def info(self):
        print(f"System:\t\t\t{self.name}\nInteractiontype:\t{self.interaction}\nFrequencies from:\t{self.filename}")
    
    def compatible(self, other):
        return self.interaction == other.interaction and self.selection == other.selection
    
    def __eq__(self, other):
        return self.interaction == other.interaction and \
                self.selection == other.selection and \
                self.df.values is other.df.values
    
    def __repr__(self):
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
        resize(data.T.shape, y_size)
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
        
    fingerprint = sns.clustermap(
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
    fingerprint.fig.set_size_inches(
        resize(data.T.shape, y_size)
    )
    
    #fingerprint.fig.axes[2].set_yticklabels([f"{a}-{b}" for a, b in data.index])
    fingerprint.ax_heatmap.yaxis.set_label_text(None)
    
    if title:
        fingerprint.fig.suptitle(title.upper())
    #fingerprint.fig.supylabel('')

    plt.setp(fingerprint.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.setp(fingerprint.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    
    return fingerprint

def resize(ls, to=5):
    smallest = min(ls)
    return [i/smallest*to for i in ls]


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
        'ILE' :
        'I', 
        'PRO': 'P', 
        'THR':'T', 
        'PHE':'F', 
        'GLY':'G',        
        'LEU':'L', 
        'ARG':'R',
        'TRP':'W', 
        'ALA':'A',   
        'VAL': 'V',        
        'TYR': 'Y', 
        'MET' : 'M' 
    }
