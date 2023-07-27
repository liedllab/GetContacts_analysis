#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

import gc_analysis


PARAMS = {'font.size': 20}


def make_flareplot(datafile, output):
    hbond = pd.read_csv(datafile, sep=r'\t', index_col=False)
    
    sorting = ["D1", "A1", "D2", "A2", "D3", "A3", "D4"]
    test = {val: number for number, val in enumerate(sorting)}
    hbond = hbond.replace(test)
    
    hbond.nhb=hbond.nhb/500000.0
    hbond = hbond.set_index(['acceptor', 'donor'])['nhb']

    flareplot = gc_analysis.flareplot(hbond)
    flareplot.replace_labels(test.keys())
    flareplot.fig.savefig(output)

if __name__ == '__main__':
    with plt.rc_context(PARAMS):
        make_flareplot('example.dat', 'example.png')
