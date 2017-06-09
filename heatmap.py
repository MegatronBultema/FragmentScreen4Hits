from math import pi
import pandas as pd
'''
from bokeh.io import show
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
)
from bokeh.plotting import figure
'''
#should double check about scaling this
scaler = preprocessing.StandardScaler().fit(W)
scale_a = scaler.transform(W)


import matplotlib.pyplot as plt
import numpy as np
plt.close()
#a = np.random.random((16, 16))
plt.imshow(W,aspect='auto',cmap='hot', interpolation='nearest')
plt.savefig('heatmap_W3_noscale.png')
