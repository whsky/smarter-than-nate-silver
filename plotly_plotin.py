import pandas as pd
import numpy as np
import seaborn as sns
import bokeh as bk
from bokeh import mpl
from bokeh.plotting import output_file, show
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
from plotly.graph_objs import Scatter, Figure, Layout
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from colour import Color

#############
#Color theme:
#############
ylw = 'rgb(255, 210, 59)'#ffd23b
org = 'rgb(234, 104, 52)'#ea6834
blk = 'rgb(48, 48, 48)'#303030
lt_org = 'rgb(235, 139, 100)'#eb8b64
taupe = 'rgb(122, 108, 91)'#7a6c5b

plt.style.use('ggplot')
col = [np.sign(y_pred[x]) != np.sign(y2_tourney[x]) for x in range(len(y_pred))]
xrng = np.arange(-40,40,0.1)
yrng=xrng

matchups = []
dates = df_tourney['DATE'].unique()
for date in dates:
    df_date = df_tourney[df_tourney['DATE']==date]
    matchups.extend(df_date['MATCH'].unique())
# Set up hover text - best practice is to make a DF and convert to string
#    inplace where building hovertext...
pred_txt = pd.DataFrame()
pred_txt['txt'] = np.round(y_pred).astype(int)
actual_txt = pd.DataFrame()
actual_txt['txt'] = [int(x) for x in y_tourney]
hover_txt = pd.DataFrame()
hover_txt['txt'] = '<b>' + 'Actual: ' + '</b>'\
        + actual_txt['txt'].astype(str) + '<br>' \
        + '<b>' + 'Predct: ' + '</b>' \
        + pred_txt['txt'].astype(str) +'<br>'\
        + matchups


data = [ dict(
        type='scatter',
        mode='lines',
        x = xrng,
        y = yrng,
        line = dict(color=ylw, dash=20, width=5),
        opacity=0.7,
        text = "Hello!",
        hoverinfo='none'
        ),
        dict(
        type='scatter',
        mode='markers',
        x = np.round(y_pred).astype(int),
        y = y2_tourney,
        marker = dict(symbol='circle',color=org, size=12,
            line = dict(width=3, color=blk), opacity=0.5),
        text = hover_txt['txt'],
        hoverinfo='text',
        textfont = dict(family='arial')
        )]

layout = dict(title='Predicted vs. Actual Pt. Spreads (2016)')
fig = Figure(data=data, layout=layout)
plotly.offline.plot(fig)
