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

plt.style.use('ggplot')
# col = [np.sign(y_pred[x]) != np.sign(y2_tourney[x]) for x in range(len(y_pred))]
xrng = np.arange(-40,40,0.1)
yrng=xrng
# plt.plot(xrng, yrng, 'r-', alpha=0.6)
# plt.scatter(y_pred, y2_tourney, c=col, alpha=0.5)
# trace1 = Scatter(x=xrng, y=yrng)
# data = [trace1]
matchups = []
dates = df_tourney['DATE'].unique()
for date in dates:
    df_date = df_tourney[df_tourney['DATE']==date]
    matchups.extend(df_date['MATCH'].unique())
#Set up hover text - best practice is to make a DF and convert to string inplace where building hovertext...
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

# def clr_range(col_start, col_end, n, sat=1, min=0, max=1):
#     col1 = Color(col_start)
#     col1.saturation = sat
#     col2 = Color(col_end)
#     col2.saturation = sat
#     val = np.linspace(min, max, n)
#     col_range = col1.range_to(col2, n)
#     col_range = [c.rgb for c in col_range]
#     col_range = [tuple(int(c*255) for c in item) for item in col_range]
#     col_range = ['rgb'+str(c) for c in col_range]
#     return [list(c) for c in zip(val, col_range)]
#
# #Plotly choropleth map of crime stats:
# #plot needs states as two letter code not full name
# crime['code'] = [state_to_code[state] for state in crime['State']]
# crime['text'] = '<i>' + 'Murder: ' + crime['Murder'].astype(str) + '<br>' +\
#                 'Rape: ' + crime['Rape'].astype(str) + '<br>' +\
#                 'Aggr. Aslt.: ' + crime['Aggravated Assault'].astype(str) + '</i>'
#
# clr2 = clr_range('green', 'red', n=30, sat=0.75)
#
# data = [ dict(
#     type='choropleth',
#     colorscale = clr2,
#     autocolorscale = False,
#     locations = crime['code'],
#     z = crime['Murder'] + crime['Rape'] + crime['Aggravated Assault'],
#     locationmode = 'USA-states',
#     text=crime['text'],
#     marker = dict(
#         line = dict (
#             color = 'rgb(255,255,255)',
#             width = 2
#         ) ),
#     colorbar = dict(
#         title = "Crime per 100,000")
#     ) ]
#
#
# layout = dict(title='Crime Stats - Total Violent Crime Rate', geo=dict(scope='usa', \
#                 projection=dict(type='albers usa'), showlakes=True, lakecolor='rgb(255, 255, 255)'))
#
# fig = Figure(data=data, layout=layout)
# plot(fig)
