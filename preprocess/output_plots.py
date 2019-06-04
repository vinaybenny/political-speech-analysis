# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:07:43 2019
@author: vinay.benny

Credits: Contains a few snippets of code from Andrew Maguire's blog: 
    https://medium.com/@andrewm4894/bokeh-battles-part-1-multi-line-plots-311109992fdc
"""

from bokeh.plotting import figure, output_file, show
from bokeh.models.widgets import Select, MultiSelect, CheckboxGroup
from bokeh.models import ColumnDataSource, Legend, HoverTool
from bokeh.io import curdoc
from bokeh.layouts import row, column, layout, widgetbox, gridplot
from bokeh.models.callbacks import CustomJS
from bokeh.palettes import Category10

import pandas as pd
import numpy as np
import random 
import string
import itertools


def load_data(df, topic_num):
    topic = df[(df["topic"] == topic_num)]
    column_order = ['topic', 'time']
    column_order.extend(topic.groupby('word').sum().sort_values(by=['prob'], ascending=[False]).index.to_list())
    output = DataFrame(pivot_table(topic, values='prob', index=['topic', 'time'], columns=['word'] ).to_records()).fillna(0)
    output = output.reindex(column_order, axis=1)
    return output    
    

def word_list(df, topic_num):
    return list(df[(df["topic"] == topic_num)].word.unique())

def convert_df(df):
    source = {}   
    for column in df.columns:
        source[column] = [df[column].values]
    return source

def color_gen():
    yield from itertools.cycle(Category10[10])

def make_data(df, topic):
    ''' Function to make some data and put it in a df
    '''
    df = load_data(df, topic)
    df['time'] = df['time'].apply(lambda x: x.to_timestamp().date())
    df = df.set_index('time')
    df = df.drop(columns = ['topic'], axis = 1)
    df.index.names = ['index']
    return df


def plot_lines_multi(df,lw=2,pw=700,ph=400,t_str="hover,save,pan,box_zoom,reset,wheel_zoom",t_loc='above'):
    '''...
    '''
    
    source = ColumnDataSource(df)
    col_names = df.columns.tolist()
    p = figure(x_axis_type="datetime",plot_width=pw, plot_height=ph,toolbar_location=t_loc, tools=t_str)
    p_dict = dict()
    for col, c, col_name in zip(df.columns,color,col_names):
        p_dict[col_name] = p.line('index', col, source=source, color=c,line_width=lw)
        p_dict[col_name].visible = False
        p.add_tools(HoverTool(
            renderers=[p_dict[col_name]],
            tooltips=[('datetime','@index{%Y-%m-%d}'),(col, f'@{col}')],
            formatters={'index': 'datetime'}
        ))
    legend = Legend(items=[(x, [p_dict[x]]) for x in p_dict])
    p.add_layout(legend,'right')
    p.legend.click_policy="hide"
    return p

if __name__== "__main__":
    
    layout=[]
    for topic_num in topic_word_prob.topic.unique():        
        words = word_list(topic_word_prob, topic_num)
        df = make_data(topic_word_prob, topic_num)
        color = color_gen()
        layout.append(plot_lines_multi(df._get_numeric_data()))
    
    grid = gridplot(layout, ncols = 2)
    show(grid)
    output_file('./output/topic_word_trends.html')
    


