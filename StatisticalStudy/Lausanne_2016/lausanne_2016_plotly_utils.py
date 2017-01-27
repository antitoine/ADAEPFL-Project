# ----------------------------------------------------------------------------------------------------------
# Imports

import datetime
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from collections import Counter
from collections import OrderedDict
from itertools import count
import plotly
from plotly import tools
import plotly.graph_objs as go
from plotly.graph_objs import Annotation, Annotations
import sys
sys.path.append('..')
import study_utils
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------
# Constants

# Information about Canton of Vaud can be found on official government website
# https://www.bfs.admin.ch/bfs/fr/home/statistiques/population.assetdetail.1500543.html.
TOTAL_RESIDENT_VAUD = 778365
TOTAL_RESIDENT_MALE = 381864
TOTAL_RESIDENT_FEMALE = 396501

YEAR_CATEGORIES = ['15-25 years', '26-30 years', '31-35 years', '36-40 years', '41-45 years', '46-50 years', '51-55 years', '56-60 years', '61-65 years', '65+ years']
FEMALE_COLOR = '#f442e8'
MALE_COLOR = '#4286f4'
ALL_GENDERS_COLOR = '#f4df42'
INDIVIDUAL_COLOR = '#42f4c5'
TEAM_COLOR = '#a142f4'
KM_10_COLOR = '#f99740'
KM_21_COLOR = '#40aff9'
KM_42_COLOR = '#f94040'

# ----------------------------------------------------------------------------------------------------------
# Functions

def plot_gender_distributions(df):
    '''
    This functions displays graph representing the gender distribution of Canton of Vaud and Lausanne Marathon 2016 for comparison.

    Parameters
        - df: DataFrame containing information on runners for Lausanne Marathon 2016

    Return
        - figure: Plotly figure
    '''

    # Building of DataFrame for ploting
    CANTON_VAUD = 'Canton of Vaud'
    LAUSANNE_MARATHON = 'Lausanne Marathon'
    total_runners = len(df)
    total_runners_male = len(df[df['sex'] == 'male'])
    total_runners_female = len(df[df['sex'] == 'female'])
    vaud_information_population = pd.Series({ 'male': TOTAL_RESIDENT_MALE/TOTAL_RESIDENT_VAUD * 100, 'female': TOTAL_RESIDENT_FEMALE/TOTAL_RESIDENT_VAUD * 100 }) 
    marathon_information_runner = pd.Series({ 'male': total_runners_male/total_runners * 100, 'female': total_runners_female/total_runners * 100 }) 
    information_population = pd.DataFrame({ CANTON_VAUD: vaud_information_population, LAUSANNE_MARATHON: marathon_information_runner })
    information_population.sort_index(axis=0, level=None, ascending=False, inplace=True)

    text_vaud = ['<b>' + CANTON_VAUD + '</b><br>' + str(TOTAL_RESIDENT_MALE) + ' residents', '<b>' + CANTON_VAUD + '</b><br>' + str(TOTAL_RESIDENT_FEMALE) + ' residents']
    text_marathon = ['<b>' + LAUSANNE_MARATHON + '</b><br>' + str(total_runners_male) + ' runners', '<b>' + LAUSANNE_MARATHON + '</b><br>' + str(total_runners_female) + ' runners']
    vaud_trace = go.Bar(x=information_population.index.values, y=information_population[CANTON_VAUD], name=CANTON_VAUD, hoverinfo='text', text=text_vaud)
    marathon_trace = go.Bar(x=information_population.index.values, y=information_population[LAUSANNE_MARATHON], name=LAUSANNE_MARATHON, hoverinfo='text', text=text_marathon)
    data = [vaud_trace, marathon_trace]

    annotations = [Annotation(y=1.1, text='Total residents: ' + str(TOTAL_RESIDENT_VAUD) + ' | Total runners: ' + str(total_runners), xref='paper', yref='paper', showarrow=False)]
    figure = study_utils.create_plotly_legends_and_layout(data, title='Gender distribution Lausanne Marathon vs Canton of Vaud', x_name='Gender', y_name='Percentage (%)', barmode='group', annotations=annotations)
    plotly.offline.iplot(figure)
    return figure


def plot_gender_distribution_according_to_running_type(df, runnings=None, sex_column_name='sex'):
    '''
    This function displays the gender distribution for the different runnings.

    Parameters
        - df: DataFrame containing data
        - runnings: Dict containing name of column containing runnings (key: column_name) and set of runnings (key: values, value: dict() with following keys: name, color)
                    By default, None. If None, default values will be set by function.
        - sex_column_name: Name of column containing gender of participants (by default, 'sex')

    Return
        - figure: Plotly figure
    '''

    if not runnings:
        runnings = {'column_name': 'distance (km)', 'values': OrderedDict([(10, {'name': '10 km', 'color': KM_10_COLOR}), (21, {'name': 'Semi-marathon', 'color': KM_21_COLOR}), (42, {'name': 'Marathon', 'color': KM_42_COLOR})])}

    data = []

    annotations_texts = []

    for key, attributes in runnings['values'].items():
        filtered_df = df[df[runnings['column_name']] == key]
        nb_runners_running = len(filtered_df)
        x_values, y_values, texts = [[] for i in range(3)]
        for sex in filtered_df[sex_column_name].unique():
            x_values.append(sex)
            nb_runners = len(filtered_df[filtered_df[sex_column_name] == sex])
            y_values.append(nb_runners)
            texts.append('<b>' + attributes['name'] + '</b><br>' + sex.capitalize() + ' runners: ' + str(nb_runners) + ' (' + '{:.1f}%'.format(nb_runners*100/nb_runners_running) + ')')
        annotations_texts.append(attributes['name'] + ': ' + str(nb_runners_running) + ' runners')
        data.append(go.Bar(x=x_values, y=y_values, name=attributes['name'], text=texts, hoverinfo='text', marker={'color': attributes['color']}))

    annotations = [Annotation(y=1.1, x=0, text=' | '.join(annotations_texts), xref='paper', yref='paper', showarrow=False)]
    figure = study_utils.create_plotly_legends_and_layout(data, title='Gender distribution by distance', x_name='Gender', y_name='Number of runners', barmode='group', annotations=annotations)
    plotly.offline.iplot(figure)
    return figure


def plot_distribution_between_types_of_participants(df, type_column_name='type'):
    '''
    This functions displays the distribution of runners between types of participants.

    Parameters
        - df: DataFrame containing information about runners
        - type_column_name: Name of column containing type of runners (by default, 'type')

    Return
        - figure: Plotly figure
    '''

    x_values, y_values, texts = [[] for i in range(3)]
    nb_total_participants = len(df)
    for type_participants in df[type_column_name].unique():
        x_values.append(type_participants)
        nb_participants = len(df[df[type_column_name] == type_participants])
        y_values.append(nb_participants)
        texts.append('<b>' + type_participants.capitalize() + '</b><br>Number of participants: ' + str(nb_participants) + ' (' + '{:.1f}%'.format(nb_participants*100/nb_total_participants) + ')')
        
    bar = go.Bar(x=x_values, y=y_values, name=type_participants, text=texts, hoverinfo='text')

    figure = study_utils.create_plotly_legends_and_layout([bar], title='Distribution by type of runners', x_name='Type of runner', y_name='Number of runners', barmode='group')
    plotly.offline.iplot(figure)
    return figure


def plot_age_distribution(df, age_column_name='age', sex_column_name='sex'):
    '''
    This function displays the distribution of runners according to their age.

    Parameters:
        - df: DataFrame containing information about runners
        - age_column_name: Name of column containing age of runners
    '''

    # Calculation of age distribution statistics by gender
    statistics = []
    all_genders = ['all']
    all_genders.extend(df[sex_column_name].unique())
    for sex in all_genders:
        if sex == 'all':
            ages = df[age_column_name]
        else:
            ages = df[df[sex_column_name] == sex][age_column_name]
        statistics.append('<b>Mean age of ' + sex + ' runners: ' + str(round(np.mean(ages), 2)) + ' (STD: ' + str(round(np.std(ages), 2)) + ')</b>')

    data = [go.Histogram(x=df[age_column_name])]
    annotations = [Annotation(y=1, x=1, text='<br>'.join(statistics), xref='paper', yref='paper', showarrow=False)]
    shapes = [{'type': 'line', 'yref': 'paper', 'x0': np.mean(df[age_column_name]), 'y0': 0, 'x1': np.mean(df[age_column_name]), 'y1': 1, 'line': {'color': '#f44242', 'width': 2, 'dash': 'dash'}}]
    figure = study_utils.create_plotly_legends_and_layout(data, title='Age distribution of runners', x_name='Age', y_name='Number of runners', barmode='group', bargap=0.25, annotations=annotations, shapes=shapes)
    plotly.offline.iplot(figure)
    return figure


def plot_distribution_age_distance(data, runnings=None, title='Distribution of runners by age categories', age_column_name='age', sex_column_name='sex'):
    '''
    This function plots, for each running, the distribution of ages of runners based on the genders of participants.

    Parameters
        - data: DataFrame to use during generation of the distribution
        - runnings: Dict containing name of column containing runnings (key: column_name) and set of runnings (key: values, value: dict() with following keys: name, color)
                    By default, None. If None, default values will be set by function.
        - title: Title of the graph (by default, 'Distribution of runners by age categories')
        - age_column_name: Name of the column containing age of participants('age' or 'age category', by default, 'age')
        - sex_column_name: Name of the column containing sex of participants (by default, 'sex')

    Return
        - figure: Plotly figure
    '''

    if not runnings:
        runnings = {'column_name': 'distance (km)', 'values': OrderedDict([(10, {'name': '10 km', 'color': KM_10_COLOR, 'position': 1}), (21, {'name': 'Semi-marathon', 'color': KM_21_COLOR, 'position': 2}), (42, {'name': 'Marathon', 'color': KM_42_COLOR, 'position': 3})])}
    colors = {'female': FEMALE_COLOR, 'male': MALE_COLOR, 'all': ALL_GENDERS_COLOR}
    statistics = {}
    with study_utils.ignore_stdout():
        figure = tools.make_subplots(rows=3, cols=1, subplot_titles=([attributes['name'] for km, attributes in runnings['values'].items()]))

    for key, attributes in runnings['values'].items():
        filtered_df = data[data[runnings['column_name']] == key]
        statistics[attributes['name']] = 'Mean age: ' + str(round(np.mean(filtered_df[age_column_name]), 2)) + ' years (SD: ' + str(round(np.std(filtered_df[age_column_name]), 2)) + ')'
        for sex in np.concatenate((filtered_df[sex_column_name].unique(), ['all']), axis=0):
            if sex == 'all':
                x = filtered_df[age_column_name]
            else:
                x = filtered_df[filtered_df[sex_column_name] == sex][age_column_name]
            nbinsx = ((np.max(x)-np.min(x))+1) if (age_column_name == 'age') else len(x)
            figure.append_trace(go.Histogram(nbinsx=nbinsx, x=x, name=sex.capitalize() + ' runners', legendgroup=sex, showlegend=(attributes['position'] == 1), marker={'color': colors[sex]}, opacity=0.75), attributes['position'], 1)
    
    # Format of axes and layout
    if age_column_name == 'age category':
        for axis, attributes in {k: v for k, v in figure['layout'].items() if 'xaxis' in k}.items():
            figure['layout'][axis].update(categoryorder='array', categoryarray=YEAR_CATEGORIES)
    figure.layout.xaxis3.update(title='Age of participants')
    figure.layout.yaxis2.update(title='Number of participants')
    figure.layout.update(title=title, barmode='stack', bargroupgap=0.1, bargap=0, margin=go.Margin(t=100, b=50, l=50, r=50))

    # Add of statistics
    # Trick: We use position of subtitles annotations to create the ones related to statistics
    annotations_statistics = []
    for annotation in figure['layout']['annotations']:
        annotations_statistics.append(Annotation(y=annotation['y'], x=1, text=statistics[annotation['text']], xref='paper', yref='paper', yanchor='bottom', showarrow=False))
    figure['layout']['annotations'].extend(annotations_statistics)
    
    plotly.offline.iplot(figure)
    return figure


def plot_time_distribution_by_age(data, runnings=None, age_column_name='age'):
    '''
    This function plots the distribution of time for all ages regarding participants of a Lausanne Marathon.
    3 subplots are displayed per rows.

    Parameters
        - data: DataFrame containing all the information of a Lausanne Marathon
        - runnings: Dict containing name of column containing runnings (key: column_name) and set of runnings (key: values, value: dict() with following keys: name, color)
                    By default, None. If None, default values will be set by function.
        - age_column_name: Name of the column containing age of participants('age' or 'age category', by default, 'age')
    '''
    
    if not runnings:
        runnings = {'column_name': 'distance (km)', 'values': {10: {'name': '10 km', 'color': KM_10_COLOR}, 21: {'name': 'Semi-marathon', 'color': KM_21_COLOR}, 42: {'name': 'Marathon', 'color': KM_42_COLOR}}}
    groups = data.groupby(age_column_name)

    figures = {}
    options = {'x_name': 'Performance time', 'y_name': 'Number of runners', 'x_type': 'date', 'x_format': '%H:%M:%S', 'barmode': 'overlay', 'bargroupgap': 0.1}
    

    for name, group in groups:
        histograms, statistics = [], []
        for km, attributes_running in runnings['values'].items():
            x = group[group[runnings['column_name']] == km]['time']
            statistics.append(attributes_running['name'] + ': ' + str(len(x)) + ' runners')
            histograms.append(go.Histogram(x=x, xbins={'start': np.min(group['time']), 'end': np.max(group['time']), 'size': 5*60000}, name=attributes_running['name'], marker={'color': attributes_running['color']}, opacity=0.5))
        statistics.append('Total: ' + str(len(group)) + ' runners')
        annotations = [Annotation(y=1.1, x=0.9, text=' | '.join(statistics), xref='paper', yref='paper', showarrow=False)]
        figure = study_utils.create_plotly_legends_and_layout(data=histograms, title='Time distribution (' + name + ')', **options, annotations=annotations)
        figures[name] = figure
    return figures


def generate_performance_by_age_and_age_category(data, runnings=None, age_column_name = 'age', age_category_column_name='age category', sex_column_name='sex'):
    '''
    This function generates figures for each running. It displays time distribution according to age and age category.
    Final Dict has the following pattern:
    {
        <running_1>: <Plotly figure>
        [, <running_2>: <Plotly figure>
        , ...]
    }

    Parameters
        - data: DataFrame containing all the information of Lausanne Marathon 2016
        - runnings: Dict containing name of column containing runnings (key: column_name) and set of runnings (key: values, value: dict() with following keys: name)
                    By default, None. If None, default values will be set by function.
        - age_column_name: Name of the column containing age of participants(by default, 'age')
        - age_category_column_name: Name of the column containing age category of participants(by default, 'age category')
        - sex_column_name: Name of the column containing gender of participants (by default, 'sex')

    Return
        - figures: Dict containing all time distribution figures
    '''

    # We create final dict and we set attributes and runnings (if not given by user)
    figures = {}

    attributes = {'colors': {'female': FEMALE_COLOR, 'male': MALE_COLOR, 'all': ALL_GENDERS_COLOR},
                'names': {'female': 'Female runners', 'male': 'Male runners', 'all': 'All runners'},
                'visibility': {'female': 'legendonly', 'male': 'legendonly', 'all': True}
                }
    
    if not runnings:
        runnings = {'column_name': 'distance (km)', 'values': {10: {'name': '10 km'}, 21: {'name': 'Semi-marathon'}, 42: {'name': 'Marathon'}}}

    # Loop over runnings
    for km, attributes_running in runnings['values'].items():
        filtered_df = data[data[runnings['column_name']] == km]
        
        # Creation of Plotly figure containing subplots (we ignore outputs here)
        with study_utils.ignore_stdout():
            figure = tools.make_subplots(rows=2, cols=1, vertical_spacing=0.1)

        # Consideration of ages and age categories
        for column_name in [age_column_name, age_category_column_name]:
            # Creation of boxplots for female and male runners
            boxplots = study_utils.create_plotly_boxplots(data=filtered_df, x=column_name, y='time', hue=sex_column_name, hue_names=attributes['names'], colors=attributes['colors'], visibility=attributes['visibility'], use_hue_names=False, use_legend_group=True, show_legend=(column_name == 'age'))
            # We add boxplots for all runners (without consideration of sex)
            boxplots.append(go.Box(y=filtered_df['time'], x=data[column_name], name=attributes['names']['all'], marker={'color': attributes['colors']['all']}, visible=attributes['visibility']['all'], legendgroup=attributes['names']['all'], showlegend=(column_name == 'age')))
            # We add each generated boxplot in the correct subplot
            for boxplot in boxplots:
                figure.append_trace(boxplot, 1 if column_name == 'age' else 2, 1)
        
        # Format of y and x axes and modification of layout
        for axis, _ in {k: v for k, v in figure['layout'].items() if 'yaxis' in k}.items():
            figure['layout'][axis].update(title='Performance time', type='date', tickformat='%H:%M:%S')
        figure['layout']['xaxis1'].update(title='Age', dtick=5)
        figure['layout']['xaxis2'].update(title='Age category', categoryorder='array', categoryarray=YEAR_CATEGORIES)
        figure['layout'].update(title='Distribution of time performance (' + attributes_running['name'] + ')', height=650, hovermode='closest')
        
        # We add newly created figure to the final dict
        figures[attributes_running['name']] = figure

    return figures


def generate_time_distribution_by_bib_numbers(data, performance_criteria):
    '''
    This function generates all BIB/performance scatters for each running of Lausanne Marathon.
    Final Dict has the following pattern:
    {
        <performance_criterion_1>: <Plotly figure>
        [, <performance_criterion_2>: <Plotly figure>
        , ...]
    }

    Parameters
        - df: DataFrame containing records about runners
        - performance_criteria: Array containing column name to use for available performance criteria (time and speed)

    Return
        - figures: Dict containing all time distribution figures
    '''


    # We define options
    runnings_names = {10: '10 km', 21: 'Semi-marathon', 42: 'Marathon'}
    colors = {10: KM_10_COLOR, 21: KM_21_COLOR, 42: KM_42_COLOR}
    default_options = {'title': 'Distribution of performance according to BIB numbers over the years', 'x_name': 'BIB numbers', 'x_format': 'f', 'hovermode':'closest'}
    time_options = {'y_name': 'Time', 'y_type': 'date', 'y_format': '%H:%M'}
    speed_options = {'y_name': 'Speed (m/s)'}
    time_options.update(default_options)
    speed_options.update(default_options)

    # We create final dict
    figures = {}

    # Loop over performance criteria (time, speed)
    for performance_criterion in performance_criteria:
        criterion = performance_criterion.lower()
        scatters = study_utils.create_plotly_scatters(data=data, x='number', y=criterion, hue='distance (km)', hue_names=runnings_names, text='name', color=colors, use_hue_names=False)
        if criterion == 'time':
            figure = study_utils.create_plotly_legends_and_layout(data=scatters, **time_options)
        else:
            figure = study_utils.create_plotly_legends_and_layout(data=scatters, **speed_options)
        figures[performance_criterion] = figure
    
    return figures


def plot_speed_distribution_by_running(data, runnings=None, title='Speed distribution by running', speed_column_name='speed (m/s)', sex_column_name='sex'):
    '''
    This function plots, for each running, the distribution of ages of runners based on the genders of participants.

    Parameters
        - data: DataFrame to use during generation of the distribution
        - runnings: Dict containing name of column containing runnings (key: column_name) and set of runnings (key: values, value: dict() with following keys: name, color)
                    By default, None. If None, default values will be set by function.
        - title: Title of the graph (by default, 'Speed distribution by running')
        - speed_column_name: Name of the column containing age of participants('age' or 'age category', by default, 'speed (m/s)')
        - sex_column_name: Name of the column containing sex of participants (by default, 'sex')

    Return
        - figure: Plotly figure

        title, x axis name, statistics, column to select, categories or not, title of xaxis, title of yaxis (not modified)
    '''

    if not runnings:
        runnings = {'column_name': 'distance (km)', 'values': OrderedDict([(10, {'name': '10 km', 'color': KM_10_COLOR, 'position': 1}), (21, {'name': 'Semi-marathon', 'color': KM_21_COLOR, 'position': 2}), (42, {'name': 'Marathon', 'color': KM_42_COLOR, 'position': 3})])}
    colors = {'female': FEMALE_COLOR, 'male': MALE_COLOR, 'all': ALL_GENDERS_COLOR}
    statistics = {}
    with study_utils.ignore_stdout():
        figure = tools.make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=([attributes['name'] for km, attributes in runnings['values'].items()]))

    for key, attributes in runnings['values'].items():
        filtered_df = data[data[runnings['column_name']] == key]
        statistics[attributes['name']] = 'Total: ' + str(len(filtered_df)) + ' runners<br>Max: ' + str(round(np.max(filtered_df[speed_column_name]), 2)) + ' m/s<br>Min: ' + str(round(np.min(filtered_df[speed_column_name]), 2)) + ' m/s<br>Median: ' + str(round(np.median(filtered_df[speed_column_name]), 2)) +' m/s | SD: ' + str(round(np.std(filtered_df[speed_column_name]), 2)) + ' m/s'
        for sex in np.concatenate((filtered_df[sex_column_name].unique(), ['all']), axis=0):
            if sex == 'all':
                x = filtered_df[speed_column_name]
            else:
                x = filtered_df[filtered_df[sex_column_name] == sex][speed_column_name]
            figure.append_trace(go.Histogram(xbins={'start': math.floor(np.min(data[speed_column_name])), 'end': math.ceil(np.max(data[speed_column_name])), 'size': 0.1}, x=x, name=sex.capitalize() + ' runners', legendgroup=sex, showlegend=(attributes['position'] == 1), marker={'color': colors[sex]}, opacity=0.75), attributes['position'], 1)
    
    # Format of axes and layout
    figure.layout.xaxis1.update(title='Speed (m/s)', tickformat='.1f')
    figure.layout.yaxis2.update(title='Number of participants')
    figure.layout.update(title=title, barmode='stack', bargroupgap=0.1, bargap=0, margin=go.Margin(t=100, b=50, l=50, r=50))

    # Add of statistics
    # Trick: We use position of subtitles annotations to create the ones related to statistics
    annotations_statistics = []
    for annotation in figure['layout']['annotations']:
        annotations_statistics.append(Annotation(y=annotation['y']-0.12, x=1, align='left', text=statistics[annotation['text']], xref='paper', yref='paper', yanchor='bottom', showarrow=False))
    figure['layout']['annotations'].extend(annotations_statistics)
    
    plotly.offline.iplot(figure)
    return figure


def generate_comparison(data, title, running_type_column_name, x_column_name, y_column_name, colors):
    '''
    This function generates comparison boxplots according to given parameters, for each running of Lausanne Marathon.

    Parameters
        - data: DataFrame containing information on runners
        - title: Title of the graph
        - running_type_column_name: Name of the column containing the runnings
        - x_column_name: Name of column to be used for x data
        - y_column_name: Name of column to be used for y data
        - colors: Dict containing colors associated to each unique value of x_column_name

    Return
        - figure: Plotly figure
    '''

    runnings = OrderedDict([(10, {'name': '10 km', 'position': 1}), (21, {'name': 'Semi-marathon', 'position': 2}), (42, {'name': 'Marathon', 'position': 3})])

    # We create final figure (outputs are ignored)
    with study_utils.ignore_stdout():
        figure = tools.make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=([attributes['name'] for km, attributes in runnings.items()]))

    # Loop over runnings of Lausanne Marathon
    for running, attributes in runnings.items():
        filtered_df = data[data[running_type_column_name] == running]

        # We loop over gender of participants
        for value in np.concatenate((['all'], filtered_df[x_column_name].unique()), axis=0):
            if value == 'all':
                filtered_df_final = filtered_df
            else:
                filtered_df_final = filtered_df[filtered_df[x_column_name] == value]

            # We append a new boxplot corresponding to a gender (or all participants) for the considered running
            figure.append_trace(go.Box(y=filtered_df_final[y_column_name], name=value.capitalize() + ' runners', marker={'color': colors[value]}, legendgroup=value, showlegend=(attributes['position'] == 1)), 1, attributes['position'])
    
    # Format of y and x axes and modification of layout
    for axis, _ in {k: v for k, v in figure['layout'].items() if 'xaxis' in k}.items():
        figure['layout'][axis].update(showticklabels=False)
    figure['layout']['yaxis1'].update(title=y_column_name.capitalize(), tickformat='.2f')
    figure['layout'].update(title=title)
    plotly.offline.iplot(figure)
    return figure


def generate_performance_comparison(data, title='Performance comparison between participants of Lausanne Marathon', running_type_column_name='distance (km)', sex_column_name='sex', speed_column_name='speed (m/s)'):
    '''
    This function plots all performance boxplots according to running type and gender of participants.

    Parameters
        - data: DataFrame containing records about runners
        - title: Title of the graph (by default, 'Performance comparison between participants of Lausanne Marathon')
        - running_type_column_name: Name of the column containing type of running (by default, 'distance (km)')
        - sex_column_name: Name of the column containing gender of participants (by default, 'sex')
        - speed_column_name: Name of the column containing speed of participants (by default, 'speed (m/s)')
    
    Return
        - figure: Plotly figure
    '''

    runnings = OrderedDict([(10, {'name': '10 km', 'position': 1}), (21, {'name': 'Semi-marathon', 'position': 2}), (42, {'name': 'Marathon', 'position': 3})])
    colors = {'female': FEMALE_COLOR, 'male': MALE_COLOR, 'all': ALL_GENDERS_COLOR}
    return generate_comparison(data=data, title=title, running_type_column_name=running_type_column_name, x_column_name=sex_column_name, y_column_name=speed_column_name, colors=colors)


def plot_runners_teams_individual_distribution_according_to_running_type(df, title='Team/individual runners composition', runnings=None, team_column_name='profile'):
    '''
    This function displays the distribution of participants according to their profiles (individual runners/runners in team) for the different runnings.

    Parameters
        - df: DataFrame containing data
        - title: Title of the graph (by default, 'Team/individual runners composition')
        - runnings: Dict containing name of column containing runnings (key: column_name) and set of runnings (key: values, value: dict() with following keys: name, color)
                    By default, None. If None, default values will be set by function.
        - team_column_name: Name of column containing type of participants (by default, 'profile')

    Return
        - figure: Plotly figure
    '''

    if not runnings:
        runnings = {'column_name': 'distance (km)', 'values': OrderedDict([(10, {'name': '10 km', 'color': KM_10_COLOR}), (21, {'name': 'Semi-marathon', 'color': KM_21_COLOR}), (42, {'name': 'Marathon', 'color': KM_42_COLOR})])}

    data = []

    annotations_texts = []

    for key, attributes in runnings['values'].items():
        filtered_df = df[df[runnings['column_name']] == key]
        nb_runners_running = len(filtered_df)
        x_values, y_values, texts = [[] for i in range(3)]

        for profile in filtered_df[team_column_name].unique():
            x_values.append(profile)
            nb_runners = len(filtered_df[filtered_df[team_column_name] == profile])
            y_values.append(nb_runners)
            texts.append('<b>' + attributes['name'] + '</b><br>' + profile.capitalize() + ' runners: ' + str(nb_runners) + ' (' + '{:.1f}%'.format(nb_runners*100/nb_runners_running) + ')')
        annotations_texts.append(attributes['name'] + ': ' + str(nb_runners_running) + ' runners')
        data.append(go.Bar(x=x_values, y=y_values, name=attributes['name'], text=texts, hoverinfo='text', marker={'color': attributes['color']}))

    annotations = [Annotation(y=1.1, x=0, text=' | '.join(annotations_texts), xref='paper', yref='paper', showarrow=False)]
    figure = study_utils.create_plotly_legends_and_layout(data, title=title, x_name='Composition', y_name='Number of runners', barmode='group', annotations=annotations)
    plotly.offline.iplot(figure)
    return figure


def generate_performance_comparison_by_profile(data, title='Performance comparison between types of participants of Lausanne Marathon', running_type_column_name='distance (km)', profile_column_name='profile', speed_column_name='speed (m/s)'):
    '''
    This function plots all performance boxplots according to running type and gender of participants.

    Parameters
        - data: DataFrame containing records about runners
        - title: Title of the graph (by default, 'Performance comparison between participants of Lausanne Marathon')
        - running_type_column_name: Name of the column containing type of running (by default, 'distance (km)')
        - profile_column_name: Name of the column containing profile of participants (by default, 'profile')
        - speed_column_name: Name of the column containing speed of participants (by default, 'speed (m/s)')
    
    Return
        - figure: Plotly figure
    '''

    colors = {'Individual': INDIVIDUAL_COLOR, 'Team-mate': TEAM_COLOR, 'all': ALL_GENDERS_COLOR}
    return generate_comparison(data=data, title=title, running_type_column_name=running_type_column_name, x_column_name=profile_column_name, y_column_name=speed_column_name, colors=colors)


def display_information_speed(data):
    '''
    This function generates median of the speed's distribution of runners in team and of individual ones.
    
    Parameters
        - data: DataFrame containing information on runners

    Return
        - string (medians)
    '''

    distances = [10, 21, 42]
    type_runners = ['Individual', 'Team-mate']
    
    for distance in distances: 
        median_distance = []
        lausanne_by_distance = data[data['distance (km)'] == distance]
        for type_runner in type_runners:      
            median_distance.append(np.median(lausanne_by_distance['speed (m/s)'][lausanne_by_distance['profile'] == type_runner]))
        
        print(str(distance) + '-km running' \
               '\nMedian for individual runners: ' + str(median_distance[0]) + ' m/s' + \
               '\nMedian for runners in team: ' + str(median_distance[1]) +' m/s' + \
               '\n*************************************************')


def plot_time_difference_distribution(df, title='Time difference with the best runner in team', time_difference_column_name='time difference team'):
    '''
    This function displays distribution representing time difference bewteen performance of team members and best performance within the team.
    
    Parameters
        - df: DataFrame containing information on runners
        - title: Title of the graph (by default, 'Time difference with the best runner in team')
        - time_difference_column_name: Name of column containing time differencies (by default, 'time_difference_column_name')

    Return
        - figure: Plotly figure
    '''

    mean_difference = np.mean(df[time_difference_column_name])
    mean_difference_dt = datetime.datetime.strptime(study_utils.convert_seconds_to_time(mean_difference), '%H:%M:%S')
    max_difference = np.max(df[time_difference_column_name])
    statistics = ['Mean difference of time: ' + str(study_utils.convert_seconds_to_time(mean_difference)),
                'Maximum difference of time: ' + str(study_utils.convert_seconds_to_time(max_difference))]

    data = df.copy()
    data[time_difference_column_name] = pd.to_datetime([study_utils.convert_seconds_to_time(t) for t in data[time_difference_column_name]], format='%H:%M:%S')
    histogram = [go.Histogram(x=data[time_difference_column_name], xbins={'start': np.min(data[time_difference_column_name]), 'end': np.max(data[time_difference_column_name]), 'size': 5*60000})]
    annotations = [Annotation(y=1, x=1, text='<br>'.join(statistics), align='right', xref='paper', yref='paper', showarrow=False)]
    shapes = [{'type': 'line', 'yref': 'paper', 'x0': mean_difference_dt, 'y0': 0, 'x1': mean_difference_dt, 'y1': 1, 'line': {'color': '#f44242', 'width': 2, 'dash': 'dash'}}]
    figure = study_utils.create_plotly_legends_and_layout(histogram, title=title, x_name='Performance gap', x_type='date', x_format='%H:%M:%S', y_name='Number of runners', barmode='group', bargap=0.1, annotations=annotations, shapes=shapes)
    plotly.offline.iplot(figure)
    return figure


######################################################
# NOT PLOTLY | FUNCTIONS FROM LAUSANNE_2016_UTILS.PY #
######################################################


def display_legend(dict_team_runner, plot):
    '''
    This function displays legend in plot following counter results.

    Parameters
        - dict_team_runner: Counter containing number of individual runners and paired runners
        - plot: plot on which information must be added
    '''

    # Creation of string containing the statistics
    pairs_runners = 'Pair runners: ' + str(dict_team_runner.get(1))   + ' runners'
    individual_runners = 'Individual runners: ' + str(dict_team_runner.get(0)) + ' runners'
    stats_str = pairs_runners + '\n' + individual_runners 

    # Add of information in the graph
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plot.text(.95, .95, stats_str, fontsize=12, transform=plot.transAxes, va='top', ha='center', bbox=props, multialignment='left')


def plot_scatter_difference_time_number(fig, data, distance, subplot_idx, annotation=[], time_mini=1000):
    '''
    This function plots the difference time between team members who have finished late compared to the best time of the team.

    Parameters
        - fig: Figure on which subplots are displayed
        - data: DataFrame containing the data relative to a given running
        - distance: number of kilometers of the considered running (10/21/42)
        - subplot_idx: Index of the subplot in the figure
        - annotation: Annotation to add in the graph (by default, no annotation)
        - time_mini: Minimal time to consider (how much runners are late compare to the first)
                    (1000 by default) 
    '''

    # select runner in teams and with the selected distance
    race_team = data[ (data['team'].notnull()) & (data['distance (km)'] == distance)]


    # Remove of times which are lower than minimal time considered
    race_team = (race_team[race_team['time difference team'] > time_mini])

    # Remove of teams with only one runner
    for team in race_team['team']: 
        race_team_selected = race_team[race_team['team'] == team]
        if len(race_team_selected['team']) == 1:
            race_team = race_team[race_team['team'] != team]


    # Map team name with team number
    team_label_encode = preprocessing.LabelEncoder()
    team_label = team_label_encode.fit_transform(race_team['team'])
    race_team['team_code'] = team_label
    
    # Computation of runners in pair
    number_runner_in_pair = race_team.apply(study_utils.compute_pair_runner, args=(race_team,60), axis=1)
    counter_pair = Counter(number_runner_in_pair)
    
    # Plotting the results
    plot = fig.add_subplot(subplot_idx)
    sns.swarmplot(x="team_code", y="time difference team", hue="sex", data=race_team, ax = plot )
    plot.set_title('Distance = '+ str(distance))
    plot.set_xlabel('')
    plot.set_ylabel('')
    plot.legend(loc='upper left')
    
    # Add annotation if any given
    if len(annotation) != 0 :
        if subplot_idx == annotation [0]:
            plot.annotate(annotation[1], annotation[2], annotation[3], arrowprops=dict(facecolor='red', shrink=0.05))
        
    # Manage of legends
    if subplot_idx != 311: 
        plot.legend_.remove()
    if subplot_idx == 312:
        plot.set_ylabel('Time difference with the best runners in the team')
    if subplot_idx == 313:
        plot.set_xlabel('Team number')
        
    plt.yticks(plot.get_yticks(), [study_utils.convert_seconds_to_time(label) for label in plot.get_yticks()])
    display_legend(counter_pair, plot)
