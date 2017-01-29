import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt    
import re
from dateutil import parser
import datetime as dt
from datetime import date
from scipy import stats
import warnings

import math
import scipy

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import plotly.plotly as py
import plotly.graph_objs as gobj
import plotly.tools as tls
import plotly.graph_objs as go

import plotly
from sklearn import preprocessing
plotly.offline.init_notebook_mode()

from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell
import io, os, sys, types
from plotly.graph_objs import Annotation, Annotations

warnings.filterwarnings('ignore')


QUART_MARATHON = '(Quart)|(1\/4)|(Filles)' 
SEMI_MARATHON = '(Semi)|(Halbmarathon)|(1\/2)'
MARATHON = 'Marathon'
KILOMETER = '\d*(\.?|\,?)\d*(\s?|-?)[kK][mM]'

OTHER_SPORT = 'Triathlon|Skating|Walking|Duathlon|Marche|Cycling' 

OTHER_RACE = 'Courir pour le plaisir'

# https://live.escalade.ch/the-race/timetable-prizes-and-courses
GENEVE_RACE = 'Course de l\'Escalade, Genève'
# http://www.christmasmidnightrun.ch/cms/index.php?option=com_content&view=category&layout=blog&id=22&Itemid=107
#LAUSSANE_RACE = 'Christmas Midnight Run, Lausanne' 
# http://www.lausanne.ch/course_co
OLYMPIC_RACE = 'Course capitale olympique'
# http://www.sierre-zinal.com/la-course/
SIERRE_ZINAL = 'Sierre-Zinal, Sierre'
# http://www.thyon-dixence.ch/le-parcours/description-2/
THYON_DIXENCE = 'Course pédestre Thyon-Dixence, Thyon'
# https://www.coursedenoel.ch/3222-trail-des-chateaux-2018!.html
SION = 'Course Titzé de noël Sion'
# http://www.morat-fribourg.ch/francais/classique.aspx
MORAT = "Morat-Fribourg"
# http://www.gruyere-cycling-tour.ch/fr/1-accueil
CYCLIC_RACE = 'Pascal Richard Classique'

FEMALE_CATEGORY_REGEX = '[D]'
MALE_CATEGORY_REGEX = '[H]'

def compute_distance_from_category(runner):
    '''
    Returns the distance of the race based on the categoryName, eventName or overCategoryName in which
    runner has done the race.
    
    Parameters
        - runner: row representing the runner
    
    Return
        - int round(distance (km)) or None if sex was not retrieved
    '''
    
    # There are the attribute useful to find the distance.
    attributes = ['categoryName', 'eventName', 'overCategoryName']
        
    # Remove all other sports
    for attribute in attributes:
        
        if pd.isnull(runner[attribute]):
            break
            
        elif (re.search(OTHER_SPORT, runner[attribute]) != None):
            return None
        
        elif (re.search(CYCLIC_RACE, runner[attribute]) != None):
            return None
        
        elif (re.search(OTHER_RACE, runner[attribute]) != None):
            return None
    
    for attribute in attributes:
        if pd.isnull(runner[attribute]):
            break
        
        # We check if the string contains information about known race.
        elif (re.search(QUART_MARATHON, runner[attribute]) != None):
            return 10
        
        elif (re.search(SEMI_MARATHON, runner[attribute]) != None):
            return 21
        
        elif (re.search(MARATHON, runner[attribute]) != None):
            return 42
        
        #elif (re.search(LAUSSANE_RACE, runner[attribute]) != None):
        #    return 2.4
    
        elif (re.search(GENEVE_RACE, runner[attribute]) != None):
            return 7.3
        
        elif (re.search(OLYMPIC_RACE, runner[attribute]) != None):
            return 5
        
        elif (re.search(SIERRE_ZINAL, runner[attribute]) != None):
            return 31
        
        elif (re.search(THYON_DIXENCE, runner[attribute]) != None):
            return 16
        
        elif (re.search(SION, runner[attribute]) != None):
            return 7
        
        elif (re.search(MORAT, runner[attribute]) != None):
            return 17
        
        # We check if the categorie contain km expression
        match = re.search(KILOMETER, runner[attribute])
        # We match we know the distance.
        if match:
            result = match.group(0).replace(',','.')
            result = re.sub('Km|km|-','',result)
            return round(float(result))
      
    return None


def compute_sex(runner,runner_dataframe):
    '''
    Returns the sex of runner based on the category in which runner has done the marathon.
    
    Parameters
        - runner: row representing the runner
    
    Return
        - string ('female'/'male') or None if sex was not retrieved
    '''

    # We remove the SettingWithCopyWarning
    pd.options.mode.chained_assignment = None 
    
    # We have already found the sex of this runner.
    if 'NaN' == str(runner_dataframe['gender'][runner_dataframe['acode'] == runner['acode']]):
        return
               
    if (re.search(FEMALE_CATEGORY_REGEX, runner['categoryName']) != None):
        runner_dataframe['gender'][runner_dataframe['acode'] == runner['acode']] = 'female'
   
    elif (re.search(MALE_CATEGORY_REGEX, runner['categoryName']) != None):
        runner_dataframe['gender'][runner_dataframe['acode'] == runner['acode']] = 'male'
    
    # We set the  default SettingWithCopyWarning
    pd.options.mode.chained_assignment = 'warn' 
    return
        
def select_runners_by_numbers_of_runs(data, nb_min_runs):
    '''
    Returns the age of runner, based on this year of birth.
    
    Parameters
        - runner: row representing the runner
        
    Return
        - age of runner (int)
    '''
    
    # We get the acode of runners where the number of runs is greater than nb_min_runs
    group_by_acode = data.groupby(['acode'], sort=False).count()
    group_by_acode = group_by_acode[group_by_acode['distance (km)'] >= nb_min_runs]
    
    # return the data with only runners with nb_min_runs minimal numbers of runs.
    return data[data['acode'].isin(group_by_acode.index.values)]


def compute_dataframe_groupby(data, data_before_Processing):
    '''
    Compute new features like 'overall number race' and 'number abandon'.
    
    Parameters
        - data: Dataframe of all races
        - data_before_Processing : Dataframe before the preprocessing
        
    Return
        - Dataframe with total runs and total abandon
    '''
    
    group_by_runs = data.groupby(['acode']).size().reset_index().groupby('acode')[[0]].max()
    group_by_runs_before_process = data_before_Processing.groupby(['acode']).size().reset_index().groupby('acode')[[0]].max()
    group_by_runs_resign = data_before_Processing[(data_before_Processing['resultState'] == 'non classé')].groupby(['acode']).size().reset_index().groupby('acode')[[0]].max()

    group_by_runs.columns = ['number_race']
    group_by_runs_before_process.columns = ['overall number race']
    group_by_runs_resign.columns = ['number abandon']

    result = group_by_runs.join(group_by_runs_before_process)
    result = pd.merge(result, group_by_runs_resign, how='left', right_index=True, left_index=True)
    result.fillna(value=0,  inplace=True)
    return result

def transform_string_to_second(runner):
    '''
    Returns the time in second of runner, based on this string time.
    
    Parameters
        - runner: row representing the runner
        
    Return
        - time of runner for the race (int)
    '''
    
    hour = '0'
    split_hour = ':'
    
    remove_quart_second = re.sub(',\d?','',runner['runtime'])
    
    # We add an hour if necessary.
    if not (re.search(split_hour, remove_quart_second) != None):
        remove_quart_second = hour + split_hour + remove_quart_second
    
    time_formated = dt.datetime.strptime(remove_quart_second, '%H:%M.%S')
    
    return dt.timedelta(hours=time_formated.hour,
                                  minutes=time_formated.minute, seconds=time_formated.second).total_seconds()


def compute_date_event(runner): 
    '''
    Returns the date of the event.
    
    Parameters
        - runner: row representing the runner
        
    Return
        - date of race (date)
    '''
    return dt.datetime.strptime(runner['eventDate'], '%d.%m.%Y').date()

def compute_age(runner, runner_dataframe):
    '''
    Returns the age of runner, based on this year of birth.
    
    Parameters
        - runner: row representing the runner
        
    Return
        - age of runner (int)
    '''
    
    # Get the year
    birthyear = (runner_dataframe['birthyear'][runner_dataframe['acode'] == runner['acode']]).values[0]
    date_event = parser.parse(runner['eventDate'])

    return date_event.year - birthyear

def remove_outliers(df):
    '''
    Remove outliers from the data.
    
    Parameters
        - df: DataFrame containing data
        
    Return
        - dataFrame with only finishers
    '''
    df = df[df['runtime'].notnull()]
    return df[df['resultState'] == 'classé']


def remove_useless_columns(df):
    '''
    Remove useless columns.
    
    Parameters
        - df: DataFrame containing data
    '''
    df.drop('entryArt', axis=1, inplace=True)
    df.drop('entryPayart', axis=1, inplace=True)
    df.drop('provider', axis=1, inplace=True)
    df.drop('startNumber', axis=1, inplace=True)
    df.drop('raceNr', axis=1, inplace=True)
    df.drop('eventRaceNr', axis=1, inplace=True)
    df.drop('racePayload', axis=1, inplace=True)
    df.drop('resultState', axis=1, inplace=True)
    
def preprocess_runners(df):  
    '''
    create columns gender.
    
    Parameters
        - df: DataFrame containing data
    '''
    df['gender'] = np.nan
    df.drop('name', axis=1, inplace=True)
    df['number_acode'] = df.index.values
    
    
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2
    
    
def presentation_performance_runners(fig, data, annotation = []):
    '''
    This function plots the speeds of a specific runner by distance.

    Parameters
        - fig: Figure on which subplots are displayed
        - data: DataFrame containing the data relative to a given running
        - annotation: Annotation to add in the graph (by default, no annotation)
    '''
    
    plot1 = fig.add_subplot(311)
    sns.swarmplot(x="distance (km)", y="speed (m/s)", data=data[0], ax=plot1)
    plot1.set_title('Runner 1')
    plot1.set_xlabel('')
    plot1.set_ylabel('')

    plot2 = fig.add_subplot(312)
    sns.swarmplot(x="distance (km)", y="speed (m/s)", data=data[1], ax=plot2)
    plot2.set_title('Runner 2')
    plot2.set_xlabel('')
    
    # Add annotation if any given
    if len(annotation) != 0 :
        plot2.annotate(annotation[0], annotation[1], annotation[2], arrowprops=dict(facecolor='red', shrink=0.05))

    plot3 = fig.add_subplot(313)
    sns.swarmplot(x="distance (km)", y="speed (m/s)", data=data[2], ax=plot3)
    plot3.set_title('Runner 3')
    plot3.set_ylabel('')
    
    
def compute_dataframe_marathon_performance(df_runs, df_overall):
    '''
    This function compute new pamareter like 'distance (km)', number of event, year of each event for each runners.

    Parameters
        - df_runs: DataFrame containing data
        - df_overall: DataFrame containing all data collected
        - eventName: Name of the event 
        - distance: Distance related to the event.
        
    return dataframe selecting of all runners.
    '''
    
    # compute the sum of kilimoter in one year for the preparation of marathon.
    df_runs['year'] = df_runs['eventDate'].apply(lambda x: int(x.year))
    group_by_runs_sum_kilometer = df_runs.groupby(['acode', 'year']).sum()[['distance (km)']]
    group_by_runs_sum_kilometer.columns = ['sum distance (km)']
    
    # compute the number of events
    group_by_runs_before_process = df_overall.groupby(['acode', 'year']).size().reset_index().groupby(['acode', 'year'])[[0]].max()
    group_by_runs_before_process.columns = ['overall number events']
    
    # compute the returning dataframe
    result = pd.concat([group_by_runs_sum_kilometer,group_by_runs_before_process], axis = 1)
    result.fillna(value=0,  inplace=True)
    result.reset_index(inplace=True)

    return pd.merge(result, df_runs, on=['acode', 'year'])

def display_age_distribution(data, title):
    '''
    Display the age distribution.

    Parameters :
        - data: Serie of age
        - title: title of the graph
    '''
    data_fig = [
        go.Histogram(
            x=data,
            histnorm='count',
            name='runners'
        )
    ]

    layout = go.Layout(
        title=title,
        xaxis=dict(
            title='Age',
            titlefont=dict(
                size=18
            )
        ),
        yaxis=dict(
            title='Number of runners',
            titlefont=dict(
                size=18
            )
        ),
        showlegend=False,
        annotations = [Annotation(
            y=120,
            x=75,
            text='mean age : ' + str(round(data.mean(),2)),
            showarrow=False
        )]
    )
    
    fig = go.Figure(data=data_fig, layout=layout)
    plotly.offline.iplot(fig)
    return fig

def diplay_comparaison_runners_performance(fig, data): 
    '''
    Plot a scatter plot between the total number of kilometer and the speed of each runners.

    Parameters
        - fig: Figure on which subplots are displayed
        - data: DataFrame containing the data relative to a given running
        
    '''
    
    # This is the first plot with the total number (Km) raced in the year.
    ax1  = fig.add_subplot(211)
    ax1.scatter(data['speed (m/s)'], data['sum distance (km)'], c=data['number_acode'], cmap=plt.cm.Paired)
    plt.ylabel('Sum distance (Km)')

    # This is the second plot with the total event in the year.
    ax2  = fig.add_subplot(212)
    ax2.scatter(data['speed (m/s)'], data['overall number events'], c=data['number_acode'], cmap=plt.cm.Paired)
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Total number Event')
    
def compute_coefficient(runner, runs_dataFrame, colums_selection):
    '''
    compute coefficient dependending of colums_selection.
    - The coefficient is defined by the difference between value for the current row and the value obtained for best time race
    for the same specific race.
    
    The value return is None is the curent race is the best time race.
    
    Parameters
        - runner: row represents a race.
        - runs_dataFrame: dataframe containing all runs of the
        - colums_selection: select columns for comparaison
        
    '''
    # find the best time for the coresponding entry
    same_race = runs_dataFrame[(runs_dataFrame['eventName'] == runner['eventName']) & (runs_dataFrame['distance (km)'] == runner['distance (km)']) & (runs_dataFrame['acode'] == runner['acode'])]
    best_time_index = same_race['time (s)'].idxmin()
    
    # compute difference between the sum kilometer
    #    - If the runner have run less in the year than his best time the coeficient is positive
    #    - if the runner have run more in the year than his best time the coeficient is negative 
    if runner['eventDate'] == same_race.ix[[best_time_index]]['eventDate'].values[0]:
        return None
    
    return (runner[colums_selection] - same_race.ix[[best_time_index]][colums_selection].values[0])

def compute_best_time (runner, runs_dataFrame):
    '''
    Compute the difference between the current row and the best time.
    
    Parameters
        - runner: row represents a race.
        - runs_dataFrame: dataframe containing all runs of the
        
    '''
    # find the best time for the coresponding entry
    same_race = runs_dataFrame[(runs_dataFrame['eventName'] == runner['eventName']) & (runs_dataFrame['distance (km)'] == runner['distance (km)']) & (runs_dataFrame['acode'] == runner['acode'])]
    best_time_index = same_race['time (s)'].idxmin()
    
    return (runner['time (s)'] - same_race.ix[[best_time_index]]['time (s)'].values[0])


def plot_coefficient_distribution(data, inexperienced_runners, experienced_runners, name_coefficient, display=True, bin_size=0.05):
    '''
    Plot the distribution of coefficient 'name_coefficient'.
        - the distibution is splitted into two group experience runners and inexperience runner 
        - the plot contain:
            - one table (Kolmogorov-Smirnov statistic test) on the total races.
            - one displot on experienced runners
            - one displot on inexperienced runners

    Parameters
        - data: DataFrame containing information about the race.
        - inexperienced_runners: List containing acode of inexperienced runner
        - experienced_runners: List containing acode of experienced runner
        - name_coefficient: Name of the coefficient studied
        - bin_size: Bin_size of the displot
    '''
        
    data_copied = data.copy()
    
    # remove null value.
    data_copied = data_copied[data_copied[name_coefficient].notnull()]
    
    # We do an kolmogorov test
    #kolmogorov_results = scipy.stats.kstest(normalize_data, cdf='norm')
    shapiro_results = scipy.stats.shapiro(data_copied[name_coefficient])

    matrix_sw = [
        ['', 'DF', 'Test Statistic', 'p-value'],
        ['Sample Data', len(data_copied[name_coefficient]) - 1, shapiro_results[0], shapiro_results[1]]
    ]

    
    # apply max min scaller to have the same disparity in coefficient.
    min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
    data_copied[name_coefficient] = min_max_scaler.fit_transform(data_copied[name_coefficient])
    runs_experienced = data_copied[name_coefficient][data_copied['acode'].isin(experienced_runners)]
    runs_inexperienced = data_copied[name_coefficient][data_copied['acode'].isin(inexperienced_runners)]

    # We compute differents parameters for siplaying usefull informations
    mean_experienced = round(runs_experienced.mean(), 3)
    mean_inexperienced = round(runs_inexperienced.mean(), 3)
    lenght_experienced = len(runs_experienced)
    lenght_inexperienced = len(runs_inexperienced)

    #build the graph
    displot_experienced = FF.create_distplot([runs_experienced], [''], bin_size=bin_size)
    displot_inexperienced = FF.create_distplot([runs_inexperienced], [''], bin_size=bin_size)
    shapiro_table = FF.create_table(matrix_sw, index=False)
    
    # We make the subplot.
    my_fig = tls.make_subplots( subplot_titles=('Experienced Runners', 'Inexperienced runners'),                       
            rows = 1,
            cols = 2,
            print_grid=False,
            vertical_spacing = 1
    )
    
    add_figure_from_displot(my_fig, displot_experienced, displot_inexperienced)

    # We compute the maximum of y axis
    max_yaxis = math.ceil(max([max(my_fig['data'][1]['y'] + 1), max(my_fig['data'][3]['y'] + 1)]))
    

    # We add informations to the graph.
    add_text_to_plot(my_fig, mean_experienced, lenght_experienced, mean_inexperienced, lenght_inexperienced, max_yaxis)
    
    
    # We add vertical line to visualize easily
    add_line_to_plot(my_fig, mean_experienced, mean_inexperienced, max_yaxis)
    
    # modify legend and label
    modify_layout(my_fig, max_yaxis, name_coefficient)
    
    # display plot.
    if display:
        plotly.offline.iplot(shapiro_table)
        plotly.offline.iplot(my_fig)
    return my_fig

def display_scatter_matrix(data): 
    '''
    plot a scattter mtrix of data
    
    Parameters
        - data : Dataframe containing the data.
    '''
    fig = FF.create_scatterplotmatrix(data, height=800, width=800)
    plotly.offline.iplot(fig)
    return


def add_figure_from_displot(my_fig, displot_experienced, displot_inexperienced):
    '''
    Add the displot to the subplot my_fig.
    
    Parameters
        - my_fig                : Figure containing the two displot.
        - displot_experienced   : Displot of experienced runners
        - displot_inexperienced : Displot of inexperienced runners
    '''
    
    data_displot_experienced = displot_experienced['data']
    for item in data_displot_experienced:
        item.pop('xaxis', None)
        item.pop('yaxis', None)
        
    data_displot_inexperienced = displot_inexperienced['data']
    for item in data_displot_inexperienced:
        item.pop('xaxis', None)
        item.pop('yaxis', None)
        
    # We add the displot in the subplot.    
    my_fig.append_trace(data_displot_experienced[0], 1, 1)
    my_fig.append_trace(data_displot_experienced[1], 1, 1)

    my_fig.append_trace(data_displot_inexperienced[0], 1, 2)
    my_fig.append_trace(data_displot_inexperienced[1], 1, 2)
    return

def modify_layout(my_fig, max_yaxis, name_coefficient):
    '''
    Modify axis value of the subplot my_fig.
        - Change the value of yaxis1 and yaxis2
        - Change title and value of xaxis1 and xaxis2, the title is the name of the coefficient
        - remove the legend
    
    Parameters
        - my_fig           : Figure containing the two displot.
        - max_yaxis        : maximum value in y axis
        - name_coefficient : name of the coefficient studied
    '''
    my_fig['layout']['yaxis1'].update(range=[0, max_yaxis])
    my_fig['layout']['yaxis2'].update(range=[0, max_yaxis])
    my_fig['layout']['xaxis1'].update(title=name_coefficient, range=[-1, 1])
    my_fig['layout']['xaxis2'].update(title=name_coefficient, range=[-1, 1])
    my_fig['layout']['annotations'][0].update(y=1.04)
    my_fig['layout']['annotations'][1].update(y=1.04)
    my_fig['layout']['yaxis1'].update(title='density')
    my_fig['layout'].update(title='Coefficient distribution')
    my_fig['layout']['showlegend'] = False
    return

def add_text_to_plot(my_fig, mean_experienced, lenght_experienced, mean_inexperienced, lenght_inexperienced, max_yaxis):
    '''
    Add text information on the subplot.
        - add information on mean
        - add information on number of races
    
    Parameters
        - my_fig                : Figure containing the two displot.
        - mean_experienced      : mean of the experienced runners value
        - lenght_experienced    : lenght of experienced runners value
        - mean_inexperienced    : mean of the inexperienced runners
        - lenght_inexperienced  : lenght of inexperienced runners value
        - max_yaxis             : maximum value in y axis
    '''
    
    information_experienced = go.Scatter(
        x=[1 - 0.5, 1 - 0.5 ],
        y=[max_yaxis - 0.5, max_yaxis - 0.7 ],
        mode='text',
        text=['mean = ' + str(mean_experienced), 'total races = ' + str(lenght_experienced)]
    )

    information_inexperienced = go.Scatter(
        x=[1 - 0.5, 1 - 0.5 ],
        y=[max_yaxis - 0.5, max_yaxis - 0.7 ],
        mode='text',
        text=['mean = ' + str(mean_inexperienced) ,' total races = ' + str(lenght_inexperienced)]
    )

    my_fig.append_trace(information_experienced, 1, 1)
    my_fig.append_trace(information_inexperienced, 1, 2)
    return

def add_line_to_plot(my_fig, mean_experienced, mean_inexperienced, max_yaxis):
    '''
    Add line indicating the mean of the distribution.
    
    Parameters
        - my_fig                : Figure containing the two displot.
        - mean_experienced      : mean of the experienced runners value
        - mean_inexperienced    : mean of the inexperienced runners value
        - max_yaxis             : maximum value in y axis
    '''
    
    line_experience = go.Scatter(
        x = [mean_experienced, mean_experienced],
        y = [0, max_yaxis],
        mode = 'lines',
        name = 'lines'
    )
    
    line_inexperience = go.Scatter(
        x = [mean_inexperienced, mean_inexperienced],
        y = [0, max_yaxis],
        mode = 'lines',
        name = 'lines'
    )

    my_fig.append_trace(line_experience, 1, 1)
    my_fig.append_trace(line_inexperience, 1, 2)
    return


def generate_plotly_figure(data, list_coefficient, inexperienced_runners, experienced_runners, bin_size=0.05):
    '''
    create dictionnary for plotly export.
    
    Parameters
        - data                  : data which contain information on coefficient.
        - list_coefficient      : list name coefficient
    '''
    result_dict = {}
    for coefficient in list_coefficient:
        result_dict[coefficient] = plot_coefficient_distribution(data, inexperienced_runners, experienced_runners, coefficient, display=False, bin_size=bin_size)
    
    return result_dict
    

#######################################################################################
#                        Import Notebook as submodule
#######################################################################################

#######################################################################################
#  credit to : 
#  http://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Importing%20Notebooks.ipynb
#######################################################################################

class NotebookLoader(object):
    """Module Loader for Jupyter Notebooks"""
    def __init__(self, path=None):
        self.shell = InteractiveShell.instance()
        self.path = path
    
    def load_module(self, fullname):
        """import a notebook as a module"""
        path = find_notebook(fullname, self.path)
        
        print ("importing Jupyter notebook from %s" % path)
                                       
        # load the notebook object
        with io.open(path, 'r', encoding='utf-8') as f:
            nb = read(f, 4)
        
        
        # create the module and add it to sys.modules
        # if name in sys.modules:
        #    return sys.modules[name]
        mod = types.ModuleType(fullname)
        mod.__file__ = path
        mod.__loader__ = self
        mod.__dict__['get_ipython'] = get_ipython
        sys.modules[fullname] = mod
        
        # extra work to ensure that magics that would affect the user_ns
        # actually affect the notebook module's ns
        save_user_ns = self.shell.user_ns
        self.shell.user_ns = mod.__dict__
        
        try:
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    # transform the input to executable Python
                    code = self.shell.input_transformer_manager.transform_cell(cell.source)
                    # run the code in themodule
                    exec(code, mod.__dict__)
        finally:
            self.shell.user_ns = save_user_ns
        return mod
    
    
    
def find_notebook(fullname, path=None):
    """find a notebook, given its fully qualified name and an optional path
    
    This turns "foo.bar" into "foo/bar.ipynb"
    and tries turning "Foo_Bar" into "Foo Bar" if Foo_Bar
    does not exist.
    """
    
    print(fullname)
    name = fullname.rsplit('.', 1)[-1]
    print(name)
    if not path:
        path = ['']
    for d in path:
        nb_path = os.path.join(d, name + ".ipynb")
        print(nb_path)
        if os.path.isfile(nb_path):
            return nb_path
        # let import Notebook_Name find "Notebook Name.ipynb"
        nb_path = nb_path.replace("_", " ")
        if os.path.isfile(nb_path):
            return nb_path
    