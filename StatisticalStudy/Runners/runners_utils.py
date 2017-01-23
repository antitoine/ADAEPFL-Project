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

from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell
import io, os, sys, types

warnings.filterwarnings('ignore')


QUART_MARATHON = '(Quart)|(1\/4)|(Filles)' 
SEMI_MARATHON = '(Semi)|(Halbmarathon)|(1\/2)'
MARATHON = 'Marathon'
KILOMETER = '\d*(\.?|\,?)\d*(\s?|-?)[kK][mM]'

OTHER_SPORT = 'Triathlon|Skating|Walking|Duathlon|Marche|Cycling' 

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
    
def compute_weakness_coefficient(runner, runs_dataFrame, colums_selection):
    '''
    compute coefficient dependending of columns.
    
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
    return (runner[colums_selection] - same_race.ix[[best_time_index]][colums_selection].values[0])


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
    