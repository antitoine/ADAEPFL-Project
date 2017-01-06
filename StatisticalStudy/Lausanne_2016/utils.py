# Data analysis 
import pandas as pd
import numpy as np
import re
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import FloatProgress
import collections
from IPython.display import display
from datetime import date
from dateutil.relativedelta import relativedelta
from matplotlib.pyplot import show

# ----------------------------------------------------------------------------------------------------------
# Constants

# category can be find at http://fr.lausanne-marathon.com/palmares/
ILINE_REGEX = '[R]'#Iline male / Iline female
KIDS_REGEX = '(K)|(Fille)|(Gar)'
WHEEL_CHAIR_REGEX = '(FD)|(FH)|(HB)|(Hand)' # Wheelchair male / Wheelchair female / handybike
FEMALE_CATEGORY_REGEX = '([D])|(JunF)'
MALE_CATEGORY_REGEX = '([H])|(JunG)'

MARATHON_DISTANCE_REGEX = '(42)|(M)|(52)' # 52 come from a mistake on datasport site
# https://services.datasport.com/2010/lauf/lamara/rang035.htm

SEMI_MARATHON_DISTANCE_REGEX = '(21)|(S)'
QUARTER_MARATHON_DISTANCE_REGEX = '(10)|(Q)'

# ----------------------------------------------------------------------------------------------------------
# Functions
    
def filter_participants(runner):
    '''
    If participant is part of a specific category, function returns false (i.e. participant is excluded).
    Specific categories are: people in wheelchair and kids (not representative) and categories such as Pink_Ch, 10W-Walk or 10W-NW (not part of Lausanne Marathon).

    Parameters
        - runner: row representing the runner

    Return
        - boolean: true if participant is part of analysis, false otherwise
    '''

    if ((runner['category'] == '10W-NW' or runner['category'] == '10W-Walk' or runner['category'] == 'Pink-Ch')
        or (re.search(WHEEL_CHAIR_REGEX, runner['category']) != None)
        or (re.search(KIDS_REGEX, runner['category']) != None)
        or (re.search(ILINE_REGEX, runner['category']) != None)):
        return False
    else:
        return True

    
def get_sex_of_runner(runner):
    '''
    Returns the sex of runner based on the category in which runner has done the marathon.
    
    Parameters
        - runner: row representing the runner
    
    Return
        - string ('female'/'male') or None if sex was not retrieved
    '''
    
    if (re.search(FEMALE_CATEGORY_REGEX, runner['category']) != None):
        return 'female'
    elif (re.search(MALE_CATEGORY_REGEX, runner['category']) != None):
        return 'male'
    else:
        return None


def get_type_of_runner(runner):
    '''
    Returns the type of the runner.
    
    Parameters
        - runner: row representing the runner
        
    Return
        - string ('junior'/'adult')
    '''
    
    if (runner['category'].find('Jun') != -1):
        return 'junior'
    else:
        return 'adult'
    
    
def compute_distance_from_category (runner):
    '''
    Returns the category distance of the runner, based on category
    
    Parameters
        - runner: row representing the runner
        
    Return
        - distance of runner (int)
    '''
    
    if (re.search(MARATHON_DISTANCE_REGEX, runner['category']) != None):
        return 42
    elif (re.search(SEMI_MARATHON_DISTANCE_REGEX, runner['category']) != None):
        return 21
    elif (re.search(QUARTER_MARATHON_DISTANCE_REGEX, runner['category']) != None):
        return 10
    
    return None


def compute_age_of_runner(runner):
    '''
    Returns the age of runner, based on this year of birth.
    
    Parameters
        - runner: row representing the runner
        
    Return
        - age of runner (int)
    '''
    
    today = date.today()
    birth_year = runner['birthday']
    return today.year - birth_year.year - ((today.month, today.day) < (birth_year.month, birth_year.day))


def compute_run_in_team(runner):
    '''
    Returns the age of runner, based on this year of birth.
    
    Parameters
        - runner: row representing the runner
        
    Return
        - age of runner (int)
    '''
    
    if pd.isnull(runner['team']):
        return 'Individual runners'
    else:
        return 'Runners in teams'

    
def format_time(runner):
    '''
    Returns the number of seconds of running time of current runner
    
    Parameters
        - runner: row representing the runner
        
    Return
        - total running time in seconds (int)
    '''
    
    time = runner['time']
    formatted_time = time.time()
    if time:
        return datetime.timedelta(hours=formatted_time.hour, minutes=formatted_time.minute, seconds=formatted_time.second).total_seconds()
    

def convert_seconds_to_time(seconds):
    '''
    Returns formatted time according to a given number of seconds
    
    Parameters
        - seconds: number of seconds of a given time
        
    Return
        - formatted time (HH:mm:ss format, string)
    '''
    
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def plot_performance_according_to_running_type(data, nb_km):
    '''
    Plots the performance according to age of participants for a given running
    
    Parameters
        - data: DataFrame containing records for a given running
        - nb_km: km of the running
    '''
    
    g = sns.factorplot(data=data, x='age', y='time', kind='box', size=10, aspect=1.5)
    for ax in g.axes.flat:
        labels = []
        for label in ax.get_yticklabels():
            formatted_label = convert_seconds_to_time(int(float(label._y)))
            labels.append(formatted_label)
        ax.set_yticklabels(labels)
    plt.title('Distribution of time of ' + str(nb_km) + 'km running following age of participants')
    plt.show()


def plot_speed_distribution_by_running(fig, running, running_type, nb_plot, y_range=np.arange(0, 900, 100)):
    '''
    This function adds plot of speed distribution for a given running, in a figure.

    Parameters
        - fig: figure in which plot will be added
        - running: data for a given running
        - running_type: number of kilometers of the running (for display)
        - nb_plot: index of the plot in the figure
        - y_range: range for ordinate (by default, (0, 900, 100))
    '''

    ax = fig.add_subplot(310 + nb_plot)
    race = running['Speed (m/s)'].tolist()
    ax.hist(race, bins=25)
    ax.set_ylabel('Number of Runners')
    ax.set_title('Distance = ' + running_type, fontsize=20)
    ax.set_xlabel('Speed (m/s)')
    ax.xaxis.set_label_coords(1.15, -0.025)

    # Set of axis
    plt.xticks(np.arange(0,6.5,0.5))
    plt.yticks(y_range)

    # Computing of important information
    avg_time = round(np.mean(race), 4)
    median_time = round(np.median(race), 4)
    max_speed = round(np.max(race), 2)
    min_speed = round(np.min(race), 2)
    total = len(race)
    ax.axvline(median_time, 0, 1750, color='r', linestyle='--')

    # Creation of string with statistics
    mean_str = 'Mean: ' + str(avg_time) + ' m/s'
    median_str = 'Median: ' + str(median_time)  + ' m/s'
    max_str = 'Max: ' + str(max_speed)   + ' m/s'
    min_str = 'Min: ' + str(min_speed)   + ' m/s'
    total_str = 'Total: ' + str(total)   + ' runners'
    std_str = 'SD: ' + str(round(np.std(race), 2)) + 'm/s'
    stats_str = total_str + '\n' + mean_str + '\n' + median_str + '\n' + max_str + '\n' + min_str + '\n' + std_str

    # Add of information in the graph
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(.95, .95, stats_str, fontsize=12, transform=ax.transAxes, va='top', ha='right', bbox=props, multialignment='left')
    
    
def plot_distribution_age_distance(fig, data, title, place):

    ax  = fig.add_subplot(place)
    ax.hist([data['age'][data['sex'] == 'male'], data['age'][data['sex'] == 'female']],
         bins=30,
         stacked=True,
         rwidth=1.0,
         label=['male', 'female'])

    plt.xticks(np.arange(10,100,10))
    
    # Legend.
    if place == 311: 
        ax.legend(loc='upper left')
        
    # Computing of the mean of age selected by gender
    mean = np.mean(data['age'])

    # Display of the median and titles
    ax.axvline(mean, 0, 1750, color='r', linestyle='--')
    ax.set_title(title)
    
    # Legend
    if place == 313:
        ax.set_xlabel('Age')
    
    # Legend
    if place == 312:
        ax.set_ylabel('Number of runners')

    # Calculation of age distribution statistics by gender
    age_stats = 'Mean Age: ' + str(round(mean, 2)) + ' years\n' + 'SD: ' + str(round(np.std(data['age']), 2)) 
    age_stats = age_stats

    # Add of legend text
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(.95, .95, age_stats, fontsize=11, transform=ax.transAxes, va='top', ha='right', bbox=props, multialignment='left')
    
    
def compute_overall_rank(data):
    '''
    compute the overall rank by distance
    
    Parameters
        - data: DataFrame containing records for a given running
    '''
    
    # remove the SettingWithCopyWarning
    pd.options.mode.chained_assignment = None 
    
    # discriminators for the selection
    distances = [42,21,10]
    type_runners =  ['Individual runners','Runners in teams']
    genders = ['male','female']

    # list containing each dataframe by distance.
    all_races = []
    
    # loop on distance
    for distance in distances:
        all_type = []
        distance_selection  = data[data['distance (km)'] == distance]
        
        # loop on runner type
        for type_runner in type_runners:
            all_gender = []
            type_selection = distance_selection[distance_selection['type_team'] == type_runner]
            
            # loop on gender
            for gender in genders:
                gender_selection = type_selection[type_selection['sex'] == gender]
                
                # sorting by gender.
                gender_selection.sort_values('time', ascending=True, inplace=True)
                
                # compute the overall rank in the distance.
                gender_selection['overall_rank'] = range (1,len(gender_selection)+1)
                
                # add the dataframe.
                all_gender.append(gender_selection)
           
            # compute the dataframe by gender
            all_type.append(pd.concat(all_gender))
        
        # compute the dataframe by type.
        all_races.append(pd.concat(all_type))

    pd.options.mode.chained_assignment = 'warn'
    
    # return the all dataframe.
    return pd.concat(all_races)

