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
from sklearn import preprocessing
from collections import Counter
from scipy import stats
from io import StringIO
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
    
    
def compute_time_to_best_in_team(runner, data):
    
    if runner.type_team == 'Individual runners':
        return None
    
    else:
        # select best performances in the team by sex and  distance
        team_performance = np.min(data['time'][(data['team'] == runner.team) & (data['distance (km)'] == runner['distance (km)'])])        
        return abs(team_performance - runner.time)
    

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


def plot_performance_according_to_running_type(data, nb_km, column, size=4.5, aspect=2):
    '''
    Plots the performance according to age of participants for a given running
    
    Parameters
        - data: DataFrame containing records for a given running
        - nb_km: km of the running
        - column: column to use for x axis
        - size: size of the graph (by default, 4.5)
        - aspect: aspect of the graph (by default, 2)
    '''
    
    g = sns.factorplot(data=data, x=column, y='time', kind='box', size=size, aspect=aspect)
    for ax in g.axes.flat:
        labels = []
        for label in ax.get_yticklabels():
            formatted_label = convert_seconds_to_time(int(float(label._y)))
            labels.append(formatted_label)
        ax.set_yticklabels(labels)
    plt.title('Distribution of time of ' + str(nb_km) + 'km running following age of participants')
    plt.show()


def plot_time_distribution(ax, running, name):
    '''
    This function create a subplot containing the time distribution for a given age, and for the 3 types of runnings (10 km, 21km, 42km).

    Parameters
        - ax: subplot to use for the histogram (Matplotlib axes)
        - running: data for a given set of participants of same age (DataFrameGroupby)
        - name: name of the category
    '''

    # Creation of histogram
    running_10k = running[running['distance (km)'] == 10]
    race_10k = running_10k['time'].tolist()
    running_21k = running[running['distance (km)'] == 21]
    race_21k = running_21k['time'].tolist()
    running_42k = running[running['distance (km)'] == 42]
    race_42k = running_42k['time'].tolist()
    ax.hist([race_10k, race_21k, race_42k], bins=30, stacked=True, rwidth=1.0, label=['10 km', '21 km', '42 km'])
    ax.legend()
    ax.set_ylabel('Number of Runners')
    ax.set_title('Time distribution (' + str(name) + ')')
    ax.xaxis.set_label_coords(1.15, -0.025)
    
    # Creation of texts
    total_10k = len(race_10k)
    total_10k_str = '10 km: ' + str(total_10k)   + ' runners'
    total_21k = len(race_21k)
    total_21k_str = '21 km: ' + str(total_21k)   + ' runners'
    total_42k = len(race_42k)
    total_42k_str = '42 km: ' + str(total_42k)   + ' runners'
    total = len(running['time'].tolist())
    total_str = 'Total: ' + str(total)   + ' runners'
    stats_str = total_10k_str + '\n' + total_21k_str + '\n' + total_42k_str + '\n' + total_str
    plt.xticks(ax.get_xticks(), [convert_seconds_to_time(label) for label in ax.get_xticks()], rotation=90)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5) 
    ax.annotate(stats_str, xy=(0, 1), xytext=(12, -12), va='top', xycoords='axes fraction', textcoords='offset points', bbox=props)


def plot_time_distribution_by_age(df, column):
    '''
    This function plots the distribution of time for all ages regarding participants of a Lausanne Marathon.
    3 subplots are displayed per rows.

    Parameters
        - df: DataFrame containing all the information of a Lausanne Marathon
    '''

    distributions = df.groupby(column)
    i = 1
    for name, group in distributions:
        plt.subplots_adjust(left=None, bottom=None, right=2, top=None, wspace=None, hspace=None)
        if i % 3 == 1:
            ax = plt.subplot(1, 3, 1)
        elif i % 3 == 2:
            ax = plt.subplot(1, 3, 2)
        else:
            ax = plt.subplot(1, 3, 3)
        plot_time_distribution(ax, group, name)
        if i % 3 == 0:
            plt.show()
        i += 1


def compute_anova_and_tukey_hsd(df, categories, values):
    '''
    This function computes the ANOVA test for different distributions.

    Parameters
        - df: DataFrame containing data
        - categories: column containing the categories on which ANOVA will be performed
        - values: column containing the values
    '''

    results = {}
    all_values_by_category = [df[df[categories] == category][values] for category in pd.unique(df[categories].unique())]
    results['f_value'], results['p_value'] = stats.f_oneway(*all_values_by_category)
    tukey_hsd_string = StringIO(pairwise_tukeyhsd(df[values], df[categories]).summary().as_csv())
    results['tukey_hsd'] = pd.read_csv(tukey_hsd_string, skiprows=1)
    return results


def plot_time_distribution_by_bib_numbers(df):
    '''
    This function display the time distribution of runners according to the BIB numbers

    Parameters
        - df: DataFrame containing data to use to generate the graph
    '''

    ax = df.plot(kind='scatter', x='number', y='time', xlim=(-1000, 18000));
    lines = [-200, 2000, 8800, 9800, 17000]
    annotations = [('10 km', 12500), ('21 km', 4500), ('42 km', 150)]
    for line in lines:
        ax.axvline(line, color='b', linestyle='--')
    for annotation in annotations:
        annotation_obj = [annotation[0],(0, 0), (annotation[1], 28000)]
        ax.annotate(annotation_obj[0], annotation_obj[1], annotation_obj[2], color='b')
    plt.yticks(ax.get_yticks(), [convert_seconds_to_time(label) for label in ax.get_yticks()])
    plt.title('Running time according to BIB number of participants')
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
    running_male =  running[running['sex'] == 'male']
    running_female =  running[running['sex'] == 'female']
    race_female = running_female['Speed (m/s)'].tolist()
    race_male = running_male['Speed (m/s)'].tolist()
    race = running['Speed (m/s)'].tolist()
    
    
    ax.hist([race_male, race_female],
         bins=30,
         stacked=True,
         rwidth=1.0,
         label=['male', 'female'])
    
    # Legend.
    if nb_plot == 1: 
        ax.legend(loc='upper left')

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

def plot_time_difference_distribution (data):

    ax = data['time difference team'].hist(bins=30,figsize=(10,6))

    # Computing of the mean of age selected by gender
    mean = np.mean(data['time difference team'])
    max_time_diff = np.max(data['time difference team'])

    # Display of the median and titles
    ax.axvline(mean, 0, 1750, color='r', linestyle='--')
    ax.set_title('Time difference with best runner in the team distribution')

    plt.xticks(ax.get_xticks(), [convert_seconds_to_time(label) for label in ax.get_xticks()])

    # Calculation of age distribution statistics by gender
    time_stats = 'Mean time difference : ' + convert_seconds_to_time(mean) + '\n' + 'Max time difference : ' +  convert_seconds_to_time(max_time_diff)

    # Add of legend text
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(.95, .95, time_stats, fontsize=11, transform=ax.transAxes, va='top', ha='right', bbox=props, multialignment='left')

    
def display_legend(dict_team_runner, plot):
    
    # Creation of string with statistics
    pairs_runners = 'Pair runners: ' + str(dict_team_runner.get(1))   + ' runners'
    individual_runners = 'individual runners: ' + str(dict_team_runner.get(0)) + ' runners'
    stats_str = pairs_runners + '\n' + individual_runners 

    # Add of information in the graph
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plot.text(.95, .95, stats_str, fontsize=12, transform=plot.transAxes, va='top', ha='center', bbox=props, multialignment='left')

def compute_pair_runner (runner, data, time_second):
    
    # get all time in the team execpt its own personnal time
    team_performance = data['time difference team'][(data['team'] == runner.team) & (data['acode'] != runner.acode)] 
       
    min_time = abs(min(team_performance, key=lambda x:abs(x-runner['time difference team'])) - runner['time difference team'])

    if min_time > time_second:
        return 0
    
    return 1

def plot_scatter_difference_time_number (fig, data, distance, time_mini, place, annotation = []):
    
    # Map string name of the team to 
    race_team = data[data['team'].notnull()].copy()
    race_team = race_team[race_team["distance (km)"] == distance]

    # remove time equal to zero not interesting for the study.
    race_team = (race_team[race_team['time difference team'] > time_mini])

    # remove team with one runner.
    for team in race_team['team']: 
        race_team_selected = race_team[race_team['team'] == team]
        if len(race_team_selected['team']) == 1:
            race_team = race_team[race_team['team'] != team]


    # map team name to team number.
    team_label_encode = preprocessing.LabelEncoder()
    team_label = team_label_encode.fit_transform(race_team['team'])
    race_team['team_code'] = team_label
    
    # compute paire runner
    number_runner_in_pair = race_team.apply(compute_pair_runner,args=(race_team,60), axis=1)
    counter_pair = Counter(number_runner_in_pair)
    
    # plotting the results.
    plot = fig.add_subplot(place)
    sns.swarmplot(x="team_code", y="time difference team", hue="sex", data=race_team, ax = plot )
    
    plot.set_title('Distance = '+ str(distance))
    plot.set_xlabel('')
    plot.set_ylabel('')
    plot.legend(loc='upper left')
    
    # add annotation
    if len(annotation) != 0 :
        if place == annotation [0]:
            plot.annotate(annotation[1], annotation[2], annotation[3], arrowprops=dict(facecolor='red', shrink=0.05))
        
    # Legend.
    if place != 311: 
        plot.legend_.remove()
        
    if place == 312:
        plot.set_ylabel('time difference with the best runners in the team')
    
    if place == 313:
        plot.set_xlabel('Team number')
        
    plt.yticks(plot.get_yticks(), [convert_seconds_to_time(label) for label in plot.get_yticks()])
    display_legend(counter_pair, plot)
    
    
def display_information_speed(data):
    '''
    dsiplay data on the speed repartition between team/individuals runners.
    
    Parameters
        - data: DataFrame containing records for a given running
    '''
    distances = [10,21,42]
    type_runners = ['Individual runners','Runners in teams']
    
    for distance in distances: 
        median_distance = []
        lausanne_by_distance = data[data['distance (km)'] == distance]
        for type_runner in type_runners:      
            median_distance.append(np.median(lausanne_by_distance['Speed (m/s)'][
                                                                        lausanne_by_distance['type_team'] == type_runner]))
        print ('Median speed for '+ str(distance) +
               'Km race is for individual runners = ' + str(median_distance[0]) + ' m/s' +
               ' and for team runners = ' + str(median_distance[1]) +' m/s ')
    
    return

