# ----------------------------------------------------------------------------------------------------------
# Import

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import Counter
import sys
sys.path.append('..')
import study_utils

# ----------------------------------------------------------------------------------------------------------
# Constants

# Information about Canton of Vaud can be found on official government website
# https://www.bfs.admin.ch/bfs/fr/home/statistiques/population.assetdetail.1500543.html.
TOTAL_RESIDENT_VAUD = 778365
TOTAL_RESIDENT_MALE = 381864
TOTAL_RESIDENT_FEMALE = 396501

# ----------------------------------------------------------------------------------------------------------
# Functions

def plot_gender_distributions(df):
    '''
    This functions displays graph representing the gender distribution of Canton of Vaud and Lausanne Marathon 2016 for comparison.

    Parameters
        - df: DataFrame containing information on runners for Lausanne Marathon 2016
    '''

    # Building of DataFrame for ploting
    total_runners = len(df)
    total_runners_male = len(df[df['sex'] == 'male'])
    total_runners_female = len(df[df['sex'] == 'female'])
    vaud_information_population = pd.Series({ 'male': TOTAL_RESIDENT_MALE/TOTAL_RESIDENT_VAUD * 100, 'female': TOTAL_RESIDENT_FEMALE/TOTAL_RESIDENT_VAUD * 100 }) 
    marathon_information_runner = pd.Series({ 'male': total_runners_male/total_runners * 100, 'female': total_runners_female/total_runners * 100 }) 
    information_population = pd.DataFrame({ 'Canton of Vaud': vaud_information_population, 'Lausanne Marathon': marathon_information_runner })
    information_population.sort_index(axis=0, level=None, ascending=False, inplace=True)

    # Displaying data
    plot = information_population.plot.bar(figsize=(10,6), rot=0)
    plot.set_title('Gender distribution Lausanne Marathon vs Canton of Vaud')   
    plot.set_ylabel('Percentage (%)')
    plot.set_xlabel('Gender')

    # Add of annotations
    annotations = [
                    str(TOTAL_RESIDENT_MALE) + '\nresidents',
                    str(TOTAL_RESIDENT_FEMALE)+ '\nresidents',
                    str(total_runners_male) + ' runners',
                    str(total_runners_female) + ' runners'
                 ]
    
    index = 0
    for p in plot.patches:
        x_position = p.get_x()
        if index > 1 :
            x_position = x_position + 0.012
        plot.annotate(annotations[index], (x_position * 1.005, p.get_height() * 1.005))
        index = index + 1

    plot.legend(loc='upper left')

    # Add of box information on total runners/residents
    # Creation of string for displaying important information
    total_runners_str = 'Total residents: ' + str(TOTAL_RESIDENT_VAUD)
    total_residents_str = 'Total runners: ' + str(total_runners)
    stats = total_runners_str + ' \n' + total_residents_str 

    # Add of information in the graph
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plot.text(.95, .95, stats, fontsize=12, transform=plot.transAxes, va='top', ha='right', bbox=props, multialignment='left')


def plot_gender_distribution_according_to_running_type(df_10km, df_21km, df_42km):
    '''
    This function displays the gender distribution for the different runnings.

    Parameters
        - df_10km: DataFrame containing information of runners for 10 km running
        - df_21km: DataFrame containing information of runners for semi-marathon
        - df_42km: DataFrame containing information of runners for marathon
    '''

    # Building of DataFrame for ploting
    total_runners_10 = len(df_10km)
    total_runners_21 = len(df_21km)
    total_runners_42 = len(df_42km)

    race_information_10 = pd.Series({ 
                                        'male': len(df_10km[df_10km['sex'] == 'male']) ,
                                        'female': len(df_10km[df_10km['sex'] == 'female'])
                                    }) 

    race_information_21 = pd.Series({ 
                                        'male': len(df_21km[df_21km['sex'] == 'male']) ,
                                        'female': len(df_21km[df_21km['sex'] == 'female'])
                                    })

    race_information_42 = pd.Series({ 
                                        'male': len(df_42km[df_42km['sex'] == 'male']) ,
                                        'female': len(df_42km[df_42km['sex'] == 'female'])
                                    })


    information_gender_race = pd.DataFrame({
                                           'Marathon': race_information_42,
                                           'Semi-marathon': race_information_21,
                                           '10 km': race_information_10
                                          })

    information_gender_race = information_gender_race[['Marathon', 'Semi-marathon', '10 km']]

    # Displaying of data
    plot = information_gender_race.plot.bar(figsize=(10, 6), rot=0)
    plot.set_title('Gender distribution by distance')
    plot.set_ylabel('Number of runners')
    plot.set_xlabel('Gender')

    # Displaying of the percentage for each race
    totals = [total_runners_42, total_runners_21, total_runners_10]
    race_distance = 0
    index = 0

    # Loop (displaying percentages)
    for p in plot.patches:
            if race_distance == 2:
                race_distance = 0
                index = index + 1
            plot.annotate('{:.1f}%'.format(p.get_height()*100/totals[index]), (p.get_x(), p.get_height() + 10))
            race_distance = race_distance + 1
    plot.legend(loc='upper left')

    # Add of box information on total runners/residents
    # Creation of string for displaying important informations
    total_marathon = 'Marathon: ' + str(total_runners_42) + ' runners'
    total_semi_marathon = 'Semi-marathon: ' + str(total_runners_21) + ' runners'
    total_10 = '10 km: ' + str(total_runners_10) + ' runners'
    stats = total_marathon + ' \n' + total_semi_marathon + ' \n' + total_10

    # Add of information in the graph
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plot.text(.95, .95, stats, fontsize=12, transform=plot.transAxes, va='top', ha='left', bbox=props, multialignment='left')


def plot_distribution_between_adults_and_juniors(df):
    '''
    This functions display the distribution of runners between adult and juniors ones.

    Parameters
        - df: DataFrame containing information about runners
    '''

    plot = sns.countplot(data=df, x='type')
    plot.figure.set_size_inches(10,6)
    total = len(df)

    for p in plot.patches:
            plot.annotate('{:.1f}%'.format(p.get_height()*100/total), (p.get_x()+0.35, p.get_height()+50))

    # Manage of legends
    plot.set_xlabel('')
    plot.set_ylabel('Number of runners')
    plot.set_title('Distribution of runners (types)')


def plot_age_distribution(df):
    '''
    This function displays the distribution of runners according to their age.

    Parameters:
        - df: DataFrame containing information about runners.
    '''

    fig, ax = plt.subplots()
    fig.set_size_inches(10,6)
    ax.hist(df['age'], bins=30)

    # Computation of the mean of ages selected by gender
    mean_age_M = np.mean(df['age'][df['sex'] == 'male'])
    mean_age_W = np.mean(df['age'][df['sex'] == 'female'])
    mean_age_all = np.mean(df['age'])

    # Display of the median and titles
    ax.axvline(mean_age_all, 0, 1750, color='r', linestyle='--')
    ax.set_title('Age distribution of runners')
    ax.set_xlabel('Age')
    ax.set_ylabel('Number of runners')

    # Calculation of age distribution statistics by gender
    age_stats = 'Mean age: ' + str(round(mean_age_all, 2)) + ' years\n' + 'SD: ' + str(round(np.std(df['age']), 2)) 
    age_statsf = 'Mean age (female): ' + str(round(mean_age_M, 2)) + ' years\n' + 'SD: ' + str(round(np.std(df['age'][df['sex'] == 'female']), 2))                                                                       
    age_statsm = 'Mean age (male): ' + str(round(mean_age_W, 2)) + ' years\n' + 'SD: ' + str(round(np.std(df['age'][df['sex'] == 'male']), 2))
    age_stats = age_stats + '\n' + age_statsf + '\n' + age_statsm

    # Add of text
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(.95, .95, age_stats, fontsize=11, transform=ax.transAxes, va='top', ha='right', bbox=props, multialignment='left')


def plot_distribution_age_distance(fig, data, title, subplot_idx):
    '''
    This function plots the distribution of ages of runners according to a running type, and following sex of participants.

    Parameters
        - fig: Figure containing the subplots
        - data: DataFrame to use during generation of the distribution
        - title: Title of subplot
        - subplot_idx: Index of subplot
    '''

    ax  = fig.add_subplot(subplot_idx)
    ax.hist([data['age'][data['sex'] == 'male'], data['age'][data['sex'] == 'female']],
         bins=30,
         stacked=True,
         rwidth=1.0,
         label=['male', 'female'])

    plt.xticks(np.arange(10,100,10))
    
    # Manage of legends
    if subplot_idx == 313:
        ax.set_xlabel('Age')
    if subplot_idx == 312:
        ax.set_ylabel('Number of runners')

    ax.legend(loc='upper left')

    # Computation of the mean of ages selected by gender
    mean = np.mean(data['age'])

    # Calculation of age distribution statistics by gender
    age_stats = 'Mean age: ' + str(round(mean, 2)) + ' years\n' + 'SD: ' + str(round(np.std(data['age']), 2)) 
    age_stats = age_stats

    # Add of legend text
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(.95, .95, age_stats, fontsize=11, transform=ax.transAxes, va='top', ha='right', bbox=props, multialignment='left')

    # Display of the median and titles
    ax.axvline(mean, 0, 1750, color='r', linestyle='--')
    ax.set_title(title)

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
            formatted_label = study_utils.convert_seconds_to_time(int(float(label._y)))
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
    plt.xticks(ax.get_xticks(), [study_utils.convert_seconds_to_time(label) for label in ax.get_xticks()], rotation=90)
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
    plt.yticks(ax.get_yticks(), [study_utils.convert_seconds_to_time(label) for label in ax.get_yticks()])
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


def plot_distribution_team_individuals(df, total_10, total_21, total_42):
    '''
    This function display the number of runners that share a team and the number of participants who do the runnings individually.

    Parameters
        - df: DataFrame containing information on runners
        - total_10: Number of runners for the 10 km running
        - total_21: Number of runners for the semi-marathon
        - total_42: Number of runners for the marathon
    '''

    fig = plt.figure()
    fig.tight_layout
    fig.set_size_inches(10, 6)

    ax = fig.add_subplot(111)
    ax = sns.countplot(x='type_team', hue='distance (km)', data=df)
    ax.set_xlabel('')
    ax.set_ylabel('Number of Runners')
    ax.set_title('Team/indivual runners composition')

    totals = [total_10, total_21, total_42]
    race_type = 0
    index = 0

    # Loop (displaying of the percentages)
    for p in ax.patches:
            if race_type == 2:
                race_type = 0
                index = index + 1
            ax.annotate('{:.1f}%'.format(p.get_height()*100/totals[index]), (p.get_x()+0.05, p.get_height()+50))
            race_type = race_type + 1

def plot_time_difference_distribution(data):
    '''
    # TODO: Add description of the function
    '''

    ax = data['time difference team'].hist(bins=30,figsize=(10,6))

    # Computing of the mean of ages selected by gender # TODO: check comment
    mean = np.mean(data['time difference team'])
    max_time_diff = np.max(data['time difference team'])

    # Display of the median and title
    ax.axvline(mean, 0, 1750, color='r', linestyle='--')
    ax.set_title('Time difference with best runner in the team distribution')

    # Display of x ticks in HH:mm:ss format
    plt.xticks(ax.get_xticks(), [study_utils.convert_seconds_to_time(label) for label in ax.get_xticks()])

    # Calculation and display of age distribution statistics by gender
    time_stats = 'Mean time difference : ' + study_utils.convert_seconds_to_time(mean) + '\n' + 'Max time difference : ' + study_utils.convert_seconds_to_time(max_time_diff)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(.95, .95, time_stats, fontsize=11, transform=ax.transAxes, va='top', ha='right', bbox=props, multialignment='left')

    
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
    # TODO: Check description
    This function plots the difference between time according to the different groups.

    Parameters
        - fig: Figure on which subplots are displayed
        - data: DataFrame containing the data relative to a given running
        - distance: number of kilometers of the considered running (10/21/42)
        - subplot_idx: Index of the subplot in the figure
        - annotation: Annotation to add in the graph (by default, no annotation)
        - time_mini: Minimal time to consider (1000 by default)
    '''

    # Map string name of the team to # TODO: Check comment
    race_team = data[data['team'].notnull()].copy()
    race_team = race_team[race_team["distance (km)"] == distance]

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
    
    
def display_information_speed(data):
    '''
    This function generates medians of the speed's distribution for runners in team and individual ones.
    
    Parameters
        - data: DataFrame containing records for a given running

    Return
        - string containing medians
    '''

    distances = [10, 21, 42]
    type_runners = ['Individual runner', 'Runner in teams']
    
    for distance in distances: 
        median_distance = []
        lausanne_by_distance = data[data['distance (km)'] == distance]
        for type_runner in type_runners:      
            median_distance.append(np.median(lausanne_by_distance['Speed (m/s)'][lausanne_by_distance['type_team'] == type_runner]))
        
        return 'Median speed for ' + str(distance) + \
               'Km race is for individual runners = ' + \
               str(median_distance[0]) + ' m/s' + \
               ' and for team runners = ' + \
               str(median_distance[1]) +' m/s'
