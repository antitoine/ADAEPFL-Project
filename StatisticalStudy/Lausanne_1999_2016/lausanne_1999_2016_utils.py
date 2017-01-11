# ----------------------------------------------------------------------------------------------------------
# Imports

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ----------------------------------------------------------------------------------------------------------
# Functions

def compute_overall_rank_all(df):
    '''
    This function computes the overall rank by distance for all years.
    
    Parameters
        - df: DataFrame containing records

    Return
        - DataFrame with overall rank calculated for all the years
    '''
    
    years = df['year'].unique()
    df_with_overall_rank = []
    
    for year in years:
        marathon_current_year = df[df['year'] == year]
        df_with_overall_rank.append(compute_overall_rank(marathon_current_year))
    
    return pd.concat(df_with_overall_rank)


def compute_overall_rank(data):
    '''
    This function computes the overall rank for a given running.
    
    Parameters
        - data: DataFrame containing records for a given running
    '''
    
    # We remove the SettingWithCopyWarning
    pd.options.mode.chained_assignment = None 
    
    # Definition of the discriminators for the selection
    distances = [42,21,10]
    type_runners =  ['Individual runners', 'Runner in teams']
    genders = ['male', 'female']

    # List containing each dataframe by distance
    all_races = []
    
    # Loop on distance
    for distance in distances:
        all_type = []
        distance_selection  = data[data['distance (km)'] == distance]
        
        # Loop on runner type
        for type_runner in type_runners:
            all_gender = []
            type_selection = distance_selection[distance_selection['type_team'] == type_runner]
            
            # Loop on gender
            for gender in genders:
                gender_selection = type_selection[type_selection['sex'] == gender]
                
                # Sorting by gender
                gender_selection.sort_values('time', ascending=True, inplace=True)
                
                # Computation of the overall rank for the running
                gender_selection['overall_rank'] = range (1, len(gender_selection)+1)
                
                # We append result
                all_gender.append(gender_selection)
           
            # We add results for all genders
            all_type.append(pd.concat(all_gender))
        
        # We add results for all types
        all_races.append(pd.concat(all_type))
    
    return pd.concat(all_races)


def generate_gender_distributions_over_years(df_10km, df_21km, df_42km):
    '''
    This function generates the gender distributions for the different runnings over the years.

    Parameters
        - df_10km: DataFrame containing all the runners over the years for the 10 km running
        - df_21km: DataFrame containing all the runners over the years for the semi-marathon
        - df_42km: DataFrame containing all the runners over the years for the marathon

    Return
        DataFrame containing gender distributions over the years
    '''

    data = {'10km': df_10km, '21km': df_21km, '42km': df_42km}
    series = {key: {} for key in data}
    for year in range(1999, 2017):
        for sex in ['female', 'male']:
            for df_name, df in data.items():
                series[df_name][(str(year), sex)] = len(df[(df['year'] == year) & (df['sex'] == sex)]) 
    return pd.DataFrame({'Marathon': pd.Series(series['42km']), 'Semi-marathon': pd.Series(series['21km']), '10 km': pd.Series(series['10km'])})


def plot_gender_distributions_over_years(data, title="Gender distribution over the years for different runnings of Lausanne Marathon"):
    '''
    This function generates a graph representing gender distributions over the years, given a data set.
    The following code was adapted from jrjc's code: http://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas

    Parameters
        - data: Array containing all DataFrames to consider (one by gender)
        - title: Title of the graph (by default, 'Gender distribution over the years for different runnings of Lausanne Marathon')

    Return
        - axe: Matplotlib axes
    '''

    nb_df = len(data)
    runnings = len(data[0].columns)
    years = len(data[0].index) # Années
    figure = plt.figure(figsize=(15, 5))
    axe = plt.subplot(111)

    for df in data : # for each data frame
        axe = df.plot(kind="bar", linewidth=0, stacked=True, ax=axe, legend=False, grid=False)


    handles, labels = axe.get_legend_handles_labels()
    for i in range(0, nb_df * runnings, runnings):
        for j, pa in enumerate(handles[i:i+runnings]):
            # Hack: we add other rectangles
            for rect in pa.patches:
                rect.set_x(rect.get_x() + 1 / float(nb_df + 1) * i / float(runnings))   
                rect.set_width(1 / float(nb_df + 1))

    # We add legends and title
    axe.set_xticks((np.arange(0, 2 * years, 2) + 1 / float(nb_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_ylabel('Number of runners')
    axe.set_title(title)
    axe.legend(handles[:runnings], labels[:runnings], loc='upper left')
    
    # For x axis, we manually add sex in labels
    labels = []
    for label in axe.get_xticklabels():
        formatted_label = "$\u2640$     $\u2642$\n" + label._text
        labels.append(formatted_label)
    axe.set_xticklabels(labels)
    return axe