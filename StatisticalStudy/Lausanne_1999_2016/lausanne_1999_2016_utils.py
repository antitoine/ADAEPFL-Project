# ----------------------------------------------------------------------------------------------------------
# Imports

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ----------------------------------------------------------------------------------------------------------
# Functions

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
    genders = ['male', 'female']

    # List containing each dataframe by distance
    all_years = []
    
    for year in data['year'].unique():
        all_races = []

        # Year selection
        year_selection = data[data['year'] == year]
        
        # Loop on distance
        for distance in distances:
            all_gender = []

            # Distance selction
            distance_selection  = year_selection[year_selection['distance (km)'] == distance]

            for gender in genders:
                # Sex Selection
                gender_selection = distance_selection[distance_selection['sex'] == gender]

                # Sorting by gender
                gender_selection.sort_values('time', ascending=True, inplace=True)

                # Computation of the overall rank for the running
                gender_selection['overall_rank'] = range (1, len(gender_selection)+1)

                # We append result
                all_gender.append(gender_selection)

            all_races.append(pd.concat(all_gender))
            
        all_years.append(pd.concat(all_races))
       
    # We remove the SettingWithCopyWarning
    pd.options.mode.chained_assignment = 'warn' 
    
    return pd.concat(all_years)


def remove_outliers(data):
    '''
    The method removes outliers present in the data. More precisely, for each category, the method removes all times which are smaller than the best runner in the considered category.
    Nota-bene: Remove of such outliers is necessary as certain people have resigned after the first loop, while their time was still considered.
    
    Parameters
        - data: DataFrame containing records for a given running
    '''
    
    # remove resigners runners.
    data = data[~(data['rank'].isin(['DNF', 'OUT']))]
    
    # convert to float.
    data['rank'] = data['rank'].apply(lambda x : int(float(x)))
    
    all_races = []

    # Loop over years
    for year in data['year'].unique():
        total_remove = 0
        all_cate = []
        
        # We select year
        year_selected = data[data['year'] == year]
        
        # Loop over categories
        for category in data['category'].unique():
            # We select by category
            category_selection = year_selected[year_selected['category'] == category]
            
            # We retrieve best time of the category
            best_time = (category_selection['time']
                                 [(category_selection['category'] == category) & (category_selection['rank'] == 1)])
            
            # There is no person first ranked in this category
            if best_time.empty:
                best_time = np.min(category_selection['time'][category_selection['category'] == category])
            else:
                best_time = best_time.values[0]
                
            # We remove all times smaller than the best time of the category
            without_outliers = category_selection[(category_selection['time'] >= best_time )] 
            
            # Compute numbers rank in
            total_remove = total_remove + (len(category_selection.index) - len(without_outliers.index))
            
            all_cate.append(without_outliers)

        print('Number of outliers removed for ' + str(year) + ': ' + str(total_remove) + ' runners')
        
        if len(all_cate) == 0 : 
            continue
        all_races.append(pd.concat(all_cate))   
                              
    return pd.concat(all_races)


def filter_by_years(data, series):
    '''
    This function iterates over a set of data and creates series by filtering by years

    Parameters
        - data: DataFrame containing information about runners
        - series: Dictionary to fill

    Return
        - series: Dictionary containing sets of data separated according to filter
    '''

    for year in range(1999, 2017):
        for df_name, df in data.items():
            series[df_name][str(year)] = len(df[df['year'] == year])
    return series


def filter_by_sex_and_years(data, series):
    '''
    This function iterates over a set of data and creates series by filtering by years and sex of runners

    Parameters
        - data: DataFrame containing information about runners
        - series: Dictionary to fill

    Return
        - series: Dictionary containing sets of data separated according to filter
    '''

    for year in range(1999, 2017):
        for sex in ['female', 'male']:
            for df_name, df in data.items():
                series[df_name][sex, str(year)] = len(df[(df['year'] == year) & (df['sex'] == sex)])
    return series


def generate_distributions(df_10km, df_21km, df_42km, filter):
    '''
    This function generates distributions for the different runnings according to a given filter.

    Parameters
        - df_10km: DataFrame containing all the runners over the years for the 10 km running
        - df_21km: DataFrame containing all the runners over the years for the semi-marathon
        - df_42km: DataFrame containing all the runners over the years for the marathon
        - filter: Filter to apply in order to generate distributions (function)

    Return
        - DataFrame containing gender distributions over the years
    '''

    data = {'10km': df_10km, '21km': df_21km, '42km': df_42km}
    series = {key: {} for key in data}
    series = filter(data, series)
    return pd.DataFrame({'Marathon': pd.Series(series['42km']), 'Semi-marathon': pd.Series(series['21km']), '10 km': pd.Series(series['10km'])})


def plot_distribution_over_years(data, title='Distributions of runners over the years for Lausanne Marathon'):
    '''
    This function generates a graph representing distribution of runner over the years, given data.

    Parameters
        - data: DataFrame containing distribution of runners over the years
        - title: Title of the graph (by default, 'Distributions of runners over the years for Lausanne Marathon')

    Return
        - axe: Matplotlib axes
    '''

    figure = plt.figure(figsize=(15, 5))
    axe = plt.subplot(111)
    axe = data.plot(kind="bar", rot=0, linewidth=0, ax=axe, stacked=True, grid=False)
    axe.set_title(title)
    axe.set_xlabel('Years')
    axe.set_ylabel('Number of runners')
    return axe


def plot_gender_distributions_over_years(data, runnings=['Marathon', '10 km', 'Semi-marathon'], years=range(1999, 2017), title="Gender distribution over the years for different runnings of Lausanne Marathon"):
    '''
    This function generates a graph representing gender distributions over the years, given a data set.
    The following code was adapted from jrjc's code: http://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas

    Parameters
        - data: Array containing all DataFrames to consider (one by gender)
        - runnings: Array of unnings to be considered and available in data (by default, 'Marathon', '10 km', 'Semi-marathon')
        - years: Years to be considered and available in data (by default, 1999 to 2016)
        - title: Title of the graph (by default, 'Gender distribution over the years for different runnings of Lausanne Marathon')

    Return
        - axe: Matplotlib axes
    '''

    nb_df = len(data)
    nb_runnings = len(runnings)
    nb_years = len(years)
    figure = plt.figure(figsize=(15, 5))
    axe = plt.subplot(111)

    for sex, df in data.items():
        axe = df.plot(kind="bar", linewidth=0, stacked=True, ax=axe, legend=False, grid=False)


    handles, labels = axe.get_legend_handles_labels()
    for i in range(0, nb_df * nb_runnings, nb_runnings):
        for j, pa in enumerate(handles[i:i+nb_runnings]):
            # Hack: we add other rectangles
            for rect in pa.patches:
                rect.set_x(rect.get_x() + 1 / float(nb_df + 1) * i / float(nb_runnings))   
                rect.set_width(1 / float(nb_df + 1))

    # We add legends and title
    axe.set_xticks((np.arange(0, 2 * nb_years, 2) + 1 / float(nb_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_ylabel('Number of runners')
    axe.set_title(title)
    axe.legend(handles[:nb_runnings], labels[:nb_runnings], loc='upper left')
    
    # For x axis, we manually add sex in labels
    labels = []
    for label in axe.get_xticklabels():
        formatted_label = "$\u2640$     $\u2642$\n" + label._text
        labels.append(formatted_label)
    axe.set_xticklabels(labels)

    plt.show()
    
    i = 1
    figure = plt.figure(figsize=(15, 5))
    for sex, df in data.items():
        df = df[runnings]
        axe = plt.subplot(1, 2, i)
        axe = df.plot(kind="bar", linewidth=0, stacked=True, ax=axe, grid=False)
        axe.set_title('Distribution of ' + sex + ' runners over the years by runnings')
        axe.set_xlabel('Years')
        axe.set_ylabel('Number of runners')
        i += 1
    
    plt.show()
