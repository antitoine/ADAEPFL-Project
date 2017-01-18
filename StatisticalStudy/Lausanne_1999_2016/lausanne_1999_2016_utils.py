# ----------------------------------------------------------------------------------------------------------
# Imports

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import collections
import sys
sys.path.append('..')
import study_utils

# ----------------------------------------------------------------------------------------------------------
# Constants

RUNNINGS_KM = {'10 km': 10, 'Semi-marathon': 21, 'Marathon': 42}

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


def remove_outliers(df):
    '''
    The method removes outliers present in the data. More precisely, for each category, the method removes all times which are 
    smaller than the best runner in the considered category.
    
    Parameters
        - df : DataFrame containing records for a given running
    '''
    
        
    # We remove the SettingWithCopyWarning
    pd.options.mode.chained_assignment = None 
    
    # We remove runners who abandonned
    df = df[~(df['rank'].isin(['DNF', 'OUT']))]
    
    # We convert rank values to float
    df['rank'] = df['rank'].apply(lambda x : int(float(x)))
    
    all_races = []

    # Loop over the years
    for year in df['year'].unique():
        total_remove = 0
        all_cate = []
        
        # We select current year
        year_selected = df[df['year'] == year]
        
        # Loop over categories
        for category in df['category'].unique():
            
            # We select current category
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

            # Handle specific problems due to late withdrawn runners
            without_outliers.sort_values('time', ascending=True, inplace=True)
            rank_list = without_outliers['rank'].tolist()
            
            # We remove outliers if the rank is not stricly increasing
            if len(rank_list) > 1:
                # Test if the list is strictly increasing or not
                if not (all(x < y for x, y in zip(rank_list, rank_list[1:]))):
                    # We remove additional outliers
                    without_outliers = without_outliers[remove_outliers_in_increasing_series(rank_list)]
              
            # Compute numbers rank in
            total_remove = total_remove + (len(category_selection.index) - len(without_outliers.index))
            all_cate.append(without_outliers)

        print('Number of outliers removed for ' + str(year) + ': ' + str(total_remove) + ' runners')
        
        if len(all_cate) == 0 : 
            continue
        all_races.append(pd.concat(all_cate))  
        
                
    # We remove the SettingWithCopyWarning
    pd.options.mode.chained_assignment = 'warn'
                              
    return pd.concat(all_races)


def remove_outliers_in_increasing_series(list_rank):
    '''
    This function allows to find values at the wrong positon in order to have a list with stricly increasing values.
    
    Parameters
        - list_rank: List of ranks of runners to clean
        
    Return
        ordered: List filled with boolean values (True/False) for each rank. False means that associated rank is greater than the following (outlier)
    '''
    
    ordered = []
    
    for idx, rank in enumerate(list_rank):
        # We mark values with False-value boolean when they are not strictly inferior to the previous ones
        if idx < (len(list_rank)-1):
            # Current value is smaller than the following one: ok
            if(list_rank[idx] < list_rank[idx + 1]):
                ordered.append(True)
            # Value is greater than the following one: error (outlier)
            else:
                ordered.append(False) 

    ordered.append(True)
    return ordered


def filter_by_years(data, series):
    '''
    This function iterates over a set of data and creates series by filtering by years.

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
    This function iterates over a set of data and creates series by filtering by years and sex of runners.

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

    # Creation of subgraphes
    i = 1
    figure = plt.figure(figsize=(15, 5))
    for sex, df in data.items():
        df = df[runnings]
        axe = plt.subplot(1, 2, i)
        axe = df.plot(kind="bar", linewidth=0, stacked=True, ax=axe, grid=False)
        axe.set_title('Distribution of ' + sex + ' runners over the years')
        axe.set_xlabel('Years')
        axe.set_ylabel('Number of runners')
        i += 1
    plt.show()


def generate_median_age_statistics(df):
    '''
    This function generates median age statistics for Lausanne Marathon.

    Parameters:
        - df: DataFrame containing all the information about runners

    Return
        - statistics: Global and detailed results by gender for median age of runners
    '''

    data = {'10km': df[df['distance (km)'] == 10], '21km': df[df['distance (km)'] == 21], '42km': df[df['distance (km)'] == 42], 'All runnings': df}
    series, series_with_gender = [{key: {} for key in data} for i in range(1, 3)]
    for year in range(1999, 2017):
        for df_name, df in data.items():
            series[df_name][str(year)] = df[(df['year'] == year)]['age'].median()
            for sex in ['female', 'male']:
                series_with_gender[df_name][(sex, str(year))] = df[(df['year'] == year) & (df['sex'] == sex)]['age'].median()
    statistics = {}
    for stats, results in {'global': series, 'detailed': series_with_gender}.items():
        statistics[stats] = pd.DataFrame(collections.OrderedDict([('Median age (10 km)', pd.Series(results['10km'])), ('Median age (semi-marathon)', pd.Series(results['21km'])), ('Median age (marathon)', pd.Series(results['42km'])), ('Median age (all runnings)', pd.Series(results['All runnings']))]))
    return statistics


def plot_median_age_evolution(data, x=None, y='Median age (all runnings)', groupby_column='Gender', title='Evolution of median age over the years'):
    '''
    This function displays a graph showing evolution of median ages for male and female runners over the years.

    Parameters
        - data: DataFrame containing data to use for graph
        - x: Name of the column to use for x axis (by default None / if None, index will be used)
        - y: Name of the column to use for y axis (by default, 'Median age (all runnings)')
        - groupby_column: Name of the column to use for grouping data (by default, 'Gender')
        - title: Title of the graph (by default, 'Evolution of median age over the years')
    '''

    fig, ax = plt.subplots()
    labels = []
    for key, group in data.groupby(['Gender']):
        if not x:
            x_axis = x if x else group.index
        ax = group.plot(ax=ax, kind='line', x=x_axis, y=y)
        labels.append(key + ' runners')
    lines, _ = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='best')
    ax.set_title(title)
    ax.set_ylabel('Mean age')
    plt.show()


def update_plotly_figure_according_to_parameters(figure, df, age_category, performance_criterion):
    '''
    This function updates Plotly figure according to selected age category.

    Parameters
        - figure: Plotly figure to update
        - df: DataFrame containing information on runners
        - age_category: String representing selected age category
        - performance_criterion: String representing selected performance criterion

    Return
        - figure: Updated Plotly figure
    '''
    
    figure_data = figure['data']
    
    if age_category == 'All':
        data = df
    else:
        data = df[df['age category'] == age_category]
    
    for d in figure_data:
        filtered_data = data[data['distance (km)'] == RUNNINGS_KM[d['name']]]
        current_x = filtered_data['year']
        current_y = filtered_data[performance_criterion.lower()]
        d['x'] = current_x
        d['y'] = current_y
    
    figure['data'] = figure_data

    return figure

def generate_all_performance_figures(df, age_categories, performance_criteria):
    '''
    This function generates all performance figures according a set of age categories and a set of performance criteria.

    Parameters
        - df: DataFrame containing records about runners
        - age_categories: Array containing age categories to be displayed
        - performance_criteria: Array containing performance criteria to consider

    Return
        - figures: Dict containing all performance figures
    '''

    # We define the considered runnings and the years interval
    runnings = {10: '10 km', 21: 'Semi-marathon', 42: 'Marathon'}
    year_values = [v for v in df['year'].unique() if v]
    
    # We define options and the final Dict
    figures = {}
    default_options = {'title': 'Performance over years for runnings of Lausanne Marathon', 'x_name': 'Years', 'x_values': year_values}
    time_options = {'y_name': 'Time', 'y_type': 'date', 'y_format': '%H:%M:%S'}
    speed_options = {'y_name': 'Speed (m/s)'}
    time_options.update(default_options)
    speed_options.update(default_options)

    for age_category in age_categories:
        # We select data according to age category
        if age_category == 'All':
            data = df
        else:
            data = df[df['age category'] == age_category]
        figures[age_category] = {}

        # We create a figure for each performance criterion
        for performance_criterion in performance_criteria:
            criterion = performance_criterion.lower()
            boxplots = study_utils.create_plotly_boxplots(data=data, x='year', y=criterion, hue='distance (km)', hue_names=runnings)
            if criterion == 'time':
                figure = study_utils.create_plotly_legends_and_layout(data=boxplots, **time_options)
            elif criterion == 'speed (m/s)':
                figure = study_utils.create_plotly_legends_and_layout(data=boxplots, **speed_options)
            else:
                # By default, two specific criteria are allowed: 'time' and 'speed (m/s)'. If any other criterion is provided, we throw an exception.
                raise ValueError('Invalid performance criterion encountered. Performance criterion must be either \'Time\' or \'Speed (m/s)\'')
            figures[age_category][performance_criterion] = figure

    return figures


def generate_all_bib_performance_figure(df):
    '''
    This function generates all BIB/performance scatters for each year of Lausanne Marathon.

    Parameters
        - df: DataFrame containing records about runners

    Return
        - figure: Dict containing all BIB/performance figures
    '''

    # We define the considered the years interval, colors and visibility
    years_range = range(1999, 2017)
    years = {year: str(year) for year in years_range}
    colors = {str(year): 'hsl(' + str(np.linspace(0, 360, len(years_range))[index]) + ', 50%' + ', 50%)' for index, year in enumerate(years_range)}
    visibility = {str(year): (True if year > 2015 else 'legendonly') for year in years_range}

    # We define options and the final Dict
    figures = {}
    default_options = {'title': 'Distribution of performance according to BIB numbers over the years', 'x_name': 'BIB numbers', 'hovermode':'closest'}
    time_options = {'y_name': 'Time', 'y_type': 'date', 'y_format': '%H:%M'}
    time_options.update(default_options)

    scatters = study_utils.create_plotly_scatters(data=df, x='number', y='time', hue='year', hue_names=years, text='name', color=colors, visibility=visibility)
    figure = study_utils.create_plotly_legends_and_layout(data=scatters, **time_options)

    return figure