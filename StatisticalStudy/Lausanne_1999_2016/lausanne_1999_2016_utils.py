# ----------------------------------------------------------------------------------------------------------
# Imports

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import copy
import sys
sys.path.append('..')
import study_utils
import plotly
from collections import OrderedDict
from plotly import tools
import plotly.graph_objs as go
from plotly.graph_objs import Annotation

# ----------------------------------------------------------------------------------------------------------
# Constants

KM_10_COLOR = '#f99740'
KM_21_COLOR = '#40aff9'
KM_42_COLOR = '#f94040'

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


def plot_distribution_over_years(data, title='Distribution of runners over the years for Lausanne Marathon'):
    '''
    This function generates a graph representing distribution of runners over the years, given data.

    Parameters
        - data: DataFrame containing distribution of runners over the years
        - title: Title of the graph (by default, 'Distribution of runners over the years for Lausanne Marathon')

    Return
        - figure: Plotly figure
    '''

    colors = {'10 km': KM_10_COLOR, 'Semi-marathon': KM_21_COLOR, 'Marathon': KM_42_COLOR}
    bars = []

    for running in data.columns:
        bars.append(go.Bar(x=[year for year in data.index], y=data[running], name=running, marker={'color': colors[running]}))

    figure = study_utils.create_plotly_legends_and_layout(bars, title=title, x_name='Years', y_name='Number of runners', barmode='stack')
    plotly.offline.iplot(figure)
    return figure


def plot_gender_distributions_over_years(data, title='Gender distribution over the years for the different runnings of Lausanne Marathon'):
    '''
    This function generates a graph representing gender distributions over the years, given a data set.

    Parameters
        - data: Array containing all DataFrames to consider (one by gender)
        - title: Title of the graph (by default, 'Gender distribution over the years for different runnings of Lausanne Marathon')

    Return
        - figure: Plotly figure
    '''

    colors = {'10 km': KM_10_COLOR, 'Semi-marathon': KM_21_COLOR, 'Marathon': KM_42_COLOR}

    # We ignore outputs of Plotly
    with study_utils.ignore_stdout():
        figure = tools.make_subplots(rows=1, cols=2, subplot_titles=([(sex.capitalize() + ' runners') for sex in data]))

    i = 1
    for sex, df in data.items():
        bars = []

        for running in df.columns:
            figure.append_trace(go.Bar(x=[year for year in df.index], y=df[running], name=running, marker={'color': colors[running]}, legendgroup=running, showlegend=(i == 1)), 1, i)

        i += 1

    figure['layout'].update(title=title, barmode='stack')
    plotly.offline.iplot(figure)
    return figure


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
        statistics[stats] = pd.DataFrame(OrderedDict([('Median age (10 km)', pd.Series(results['10km'])), ('Median age (semi-marathon)', pd.Series(results['21km'])), ('Median age (marathon)', pd.Series(results['42km'])), ('Median age (all runnings)', pd.Series(results['All runnings']))]))
    return statistics


def plot_median_age_evolution(data, x=None, y='Median age (all runnings)', title='Evolution of median age over the years', groupby_column='Gender', groupby_attributes=None):
    '''
    This function displays a graph showing evolution of median ages for male and female runners over the years.

    Parameters
        - data: DataFrame containing data to use for graph
        - x: Name of the column to use for x axis (by default None / if None, index will be used)
        - y: Name of the column to use for y axis (by default, 'Median age (all runnings)')
        - title: Title of the graph (by default, 'Evolution of median age over the years')
        - groupby_column: Name of the column to use for grouping data (by default, 'Gender')
        - groupby_attributes: Dictionary containing options for each unique value in column groupby_column (at present, 'colors' and 'name' are supported / by default, None)
    
    Return
        - figure: Plotly figure
    '''

    lines = []

    for key, group in data.groupby([groupby_column]):
        x_values = group[x] if x else group.index
        line = go.Scatter(x=x_values, y=group[y], mode='lines', name=(groupby_attributes[key].get('name', key) if groupby_attributes else key), marker={'color': (groupby_attributes[key].get('color', None) if groupby_attributes else None)})
        lines.append(line)

    figure = study_utils.create_plotly_legends_and_layout(lines, title=title, x_name=(x if x else data.index.name), y_name=y)
    plotly.offline.iplot(figure)
    return figure


def plot_ols_fitted_and_true_values(data, ols_results, x=None, y='Median age (all runnings)', title='Fitted and original values for median ages of participants of Lausanne Marathon editions', groupby_column='Gender', markers_attributes=None):
    '''
    This function plots fitted and true values for a given dataset and associated ols results.

    Parameters
        - data: DataFrame containing data to use for original values
        - x: Name of the column to use for x axis for original values and fitted values (by default None / if None, index will be used for original values and index name for retrieve_ols_predictions_and_errors)
        - y: Name of the column to use for y axis (by default, 'Median age (all runnings)')
        - title: Title of the graph (by default, 'Fitted and original values for median ages of participants of Lausanne Marathon editions')
        - groupby_column: Name of the column to use for grouping data (by default, 'Gender')
        - markers_attributes: Dictionary containing options for each unique value in column groupby_column (by default, None)
        (at present, 'title' (title of subplot), 'color_true' (color of markers of original values), 'color_fitted' (color of markers of fitted values), 'color_errors' (color of errors bars), 'name_true' (name of markers' legend of original values) and 'name_fitted' (name of of markers' legend of fitted values) are supported)
    
    Return
        - figure: Plotly figure
    '''

    position = {}

    predictions_and_errors = study_utils.retrieve_ols_predictions_and_errors(ols_results=ols_results, regressor=(x if x else data.index.name))

    # We ignore outputs of Plotly
    with study_utils.ignore_stdout():
        figure = tools.make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=([(markers_attributes[key].get('title', None) if markers_attributes else None) for key in predictions_and_errors]))

    i = 1
    for key, results in predictions_and_errors.items():
        position[key] = i
        name=(markers_attributes[key].get('name_fitted', 'Fitted values') if markers_attributes else 'Fitted values')
        errors = {'type': 'data', 'symmetric': False, 'array': np.subtract(np.array(results['errors']['max']), np.array(results['predictions'])), 'arrayminus': abs(np.subtract(np.array(results['errors']['min']), np.array(results['predictions']))), 'color': (markers_attributes[key].get('color_errors', None) if markers_attributes else None)}
        markers = go.Scatter(x=results['x'], y=results['predictions'], mode='markers', hoverinfo='y+text', name=name, text=name, marker={'color': (markers_attributes[key].get('color_fitted', None) if markers_attributes else None)}, legendgroup='fitted', error_y=errors, showlegend=(position[key] == 1))
        figure.append_trace(markers, 1, position[key])
        i += 1

    for key, group in data.groupby([groupby_column]):
        x_values = group[x] if x else group.index
        name = (markers_attributes[key].get('name_true', 'Original values (' + key + ')') if markers_attributes else 'Original values (' + key + ')')
        markers = go.Scatter(x=x_values, y=group[y], mode='markers', hoverinfo='y+text', name=name, text=name, marker={'color': (markers_attributes[key].get('color_true', None) if markers_attributes else None)})
        figure.append_trace(markers, 1, position[key])

    # Add of title and format of axes
    figure['layout'].update(title=title)
    figure['layout']['yaxis'].update(title=y)
    for axis, attributes in {k: v for k, v in figure['layout'].items() if 'xaxis' in k}.items():
        figure['layout'][axis].update(title=(x if x else data.index.name))
    
    plotly.offline.iplot(figure)
    return figure


def generate_all_evolution_figures(df, age_categories, sex_categories):
    '''
    This function generates all evolution figures according sets of age categories and sex categories.
    Final Dict has the following pattern:
    {
        <age_category_1: {
            <sex_1>: <Plotly figure
            [, <sex_2>: <Plotly figure, ...]
        }
        [, <age_category_2: {
            <sex_1>: <Plotly figure
            [, <sex_2>: <Plotly figure, ...]
        }, ...]
    }

    Parameters
        - df: DataFrame containing records about runners
        - age_categories: Array containing age categories to be displayed (if 'All'/'all', no filter is done on df)
        - sex_categories: Array containing sex categories to be displayed (if 'All'/'all', no filter is done on df)

    Return
        - figures: Dict containing all evolution figures
    '''

    # We define the considered runnings and the years interval
    runnings = {'column_name': 'distance (km)', 'values': OrderedDict([(10, {'name': '10 km', 'color': KM_10_COLOR}), (21, {'name': 'Semi-marathon', 'color': KM_21_COLOR}), (42, {'name': 'Marathon', 'color': KM_42_COLOR})])}

    year_values = [year for year in df['year'].unique() if year]
    
    # We define options and the final Dict
    figures = {}
    options = {'title': 'Evolution of number of participants over years for runnings of Lausanne Marathon', 'x_name': 'Years', 'x_values': year_values, 'y_format': 'f'}

    for age_category in age_categories:
        # We select data according to age category
        if age_category.lower() == 'all':
            data = df
        else:
            data = df[df['age category'] == age_category]
        figures[age_category] = {}

        for sex_category in sex_categories:
            # We select data according to sex category
            if sex_category.lower() == 'all':
                data_final = data
            else:
                data_final = data[data['sex'] == sex_category.lower()]

            lines = []

            for km, attributes in runnings['values'].items():
                data_running = data_final[data_final[runnings['column_name']] == km]
                line = go.Scatter(x = year_values, y = [len(data_running[data_running['year'] == y]) for y in year_values], mode = 'lines', name = attributes['name'], marker={'color': attributes['color']})
                lines.append(line)

            annotations = [Annotation(y=1.1, text='Age category: ' + age_category + '    Sex category: ' + sex_category + ' runners', xref='paper', yref='paper', showarrow=False)]
            
            figure = study_utils.create_plotly_legends_and_layout(data=lines, **options, annotations=annotations)
            figures[age_category][sex_category] = figure

    return figures


def generate_all_performance_figures(df, age_categories, sex_categories, performance_criteria):
    '''
    This function generates all performance figures according sets of age categories and sex categories and a set of performance criteria.
    Final Dict has the following pattern:
    {
        <age_category_1: {
            <sex_1>: {
                <performance_criterion_1>: <Plotly figure>
                [, <performance_criterion_2>: <Plotly figure>
                , ...]
            }
            [, <sex_2>: {
                <performance_criterion_1>: <Plotly figure>
                [, <performance_criterion_2>: <Plotly figure>
                , ...]
            }, ...]   
        }
        [, <age_category_2: {
            <sex_1>: {
                <performance_criterion_1>: <Plotly figure>
                [, <performance_criterion_2>: <Plotly figure>
                , ...]
            }
            [, <sex_2>: {
                <performance_criterion_1>: <Plotly figure>
                [, <performance_criterion_2>: <Plotly figure>
                , ...]
            }, ...]   
        }, ...]
    }

    Parameters
        - df: DataFrame containing records about runners
        - age_categories: Array containing age categories to be displayed (if 'All'/'all', no filter is done on df)
        - sex_categories: Array containing sex categories to be displayed (if 'All'/'all', no filter is done on df)
        - performance_criteria: Array containing performance criteria to consider

    Return
        - figures: Dict containing all performance figures
    '''

    # We define the considered runnings and the years interval, as colors for boxplots
    runnings = {10: '10 km', 21: 'Semi-marathon', 42: 'Marathon'}
    year_values = [year for year in df['year'].unique() if year]
    colors = {'10 km': KM_10_COLOR, 'Semi-marathon': KM_21_COLOR, 'Marathon': KM_42_COLOR}
    
    # We define options and the final Dict
    figures = {}
    default_options = {'title': 'Performance over years for runnings of Lausanne Marathon', 'x_name': 'Years', 'x_values': year_values, 'boxmode': 'group'}
    time_options = {'y_name': 'Time', 'y_type': 'date', 'y_format': '%H:%M:%S'}
    speed_options = {'y_name': 'Speed (m/s)'}
    time_options.update(default_options)
    speed_options.update(default_options)

    for age_category in age_categories:
        # We select data according to age category
        if age_category.lower() == 'all':
            data = df
        else:
            data = df[df['age category'] == age_category]
        figures[age_category] = {}

        for sex_category in sex_categories:
            # We select data according to sex category
            if sex_category.lower() == 'all':
                data_final = data
            else:
                data_final = data[data['sex'] == sex_category.lower()]
            figures[age_category][sex_category] = {}

            annotations = [Annotation(y=1.1, text='Age category: ' + age_category + '    Sex category: ' + sex_category + ' runners', xref='paper', yref='paper', showarrow=False)]

            # We create a figure for each performance criterion
            for performance_criterion in performance_criteria:
                criterion = performance_criterion.lower()
                boxplots = study_utils.create_plotly_boxplots(data=data_final, x='year', y=criterion, hue='distance (km)', hue_names=runnings, colors=colors)
                if criterion == 'time':
                    figure = study_utils.create_plotly_legends_and_layout(data=boxplots, **time_options, annotations=annotations)
                elif criterion == 'speed (m/s)':
                    figure = study_utils.create_plotly_legends_and_layout(data=boxplots, **speed_options, annotations=annotations)
                else:
                    # By default, two specific criteria are allowed: 'time' and 'speed (m/s)'. If any other criterion is provided, we throw an exception.
                    raise ValueError('Invalid performance criterion encountered. Performance criterion must be either \'Time\' or \'Speed (m/s)\'')
                figures[age_category][sex_category][performance_criterion] = figure

    return figures


def generate_all_bib_performance_figure(df):
    '''
    This function generates all BIB/performance scatters for each year of Lausanne Marathon.

    Parameters
        - df: DataFrame containing records about runners

    Return
        - figure: Plotly figure
    '''

    # We define the considered the years interval, colors and visibility
    years_range = range(1999, 2017)
    years = {year: str(year) for year in years_range}
    colors = study_utils.generate_colors_palette(data=years_range, isDict=False, forceString=True)
    visibility = {str(year): (True if year > 2015 else 'legendonly') for year in years_range}

    # We define options
    default_options = {'title': 'Distribution of performance according to BIB numbers over the years', 'x_name': 'BIB numbers', 'hovermode':'closest'}
    time_options = {'y_name': 'Time', 'y_type': 'date', 'y_format': '%H:%M'}
    time_options.update(default_options)

    scatters = study_utils.create_plotly_scatters(data=df, x='number', y='time', hue='year', hue_names=years, text='name', color=colors, visibility=visibility)
    figure = study_utils.create_plotly_legends_and_layout(data=scatters, **time_options)
    plotly.offline.iplot(figure)
    return figure


def generate_performance_comparison(df, age_categories, performance_criteria):
    '''
    This function generates all performance boxplots according to a set of performance criteria and according to a given year to use for comparison.
    Final Dict has the following pattern:
    {
        <age_category_1: {
            <year_1>: {
                <performance_criterion_1>: {'10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]}
                [, <performance_criterion_2>: {'10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]}
                , ...]
            }
            [, <year_2>: {
                <performance_criterion_1>: {'10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]}
                [, <performance_criterion_2>: {'10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]}
                , ...]
            }, ...]   
        }
        [, <age_category_2: {
            <year_1>: {
                <performance_criterion_1>: {'10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]}
                [, <performance_criterion_2>: {'10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]}
                , ...]
            }
            [, <year_2>: {
                <performance_criterion_1>: {'10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]}
                [, <performance_criterion_2>: {'10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]}
                , ...]
            }, ...]   
        }, ...]
    }

    Parameters
        - df: DataFrame containing records about runners
        - age_categories: Array containing age categories to be displayed (if 'All'/'all', no filter is done on df)
        - performance_criteria: Array containing performance criteria to consider

    Return
        - data: Dict containing all boxplots
    '''

    # We define the considered runnings and the years interval
    runnings = {10: '10 km', 21: 'Semi-marathon', 42: 'Marathon'}
    year_values = [year for year in df['year'].unique() if year]
    
    data = {}

    # Loop over age categories
    for age_category in age_categories:
        if age_category.lower() == 'all':
            df_filtered = df
        else:
            df_filtered = df[df['age category'] == age_category]
        data[age_category] = {}

        for year in year_values:
            data[age_category][year] = {}

            # Loop over performance criteria
            for performance_criterion in performance_criteria:
                criterion = performance_criterion.lower()
                data[age_category][year][criterion] = {}
                i = 0 # Integer used to manage legends (display of first legend only for each group, for better readability)

                # Loop over runnings of Lausanne Marathon
                for km, running in runnings.items():
                    boxplots = []

                    # We filter initial DataFrame to select current running and current year used for comparison
                    df_filtered_final = df_filtered[df_filtered['distance (km)'] == km]
                    df_filtered_final_year = df_filtered_final[df_filtered_final['year'] == year]
                    
                    # We create x values (see definition of generate_x_data for more information)
                    x_variables = OrderedDict([(variable, variable.capitalize()) for variable in ['all', 'female', 'male']])
                    x_all = study_utils.generate_x_data(df_filtered_final, x_variables, 'sex')
                    x_filtered = study_utils.generate_x_data(df_filtered_final_year, x_variables, 'sex')

                    # We retrieve y values (see definition of generate_y_data for more information)
                    y_all = study_utils.generate_y_data(df_filtered_final, ['all', 'female', 'male'], 'sex', criterion)
                    y_filtered = study_utils.generate_y_data(df_filtered_final_year, ['all', 'female', 'male'], 'sex', criterion)

                    # We create boxplot for All years and for current year used for comparison
                    boxplots.append(go.Box(y=y_all, x=x_all, name='All', legendgroup='All', marker=dict(color='rgb(102, 179, 255)'), showlegend=(i == 0)))
                    boxplots.append(go.Box(y=y_filtered, x=x_filtered, name=str(year), legendgroup=str(year), marker=dict(color='rgb(255, 179, 102)'), showlegend=(i == 0)))

                    # We append all the boxplots for current running, and according to considered criterion
                    data[age_category][year][criterion][running] = boxplots

                    i += 1
    
    return data


def plot_performance_comparison(data, age_category, year, performance_criterion, silent=False):
    '''
    This function displays performance comparison graph given a set of data and a performance criterion.

    Parameters
        - data: Dict containing all the data used for performance comparison (see generate_performance_comparison)
        - age_category: string representing age category considered
        - year: year to compare with all other Lausanne Marathon editions
        - performance_criterion: Criterion to use for y axis
        - silent: If True, outputs are not displayed and figure is only returned
    '''
   
    criterion = performance_criterion.lower()

    # Creation of figure (output is ignored)
    with study_utils.ignore_stdout():
        figure = tools.make_subplots(rows=1, cols=3, subplot_titles=([key for key, value in data[criterion].items()]))

    for index, boxplots in enumerate(data[criterion]):
        for boxplot in data[criterion][boxplots]:
            figure.append_trace(boxplot, 1, index+1)
    if criterion == 'time':
        for axis, attributes in {k: v for k, v in figure['layout'].items() if 'yaxis' in k}.items():
            figure['layout'][axis].update(type='date', tickformat='%H:%M:%S')
    figure['layout']['yaxis1'].update(title=performance_criterion)
    figure['layout'].update(title='Performance comparison between Lausanne Marathon ' + str(year) + ' and all Lausanne Marathon<br>(Age category: ' + age_category +')')
    if not silent:
        plotly.offline.iplot(figure)
    return figure


def join_evolution_and_performance_data(evolution_figures, performance_figures, age_categories, sex_categories, performance_criteria):
    '''
    This function uses evolution and performance figures in order to combinate evolution and performance data in a single Dictionary.
    Final Dict has the following pattern:
    {
        <age_category_1: {
            <sex_1>: {
                'evolution': {
                    '10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]
                },
                'performance': {
                    <performance_criterion_1>: {'10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]}
                    [, <performance_criterion_2>: {'10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]}
                    , ...]
                }
            }
            [, <sex_2>: {
                'evolution': {
                    '10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]
                },
                'performance': {
                    <performance_criterion_1>: {'10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]}
                    [, <performance_criterion_2>: {'10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]}
                    , ...]
                }
            }, ...]
        }
        [, <age_category_2: {
            <sex_1>: {
                'evolution': {
                    '10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]
                },
                'performance': {
                    <performance_criterion_1>: {'10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]}
                    [, <performance_criterion_2>: {'10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]}
                    , ...]
                }
            }
            [, <sex_2>: {
                'evolution': {
                    '10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]
                },
                'performance': {
                    <performance_criterion_1>: {'10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]}
                    [, <performance_criterion_2>: {'10 km': [boxplots], 'Semi-marathon': [boxplots], 'Marathon': [boxplots]}
                    , ...]
                }
            }, ...]
        }, ...]

    Note: Function uses deep copy to have distinct dict at the end of manipulation. This is important to preserve correctness of all Notebook!
    
    Parameters
        - evolution_figures: Dictionary containing multiple evolution figures (see generate_all_evolution_figures)
        - performance_figures: Dictionary containing multiple performance figures (see generate_all_performance_figures)
        - age_categories: Age categories contained in the dictionaries (i.e. categories used to build evolution and performance figures)
        - sex_categories: Sex categories contained in the dictionaries (i.e. categories used to build evolution and performance figures)
        - performance_criteria: Performance criteria contained in performance's dictionary (i.e. crtieria used to build performance figures)

    Return
        - data: Final dictionary combining evolution and performance data
    '''

    data = {}
    evolution_figures_copy = copy.deepcopy(evolution_figures)
    performance_figures_copy = copy.deepcopy(performance_figures)

    for age_category in age_categories:
        data[age_category] = {}

        for sex_category in sex_categories:
            # Add evolution data for given age category and sex
            data[age_category][sex_category] = {}
            data[age_category][sex_category]['evolution'] = evolution_figures_copy[age_category][sex_category]['data']
            
            # Add performance data for given age category and sex
            data[age_category][sex_category]['performance'] = {}
            for criterion in performance_criteria:
                data[age_category][sex_category]['performance'][criterion] = performance_figures_copy[age_category][sex_category][criterion]['data']

    return data


def generate_evolution_and_performance_figures(data, age_categories, sex_categories, performance_criteria):
    '''
    This function generates all figures containing evolution and performance subplots, given age categories and sex categories.

    Parameters
        - data: Dictionary containing evolution and performance data (see join_evolution_and_performance_data)
        - age_categories: Age categories contained in the dictionaries (i.e. categories used to build evolution and performance data)
        - sex_categories: Sex categories contained in the dictionaries (i.e. categories used to build evolution and performance data)
        - performance_criteria: Performance criteria contained in performance's dictionary (i.e. crtieria used to build performance data)

    Return
        - figures: Dictionary containing all the figures
    '''
    
    figures = {}

    for age_category in age_categories:
        figures[age_category] = {}

        for sex_category in sex_categories:
            figures[age_category][sex_category] = {}

            for criterion in performance_criteria:
                figures[age_category][sex_category][criterion] = generate_evolution_and_performance_figure(data[age_category][sex_category], criterion)

    return figures


def generate_evolution_and_performance_figure(data, criterion):
    '''
    This function generates single evolution and performance figure given a set of data (see generate_evolution_and_performance_figures).

    Parameters
        - data: Data to be used for generation of figure
        - criterion: Performance criterion to be used

    Return
        - figure: Plotly figure representing evolution and performance in same graph
    '''

    year_values = [year for year in range(1999, 2017)]
    year_labels = [v for v in year_values]
    colors = {'10 km': KM_10_COLOR, 'Semi-marathon': KM_21_COLOR, 'Marathon': KM_42_COLOR}

    # We ignore outputs of Plotly
    with study_utils.ignore_stdout():
        figure = tools.make_subplots(rows=4, specs=[[{'rowspan': 2}], [None], [{'rowspan': 2}], [None]], subplot_titles=('Evolution of runners over the years', 'Performance comparison by years'))

        # Add all lines in first subplot
        for line in data['evolution']:
            line['legendgroup'] = line['name']
            line['marker'] = {'color': colors[line['name']]}
            figure.append_trace(line, 1, 1)

        # Add all boxplots in second subplot
        for boxplot in data['performance'][criterion]:
            boxplot['legendgroup'] = boxplot['name']
            boxplot['marker'] = {'color': colors[boxplot['name']]}
            boxplot['name'] = ''
            figure.append_trace(boxplot, 3, 1)

        # Format of x axes
        for axis, attributes in {k: v for k, v in figure['layout'].items() if 'xaxis' in k}.items():
            figure['layout'][axis].update(tickvals=year_values, ticktext=year_labels)

        # Format of y axes
        figure['layout']['yaxis1'].update(title='Number of runners', tickformat='f')
        figure['layout']['yaxis2'].update(title=criterion)
        if criterion.lower() == 'time':
            figure['layout']['yaxis2'].update(type='date', tickformat='%H:%M')

        # Use of group for boxplots to avoid superposition
        figure['layout'].update(boxmode='group')

        # Update of size, margins and legend attributes of figure
        figure['layout'].update(width=1000, height=600, margin=go.Margin(t=50, b=50, l=50, r=50))
        figure['layout']['legend'].update(y=0.5)

        return figure


def generate_performance_distribution_figures(df, age_categories, sex_categories, performance_criteria):
    '''
    This function generates all performance distribution figures according sets of age categories and sex categories and a set of performance criteria.
    Final Dict has the following pattern:
    {
        <age_category_1: {
            <sex_1>: {
                <performance_criterion_1>: <Plotly figure>
                [, <performance_criterion_2>: <Plotly figure>
                , ...]
            }
            [, <sex_2>: {
                <performance_criterion_1>: <Plotly figure>
                [, <performance_criterion_2>: <Plotly figure>
                , ...]
            }, ...]   
        }
        [, <age_category_2: {
            <sex_1>: {
                <performance_criterion_1>: <Plotly figure>
                [, <performance_criterion_2>: <Plotly figure>
                , ...]
            }
            [, <sex_2>: {
                <performance_criterion_1>: <Plotly figure>
                [, <performance_criterion_2>: <Plotly figure>
                , ...]
            }, ...]   
        }, ...]
    }

    Parameters
        - df: DataFrame containing records about runners
        - age_categories: Array containing age categories to be displayed (if 'All'/'all', no filter is done on df)
        - sex_categories: Array containing sex categories to be displayed (if 'All'/'all', no filter is done on df)
        - performance_criteria: Array containing performance criteria to consider

    Return
        - figures: Dict containing all performance distribution figures
    '''
    
    # We define options and the final Dict
    figures = {}

    for age_category in age_categories:
        # We select data according to age category
        if age_category.lower() == 'all':
            data = df
        else:
            data = df[df['age category'] == age_category]
        figures[age_category] = {}

        for sex_category in sex_categories:
            # We select data according to sex category
            if sex_category.lower() == 'all':
                data_final = data
            else:
                data_final = data[data['sex'] == sex_category.lower()]
            figures[age_category][sex_category] = {}

            # We create a figure for each performance criterion
            for performance_criterion in performance_criteria:
                criterion = performance_criterion.lower()
                figures[age_category][sex_category][performance_criterion] = generate_performance_distribution_figure(data_final, age_category, sex_category, criterion)

    return figures


def generate_performance_distribution_figure(data, age_category, sex_category, criterion):
    '''
    This function generates performance distribution figure given a dataset.

    Parameters
        - data: DataFrame containing data to use
        - age_category: Selected age category
        - sex_category: Selected sex category
        - criterion: Selected criterion

    Return
        - figure: Plotly figure
    '''

    # We define the considered runnings and the years interval
    runnings = OrderedDict([(10, {'name': '10 km', 'position': 1}), (21, {'name': 'Semi-marathon', 'position': 2}), (42, {'name': 'Marathon', 'position': 3})])
    years_range = range(1999, 2017)
    years = {year: str(year) for year in years_range}
    colors = study_utils.generate_colors_palette(years)

    # We ignore outputs of Plotly
    with study_utils.ignore_stdout():
        figure = tools.make_subplots(rows=3, subplot_titles=([attributes['name'] for km, attributes in runnings.items()]))

        for km, attributes in runnings.items():

            for year in years_range:
                data_filtered = data[(data['distance (km)'] == km) & (data['year'] == year)]

                if criterion == 'time':
                    group_data_filtered = data_filtered.set_index('time').groupby(pd.TimeGrouper(freq='5Min'))
                    x_values = [datetime.datetime.strptime(str(name), '%Y-%m-%d %H:%M:%S') for name, group in group_data_filtered]
                elif criterion == 'speed (m/s)':
                    group_data_filtered = data_filtered.round({'speed (m/s)': 1}).groupby('speed (m/s)')
                    x_values = [name for name, group in group_data_filtered]

                line = go.Scatter(mode='lines', x=x_values, y=[len(group) for name, group in group_data_filtered], name=str(year), legendgroup=str(year), marker={'color': colors[year]}, showlegend=(attributes['position'] == 1))

                figure.append_trace(line, attributes['position'], 1)

        # Format of x axes if time is selected criterion
        if criterion == 'time':
            
            for axis, attributes in {k: v for k, v in figure['layout'].items() if 'xaxis' in k}.items():
                figure['layout'][axis].update(type='date', tickformat='%H:%M')

        figure['layout'].update(width=1000, height=600, margin=go.Margin(t=100, b=50, l=50, r=50))
        figure['layout'].update(title='Performance distribution of Lausanne Marathon editions for all runnings<br>(Age category: ' + age_category +'  |  Sex category: ' + sex_category +')')
        figure['layout']['legend'].update(y=0.5)

        return figure


def generate_teams_evolution_figures(data, title='Evolution of teams performance over the years', runnings=None, team_column_name='team', year_column_name='year', min_years=6, nb_teams=8, threshold_bins_size=50, display_annotations=True):
    '''
    This function generate teams_evolution figures for all runnings.
    Final Dict has the following pattern:
    {
        <running_1: {
            <Plotly figure>
        }
        [, <running_2: {
            <Plotly figure>
        }, ...]
    }

    Parameters
        - data: DataFrame containing results
        - title: Title of figure
        - runnings: Dict containing name of column containing runnings (key: column_name) and set of runnings (key: values, value: dict() with key: value in column, value: name of running)
                    By default, None. If None, default values will be set by function.
        - team_column_name: Name of column containing teams (by default, 'teams')
        - year_column_name: Name of column containing year associated to a given result (by default, 'year')
        - min_years: Minimum of participations when considering a team (by default, 6)
        - nb_teams: Number of teams to consider among teams with number of participations > min_years (by default, 8)
                    Note: Teams are filtered by number of participants.
        - threshold_bins_size: Maximum size of a bin (by default, 25)
                    Note: Size of bin is related to number of participants of a considered team and for a given year. If None, no limitation is used.
        - display_annotations: Boolean used to display annotations (by default, True)

    Return
        - figures: Dict containing all teams evolution figures 
    '''

    # Default runnings
    if not runnings:
        runnings = {'column_name': 'distance (km)', 'values': {10: '10 km', 21: 'Semi-marathon', 42: 'Marathon'}}

    figures = {}

    # Loop over runnings
    for key, value in runnings['values'].items():
        # We retrieve data related to current running
        filtered_data = data[data[runnings['column_name']] == key]
        # We retrieve names of the <nb_teams> most important groups with at least <min_years> participations in Lausanne Marathon
        top_teams = filtered_data.groupby(team_column_name).filter(lambda x: x[year_column_name].nunique() >= min_years).groupby('team').size().sort_values(ascending=False).nlargest(nb_teams)
        # We keep only data linked with such groups
        data_top_teams = filtered_data[filtered_data[team_column_name].isin(top_teams.index.values)]
        # We finally groupby teams after complete filter
        groups_top_teams = data_top_teams.groupby(team_column_name)

        # We generate colors for each group and we initialize array that will contain traces
        colors = study_utils.generate_colors_palette(groups_top_teams.groups)
        traces = []

        # Loop over groups
        for name, groups in groups_top_teams:
            x_values, y_values, size_values, texts = [], [], [], []
            # Loop over participation years for current group
            for year, results in groups.groupby(year_column_name):
                x_values.append(year)
                y = study_utils.compute_average_time(results)
                y_values.append(y)
                text = '<b>Team: ' + name + '</b><br>Average time: ' + y.strftime('%H:%M:%S') + '<br>Participants: ' + str(len(results)) + '<br>Median age: ' + str(int(results['age'].median()))
                texts.append(text)
                size = len(results) if not threshold_bins_size or (len(results) < threshold_bins_size) else threshold_bins_size
                size_values.append(size)
            trace = go.Scatter(x=x_values, y=y_values, name=name, mode='lines+markers', hoverinfo='text', text=texts, marker=dict(size=size_values, color=colors[name], line=dict(width = 1.5, color = 'rgb(0, 0, 0)')))
            traces.append(trace)

        # For each running, we create annotations if asked by user, we set multiple options accordingly and we store figure
        if display_annotations:
            annotations = [Annotation(y=1.1, text='Running: ' + str(value) + ' | Top teams: ' + str(nb_teams) + ' | Minimum participations: ' + str(min_years) + ' | Maximum bins size: ' + str(threshold_bins_size), xref='paper', yref='paper', showarrow=False)]
        else:
            annotations = None
        options = {'title': title, 'hovermode': 'closest', 'x_name': 'Year', 'y_name': 'Median time', 'y_type': 'time', 'y_format': '%H:%M:%S', 'annotations': annotations}
        figure = study_utils.create_plotly_legends_and_layout(data=traces, **options)
        figures[value] = figure

    return figures
