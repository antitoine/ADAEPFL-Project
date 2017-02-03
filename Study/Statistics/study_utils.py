# ----------------------------------------------------------------------------------------------------------
# Imports

import contextlib
import sys
import pandas as pd
import numpy as np
import itertools
import re
import colorlover as cl
import statsmodels.api as sm
import datetime
import json
import os
from datetime import date
from sklearn import preprocessing
from scipy import stats
from io import StringIO
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.graphics import utils
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from plotly.graph_objs import Annotations

# ----------------------------------------------------------------------------------------------------------
# Constants

# category can be find at http://fr.lausanne-marathon.com/palmares/
ILINE_REGEX = '[R]'#Iline male / Iline female
KIDS_REGEX = '(K)|(Fille)|(Gar)'
WHEEL_CHAIR_REGEX = '(FD)|(FH)|(HB)|(Hand)' # Wheelchair male / Wheelchair female / handybike
FEMALE_CATEGORY_REGEX = '([D])|(JunF)'
MALE_CATEGORY_REGEX = '([H])|(JunG)'

OTHER_SPORT = '(Bike)'

MARATHON_DISTANCE_REGEX = '(42)|(M)|(52)' # 52 come from a mistake on datasport site
# https://services.datasport.com/2010/lauf/lamara/rang035.htm

SEMI_MARATHON_DISTANCE_REGEX = '(21)|(S)'
QUARTER_MARATHON_DISTANCE_REGEX = '(10)|(Q)'

# ----------------------------------------------------------------------------------------------------------
# Functions

# Inspired from: http://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
@contextlib.contextmanager
def ignore_stdout():
    '''
    This function hides all outputs generated by a called function.
    '''

    original_stdout = sys.stdout
    sys.stdout = open('trash', 'w')
    try:
        yield
    finally:
        sys.stdout = original_stdout
        os.remove('trash')


def get_data(path, common_name='Lausanne_Marathon_', extension='pickle', identifiers=range(1999, 2017), id_name='year'):
    '''
    This function retrieves all data stored given files and concatenate them to create a single DataFrame.

    Parameters
        - path: Directory containing files
        - common_name: Common beginning name between the different files to import (by default, 'LAUSANNE_MARATHON_')
        - extension: extension of considered files ('csv' or 'pickle' / by default, 'pickle')
        - identifiers: Unique identifiers of files to import (by default, range from 1999 to 2017)
        - id_name: Name of identifier (by default, 'year' / if value is set, column will be added in the DataFrame accordingly to identify origin of row)

    Return
        - DataFrame containing all the data for the given files to import
    '''

    if (extension != 'pickle') and (extension != 'csv'):
        raise ValueError('Incorrect extension. Extension must be \'csv\' or \'pickle\'.')

    df = []

    for identifier in identifiers:
        filename = path + common_name + str(identifier) + '.' + extension
        if extension == 'pickle':
            current_df = pd.read_pickle(filename)
        else:
            current_df = pd.read_csv(filename)
        if id_name:
            current_df[id_name] =  identifier
        df.append(current_df)
    
    return pd.concat(df)


def apply_computations(df):
    '''
    This function applies different computations in order to clean DataFrame.

    Parameters
        - df: DataFrame on which computations will be applied

    Return
        - df_cleaned: Cleaned DataFrame
    '''
    
    df_cleaned = df.copy()

    # We compute gender of runners and exclude those for whom sex was not retrieved accordingly
    df_cleaned['sex'] = df_cleaned.apply(get_sex_of_runner, axis=1)
    df_cleaned = df_cleaned[df_cleaned['sex'].notnull()]


    # We then compute age using birthdate of runners
    # Runners without birthday are excluded from analysis
    df_cleaned = df_cleaned[df_cleaned['birthday'].notnull()]
    df_cleaned['age'] = df_cleaned.apply(compute_age_of_runner, axis=1)
    df_cleaned['age'] = df_cleaned['age'].apply(lambda x : int(float(x)))
    df_cleaned['age category'] = pd.cut(df_cleaned['age'], [10, 26, 31, 36, 41, 46, 51, 56, 61, 66, 100], labels=['10-25 years', '26-30 years', '31-35 years', '36-40 years', '41-45 years', '46-50 years', '51-55 years', '56-60 years', '61-65 years', '65+ years'], right=False)
    
    # We then format time
    # Runners without time are excluded from analysis
    df_cleaned = df_cleaned[df_cleaned['time'].notnull()]
    df_cleaned['time'] = df_cleaned.apply(format_time, axis=1)
    
    # We create global categories (Adult / Junior) and mark type of runners (in temas/individual)
    df_cleaned['type'] = df_cleaned.apply(get_type_of_runner, axis=1)
    df_cleaned['profile'] = df_cleaned.apply(compute_run_in_team, axis=1)
    
    # We also compute running type and average speed
    df_cleaned['distance (km)'] = df_cleaned.apply(compute_distance_from_category, axis=1)
    df_cleaned['speed (m/s)'] = df_cleaned['distance (km)'] * 1000 / df_cleaned['time']
    
    return df_cleaned


def get_statistics_outliers(df, columns, check_subsets=True):
    '''
    This function displays statistics about outliers for given set of parameters.

    Parameters
        - df: DataFrame containing datas
        - columns: columns to check for missing values (i.e. to consider runner as outlier)
        - check_subsets: Check of all subsets when considering outliers (by default, True)
    '''

    combinations = []
    if check_subsets:
        for i in range(1, len(columns)+1):
            elements = [list(x) for x in itertools.combinations(columns, i)]
            combinations.append(elements)
    else:
        combinations.append(columns)

    for subsets in combinations:
        for subset in subsets:
            print('Subset: ' + str(subset))
            print('Runners with NAN: ' + str(len(df[pd.isnull(df[subset]).all(axis=1)])))


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
        or (re.search(OTHER_SPORT, runner['category']) != None)
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


def compute_age_of_runner(runner, ref=None):
    '''
    Returns the age of runner, based on his year of birth.
    
    Parameters
        - runner: row representing the runner
        - ref: Date representing the reference date to compute the age (by default, None)
        
    Return
        - age of runner (int)
    '''
    
    if ref:
        reference = ref
    elif 'year' in runner:
        reference = date(year=runner['year'], month=10, day=15)
    else:
        reference = date(year=2016, month=10, day=15)
    
    birth_year = runner['birthday']
    return reference.year - birth_year.year - ((reference.month, reference.day) < (birth_year.month, birth_year.day))


def compute_run_in_team(runner):
    '''
    Returns the profile of runner (team-mate/individual).
    
    Parameters
        - runner: row representing the runner
        
    Return
        - Profile of runner ('Individual' or 'Team-mate')
    '''
    
    if pd.isnull(runner['team']):
        return 'Individual'
    else:
        return 'Team-mate'

    
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
    seconds = int(float(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

    
def compute_time_to_best_in_team(runner, data):
    '''
    This function computes the difference between individual time and best time in a team.

    Parameters
        - runner: row representing the runner to consider for comparison
        - data: DataFrame containing the data about runners

    Return
        - Difference (absolute) between runner's performance and best performance of the team (None if runner is not part of any team)
    '''

    if runner.profile == 'Individual':
        return None
    
    else:
        # select best performances in the team by sex and  distance
        team_performance = np.min(data['time'][(data['team'] == runner.team) & (data['distance (km)'] == runner['distance (km)'])])        
        return abs(team_performance - runner.time)


def compute_pair_runner(runner, data, time_second):
    '''
    # TODO: Add description of function
    '''

    # Retrieve of all the times in the team execpt runner's personal time
    team_performance = data['time difference team'][(data['team'] == runner.team) & (data['acode'] != runner.acode)] 
       
    min_time = abs(min(team_performance, key=lambda x:abs(x-runner['time difference team'])) - runner['time difference team'])

    if min_time > time_second:
        return 0
    
    return 1


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


def run_ols_test(data, x, y):
    '''
    This function runs OLS test for each dataset given in data.

    Parameters
        - data: Dictionary containing datasets on which OLS must be executed
        - x: Name of column containing x values
        - y: Name of column containing y values

    Return
        - results: Dictionary containing results of OLS for each dataset given in data
    '''

    results = {}
    for key, values in data.items():
        y_data = values[y]
        x_data = sm.add_constant(values[x])
        results[key] = sm.OLS(y_data, x_data).fit()
    return results


def retrieve_ols_predictions_and_errors(ols_results, regressor):
    '''
    This function retrieves OLS predictions and errors from OLS results for each dataset given in ols_results.
    Final dict has the following pattern:
    {
        <first_variable>: {
            'x': <array>,
            'predictions': <array>
            'errors': {
                'min': <array>
                'max': <array>
            }
        }
        [, <second_variable>: {
            'x': <array>,
            'predictions': <array>
            'errors': {
                'min': <array>
                'max': <array>
            }
        }, ...]
    }

    Parameters
        - ols_results: Dictionary containing results for each dataset on which OLS has been run (see run_ols_test)
        - regressor: Name or index of regressor in exog matrix (see statsmodels documentation if needed)

    Return
        - results: Dictionary containing predictions and errors for each dataset given in ols_results
    '''

    results = {}

    for key, ols_results in ols_results.items():
        name, index = utils.maybe_name_or_idx(regressor, ols_results.model)
        x = ols_results.model.exog[:, index]
        x_argsort = np.argsort(x)
        x = x[x_argsort]
        prstd, iv_l, iv_u = wls_prediction_std(ols_results)
        predictions = ols_results.fittedvalues
        results[key] = {'x': x, 'predictions': predictions, 'errors': {'min': iv_l, 'max': iv_u}}

    return results


def display_boxplot(data, x, y, hue=None, title=None, x_format=None, y_format=None, size=5, aspect=2):
    '''
    This function displays a Seaborn boxplot.

    Parameters
        - data: DataFrame to use for graph
        - x: Name of column to use for x axis
        - y: Name of column to use for y axis
        - hue: Column name of the categorical data to use (by default, None)
        - title: Title of the graph (by default, None)
        - x_format: Function to use to format x labels (by default, None)
        - y_format: Function to use to format y labels (by default, None)
        - size: Size of the boxplot (by default, 5)
        - aspect: Aspect of the boxplot (by default, 2)
    '''

    g = sns.factorplot(data=data, x=x, y=y, kind='box', hue=hue, size=size, aspect=aspect)
    
    if x_format:
        for ax in g.axes.flat:
            labels = []
            for label in ax.get_xticklabels():
                formatted_label = x_format(label._x)
                labels.append(formatted_label)
            ax.set_xticklabels(labels)

    if y_format:
        for ax in g.axes.flat:
            labels = []
            for label in ax.get_yticklabels():
                formatted_label = y_format(label._y)
                labels.append(formatted_label)
            ax.set_yticklabels(labels)

    if title:
        plt.title(title)

    ax.set(xlabel=x.capitalize(), ylabel=y.capitalize())

    plt.show()


def create_plotly_legends_and_layout(data, title=None, width=None, height=None, x_name=None, y_name=None, x_values=None, y_values=None, x_values_format=None, y_values_format=None, x_categoryarray=None, y_categoryarray=None, x_type=None, y_type=None, x_format=None, y_format=None, x_dtick=None, y_dtick=None,boxmode=None, barmode=None, bargap=None, bargroupgap=None, hovermode=None, annotations=None, shapes=None):
    '''
    This function creates Plotly legends and layout.

    Parameters
        - data: Plotly data
        - title: Title of the graph (by default, None)
        - width: Width of the figure (by default, None)
        - height: Height of the figure (by default, None)
        - x_name: Name of x axis (by default, None)
        - y_name: Name of y axis (by default, None)
        - x_values: Array containing x values to display (by default, None / if None, Plotly creates axis automatically)
        - y_values: Array containing y values to display (by default, None / if None, Plotly creates axis automatically)
        - x_values_format: Function to use to format x_values (by default, None / if None, x_values is used for x labels)
        - y_values_format: Function to use to format x_values (by default, None / if None, y_values is used for y labels)
        - x_categoryarray: Array containing categories in order they must appear on x axis (by default, None)
        - y_categoryarray: Array containing categories in order they must appear on y axis (by default, None)
        - x_type: String representing type of x axis (by default, None / type must be supported by Plotly)
        - y_type: String representing type of y axis (by default, None / type must be supported by Plotly)
        - x_format: String representing format of x axis (by default, None / format must be supported by Plotly)
        - y_format: String representing format of y axis (by default, None / format must be supported by Plotly)
        - x_dtick: Integer representing the spacing between ticks for x axis (by default, None)
        - y_dtick: Integer representing the spacing between ticks for y axis (by default, None)
        - boxmode: String representing the type of boxmode to use (by default, None)
        - barmode: String representing the type of barmode to use (by default, None)
        . bargap: Float representing the gap between bars (by default, None)
        . bargroupgap: Float representing the gap between group bars (by default, None)
        - hovermode: String representing the type of mode to use on hover (by default, None)
        - annotations: Array containing all Annotation to add in layout (Annotations and Annotation are part of Plotly library / by default, None)
        - shapes: Array containing all shapes to add in layout (by default, None)

    Return
        - figure: Plotly figure
    '''

    if x_values:
        x_labels = [(x_values_format(v) if x_format else v) for v in x_values]
    else:
        x_labels = None
    
    if y_values:
        y_labels = [(y_values_format(v) if y_format else v) for v in y_values]
    else:
        y_labels = None
    
    x_axis = go.XAxis(title=x_name, categoryorder=('array' if x_categoryarray else None), categoryarray=x_categoryarray, type=x_type, tickformat=x_format, ticktext=x_labels, tickvals=x_values, dtick=x_dtick, mirror='ticks', ticks='inside', showgrid=True, showline=True, zeroline=True, zerolinewidth=2)
    y_axis = go.YAxis(title=y_name, categoryorder=('array' if y_categoryarray else None), categoryarray=y_categoryarray, type=y_type, tickformat=y_format, ticktext=y_labels, tickvals=y_values, dtick=y_dtick, mirror='ticks', ticks='inside', showgrid=True, showline=True, zeroline=True, zerolinewidth=2)

    layout = go.Layout(title=title, width=width, height=height, xaxis=x_axis, yaxis=y_axis, boxmode=boxmode, barmode=barmode, bargap=bargap, bargroupgap=bargroupgap, hovermode=hovermode, annotations=Annotations(annotations) if annotations else Annotations(), shapes=shapes if shapes else [])
    figure = go.Figure(data=data, layout=layout)

    return figure


def create_plotly_boxplots(data, x, y, hue=None, hue_names=None, colors=None, visibility=None, use_hue_names=True, use_legend_group=False, show_legend=True):
    '''
    This function creates Plotly figure containing boxplots.

    Parameters
        - data: DataFrame containing data to use for graph
        - x: Name of column used for x axis
        - y: Name of column used for y axis
        - hue: Column name of the categorical data to use (by default, None)
        - hue_names: Dictionary containing name to display for an associated value available in data[hue] (by default, None)
        - colors: Dictionary containing color for each value available in data[hue] (key must be hue name (see hue_names) or hue value (see hue) / by default, None)
        - visibility: Dictionary containing visibility to set for an associated value in data[hue] (by default, None)
        - use_hue_names: Boolean indicating if hue_names must be used to access attributes of other dictionaries (such colors) instead of value of hue_values (by default, True)
        - use_legend_group: Boolean indicating if legendgroup must be used (by default, False / if True, legends are grouped by hue)
        - show_legend: Boolean indicating if legend must be shown (by default, True)

    Return
        - fig: Plotly figure
    '''
    
    hue_values = data[hue].unique()
    all_boxes = []

    if hue:
        for value in hue_values:
            filtered_data = data[data[hue] == value]
            current_x = filtered_data[x]
            current_y = filtered_data[y]
            box = go.Box(y=current_y, x=current_x, name=(hue_names.get(value, value) if hue_names else value), marker={'color': (colors.get(hue_names.get(value, value) if (hue_names and use_hue_names) else value, None) if colors else None)}, visible=(visibility[hue_names.get(value, value) if (hue_names and use_hue_names) else value] if visibility else None), legendgroup=(value if use_legend_group else None), showlegend=show_legend)
            all_boxes.append(box)
    else:
        box = go.Box(y=data[y], x=data[x])
        all_boxes.append(box)
    
    return all_boxes


def create_plotly_scatters(data, x, y, hue=None, hue_names=None, text=None, color=None, visibility=None, use_hue_names=True):
    '''
    This function creates Plotly figure containing boxplots.

    Parameters
        - data: DataFrame containing data to use for graph
        - x: Name of column used for x axis
        - y: Name of column used for y axis
        - hue: Column name of the categorical data to use (by default, None)
        - hue_names: Dictionary containing name to display for an associated value available in data[hue] (by default, None)
        - text: Column name of data to display on hover (by default, None)
        - color: Dictionary containing color to use for an associated value in data[hue] (if hue provided) or String (rgba) representing color of bins
        - visibility: Dictionary containing visibility to set for an associated value in data[hue] (by default, None)
        - use_hue_names: Boolean indicating if hue_names must be used to access attributes of other dictionaries (such colors) instead of value of hue_values (by default, True)

    Return
        - fig: Plotly figure
    '''
    
    hue_values = data[hue].unique()
    all_scatters = []

    if hue:
        for value in hue_values:
            filtered_data = data[data[hue] == value]
            current_x = filtered_data[x]
            current_y = filtered_data[y]
            scatter = go.Scattergl(y=current_y, x=current_x, name=(hue_names.get(value, value) if hue_names else value), text=data[text], visible=(visibility[hue_names.get(value, value) if (hue_names and use_hue_names) else value] if visibility else None), mode='markers', marker=dict(size=10, color=(color[hue_names.get(value, value) if (hue_names and use_hue_names) else value] if color else None), line = dict(width = 2)))
            all_scatters.append(scatter)
    else:
        scatter = go.Scattergl(y=data[y], x=data[x], text=data[text], mode='markers', marker=dict(color=color))
        all_scatters.append(scatter)

    return all_scatters


def generate_x_data(data, variables, column_filter):
    '''
    This function instantiates an array with following pattern: [<variable1_name> * len(data[variable1_value])[, <variable2_name> * len(data[variable2_value]), ...]].

    Parameters
        - data: DataFrame containing data to use
        - variables: Dict containing variables to use for generation of array (key is used for filtering, value is used for filling final array / if 'all' as key, then no filtering in data)
        - column_filter: Column name to use during filtering in data

    Return
        - x_values: Array containing generated data
    '''

    x_values = []

    for key, value in variables.items():
        if key == 'all':
            x_values.extend([value]*len(data))
        else:
            x_values.extend([value]*len(data[data[column_filter] == key]))

    return x_values


def generate_y_data(data, variables, column_filter, column_select):
    '''
    This function instantiates a Series containing data[column_select] for all rows containing one of any variable in data[column_filter].

    Parameters
        - data: DataFrame containing data to use
        - variables: Array containing variables to use during filtering (if 'all', then no filtering in data)
        - column_filter: Column name to use during filtering in data
        - column_select: Column name to use for selection of data

    Return
        - y_values: Series containing filtered data
    '''

    y_values = pd.Series()

    for value in variables:
        if value == 'all':
            y_values = y_values.append(data[column_select])
        else:
            y_values = y_values.append(data[data[column_filter] == value][column_select])

    return y_values


def generate_colors_palette(data, isDict=True, forceString=False):
    '''
    This function builds a dict containing color for each element in data.

    Parameters
        - data: Container of objects containing name of objects which need specific color
        - isDict: Boolean which indicates if data is a Dict (if True, keys will be used / by default, True)
        - forceString: Boolean which indicates if key of colors dictionary must be converted to string (by default, False)

    Return
        - colors: Dict containing color for each object (key: name of object, value: color (string))
    '''

    palette = None
    if len(data) <=8:
        palette = cl.scales['8']['qual']['Paired']
    else:
        size = len(data)
        # Bug in colorlover (see issues on GitHub page of project)
        # We artificially increase size until no exception is thrown (i.e. mapping has been successfully created)
        while not palette:
            try:
                palette = cl.interp(cl.scales['11']['div']['Spectral'], size)
            except:
                size += 1
    colors = {(name if not forceString else str(name)): palette[index] for index, name in enumerate(data.keys() if isDict else data)}
    return colors


def compute_average_time(results, time_column_name='time'):
    '''
    This function computes the average time given a set of results.

    Parameters
        - results: DataFrame containing results for a given running
        - time_column_name: Name of column containing time results (by default, 'time')

    Return
        - datetime object representing median time of given results set
    '''

    median_in_seconds = (results[time_column_name]-results[time_column_name].min()).median() / np.timedelta64(1, 's')
    min_time = results[time_column_name].min().time()
    min_time_in_seconds = datetime.timedelta(hours=min_time.hour, minutes=min_time.minute, seconds=min_time.second).total_seconds()
    return datetime.datetime.strptime(convert_seconds_to_time(median_in_seconds + min_time_in_seconds), '%H:%M:%S')


def convert_to_JSON(object, file_name, path='.', encoder=None, indent=4, sort_keys=True, separators=(',', ':'), override=False):
    '''
    This function creates JSON file given an object.

    Parameters
        - object: Object to convert to JSON
        - file_name: Name of JSON file (without .JSON extention)
        - path: Folder path (by default, '.' (current directory))
        - encoder: JSON Encoder (by default, None)
        - indent: Number of indents (by default, 4)
        - sort_keys: Indicates if keys must be sorted or not (by default, True)
        - separators: Separators to use (by default, (',', ':'))
        - override: Indicates if existing file must be overriden (by default, False)
    '''

    full_path = path + '/' + file_name + '.json'
    file_exists = os.path.isfile(full_path)

    if file_exists and not override:
        print('File: ' + full_path)
        print('Conversion was aborded. File already exists. To force override of file, set \'override\' to True.')
        return

    with open(full_path, 'w') as outfile:
        json.dump(obj=object, cls=encoder, fp=outfile, indent=indent, sort_keys=sort_keys, separators=separators)
