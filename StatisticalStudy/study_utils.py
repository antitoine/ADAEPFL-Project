# ----------------------------------------------------------------------------------------------------------
# Imports

import pandas as pd
import numpy as np
import itertools
import re
import statsmodels.api as sm
import datetime
from datetime import date
from sklearn import preprocessing
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

OTHER_SPORT = '(Bike)'

MARATHON_DISTANCE_REGEX = '(42)|(M)|(52)' # 52 come from a mistake on datasport site
# https://services.datasport.com/2010/lauf/lamara/rang035.htm

SEMI_MARATHON_DISTANCE_REGEX = '(21)|(S)'
QUARTER_MARATHON_DISTANCE_REGEX = '(10)|(Q)'

# ----------------------------------------------------------------------------------------------------------
# Functions

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

    Parameters:
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
    df_cleaned['age category'] = pd.cut(df_cleaned['age'], [15, 26, 31, 36, 41, 46, 51, 56, 61, 66, 100], labels=['15-25 years', '26-30 years', '31-35 years', '36-40 years', '41-45 years', '46-50 years', '51-55 years', '56-60 years', '61-65 years', '65+ years'], right=False)
    
    # We then format time
    # Runners without time are excluded from analysis
    df_cleaned = df_cleaned[df_cleaned['time'].notnull()]
    df_cleaned['time'] = df_cleaned.apply(format_time, axis=1)
    
    # We create global categories (Adult / Junior) and mark type of runners (in temas/individual)
    df_cleaned['type'] = df_cleaned.apply(get_type_of_runner, axis=1)
    df_cleaned['type_team'] = df_cleaned.apply(compute_run_in_team, axis=1)
    
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
        return 'Individual runner'
    else:
        return 'Runner in teams'

    
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

    
def compute_time_to_best_in_team(runner, data):
    '''
    This function computes the difference between individual time and best time in a team.

    Parameters:
        - runner: row representing the runner to consider for comparison
        - data: DataFrame containing the data about runners

    Return
        - Difference (absolute) between runner's performance and best performance of the team (None if runner is not part of any team)
    '''

    if runner.type_team == 'Individual runner':
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
