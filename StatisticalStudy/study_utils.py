# ----------------------------------------------------------------------------------------------------------
# Import

import pandas as pd
import numpy as np
import re
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

MARATHON_DISTANCE_REGEX = '(42)|(M)|(52)' # 52 come from a mistake on datasport site
# https://services.datasport.com/2010/lauf/lamara/rang035.htm

SEMI_MARATHON_DISTANCE_REGEX = '(21)|(S)'
QUARTER_MARATHON_DISTANCE_REGEX = '(10)|(Q)'

# ----------------------------------------------------------------------------------------------------------
# Functions

def test():
    print('Hello world')

def test2():
    print('Bla')

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