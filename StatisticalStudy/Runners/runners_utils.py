import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt    
import re
from dateutil import parser
import datetime as dt
from datetime import date


SEMI_MARATHON = '(Semi)|(1/2)|(Halbmarathon)'
MARATHON = 'Marathon'
KILOMETER = '\d*(\.?|\,?)\d*(\s?|-?)[kK][mM]'

OTHER_SPORT = 'Triathlon|Skating|Walking|Duathlon|Marche' 

# https://live.escalade.ch/the-race/timetable-prizes-and-courses
GENEVE_RACE = 'Course de l\'Escalade, Genève'
# http://www.christmasmidnightrun.ch/cms/index.php?option=com_content&view=category&layout=blog&id=22&Itemid=107
LAUSSANE_RACE = 'Christmas Midnight Run, Lausanne' 
# http://www.lausanne.ch/course_co
OLYMPIC_RACE = 'Course capitale olympique'
# http://www.sierre-zinal.com/la-course/
SIERRE_ZINAL = 'Sierre-Zinal, Sierre'
# http://www.thyon-dixence.ch/le-parcours/description-2/
THYON_DIXENCE = 'Course pédestre Thyon-Dixence, Thyon'
# https://www.coursedenoel.ch/3222-trail-des-chateaux-2018!.html
SION = 'Course Titzé de noël Sion'
# http://www.morat-fribourg.ch/francais/classique.aspx
MORAT = "Morat-Fribourg"

FEMALE_CATEGORY_REGEX = '[D]'
MALE_CATEGORY_REGEX = '[H]'

def compute_distance_from_category(runner):
    '''
    Returns the distance of the race based on the categoryName, eventName or overCategoryName in which
    runner has done the race.
    
    Parameters
        - runner: row representing the runner
    
    Return
        - int round(distance (km)) or None if sex was not retrieved
    '''
    
    # There are the attribute useful to find the distance.
    attributes = ['categoryName', 'eventName', 'overCategoryName']
        
    for attribute in attributes:
        if pd.isnull(runner[attribute]):
            break
            
        elif (re.search(OTHER_SPORT, runner[attribute]) != None):
            return None
        
        # We check if the string contains information about known race.
        elif (re.search(SEMI_MARATHON, runner[attribute]) != None):
            return 21
        
        elif (re.search(MARATHON, runner[attribute]) != None):
            return 42
        
        elif (re.search(LAUSSANE_RACE, runner[attribute]) != None):
            return 2.4
    
        elif (re.search(GENEVE_RACE, runner[attribute]) != None):
            return 7.3
        
        elif (re.search(OLYMPIC_RACE, runner[attribute]) != None):
            return 5
        
        elif (re.search(SIERRE_ZINAL, runner[attribute]) != None):
            return 31
        
        elif (re.search(THYON_DIXENCE, runner[attribute]) != None):
            return 31
        
        elif (re.search(SION, runner[attribute]) != None):
            return 31
        
        elif (re.search(MORAT, runner[attribute]) != None):
            return 31
        
        # We check if the categorie contain km expression
        match = re.search(KILOMETER, runner[attribute])
        # We match we know the distance.
        if match:
            result = match.group(0).replace(',','.')
            result = re.sub('Km|km|-','',result)
            return round(float(result))
      
    return None


def compute_sex(runner,runner_dataframe):
    '''
    Returns the sex of runner based on the category in which runner has done the marathon.
    
    Parameters
        - runner: row representing the runner
    
    Return
        - string ('female'/'male') or None if sex was not retrieved
    '''

    # We remove the SettingWithCopyWarning
    pd.options.mode.chained_assignment = None 
    
    # We have already found the sex of this runner.
    if 'NaN' == str(runner_dataframe['gender'][runner_dataframe['acode'] == runner['acode']]):
        return
               
    if (re.search(FEMALE_CATEGORY_REGEX, runner['categoryName']) != None):
        runner_dataframe['gender'][runner_dataframe['acode'] == runner['acode']] = 'female'
   
    elif (re.search(MALE_CATEGORY_REGEX, runner['categoryName']) != None):
        runner_dataframe['gender'][runner_dataframe['acode'] == runner['acode']] = 'male'
    
    # We set the  default SettingWithCopyWarning
    pd.options.mode.chained_assignment = 'warn' 
    return
        
def select_runners_by_numbers_of_runs(data, nb_min_runs):
    '''
    Returns the age of runner, based on this year of birth.
    
    Parameters
        - runner: row representing the runner
        
    Return
        - age of runner (int)
    '''
    
    # We get the acode of runners where the number of runs is greater than nb_min_runs
    group_by_acode = data.groupby(['acode'], sort=False).count()
    group_by_acode = group_by_acode[group_by_acode['distance (km)'] >= nb_min_runs]
    
    # return the data with only runners with nb_min_runs minimal numbers of runs.
    return data[data['acode'].isin(group_by_acode.index.values)]


def transform_string_to_second(runner):
    '''
    Returns the time in second of runner, based on this string time.
    
    Parameters
        - runner: row representing the runner
        
    Return
        - time of runner for the race (int)
    '''
    
    hour = '0'
    split_hour = ':'
    
    remove_quart_second = re.sub(',\d?','',runner['runtime'])
    
    # We add an hour if necessary.
    if not (re.search(split_hour, remove_quart_second) != None):
        remove_quart_second = hour + split_hour + remove_quart_second
    
    time_formated = dt.datetime.strptime(remove_quart_second, '%H:%M.%S')
    
    return dt.timedelta(hours=time_formated.hour,
                                  minutes=time_formated.minute, seconds=time_formated.second).total_seconds()


def compute_date_event(runner): 
    '''
    Returns the date of the event.
    
    Parameters
        - runner: row representing the runner
        
    Return
        - date of race (date)
    '''
    return dt.datetime.strptime(runner['eventDate'], '%d.%m.%Y').date()

def compute_age(runner, runner_dataframe):
    '''
    Returns the age of runner, based on this year of birth.
    
    Parameters
        - runner: row representing the runner
        
    Return
        - age of runner (int)
    '''
    
    # Get the year
    birthyear = (runner_dataframe['birthyear'][runner_dataframe['acode'] == runner['acode']]).values[0]
    date_event = parser.parse(runner['eventDate'])

    return date_event.year - birthyear

def remove_outliers(df):
    '''
    Remove outliers from the data.
    
    Parameters
        - df: DataFrame containing data
        
    Return
        - dataFrame with only finishers
    '''
    df = df[df['runtime'].notnull()]
    return df[df['resultState'] == 'classé']


def remove_useless_columns(df):
    '''
    Remove useless columns.
    
    Parameters
        - df: DataFrame containing data
    '''
    df.drop('entryArt', axis=1, inplace=True)
    df.drop('entryPayart', axis=1, inplace=True)
    df.drop('provider', axis=1, inplace=True)
    df.drop('startNumber', axis=1, inplace=True)
    df.drop('raceNr', axis=1, inplace=True)
    df.drop('eventRaceNr', axis=1, inplace=True)
    df.drop('racePayload', axis=1, inplace=True)
    df.drop('resultState', axis=1, inplace=True)
    
def preprocess_runners(df):  
    '''
    create columns gender.
    
    Parameters
        - df: DataFrame containing data
    '''
    df['gender'] = np.nan