# ----------------------------------------------------------------------------------------------------------
# Import

import pandas as pd

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

    #pd.options.mode.chained_assignment = 'warn'
    
    return pd.concat(all_races)