# ----------------------------------------------------------------------------------------------------------
# Imports

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ----------------------------------------------------------------------------------------------------------
# Functions

# winners name can be found:

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
    The method removes the outliers on the data, precisly for each category, the method remove all time smaller than the best 
    runners in the category

    The problem come from that certain people have resinged after the first loop.
    
    Parameters
        - data: DataFrame containing records for a given running
    '''
    
    # remove resigners runners.
    data = data[~(data['rank'].isin(['DNF', 'OUT']))]
    
    # convert to float.
    data['rank'] = data['rank'].apply(lambda x : int(float(x)))
    
    all_races = []

    # loop on every years.
    for year in data['year'].unique():
        total_remove = 0
        all_cate = []
        
        # Select year
        year_selected = data[data['year'] == year]
        
        # loop on every category.
        for category in data['category'].unique():
            
            # Select by category
            category_selection = year_selected[year_selected['category'] == category]
            
            # best time of the category
            best_time = (category_selection['time']
                                 [(category_selection['category'] == category) & (category_selection['rank'] == 1)])
            
            # there is no person fist ranked in this category.
            if best_time.empty:
                best_time = np.min(category_selection['time'][category_selection['category'] == category])
            else:
                best_time = best_time.values[0]
                
            # remove all time smaller than the best time of the category
            without_outliers = category_selection[(category_selection['time'] >= best_time )] 
            
            # Compute numbers rank in
            total_remove = total_remove + (len(category_selection.index) - len(without_outliers.index))
            
            all_cate.append(without_outliers)

        print('remove outliers for ' + str(year) + ' : ' + str(total_remove) + ' runners')
        
        if len(all_cate) == 0 : 
            continue
        all_races.append(pd.concat(all_cate))   
                              
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
