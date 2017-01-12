import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt


def preprocess_runs(data):
    data.drop('entryArt', axis=1, inplace=True)
    data.drop('entryPayart', axis=1, inplace=True)
    data.drop('provider', axis=1, inplace=True)
    
def preprocess_runners(data):  
    data['gender'] = np.nan