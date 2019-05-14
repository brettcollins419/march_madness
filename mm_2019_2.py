# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:25:16 2019

@author: u00bec7
"""


### ###########################################################################
### ################# ENVIRONMENT SETUP & DATA LOAD ###########################
### ###########################################################################
import os
import socket



from __future__ import division
from os.path import join
import time
import sys
import numpy as np
import pandas as pd
import string
from win32api import GetSystemMetrics
import os
import re
from itertools import product, islice, chain, repeat, combinations
from datetime import datetime
import socket

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Colormap
import seaborn as sns

from scipy.stats import ttest_ind

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, OneHotEncoder, 
                                   LabelEncoder, QuantileTransformer, 
                                   KBinsDiscretizer, PolynomialFeatures)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import *
from sklearn.pipeline import Pipeline, FeatureUnion

# Models
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB



# Working Directory Dictionary
pcs = {
    'WaterBug' : {'wd':'C:\\Users\\brett\\Documents\\march_madness_ml',
                  'repo':'C:\\Users\\brett\\Documents\\march_madness_ml\\march_madness'},

    'WHQPC-L60102' : {'wd':'C:\\Users\\u00bec7\\Desktop\\personal\\march_madness_ml',
                      'repo':'C:\\Users\\u00bec7\\Desktop\\personal\\march_madness'},
                      
    'raspberrypi' : {'wd':'/home/pi/Documents/march_madness_ml',
                     'repo':'/home/pi/Documents/march_madness'}
    }

# Set working directory & load functions
pc = pcs.get(socket.gethostname())

del(pcs)



# Set up environment
os.chdir(pc['repo'])
from mm_functions import *

os.chdir(pc['wd'])
#execfile('{}\\000_mm_environment_setup.py'.format(pc['repo']))


# Intitial Data load
execfile('{}\\010_mm_data_load.py'.format(pc['repo']))


#%%

### ###########################################################################
### ############### ADDITIONAL IN GAME METRICS ################################
### ###########################################################################

execfile('{}\\020_mm_in_game_metrics.py'.format(pc['repo']))



### ###########################################################################
### ################## BUILD SINGLE TEAM & MODEL DATASETS #####################
### ###########################################################################

## ORGANIZE BY TEAM VS OPPONENT INSTEAD OF WTEAM VS LTEAM
# Doubles the number of observations
#   Data shape: (m,n) -> (2m,n)
map(lambda df: dataDict.setdefault('{}singleTeam'.format(df),
                                    buildSingleTeam(dataDict[df])),
    ('rGamesC', 'rGamesD', 'tGamesC', 'tGamesD'))


## REORGANIZE DATA TO MAKE HALF OF GAMES LOSSES
# Maintains same data shape as original
map(lambda df: dataDict.setdefault('{}modelData'.format(df),
                                    buildModelData(dataDict[df])),
    ('tGamesC', 'tGamesD'))



### ###########################################################################
### ############# TEAM SEASON STATISTIC METRICS ###############################
### ###########################################################################

# Season Means
# Conference Champion
# Season Means scaled within season between 0-1

execfile('{}\\030_mm_team_season_metrics.py'.format(pc['repo']))


### ###########################################################################
### ###### SEED RANK, ORDINAL RANK, AND TEAM CONFERENCES ######################
### ###########################################################################

execfile('{}\\040_mm_seeds_ordinals_conferences.py'.format(pc['repo']))


#%% DEV

### ###########################################################################
### ############################## TEAM STRENTGH METRICS ######################
### ###########################################################################

execfile('{}\\050_mm_team_strength_metrics.py'.format(pc['repo']))

### ###########################################################################
### ##################### MAP TEAM CONFERENCES ################################
### ###########################################################################


for df in map(lambda g: g + 'TeamSeasonStats', ('rGamesC', 'rGamesD')):
    
    dataDict[df].loc[:, 'ConfAbbrev'] = (
            dataDict['teamConferences'].set_index(['Season', 'TeamID'])['ConfAbbrev']
            )
    
    # New column with all small conferences grouped together
    dataDict[df].loc[:, 'confGroups'] = (
            map(lambda conf: conf if conf in 
                ('big_east', 'big_twelve', 'acc', 'big_ten', 'sec')
                else 'other',
                dataDict[df]['ConfAbbrev'].values.tolist())
            )
    
        


# =================================================================== #
# STRENGTH OF SCHEDULE AND PERFORMANCE AGAINST STRONG TEAM METRICS #
# =================================================================== #


