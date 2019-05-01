# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:25:16 2019

@author: u00bec7
"""

### ###########################################################################
### ##################### PACKAGES ############################################
### ###########################################################################

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


### ###########################################################################
### ################# ENVIRONMENT SETUP & DATA LOAD ###########################
### ###########################################################################


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
os.chdir(pc['wd'])
execfile('{}\\000_mm_environment_setup.py'.format(pc['repo']))

# Intitial Data load
execfile('{}\\010_mm_data_load.py'.format(pc['repo']))




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

df = 'rGamesC'

# Create matchup stats using team season statistics
matchups = createMatchups(matchupDF = dataDict['{}singleTeam'.format(df)],
                          statsDF = dataDict['{}TeamSeasonStats'.format(df)])


#### SUBSET FOR DEVELOPMENT
#matchups = matchups[matchups['Season'] == 2019]



# Offense and defence performance: 
#   points scored and allowed against opponent compared to opponent averages
#   Invert defense metric to make a higher number better (postive = allowed fewer points in game than opponent average)
matchups.loc[:, 'offStrength'] = (
        matchups['Score'] - matchups['opppointsAllowed']
        )

matchups.loc[:, 'defStrength'] = (
        matchups['pointsAllowed'] - matchups['oppScore']
        )  * (-1.0)

# How much did opponent win/lose by compared to their average
matchups.loc[:, 'spreadStrength'] = (
        matchups['scoreGap'] - (matchups['oppscoreGap'] * -1.0)
        )


# How much did team win/lose by versus how much they should have
#   based on averages
matchups.loc[:, 'spreadStrength2'] = (
        matchups['scoreGap'] 
            - (matchups['teamScore'] - matchups['opppointsAllowed'])
        )



# Weighted metrics (higher is better)
matchups.loc[:, 'offStrengthRatio'] = (
        matchups['offStrength'] * (1/ matchups['opppointsAllowed'])
        )


matchups.loc[:, 'defStrengthRatio'] = (
        matchups['defStrength'] * matchups['oppScore']
        )


matchups.loc[:, 'spreadStrengthRatio'] = (
        matchups['spreadStrength'] * matchups['oppscoreGap']
        )


matchups.loc[:, 'spreadStrengthRatio2'] = (
        matchups['spreadStrength2'] * matchups['oppscoreGap']
        )



# Opponent strength Metrics
matchups.loc[:, 'oppStrength'] = (
        (matchups['oppscoreGap'] * matchups['oppwin'])
        )

matchups.loc[:, 'oppStrengthWin'] = (
        (matchups['oppscoreGapWin'] * matchups['oppwin'])
        )


# Identify strength columns for aggregation and calculating team performance
strengthMetrics = filter(lambda metric: metric.find('Strength') >= 0, 
                         matchups.columns.tolist())


# Strength metrics incorporating game outcome
# Opponent Win % * if team won the game (use for calculating strength of team)
for metric in strengthMetrics + ['oppwin']:
    matchups.loc[:, '{}Win'.format(metric)] = (
            matchups[metric] * matchups['win']
            )


# Identify strength columns for aggregation and calculating team performance
strengthMetrics = filter(lambda metric: metric.find('Strength') >= 0, 
                         matchups.columns.tolist())


# Calculate team season means for each metric
strengthDF = (matchups.groupby(['Season', 'TeamID'])
                      .agg(dict(zip(strengthMetrics, 
                                    repeat(np.mean, len(strengthMetrics))))
                            )
                )
   
   
# Plot heat map of metric bins
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (0.9*GetSystemMetrics(0)//96, 0.8*GetSystemMetrics(1)//96))
sns.heatmap(strengthDF.corr(),
            annot = True, 
            fmt='.2f',
            mask = heatMapMask(strengthDF.corr()),
#            square = True,
            cmap = 'RdYlGn',
            linewidths = 1, 
            linecolor = 'k',
            ax = ax)
fig.tight_layout(rect=[0,0,1,0.97])
fig.suptitle('Correlation of Strength Metrics', fontsize = 20)
fig.show()
            
sns.pairplot(strengthDF, diag_kind='hist')


# Scale Data between 0 and 1 using minmax to avoid negatives and append values as '[metric]Rank'
# Change from merge to just replace scaled data (4/23/19)
strengthDF = (strengthDF.groupby('Season')
                        .apply(lambda m: (m - m.min()) / (m.max() - m.min()))
                        )

x, y = pcaVarCheck(strengthDF.shape[1], strengthDF)

z = x.explained_variance_.to_list()


# Create MatchUps of Tournament Games to determine which strength metrics
# Has the best performance, and which one to use for ranking teams
matchups = createMatchups(matchupDF = dataDict['tGamesCsingleTeam'][['Season', 'TeamID', 'opponentID', 'win']],
                                         statsDF = strengthDF,
                                         teamID1 = 'TeamID', 
                                         teamID2 = 'opponentID',
                                         teamLabel1 = 'team',
                                         teamLabel2 = 'opp',
                                         returnBaseCols = True,
                                         returnTeamID1StatCols = True,
                                         returnTeamID2StatCols = True,
                                         calculateDelta = True,
                                         calculateMatchup = False)

matchups.set_index(['Season', 'TeamID', 'opponentID', 'win'], inplace = True)




gb = GradientBoostingClassifier()

lg = LogisticRegressionCV(penalty = 'l1', solver = 'saga', cv = 5, max_iter = 200, random_state = 1127)



rfecv = RFECV(gb, cv = 5)

gb.fit(matchups, matchups.index.get_level_values('win'))

gb.feature_importances_

gb.score(matchups, matchups.index.get_level_values('win'))

dir(rfecv)


rfecv.fit(matchups, matchups.index.get_level_values('win'))
rfecv.score(matchups, matchups.index.get_level_values('win'))
rfecv.ranking_
rfecv.scoring

rfecv.grid_scores_

x = pd.DataFrame((matchups.columns, gb.feature_importances_)).transpose()
