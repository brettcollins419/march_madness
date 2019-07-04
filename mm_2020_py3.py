# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:25:16 2019

@author: u00bec7
"""


### PACKAGES

from __future__ import division
import os
import time
import sys
import numpy as np
import pandas as pd
import string
from win32api import GetSystemMetrics
import re
from itertools import product, islice, chain, repeat, combinations
from datetime import datetime
import socket

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Colormap
import seaborn as sns

from scipy.stats import ttest_ind


from sklearn.preprocessing import (StandardScaler, MinMaxScaler, OneHotEncoder, 
                                   LabelEncoder, QuantileTransformer, 
                                   KBinsDiscretizer, PolynomialFeatures)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.metrics import *
from sklearn.pipeline import Pipeline, FeatureUnion

# Models
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, 
                              ExtraTreesClassifier, 
                              GradientBoostingClassifier)
from sklearn.linear_model import (LogisticRegression, 
                                  LinearRegression, 
                                  LogisticRegressionCV)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

#%% FUNCTIONS
## ############################################################################


def generateDataFrameColumnSummaries(df, returnDF = False):
    '''Create summary for each column in df as a tuple:
        (column Name, column type, is object boolean)
        
    Return list of tuples.
    If returnDF == True, return a pandas dataframe'''

    colSummary = map(lambda c: (c, 
                                df[c].dtype.type,
                                df[c].dtype.hasobject),
                    df.columns.tolist()
                    )
                    
    if returnDF == True:
      colSummary = pd.DataFrame(colSummary,
                                columns = ['colName', 'colDataType', 'isObject'])
    
    return colSummary



def buildSingleTeam(df, sharedCols = ['Season', 'DayNum', 'WLoc', 'NumOT', 'scoreGap', 'WTeamID', 'LTeamID']):
    '''Create dataframe where all data is aligned from the perspective of a single team
        versus an opponent. Not organized by wins & loss categories. 
        
        Will generate a dataframe with 2x as many records'''


    colsWinTemp = filter(lambda col: col.startswith('W') & (col not in sharedCols),
                         df.columns.tolist())
    
    colsLossTemp = filter(lambda col: col.startswith('L') & (col not in sharedCols),
                          df.columns.tolist())   
    
    
    # Format base/shared columns between wins & loss data
    # Remove wLoc from colsBase (object, would need to parameterize for any value)
 
    
    winDF = df.loc[:, sharedCols + colsWinTemp]
    winDF.rename(columns = {'LTeamID':'opponentID', 
                            'WTeamID': 'TeamID',
                            'WLoc':'Loc'}, inplace = True)
    winDF.loc[:, 'win'] = 1

    lossDF = df.loc[:, sharedCols + colsLossTemp]
    lossDF.rename(columns = {'WTeamID':'opponentID', 
                             'LTeamID':'TeamID',
                             'WLoc':'Loc'}, inplace = True)
    lossDF.loc[:, 'win'] = 0
    
    # Change Location to correct location for losing team
    lossDF.loc[:, 'Loc'].replace({'H':'A', 'A':'H'}, inplace = True)
    
    # Flip scoreGap for losing Team
    lossDF.loc[:, 'scoreGap'] = lossDF.loc[:, 'scoreGap'] * -1
    
    # Drop 'W' & 'L' from labels
    winDF.rename(columns=dict(zip(colsWinTemp, map(lambda c: c.replace('W',''), 
                                                   colsWinTemp))), 
                  inplace = True)
                  
    lossDF.rename(columns=dict(zip(colsLossTemp, map(lambda c: c.replace('L',''), 
                                                     colsLossTemp))), 
                   inplace = True)
    
    # Combine wins and losses data and calculate means
    aggDF = pd.concat((winDF, lossDF), axis = 0, sort = True)
    
    
    # Calculate points allowed
    aggDF.loc[:, 'pointsAllowed'] = map(lambda g: g[0] - g[1], 
                                     aggDF.loc[:, ['Score', 'scoreGap']].values.tolist())      
        

    return aggDF



def buildModelData(gameDF):                                  
                                         
    '''Randomnly split games data frames in half and swap columns for creating
    model datasets.
        
    Return dataframe of same # of games with 1/2 of games wins and half losses
    and additional columns 'win' with boolean of 1, 0 for win / loss respectively.
    '''
    gameDF = gameDF.copy()

    # Split data
    win, loss = train_test_split(gameDF, test_size = 0.5, random_state = 1127)
 
    
    # Rename columns to flip winning and losing fields
    loss.columns = map(lambda field: re.sub('^W(?!Loc)|^L', 
                                            lambda x: 'L' if x.group(0) == 'W' else 'W', 
                                            field),
                       loss.columns)
    
    # Assign win & loss column
    win.loc[:, 'win'], loss.loc[:, 'win'] = 1, 0
    
    
    # Switch Winning location for loss dataframe
    loss.loc[:, 'WLoc'].replace({'H':'A', 'A':'H'}, inplace = True)
    
    # Combine two dataframes
    modelData = pd.concat((win, loss), axis = 0, sort = True)
    
    return modelData



def buildModelData2(gameDF, teamDF, 
                    indexCols = [],
                    label1 = 'A', 
                    label2 = 'B',
                    labelName = 'TeamID',
                    extraMergeFields = ['Season'],
                    calculateMatchupStats = True,
                    calculateDeltas = True,
                    createMatchupFields = True,
                    returnStatCols = True,
                    deltaExcludeFields = [],
                    matchupFields = [('confMatchup', 'ConfAbbrev'), 
                                     ('seedRankMatchup', 'seedRank')]):
                                         
                                         
    '''Randomnly split games data frames in half and swap columns for creating
    model datasets.  gameDF data should be base columns only 
    with shared stats and TeamIDs.  After splitting and swapping 50% of the records
    look up team metrics for filling in matchup using generateGameMatchupStats
    function.
        
    Return dataframe of team statistics matchup plus winnerA boolean column.'''

    
    baseCols = filter(lambda c: c not in ['WTeamID', 'LTeamID'],
                      gameDF)
    
    # Split data
    a, b = train_test_split(gameDF, test_size = 0.5, random_state = 1127)
    
    # Assign win & loss boolean
    a.loc[:, 'winnerA'], b.loc[:, 'winnerA'] = 1, 0    
    
    # Rename columns with new labels
    a.rename(columns = {'WTeamID':'ATeamID', 'LTeamID':'BTeamID'},
             inplace = True)
    b.rename(columns = {'LTeamID':'ATeamID', 'WTeamID':'BTeamID'},
             inplace = True)
    
    # Combine new datasets
    mdlData = pd.concat([a,b], axis = 0)   
    
    # Calculate matchup statistics if true
    if calculateMatchupStats == True:
        mdlData = generateGameMatchupStats2(mdlData, teamDF, 
                                            indexCols,
                                            teamID1 = 'ATeamID', teamID2 = 'BTeamID',
                                            labelName = labelName,
                                            extraMergeFields = extraMergeFields,
                                            label1 = label1, label2 = label2,
                                            calculateDeltas = calculateDeltas,
                                            returnStatCols = returnStatCols,
                                            createMatchupFields = createMatchupFields,
                                            deltaExcludeFields = deltaExcludeFields,
                                            matchupFields = matchupFields)

    
    return mdlData


def generateMatchupField(df, matchupName, label1, label2):
    '''Create matchup key from two fields sorted alphabetically
        Return a list of sorted tuples with label fields.'''
    
    matchup = zip(df[label1 + matchupName].values.tolist(),
                  df[label2 + matchupName].values.tolist())
    matchup = map(lambda m: list(m), matchup)
    map(lambda m: m.sort(), matchup)
    matchup = map(lambda l: tuple(l), matchup)
    
    return matchup



def createMatchupField(df, label1, label2, sort = False):
    '''Create matchup key from two fields sorted alphabetically
        Return a list of sorted tuples with label fields.'''
    
    matchup = zip(df[label1].values.tolist(),
                  df[label2].values.tolist())
    
    if sort == True:
        matchup = map(lambda m: list(m), matchup)
        map(lambda m: m.sort(), matchup)
        matchup = map(lambda l: tuple(l), matchup)
    
    return matchup



def modelAnalysis(model, data = [], 
                  targetCol = None, 
                  indCols = None, 
                  testTrainDataList = [], 
                  testTrainSplit = 0.2):
    
    if indCols == None:
        indCols = filter(lambda col: ((data[col].dtype.hasobject == False) 
                                        & (col != targetCol)), 
                         data.columns.tolist())
    
    if len(testTrainDataList) == 4:                                           
        xTrain, xTest, yTrain, yTest = testTrainDataList
    
    else:
        xTrain, xTest, yTrain, yTest = train_test_split(data[indCols], 
                                                        data[targetCol],
                                                        test_size = testTrainSplit)
    

    model.fit(xTrain, yTrain)
    predictions = model.predict(xTest)
    predProbs = np.max(model.predict_proba(xTest), axis = 1)
    aucMetric = roc_auc_score(yTest, predictions)
    accuracy = accuracy_score(yTest, predictions)
    
    return predictions, predProbs, aucMetric, accuracy



def modelAnalysisPipeline(modelPipe, data = [], 
                          targetCol = None, 
                          indCols = None, 
                          testTrainDataList = [], 
                          testTrainSplit = 0.2,
                          gridSearch = False,
                          paramGrid = None,
                          scoring = None):

    '''Perform model pipeline and perfrom grid search if necessary.
    
        Return dictionary with Pipeline, predictions, probabilities,
        test data, train data, rocCurve, auc, and accuracy'''

    # Remove all non numeric columns from model
    if indCols == None:
        indCols = filter(lambda col: ((data[col].dtype.hasobject == False) 
                                        & (col != targetCol)), 
                         data.columns.tolist())
    
    # Assign test/train datasets if defined, otherwise perform test/train split
    if len(testTrainDataList) == 4:                                           
        xTrain, xTest, yTrain, yTest = testTrainDataList
    
    else:
        xTrain, xTest, yTrain, yTest = train_test_split(data[indCols], 
                                                        data[targetCol],
                                                        test_size = testTrainSplit)
    # Perform grid search if necessary
    if gridSearch == True:
        modelPipe = GridSearchCV(modelPipe, paramGrid, scoring = scoring)
    
    # Fit pipleine
    modelPipe.fit(xTrain, yTrain)
    
    # Perform predictions and return model results
    predictions = modelPipe.predict(xTest)
    predProbs = np.max(modelPipe.predict_proba(xTest), axis = 1)
    aucMetric = roc_auc_score(yTest, predictions)
    accuracy = accuracy_score(yTest, predictions)
    rocCurve = roc_curve(yTest, modelPipe.predict_proba(xTest)[:,1])
    
    # return modelPipe, predictions, predProbs, auc, accuracy
    return {'pipe' : modelPipe,
            'predictions' : predictions,
            'probabilities' : predProbs,
            'xTrain' : xTrain, 'yTrain' : yTrain,
            'xTest' : xTest, 'yTest' : yTest,
            'rocCurve' : rocCurve,
            'auc' : aucMetric, 'accuracy' : accuracy}



def generateTeamLookupDict(teamDF, yrFilter=True, yr=2019):
    '''Generate dictionary of team statistics for looking up team in matchups'''
    
    if yrFilter == True: 
        teamDF.reset_index('Season', inplace = True)
        teamDF = teamDF[teamDF['Season'] == yr]
        teamDF.drop('Season', inplace = True, axis = 1)

    teamIDs = teamDF.index.values.tolist()
    teamData = map(lambda v: tuple(v), teamDF.values.tolist())
    teamDict = {k:v for k,v in zip(teamIDs,teamData)}

    return teamDict



def generateOldTourneyResults(tSeeds, tSlots, tGames, yr):
    '''Get matchups and results for any previous tournament.
    
    Use for model assessment.
    
    Return tSlots dataframe with StrongTeam, WeakTeam, and winner columns added.
    '''
   
    # Clean up dataframes for specific season and isolate only desired columns
    if 'Season' in tSeeds.columns.tolist():
        tSeeds = tSeeds[tSeeds['Season'] == yr][['Seed', 'TeamID']]
    else: tSeeds = tSeeds[['Seed', 'TeamID']]
    
    
    if 'Season' in tSlots.columns.tolist():
        tSlots = tSlots[tSlots['Season'] == yr][['Slot', 'StrongSeed', 'WeakSeed']]
    else: tSlots = tSlots[['Slot', 'StrongSeed', 'WeakSeed']]
    
    
    if 'Season' in tGames.columns.tolist():
        tGames = tGames[tGames['Season'] == yr][['DayNum', 'WTeamID', 'LTeamID']]
    else: tGames = tGames[['DayNum', 'WTeamID', 'LTeamID']]
    
    
    # Create dictionary with keys as seeds and values as team
    seedsDict = dict(tSeeds.values.tolist())   

    
    # Create dictionary of matchups with winners
    #   Keys: sorted tuple of TeamIDs
    #   Values: game winners
    tGames['matchup'] = generateMatchupField(tGames, 'TeamID', 'W', 'L') 
    matchupDict = dict(tGames[['matchup', 'WTeamID']].values.tolist())
  

    # Setup columns
    tSlots['StrongTeam'], tSlots['WeakTeam'], tSlots['winner'] = 'x', 'x', 'x'

    # Loop through the following sequence of steps until all matchups are complete:
    #   1. Lookup TeamIDs based on Strong and Weak Seed using seedsDict
    #   2. Create a sorted tuple of the TeamIDs for matching
    #   3. Lookup winner of the matchup from the matchupDict
    #   4. Update the seedsDict by assigning the Slot to the winner (winner takes Slot as new Seed)

    while np.any(tSlots['winner'].map(str) == 'x'):
        # Lookup Teams for Strong and Weak seeds
        for team in ['Strong', 'Weak']:
            tSlots[team + 'Team'] = map(lambda t: seedsDict.get(t, 'x'),
                                        tSlots[team + 'Seed'].values.tolist())
        
        # Generate sorted matchup Teams tuple
        tSlots['matchup'] = generateMatchupField(tSlots, 'Team', 'Strong', 'Weak')
        
        # Lookup winner
        tSlots['winner'] = map(lambda m: matchupDict.get(m, 'x'),
                               tSlots['matchup'].values.tolist())
        
        # Update seeds dict with winner & slot    
        seedsDict.update(dict(tSlots[['Slot', 'winner']].values.tolist()))
   
    return tSlots




def pcaVarCheck(n, data):
    '''Calculate PCA and return the pca object & explained variance'''
    
    pca = PCA(n_components=n)
    pca.fit(data)
    
    explainedVar = np.sum(pca.explained_variance_ratio_)
    
    return (pca, explainedVar)



def timer(sigDigits = 3):
    '''Timing function'''
    global startTime   
    if 'startTime' in globals().keys():
               
        calcTime =  time.time() - startTime
        startTime = time.time()
        
        return round(calcTime, sigDigits) 
        
    
    else:
        globals()['startTime'] = time.time()
        #global startTime        




def plotCorrHeatMap(corrData, ax = None,
                    cmap = 'coolwarm', 
                    maskHalf = True, 
                    plotTitle = None,
                    plotTitleFontSize = 24,
                    tickLabelSize = 16):

    '''Plot heatmap of correlation matrix. Only plots the lower
        left corner of the correlation matrix if maskHalf == True.'''

    if ax == None:
       fig, ax = plt.subplots(1)  

    mask = np.zeros_like(corrData)

    if maskHalf == True:  
        mask[np.triu_indices_from(mask)] = True   

    if plotTitle != None:    
        ax.set_title(plotTitle, fontsize = 24)
    
    sns.heatmap(corrData, mask = mask, square = True, cmap=cmap, ax=ax)    
    plt.tight_layout()    
    ax.tick_params(labelsize=tickLabelSize)

    return




def heatMapMask(corrData, k = 0, corner = 'upper_right'):
    
    ''' Create array for masking upper right or lower left corner of map.
        k is the offset from the diagonal.
            if 1 returns the diagonal
        Return array for masking'''
        
    # Mask lower left
    if corner == 'lower_left':
        mask = np.zeros_like(corrData) 
        mask[np.tril_indices_from(arr = mask, k = k)] = True

    # Mask upper right
    else:
        mask = np.zeros_like(corrData) 
        mask[np.triu_indices_from(arr = mask, k = k)] = True

    return mask




def colsTeamFilter(col, teamLabel):
    '''Check if column string starts with specificed letter or phrase.'''
    return col.startswith(teamLabel)



def generateMatchupDeltas(df, label1, label2, labelName = 'TeamID', excludeCols = []):
    '''Calculate deltas bewtween matching metric columns
    
        Return dataframe of delta metrics with column labels as same
        base name + 'Delta' '''    
    
    dfCols = df.columns.tolist()
    
    # Error handling of exclusion list
    if type(excludeCols) != list:
        excludeCols = list(excludeCols)
    
    # Add teamID fields to exclusion list
    excludeCols += [label1 + labelName, label2 + labelName]    
    
    # Find all numeric columns
    objectCols = filter(lambda c: df[c].dtype.hasobject, dfCols)
    excludeCols += objectCols
    
    numericCols = filter(lambda c: c not in excludeCols, dfCols)
    
    # Split numeric columns between the two teams
    label1Cols = filter(lambda c: colsTeamFilter(c, label1), numericCols)
    label2Cols = filter(lambda c: colsTeamFilter(c, label2), numericCols)

    len1, len2 = len(label1Cols), len(label2Cols)
    
    # Make sure labels are in both datasets 
    # (filter longest list to elements in shortest list)
    if len1 >= len2:
        label1Cols = filter(lambda c: c[1:] in map(lambda cc: c[1:], label2Cols), 
                            label1Cols)
    else:
        label2Cols = filter(lambda c: c[1:] in map(lambda cc: c[1:], label1Cols), 
                            label2Cols)
    
    # Sort columns for zippping 
    label1Cols.sort()
    label2Cols.sort()
    
    
    # Create dataframe of metric deltas (label1 - label2)
    l1DF = df[label1Cols]
    l2DF = df[label2Cols]

    l2DF.columns = label1Cols

    deltaDF = l1DF - l2DF    
    deltaDF.columns = map(lambda colName: colName[1:] + 'Delta', label1Cols)

    return deltaDF



    




def tourneyPredictions2(model, teamDF, tSeeds, tSlots, mdlCols, 
                        seedDF = pd.DataFrame(), 
                        includeSeedStats = True, yr = 2019, returnStatCols = False):
    
    # Create copies of DataFrames
    tSeeds, tSlots, teamDF = tSeeds.copy(), tSlots.copy(), teamDF.copy()
    
    if 'Season' in tSeeds.columns.tolist():
        tSeeds = tSeeds[tSeeds['Season'] == yr][['Seed', 'seedRank', 'TeamID']]
        #tSeeds = tSeeds.drop('Season', inplace = True, axis = 1)
    else: tSeeds = tSeeds[['Seed', 'seedRank', 'TeamID']]
    
    if 'Season' in tSlots.columns.tolist():
        tSlots = tSlots[tSlots['Season'] == yr][['Slot', 'StrongSeed', 'WeakSeed']].copy()
        #tSlots = tSlots.drop('Season', inplace = True, axis = 1)
    else: tSlots = tSlots[['Slot', 'StrongSeed', 'WeakSeed']].copy()  
    
    if 'Season' in teamDF.index.names:
        teamDF = teamDF[teamDF.index.get_level_values('Season') == yr]
        teamDF.reset_index('Season', inplace = True)
        teamDF.drop('Season', axis = 1, inplace = True)


    seedDict = dict(tSeeds[['Seed', 'TeamID']].values.tolist())
    resultProbDict = {}
    
    tSlots.loc[:,'rndWinner'], tSlots.loc[:,'winProb'] = 'x', 0
    
    
    matchupData = pd.DataFrame()
    
    # Loop through rounds making predictions
    while len(filter(lambda result: result == 'x', tSlots['rndWinner'].values.tolist())) > 0:
        
        # Match seeds to slot for matchups
        for seed in ['Strong', 'Weak']:
            tSlots[seed + 'Team'] = tSlots[seed + 'Seed'].map(lambda t: seedDict.get(t, 'x'))
        
        # Need to error handle for all numeric indexes (last round)    
        slotMatchUps = tSlots[((tSlots['StrongTeam'].map(str) != 'x')
                                 & (tSlots['WeakTeam'].map(str) != 'x')
                                 & (tSlots['winProb'] == 0))][['Slot', 'StrongTeam', 'WeakTeam']]
        
        
        # Generate matchup data for modeling
        if includeSeedStats == False:
            slotMatchUps2 = generateGameMatchupStats2(gameDF = slotMatchUps, 
                                                     teamDF = teamDF, 
                                                    teamID1='StrongTeam', 
                                                    teamID2 = 'WeakTeam',
                                                    label1 = 'A',
                                                    label2 = 'B',
                                                    calculateDeltas = True,
                                                    returnStatCols = returnStatCols,
                                                    createMatchupFields = True)
        
        else:
  
    
    
            slotMatchUps2 = genGameMatchupswSeedStats(baseCols = ['StrongTeam', 'WeakTeam'],
                                                       gameDF = slotMatchUps,
                                                       teamDF = teamDF,
                                                       seedDF = seedDF,
                                                       teamID1 = 'StrongTeam', 
                                                       teamID2 = 'WeakTeam',
                                                       label1 = 'A', 
                                                       label2 = 'B',
                                                       calculateDeltas = True,
                                                       returnStatCols = returnStatCols,
                                                       createMatchupFields = True)
    
        
        # Store matchup data
        matchupData = pd.concat([matchupData, slotMatchUps2], axis = 0)
        
        # Predict winner and winning probability
        slotMatchUps['rndWinner'] = model.predict(slotMatchUps2[mdlCols])
        slotMatchUps['winProb'] = np.max(model.predict_proba(slotMatchUps2[mdlCols]), axis = 1)
        
        # Assign TeamID to roundWinner (1 = StrongSeed/Team, 0 = WeakTeam/Seed)
        slotMatchUps['rndWinner'] = slotMatchUps.apply(lambda game: game['StrongTeam'] if game['rndWinner'] == 1 else game['WeakTeam'], 
                                                        axis = 1)
        
        # Convert results to dictionary and update base dictionaries with new results
        winnerDict = slotMatchUps.set_index('Slot').to_dict()
        seedDict.update(winnerDict['rndWinner'])
        resultProbDict.update(winnerDict['winProb'])
        
        
        # Update tSlots dataframe with winner results
        for team in [('rndWinner', 'Slot'), ('StrongTeam', 'StrongSeed'), ('WeakTeam', 'WeakSeed')]:
            tSlots[team[0]] = tSlots[team[1]].map(lambda result: seedDict.get(result, 'x'))
        tSlots['winProb'] = tSlots['Slot'].map(lambda result: resultProbDict.get(result, 0))
        
        
    # Map team name and original seed to results
    for team in ['StrongTeam', 'WeakTeam', 'rndWinner']:
        tSlots = tSlots.merge(pd.DataFrame(dataDict['teams'].set_index('TeamID')['TeamName']),
                              left_on = team, right_index = True)
        tSlots.rename(columns = {'TeamName' : team + 'Name'}, inplace = True)
                    
        tSlots = tSlots.merge(pd.DataFrame(tSeeds.set_index('TeamID')),
                              left_on = team, right_index = True)
    
        tSlots.rename(columns = {'Seed' : team + 'Seed',
                                 'seedRank' : team + 'seedRank'}, inplace = True)
    
        # Combine rank and name
        tSlots.loc[:, '{}NameAndRank'.format(team)] = map(lambda t: '{:.0f} {}'.format(t[0], t[1]),
                                                           tSlots[['{}seedRank'.format(team),
                                                                   '{}Name'.format(team)]].values.tolist())
    
    # Order & clean results 
    tSlots = tSlots.sort_values('Slot')
    tSlotsClean = tSlots[['Slot', 'StrongTeamSeed', 'WeakTeamSeed', 'rndWinnerSeed', 
                          'StrongTeamNameAndRank', 'WeakTeamNameAndRank', 'rndWinnerNameAndRank', 
                          'winProb']]

    return tSlots, tSlotsClean, matchupData


def createMatchups(matchupDF, 
                   statsDF, 
                   teamID1 = 'TeamID', 
                   teamID2 = 'opponentID', 
                   teamLabel1 = 'team',
                   teamLabel2 = 'opp',
                   calculateDelta = False, 
                   calculateMatchup = False, 
                   extraMatchupCols = [],
                   returnTeamID1StatCols = True,
                   returnTeamID2StatCols = True,
                   returnBaseCols = True,
                   reindex = True):
    ''' Create dataframe game matchups using team statistics to use for modeling
        & parameter selection / performance.
        
        Options:
            Return Statistic columns for each team (returnStatCols Boolean)
            Calculate teamStatistic deltas for the matchup (calculateMatchup Boolean)
            
            Create a tuple of object columns such as conference or seeds (calculateMatchup Boolean)
                
        
        Return dataframe with same number of recors as the matchupDF.
        '''
    baseCols = matchupDF.columns.tolist()
    team1Cols = map(lambda field: '{}{}'.format(teamLabel1, field), statsDF.columns.tolist())
    team2Cols = map(lambda field: '{}{}'.format(teamLabel2, field), statsDF.columns.tolist())

    
    # Merge team statsDF on teamID1
    matchupNew = matchupDF.merge((statsDF.reset_index('TeamID')
                                         .rename(columns = {'TeamID':teamID1})
                                         .set_index(teamID1, append = True)
                                         .rename(columns = dict(map(lambda field: (field, '{}{}'.format(teamLabel1, field)),
                                                                    statsDF.columns.tolist())))
                                         ),
                                        left_on = ['Season', teamID1],
                                        right_index = True,
                                       how = 'left'
                                        )

    # Merge team statsDS on teamID2
    matchupNew = matchupNew.merge((statsDF.reset_index('TeamID')
                                         .rename(columns = {'TeamID':teamID2})
                                         .set_index(teamID2, append = True)
                                         .rename(columns = dict(map(lambda field: (field, '{}{}'.format(teamLabel2, field)),
                                                                    statsDF.columns.tolist())))
                                         ),
                                        left_on = ['Season', teamID2],
                                        right_index = True,
                                        how = 'left'
                                        )



    # Reset index to avoid duplication 
    #   (needed for dataframes {}singleTeam since index is duplicated when
    #   data is stacked as team / opponent instead of Winning Team & Losing Team)
    if reindex == True:
        matchupNew.reset_index(drop = True, inplace = True)


     
    colCount = pd.DataFrame(map(lambda c: re.sub('^{}(?!.*ID$)|^{}(?!.*ID$)'.format(teamLabel1, teamLabel2), '', c), 
                                 matchupNew.columns.tolist()), 
                            columns = ['field']).groupby('field').agg({'field': lambda c: len(c)})

    # Identify numeric columns to calculate deltas
    deltaCols = colCount[colCount['field'] >= 2]
    
    matchupCols = filter(lambda c: matchupNew['{}{}'.format(teamLabel1, c)].dtype.hasobject == True,
                       deltaCols.index.get_level_values('field'))
    
    deltaCols = filter(lambda c: c not in matchupCols,
                       deltaCols.index.get_level_values('field'))
    
 
    
    
    # Calculate statistic deltas if necessary
    if calculateDelta == True:       
        for col in deltaCols:
            matchupNew.loc[:, '{}Delta'.format(col)] = (matchupNew.loc[:, '{}{}'.format(teamLabel1, col)]        
                                                  - matchupNew.loc[:, '{}{}'.format(teamLabel2, col)])
   
        # Update column names
        deltaCols = map(lambda col: '{}Delta'.format(col), deltaCols)
    
    else: deltaCols = []
    
    # Calculate matchup attributes if necessary
    if calculateMatchup == True:
        for col in list(set(matchupCols + extraMatchupCols)):
            matchupNew.loc[:, '{}Matchup'.format(col)] = pd.Series(createMatchupField(matchupNew, 
                                                                                       '{}{}'.format(teamLabel1, col), 
                                                                                       '{}{}'.format(teamLabel2, col), sort = False))
    

         # Update column names
        matchupCols = map(lambda col: '{}Delta'.format(col), list(set(matchupCols + extraMatchupCols)))   
    
    else: matchupCols = []
        
    
    
    # Compile columns to return
    returnCols = list((baseCols * returnBaseCols) 
                        + (team1Cols * returnTeamID1StatCols) 
                        + (team2Cols * returnTeamID2StatCols) 
                        + deltaCols 
                        + matchupCols
                        )
    
    
    
    return matchupNew[returnCols]
    
        
#    if returnStatCols == True:
#        return matchupNew
#    
#    # Don't return stat cols
#    else:
#        deltaReturnCols = filter(lambda c: (c.endswith('Delta')) | (c.endswith('Matchup')) | (c in colCount[colCount['field'] != 2].index.get_level_values('field')),
#                                                       matchupNew.columns.tolist())
#        
#        
#        return matchupNew.loc[:, deltaReturnCols]
#    
#
#    
#    return matchupNew




def pctIntersect(data1, data2):
    
    data1, data2 = np.array(data1), np.array(data2)
    
    upperBound = min(np.max(data1), np.max(data2))
    lowerBound = max(np.min(data1), np.min(data2))
    
    dataAgg = np.hstack((data1, data2))
    
    dataIntersect = filter(lambda x: (x >= lowerBound) & (x <= upperBound), data2)
    
    return len(dataIntersect) / len(dataAgg)



def independentColumnsFilter(df, excludeCols = [], includeCols = []):
    '''Filter dataframe columns down to independent columns for modeling
    
    Return list of column names of independent variables'''

    if len(includeCols) == 0:
        indCols = filter(lambda c: (df[c].dtype.hasobject == False) 
                                    & (c.endswith('ID') == False)
                                    & (c not in excludeCols),
                        df.columns.tolist())
    
    # Use input list
    else:
        indCols = filter(lambda c: (df[c].dtype.hasobject == False) 
                                    & (c.endswith('ID') == False)
                                    & (c in includeCols),
                        df.columns.tolist())
                   
    return indCols


#%% ENVIRONMENT SETUP

# Working Directory Dictionary
pcs = {
    'WaterBug' : {'wd':'C:\\Users\\brett\\Documents\\march_madness_ml',
                  'repo':'C:\\Users\\brett\\Documents\\march_madness_ml\\march_madness'},

    'WHQPC-L60102' : {'wd':'C:\\Users\\u00bec7\\Desktop\\personal\\march_madness_ml',
                      'repo':'C:\\Users\\u00bec7\\Desktop\\personal\\march_madness'},
                      
    'raspberrypi' : {'wd':'/home/pi/Documents/march_madness_ml',
                     'repo':'/home/pi/Documents/march_madness'},
    
    'WINDOWS-ASE2MLR' : {'wd':'C:\\Users\\brett\\Documents\\march_madness_ml',
                  'repo':'C:\\Users\\brett\\Documents\\march_madness_ml\\march_madness'},

}
    
    C:\Users\brett\Documents\march_madness_ml\datasets\2019

# Set working directory & load functions
pc = pcs.get(socket.gethostname())

del(pcs)



# Set up environment
os.chdir(pc['repo'])
from mm_functions import *

os.chdir(pc['wd'])
#execfile('{}\\000_mm_environment_setup.py'.format(pc['repo']))


#%% INTITIAL DATA LOAD
## ############################################################################

#execfile('{}\\010_mm_data_load.py'.format(pc['repo']))


# Read data
dataFiles = os.listdir('datasets\\2019')
dataFiles.sort()

# Remove zip files
dataFiles = list(filter(lambda f: '.csv' in f, dataFiles))


keyNames = map(lambda f: f[:-4], dataFiles)


keyNames = [ 'cities',
             'confTgames',
             'conferences',
             'gameCities',
             'MasseyOrdinals',
             'tGamesC',
             'tGamesD',
             'tSeedSlots',
             'tSeeds',
             'tSlots',
             'rGamesC',
             'rGamesD',
             'Seasons',
             'secTGamesC',
             'secTTeams',
             'teamCoaches',
             'teamConferences',
             'teamSpellings',
             'teams'
             ]

dataDict = {k : pd.read_csv('datasets\\2019\\{}'.format(data), encoding = 'latin1') 
            for k, data in zip(keyNames, dataFiles)}

#%% ADDITIONAL IN GAME METRICS
## ############################################################################

#execfile('{}\\020_mm_in_game_metrics.py'.format(pc['repo']))

# Score gap calculation
for df in ['tGamesC', 'tGamesD', 'rGamesC', 'rGamesD']:
    dataDict[df].loc[:, 'scoreGap'] = (dataDict[df]['WScore'] 
                                        - dataDict[df]['LScore'])
    
for df in ['tGamesD', 'rGamesD']:
   
    # Team Field Goal %
    for team in ['W', 'L']:
        dataDict[df].loc[:, team + 'FGpct'] = (dataDict[df][team + 'FGM'] 
                                                / dataDict[df][team + 'FGA'])
        dataDict[df].loc[:, team + 'FGpct3'] = (dataDict[df][team + 'FGM3'] 
                                                / dataDict[df][team + 'FGA3'])
        dataDict[df].loc[:, team + 'FTpct'] = (dataDict[df][team + 'FTM'] 
                                                / dataDict[df][team + 'FTA'])
        
        dataDict[df].loc[:, team + 'Scorepct'] = (dataDict[df][team + 'FGpct3'] * 3 
                                                    + dataDict[df][team + 'FGpct'] * 2 
                                                    + dataDict[df][team + 'FTpct']
                                                    ) / 6
    # Team Rebound %
    for team in [('W', 'L'), ('L', 'W')]:        
        dataDict[df].loc[:, team[0] + 'ORpct'] = (dataDict[df][team[0] + 'OR'] /
                                                (dataDict[df][team[0] + 'OR'] 
                                                + dataDict[df][team[1] + 'DR'])
                                                )
                                                
        dataDict[df].loc[:, team[0] + 'DRpct'] = (dataDict[df][team[0] + 'DR'] /
                                                (dataDict[df][team[0] + 'DR'] 
                                                + dataDict[df][team[1] + 'OR']))    

        dataDict[df].loc[:, team[0] + 'Rpct'] = ((dataDict[df][team[0] + 'DR'] 
                                                + dataDict[df][team[0] + 'OR']) /
                                            (dataDict[df][team[0] + 'DR'] 
                                              + dataDict[df][team[0] + 'OR']
                                              + dataDict[df][team[1] + 'OR']
                                              + dataDict[df][team[1] + 'DR'])) 
        
# Variable clean up
del(df, team)


#%% BUILD SINGLE TEAM & MODEL DATASETS
## ############################################################################

## ORGANIZE BY TEAM VS OPPONENT INSTEAD OF WTEAM VS LTEAM
# Doubles the number of observations
#   Data shape: (m,n) -> (2m,n)
map(lambda df: dataDict.setdefault('{}singleTeam'.format(df),
                                    buildSingleTeam(dataDict[df])),
    ('rGamesC', 'rGamesD', 'tGamesC', 'tGamesD'))


## REORGANIZE DATA TO MAKE HALF OF GAMES LOSSES
# Maintains same data shape as original
#map(lambda df: dataDict.setdefault('{}modelData'.format(df),
#                                    buildModelData(dataDict[df])),
#    ('tGamesC', 'tGamesD'))



#%% TEAM SEASON STATISTIC METRICS
## ############################################################################

# Season Means
# Conference Champion
# Season Means scaled within season between 0-1

#execfile('{}\\030_mm_team_season_metrics.py'.format(pc['repo']))

for df in ('rGamesC', 'rGamesD'):

    # Isolate score gap for only wins and losses respectively
    dataDict[df + 'singleTeam'].loc[:, 'scoreGapWin'] = (
            dataDict[df + 'singleTeam']['scoreGap'] 
            * dataDict[df + 'singleTeam']['win']
            ).replace(0, np.NaN)
    
    dataDict[df + 'singleTeam'].loc[:, 'scoreGapLoss'] = (
            dataDict[df + 'singleTeam']['scoreGap'] 
            * dataDict[df + 'singleTeam']['win'].replace({0:1, 1:0})
            ).replace(0, np.NaN)
   
    
    # Subset of columns in games dataframes to calculate metrics
    statCols = filter(lambda c: c not in ('DayNum', 'NumOT', 'FTM', 'FGM', 
                                          'FGM3', 'opponentID', 
                                          'Loc', 'TeamID', 'Season'),
                      dataDict[df + 'singleTeam'].columns)
    
    
    # Calculate season averages for each team and store results
    aggDict = dict(zip(statCols, repeat(np.mean, len(statCols))))
    
    dataDict[df+'TeamSeasonStats'] = (
            dataDict[df+'singleTeam'].groupby(['Season', 'TeamID'])
                                     .agg(aggDict)
                                     .fillna(0)
                                     )
 
    
    # Calculate win % over last 8 games
    dataDict[df + 'singleTeam'].sort_values(['Season', 'TeamID', 'DayNum'],
                                                 inplace = True)
    dataDict[df + 'TeamSeasonStats'].loc[:, 'last8'] =  (
            dataDict[df+'singleTeam'].groupby(['Season', 'TeamID'])
                                     .agg({'win': lambda games: np.mean(games[-8:])})
                                     )
    
    
    
    # Drop columns no longer needed from games dataframe
    dataDict[df + 'singleTeam'].drop(['scoreGapWin', 'scoreGapLoss'], 
                                     axis = 1, 
                                     inplace = True)     
    
    # Weight scoreGapWin by win %
#    dataDict[df+'TeamSeasonStats'].loc[:, 'scoreGapWinPct'] = dataDict[df + 'TeamSeasonStats'].loc[:, 'scoreGapWin'] * dataDict[df + 'TeamSeasonStats'].loc[:, 'win']
    

    
    # Merge ranked results with orginal team season metrics
#
#    statsRankNamesDict = {k:v for k,v in 
#                          map(lambda field: (field, '{}Rank'.format(field)), 
#                              statsRank.columns.tolist())
#                          }
#        
#    # Merge Ranking metrics with original metrics
#    dataDict[df+'TeamSeasonStats'] = (
#            dataDict[df+'TeamSeasonStats'].merge(
#                    statsRank.rename(columns = statsRankNamesDict),
#                    left_index = True, 
#                    right_index = True)
#            )
    
#%% CONFERENCE TOURNAMENT CHAMPIONS
## ############################################################################

for df in ('rGamesC', 'rGamesD'):
    
    # Identify conference champions by win in last game of conference tournament
    dataDict['confTgames'].sort_values(['Season', 'ConfAbbrev', 'DayNum'], 
                                        inplace = True)
    
    confWinners = pd.DataFrame(
                    dataDict['confTgames'].groupby(['Season', 
                                                    'ConfAbbrev'])['WTeamID']
                                          .last()
                                          .rename('TeamID')
                    )
    
    # Drop conference from index
    confWinners.reset_index('ConfAbbrev', drop = True, inplace = True)
    
    # Set flag for conference champ and fill missing values with 0
    confWinners.loc[:, 'confChamp'] = 1
    confWinners.set_index('TeamID', append = True, inplace = True)
    
    dataDict[df+'TeamSeasonStats'] = (
            dataDict[df+'TeamSeasonStats'].merge(confWinners, 
                                                 left_index = True, 
                                                 right_index = True, 
                                                 how = 'left')
            )
            
    dataDict[df+'TeamSeasonStats'].loc[:, 'confChamp'].fillna(0, inplace = True)
    
    
    
del(statCols, confWinners, aggDict, df)



#execfile('{}\\040_mm_seeds_ordinals_conferences.py'.format(pc['repo']))

#%% CONFERENCES
## ############################################################################

# Add team conference
for df in map(lambda d: '{}TeamSeasonStats'.format(d),
              ('rGamesC', 'rGamesD')):
    
    dataDict[df].loc[:, 'ConfAbbrev'] = (
            dataDict['teamConferences'].set_index(['Season', 'TeamID'])['ConfAbbrev']
            )

    # Group small conferences together
    dataDict[df].loc[:, 'confGroups'] = (
            map(lambda conf: conf if conf in ('big_east', 
                                              'big_twelve', 
                                              'acc', 
                                              'big_ten', 
                                              'sec'
                                              )
                                else 'other',
                dataDict[df]['ConfAbbrev'].values.tolist())
            )
    
        

#%% TOURNAMENT SEED RANKS
## ############################################################################

# Tourney Seed Rank
dataDict['tSeeds'].loc[:, 'seedRank'] = (
        map(lambda s: float(re.findall('[0-9]+', s)[0]), 
            dataDict['tSeeds']['Seed'].values.tolist())
        )

for df in map(lambda g: g + 'TeamSeasonStats', ('rGamesC', 'rGamesD')):

    dataDict[df].loc[:, 'seedRank'] = (
            dataDict['tSeeds'].set_index(['Season', 'TeamID'])['seedRank'])



#%% END OF SEASON MASSEY ORDINAL RANKS
## ############################################################################

# Get end of regular season rankings for each system
endRegSeason = 133

# Filter for data within regular season
rsRankings = dataDict['MasseyOrdinals'][dataDict['MasseyOrdinals']['RankingDayNum'] <=  endRegSeason]

# Get max rank date for regular season for each system and team
maxRankDate = (rsRankings.groupby(['Season', 
                                   'TeamID', 
                                   'SystemName'])['RankingDayNum']
                         .max()
                         .rename('maxRankDate')
                         )

# Merge and filter for only last rank for regular season
rsRankings = (rsRankings.set_index(['Season', 'TeamID', 'SystemName'])
                        .merge(pd.DataFrame(maxRankDate),
                               left_index = True,
                               right_index = True))


rsRankings = rsRankings[rsRankings['RankingDayNum'] == rsRankings['maxRankDate']]


# Pivot rankings on Season and TeamID
rsRankings = (rsRankings.reset_index()
                         .pivot_table(index = ['Season', 'TeamID'],  
                               columns = 'SystemName',
                               values = 'OrdinalRank',
                               aggfunc = np.mean)
                         )

# Calculate Median, Mean and Total # of end of season ranks
rsRankings.loc[:, 'median'] = rsRankings.median(axis = 1)
rsRankings.loc[:, 'mean'] = rsRankings.mean(axis = 1)
rsRankings.loc[:, 'count'] = rsRankings.count(axis = 1)




# Add team ranks to season stats    
for df in map(lambda g: g + 'TeamSeasonStats', ('rGamesC', 'rGamesD')):     
    dataDict[df].loc[:, 'OrdinalRank'] = rsRankings['median']



# Store ranking dataframe
dataDict['endSeasonRanks'] = rsRankings


#%% SCALED TEAM SEASON STATISTICS
## ############################################################################    
  
#for df in ('rGamesC', 'rGamesD'):    
#    
#    # Rank teams by each season stat metrics within 
#    rankDesc = filter(lambda field: field in ('pointsAllowed', 'TO', 'PF'), 
#                      dataDict[df + 'TeamSeasonStats'].columns.tolist())
#
#  
#    # Rank teams within season for each metric and use % for all teams between 0 and 1 (higher = better)
#    # Change from sequential ranker to minmax scale to avoid unrealistic spread (4/22/19)
#    dataDict[df+'TeamSeasonStatsRank'] = (dataDict[df + 'TeamSeasonStats']
#                                            .groupby('Season')
#                                            .apply(lambda m: (m - m.min()) 
#                                                    / (m.max() - m.min()))
#                                            )
#    
#    # Switch fields where lower values are better
#    dataDict[df+'TeamSeasonStatsRank'].loc[:, rankDesc] = (
#            1 - dataDict[df+'TeamSeasonStatsRank'][rankDesc]
#            )
    


    
#%% HANDLE MISSING DATA
## ############################################################################


# Missing Values Dict
fillDict = {'LSeed':'NA', 
            'seedRank':17,
            'OrdinalRank': 176}

    
for df in map(lambda g: g + 'TeamSeasonStats', ('rGamesC', 'rGamesD')):
    dataDict[df].fillna(fillDict, inplace = True)
    
    
    
    
# Memory Cleanup (consumes ~122 MB)
del(maxRankDate, rsRankings, dataDict['MasseyOrdinals'], 
    fillDict, endRegSeason, df)


#%% STRENGTH METRICS
## ############################################################################



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


# Win weighted by opponent win%
matchups.loc[:, 'teamStrength'] = (
        matchups['win'] * matchups['oppwin']
        )


# Opponent strength Metrics
matchups.loc[:, 'oppStrength'] = (
        (matchups['oppscoreGap'] * matchups['oppwin'])
        )

#matchups.loc[:, 'oppStrengthWin'] = (
#        (matchups['oppscoreGapWin'] * matchups['oppwin'])
#        )

# Win weighted by opponent strength
matchups.loc[:, 'teamStrength2'] = (
        matchups['win'] * matchups['oppStrength']
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





# Identify strength columns for aggregation and calculating team performance
strengthMetrics = filter(lambda metric: metric.find('Strength') >= 0, 
                         matchups.columns.tolist())


# Calculate team season means for each metric
strengthDF = (matchups.groupby(['Season', 'TeamID'])
                      .agg(dict(zip(strengthMetrics, 
                                    repeat(np.mean, len(strengthMetrics))))
                            )
                )
   
   
# Plot heat map of strength metric correlations
fig, ax = plt.subplots(nrows = 1, ncols = 1, 
                       figsize = (0.9*GetSystemMetrics(0)//96, 
                                  0.8*GetSystemMetrics(1)//96))

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
            

# Scale Data between 0 and 1 using minmax to avoid negatives and append values as '[metric]Rank'
# Change from merge to just replace scaled data (4/23/19)
strengthDF = (strengthDF.groupby('Season')
                        .apply(lambda m: (m - m.min()) / (m.max() - m.min()))
                        )

#%% STRENGTH METRIC ANALYSIS

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






nRows = int(np.ceil(len(strengthDF.columns)**0.5))
nCols = int(np.ceil(len(strengthDF.columns)/nRows))

# Plot distributions of tourament team strength metrics vs. winning team strength metrics
fig, ax = plt.subplots(nrows = nRows, 
                       ncols = nCols, 
                       figsize = (0.9*GetSystemMetrics(0)//96, 
                                  0.8*GetSystemMetrics(1)//96))

for i, metric in enumerate(strengthDF.columns):
    sns.distplot(matchups.groupby(['Season', 'TeamID'])['team{}'.format(metric)].mean(), 
                 hist = True, 
                 ax = ax[i//nCols, i%nCols], 
                 kde_kws={"shade": True}, 
                 label = 'unique')
    
    sns.distplot(matchups['team{}'.format(metric)][matchups.index.get_level_values('win') == 1], 
                 hist = True, 
                 ax = ax[i//nCols, i%nCols], 
                 kde_kws={"shade": True}, 
                 label = 'win')
   
    ax[i//nCols, i%nCols].grid(True)
    ax[i//nCols, i%nCols].legend()
 
    
if len(ax.flatten()) > len(strengthDF.columns):
    for i in range(len(strengthDF.columns), len(ax.flatten())):
        ax.flatten()[i].axis('off')   
        
fig.tight_layout(rect=[0,0,1,0.97])
fig.suptitle('Strength Metrics of Teams and Winning Teams', fontsize = 20)
fig.show()




# Plot heat maps of win % by metric bins

# Desired # of Bins
mBins = 10


fig1, ax1 = plt.subplots(nrows = nRows, 
                       ncols = nCols, 
                       figsize = (0.9*GetSystemMetrics(0)//96, 
                                  0.8*GetSystemMetrics(1)//96))

fig2, ax2 = plt.subplots(nrows = nRows, 
                       ncols = nCols, 
                       figsize = (0.9*GetSystemMetrics(0)//96, 
                                  0.8*GetSystemMetrics(1)//96))

for i, metric in enumerate(strengthDF.columns):
    
    heatMapData = pd.pivot_table(matchups.reset_index(['win', 'TeamID'])
                                        .applymap(lambda p: 
                                            round(p*mBins, 0)/ mBins),
                index = 'opp{}'.format(metric),
                columns = 'team{}'.format(metric),
                values = ['win', 'TeamID'],
                aggfunc = {'win':np.mean, 'TeamID':len}
                )
                                
    
    sns.heatmap(heatMapData.loc[:, heatMapData.columns.get_level_values(0) == 'win'], 
                annot = True, 
                fmt='.0%',
                mask = heatMapMask(heatMapData.loc[:, heatMapData.columns.get_level_values(0) == 'win'], k = -1, corner = 'lower_left'),
#                square = True,
                cmap = 'RdYlGn',
                linewidths = 1, 
                linecolor = 'k',
                ax = ax1[i//nCols, i%nCols]
                )
    ax1[i//nCols, i%nCols].invert_yaxis()
    ax1[i//nCols, i%nCols].invert_xaxis()
    ax1[i//nCols, i%nCols].set_xticklabels(
            heatMapData.loc[:, heatMapData.columns.get_level_values(0) == 'TeamID'].columns.droplevel(0)
            )
  
    sns.heatmap(heatMapData.loc[:, heatMapData.columns.get_level_values(0) == 'win'], 
                annot = heatMapData.loc[:, heatMapData.columns.get_level_values(0) == 'TeamID'], 
                fmt='.0f',
                mask = heatMapMask(heatMapData.loc[:, heatMapData.columns.get_level_values(0) == 'win'], k = -1, corner = 'lower_left'),
#                square = True,
                cmap = 'RdYlGn',
                linewidths = 1, 
                linecolor = 'k',
                ax = ax2[i//nCols, i%nCols]
                )
    ax2[i//nCols, i%nCols].invert_yaxis()
    ax2[i//nCols, i%nCols].invert_xaxis()
    ax2[i//nCols, i%nCols].set_xticklabels(
            heatMapData.loc[:, heatMapData.columns.get_level_values(0) == 'TeamID'].columns.droplevel(0)
            )
    
for ax in (ax1, ax2):
    if len(ax.flatten()) > len(strengthDF.columns):
        for i in range(len(strengthDF.columns), len(ax.flatten())):
            ax.flatten()[i].axis('off')
    
 


fig1.tight_layout(rect=[0,0,1,0.97])
fig1.suptitle('Win % for Metric Bins {}'.format(mBins), fontsize = 20)
fig1.show()

fig2.tight_layout(rect=[0,0,1,0.97])
fig2.suptitle('# of Games for Metric Bins {}'.format(mBins), fontsize = 20)
fig2.show()





#%% STRENGTH METRIC IMPORTANCE AND SELECTION


# Models for getting feature importance
treeModels = {'gb': GradientBoostingClassifier(random_state = 1127, 
                                               n_estimators = 20),
              'et': ExtraTreesClassifier(random_state = 1127, 
                                         n_estimators = 20),
              'rf': RandomForestClassifier(random_state = 1127,
                                           n_estimators = 20)}

trainIndex, testIndex = train_test_split(range(matchups.shape[0]), 
                                         test_size = 0.2)

# Create recursive feature selection models for each treeModel
rfeCVs = {k:RFECV(v, cv = 5) for k,v in treeModels.iteritems()}

# Train models
map(lambda tree: tree.fit(matchups.iloc[trainIndex,:], 
                          matchups.iloc[trainIndex,:].index.get_level_values('win'))
    , rfeCVs.itervalues())


# Score models on train & test data
map(lambda tree: 
    map(lambda idx: 
        tree.score(matchups.iloc[idx,:], 
                      matchups.iloc[idx,:].index.get_level_values('win')),
        (trainIndex, testIndex)
    )
    , rfeCVs.itervalues())

# # of features selected for each model
map(lambda rfeCV: 
    (rfeCV[0], rfeCV[1].n_features_)
    , rfeCVs.iteritems())    
    

# Get selected features for each model
featureImportance = pd.concat(
        map(lambda rfeCV:
            pd.DataFrame(
                zip(repeat(rfeCV[0], sum(rfeCV[1].support_)),
                    matchups.columns[rfeCV[1].support_],
                    rfeCV[1].estimator_.feature_importances_),
                columns = ['model', 'metric', 'importance']
                ).sort_values(['model', 'importance'], ascending = [True, False])
                , rfeCVs.iteritems())
        , axis = 0)



featureRank = pd.concat(
        map(lambda rfeCV:
            pd.DataFrame(
                zip(repeat(rfeCV[0], len(rfeCV[1].ranking_)),
                    matchups.columns,
                    rfeCV[1].ranking_),
                columns = ['model', 'metric', 'ranking']
                ).sort_values(['model', 'ranking'], ascending = [True, True])
                , rfeCVs.iteritems())
        , axis = 0)




# Aggregate Feature Importance Metrics 
featureImportanceAgg = (featureImportance.groupby('metric')
                                         .agg({'importance':np.sum,
                                               'model':len})
                        ).sort_values('importance', ascending = False)    
 
   
featureRankAgg = (featureRank.groupby('metric')
                             .agg({'ranking':np.mean})
#                             .rank()
                        ).sort_values('ranking', ascending = True) 
    




# Store strength metrics
dataDict['strengthMetrics'] = strengthDF
    

# Map only relavent features
for df in map(lambda g: g + 'TeamSeasonStats', ('rGamesC', 'rGamesD')):
    dataDict[df].loc[:, 'spreadStrength'] = strengthDF['spreadStrength']
    dataDict[df].loc[:, 'teamStrength2'] = strengthDF['teamStrength2'] 
    dataDict[df].loc[:, 'oppStrength'] = strengthDF['oppStrength']


#%% Wins Against TopN Teams

# use 'spreadStrengthDelta' as teamstrength since it highest feature importance
# Find best metric for # of wins agains topN teams
topNlist = range(10, 301, 10)


matchups = createMatchups(
        matchupDF = dataDict['rGamesCsingleTeam'][['Season', 'TeamID', 'opponentID', 'win']],
        statsDF = (strengthDF[['spreadStrength', 'teamStrength2']]
                            .groupby('Season')
                            .rank(ascending = False, method = 'min')
                            ),
        teamID1 = 'TeamID', 
        teamID2 = 'opponentID',
        teamLabel1 = 'team',
        teamLabel2 = 'opp',
        returnBaseCols = True,
        returnTeamID1StatCols = False,
        returnTeamID2StatCols = True,
        calculateDelta = False,
        calculateMatchup = False)


topNWins = (pd.concat(
        map(lambda topN: 
            (matchups[matchups['oppspreadStrength'] <= topN].groupby(['Season', 'TeamID'])
                                                            .agg({'win':np.sum})),
            topNlist),
        axis = 1))


# Fill missing values and rename columns
topNWins.fillna(0, inplace = True)
topNWins.columns = map(lambda topN: 'wins{}'.format(str(topN).zfill(3)), topNlist)



# Create tournament matchups with wins against topN counts
matchups = createMatchups(
        matchupDF = dataDict['tGamesCsingleTeam'][['Season', 'TeamID', 'opponentID', 'win']],
        statsDF = topNWins,
        teamID1 = 'TeamID', 
        teamID2 = 'opponentID',
        teamLabel1 = 'team',
        teamLabel2 = 'opp',
        returnBaseCols = True,
        returnTeamID1StatCols = False,
        returnTeamID2StatCols = False,
        calculateDelta = True,
        calculateMatchup = False)


matchups.set_index(['Season', 'TeamID', 'opponentID', 'win'], inplace = True)


# Fit and score feature selection

# Train models
map(lambda tree: tree.fit(matchups.iloc[trainIndex,:], 
                          matchups.iloc[trainIndex,:].index.get_level_values('win'))
    , rfeCVs.itervalues())


# Score models on train & test data
map(lambda tree: 
    map(lambda idx: 
        tree.score(matchups.iloc[idx,:], 
                      matchups.iloc[idx,:].index.get_level_values('win')),
        (trainIndex, testIndex)
    )
    , rfeCVs.itervalues())

# # of features selected for each model
map(lambda rfeCV: 
    (rfeCV[0], rfeCV[1].n_features_)
    , rfeCVs.iteritems())  
    
 
    
    
# Get selected features for each model
featureImportanceTopN = pd.concat(
        map(lambda rfeCV:
            pd.DataFrame(
                zip(repeat(rfeCV[0], sum(rfeCV[1].support_)),
                    matchups.columns[rfeCV[1].support_],
                    rfeCV[1].estimator_.feature_importances_),
                columns = ['model', 'metric', 'importance']
                ).sort_values(['model', 'importance'], ascending = [True, False])
                , rfeCVs.iteritems())
        , axis = 0)


# Aggregate Feature Importance Metrics 
featureImportanceTopNAgg = (featureImportanceTopN.groupby('metric')
                                                 .agg({'importance':np.sum,
                                                       'model':len})
                        ).sort_values('importance', ascending = False)    
 

    
    
    
    
fig, ax = plt.subplots(1, 
                       figsize = (0.9*GetSystemMetrics(0)//96, 
                                  0.8*GetSystemMetrics(1)//96))

sns.barplot(x = 'metric', 
            y = 'importance', 
            data = featureImportanceTopNAgg.reset_index().sort_values('metric'), 
            ax = ax)

ax.tick_params(axis='x', rotation=90)



ax2 = ax.twinx()
plt.plot((pd.melt(matchups.loc[matchups.index.get_level_values('win') == 1, 
                               map(lambda col: col in featureImportanceTopNAgg.index.get_level_values('metric'),
                                   matchups.columns)])
            .groupby('variable')
            .agg({'value': lambda data: 
                len(filter(lambda delta: delta > 0, data))/ len(data)})),
        'go--', linewidth=2, markersize=12)

ax2.grid()
ax2.set_ylabel('Win %')
fig.tight_layout(rect=[0,0,1,0.97])
fig.suptitle('Wins Against Top N Teams Feature Rank', fontsize = 20)
fig.show()


# Store topNWins
dataDict['topNWins'] = topNWins

for df in map(lambda g: g + 'TeamSeasonStats', ('rGamesC', 'rGamesD')):
    dataDict[df].loc[:, 'wins160'] = topNWins['wins160']
    dataDict[df].loc[:, 'wins090'] = topNWins['wins090']


#%% DEV
## ############################################################################

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
    
        



