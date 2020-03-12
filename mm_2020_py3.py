# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:25:16 2019

@author: u00bec7
"""


#%% PACKAGES
## ############################################################################

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

    colSummary = list(map(lambda c: (c, 
                                df[c].dtype.type,
                                df[c].dtype.hasobject),
                    df.columns.tolist()
                    ))
                    
    if returnDF == True:
      colSummary = pd.DataFrame(colSummary,
                                columns = ['colName', 'colDataType', 'isObject'])
    
    return colSummary



def buildSingleTeam(df, sharedCols = ['Season', 'DayNum', 'WLoc', 'NumOT', 'scoreGap', 'WTeamID', 'LTeamID']):
    '''Create dataframe where all data is aligned from the perspective of a single team
        versus an opponent. Not organized by wins & loss categories. 
        
        Will generate a dataframe with 2x as many records'''


    colsWinTemp = list(filter(lambda col: col.startswith('W') & (col not in sharedCols),
                         df.columns.tolist()))
    
    colsLossTemp = list(filter(lambda col: col.startswith('L') & (col not in sharedCols),
                          df.columns.tolist())) 
    
    
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
    aggDF.loc[:, 'pointsAllowed'] = list(map(lambda g: g[0] - g[1], 
                                     aggDF.loc[:, ['Score', 'scoreGap']].values.tolist()))
        

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
    loss.columns = list(
            map(lambda field: re.sub('^W(?!Loc)|^L', 
                                            lambda x: 'L' if x.group(0) == 'W' else 'W', 
                                            field),
                       loss.columns)
        )
    
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

    
    baseCols = list(filter(lambda c: c not in ['WTeamID', 'LTeamID'],
                      gameDF))
    
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
    matchup = list(map(lambda m: list(m), matchup))
    map(lambda m: m.sort(), matchup)
    matchup = list(map(lambda l: tuple(l), matchup))
    
    return matchup



def createMatchupField(df, label1, label2, sort = False):
    '''Create matchup key from two fields sorted alphabetically
        Return a list of sorted tuples with label fields.'''
    
    matchup = zip(df[label1].values.tolist(),
                  df[label2].values.tolist())
    
    if sort == True:
        matchup = list(map(lambda m: list(m), matchup))
        map(lambda m: m.sort(), matchup)
        matchup = list(map(lambda l: tuple(l), matchup))
    
    return matchup



def modelAnalysis(model, data = [], 
                  targetCol = None, 
                  indCols = None, 
                  testTrainDataList = [], 
                  testTrainSplit = 0.2):
    
    if indCols == None:
        indCols = list(filter(lambda col: ((data[col].dtype.hasobject == False) 
                                        & (col != targetCol)), 
                         data.columns.tolist()))
    
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
                          excludeCols = [],
                          indCols = None, 
                          testTrainDataList = [], 
                          testTrainSplit = 0.2,
                          gridSearch = False,
                          paramGrid = None,
                          scoring = None,
                          crossFolds = 5):

    '''Perform model pipeline and perfrom grid search if necessary.
    
        Return dictionary with Pipeline, predictions, probabilities,
        test data, train data, rocCurve, auc, and accuracy'''

    # Remove all non numeric columns from model
    if indCols == None:
        indCols = list(filter(lambda col: ((data[col].dtype.hasobject == False) 
                                        & (col != targetCol)
                                        & (col not in excludeCols)), 
                         data.columns.tolist()))
    
    # Assign test/train datasets if defined, otherwise perform test/train split
    if len(testTrainDataList) == 4:                                           
        xTrain, xTest, yTrain, yTest = testTrainDataList
    
    else:
        xTrain, xTest, yTrain, yTest = train_test_split(data[indCols], 
                                                        data[targetCol],
                                                        test_size = testTrainSplit)
    # Perform grid search if necessary
    if gridSearch == True:
        modelPipe = GridSearchCV(modelPipe, 
                                 paramGrid, 
                                 scoring = scoring, 
                                 cv = crossFolds)
    
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
            'independentVars' : indCols,
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
    teamData = list(map(lambda v: tuple(v), teamDF.values.tolist()))
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
            tSlots[team + 'Team'] = list(map(lambda t: seedsDict.get(t, 'x'),
                                        tSlots[team + 'Seed'].values.tolist()))
        
        # Generate sorted matchup Teams tuple
        tSlots['matchup'] = generateMatchupField(tSlots, 'Team', 'Strong', 'Weak')
        
        # Lookup winner
        tSlots['winner'] = list(map(lambda m: matchupDict.get(m, 'x'),
                               tSlots['matchup'].values.tolist()))
        
        # Update seeds dict with winner & slot    
        seedsDict.update(dict(tSlots[['Slot', 'winner']].values.tolist()))
   
    return tSlots




def performPCA(data, pcaExcludeCols = [], 
               scaler = StandardScaler(), 
               dataLabel = '', 
               plotComponents = True,
               labelFontSize = 20,
               titleFontSize = 24,
               tickFontSize = 16):
    
    '''Perform principle component analysis on pandas dataframe
        (excluding any defined or non-numeric columns).
        
        First scale the data (default StandardScaler) and then perform
        full PCA.
        
        Plot component weightings and explained variance if desired.
        
        Return principle component array with weightings
    '''


    # Filter columns for PCA
    pcaCols = [
        col for col in data.columns.tolist()
        if ((data[col].dtype.hasobject == False) 
                & (col not in pcaExcludeCols))
        ]


    # Build & fit PCA pipeline
    pcaPipe = Pipeline([('sScale', scaler), 
                        ('pca', PCA(n_components = len(pcaCols), 
                                    random_state = 1127))])
   
    pcaPipe.fit(data[pcaCols])


    # Plot PCA Results
    if plotComponents == True:
        
        fig, axs = plt.subplots(1, 2, figsize = (0.9*GetSystemMetrics(0)//96, 
                                                 0.8*GetSystemMetrics(1)//96)
                                )
        
        plt.suptitle(' '.join([dataLabel, 'PCA Analysis']).strip(), 
                     fontsize = 36)    
        
        
        # Determine how many labels to plot so that axis isn' cluttered
        axisLabelFreq = len(pcaCols) // 20 + 1
        
        xAxisLabelsMask =  list(
                map(lambda x: x % axisLabelFreq == 0
                    , range(len(pcaCols)))
                )
        
        xAxisLabels = data[pcaCols].columns[xAxisLabelsMask]
        
        # Plot feature weights for each component
        sns.heatmap(pcaPipe.named_steps['pca'].components_, 
                    square = False, 
                    cmap = 'coolwarm', 
                    ax=axs[0], 
                    #annot=True,
                    xticklabels = axisLabelFreq,
                    yticklabels = axisLabelFreq)
        
        # Add feature lables & format plot    
        axs[0].set_xticklabels(xAxisLabels, 
                               fontsize = tickFontSize,
                               rotation = 90)
        
        axs[0].tick_params(labelsize = tickFontSize)
        
        axs[0].set_title('PCA Components Feature Weights', 
                         fontsize = titleFontSize)
        axs[0].set_xlabel('Feature', 
                          fontsize = labelFontSize)
        axs[0].set_ylabel('PCA #', 
                          fontsize = labelFontSize)
           
        
        # Plot explained variance curve
        axs[1].plot(
            range(pcaPipe.named_steps['pca'].n_components_), 
            np.cumsum(pcaPipe.named_steps['pca'].explained_variance_ratio_), 
            '-bo', 
            markersize = 20, 
            linewidth = 10)
        
        
        # Convert y-axis to %
        axs[1].set_yticklabels(map(lambda v: '{:.0%}'.format(v), 
                                   axs[1].get_yticks()
                                   )
                               )
        
        axs[1].set_title('Explained Variance vs. # of Components', 
                         fontsize = titleFontSize)
        
        axs[1].set_xlabel('# of Features', 
                          fontsize = labelFontSize)
        
        axs[1].set_ylabel('Explained Variance', 
                          fontsize = labelFontSize)
        
        axs[1].tick_params(labelsize = tickFontSize)
        
        axs[1].grid()


    return pcaPipe



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


def logProcessTime(comment, timeLog):
    '''Append process comment and process time to timeLog and print'''
    
    timeLog.append(
            (comment
             , timer()
             , datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )
    
    print('{}\t{}\t{}'.format(*timeLog[-1]))

    return


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
    objectCols = list(filter(lambda c: df[c].dtype.hasobject, dfCols))
    excludeCols += objectCols
    
    numericCols = list(filter(lambda c: c not in excludeCols, dfCols))
    
    # Split numeric columns between the two teams
    label1Cols = list(filter(lambda c: colsTeamFilter(c, label1), numericCols))
    label2Cols = list(filter(lambda c: colsTeamFilter(c, label2), numericCols))

    len1, len2 = len(label1Cols), len(label2Cols)
    
    # Make sure labels are in both datasets 
    # (filter longest list to elements in shortest list)
    if len1 >= len2:
        label1Cols = list(filter(lambda c: c[1:] in map(lambda cc: c[1:], label2Cols), 
                            label1Cols))
    else:
        label2Cols = list(filter(lambda c: c[1:] in map(lambda cc: c[1:], label1Cols), 
                            label2Cols))
    
    # Sort columns for zippping 
    label1Cols.sort()
    label2Cols.sort()
    
    
    # Create dataframe of metric deltas (label1 - label2)
    l1DF = df[label1Cols]
    l2DF = df[label2Cols]

    l2DF.columns = label1Cols

    deltaDF = l1DF - l2DF    
    deltaDF.columns = list(map(lambda colName: colName[1:] + 'Delta', label1Cols))

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
    while len(list(filter(lambda result: result == 'x', tSlots['rndWinner'].values.tolist()))) > 0:
        
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
        tSlots.loc[:, '{}NameAndRank'.format(team)] = list(map(lambda t: '{:.0f} {}'.format(t[0], t[1]),
                                                           tSlots[['{}seedRank'.format(team),
                                                                   '{}Name'.format(team)]].values.tolist()))
    
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
    
    team1Cols = list(
        map(lambda field: '{}{}'.format(teamLabel1, field), 
            statsDF.columns.tolist()
            )
        )
    
    team2Cols = list(
        map(lambda field: '{}{}'.format(teamLabel2, field), 
            statsDF.columns.tolist()
            )
        )

    
    # Merge team statsDF on teamID1
    matchupNew = matchupDF.merge(
        (statsDF.reset_index('TeamID')
                .rename(columns = {'TeamID':teamID1})
                .set_index(teamID1, append = True)
                .rename(columns = dict(
                    map(lambda field: 
                        (field, '{}{}'.format(teamLabel1, field)),
                        statsDF.columns.tolist()
                        )
                    )
                )
            ),
        left_on = ['Season', teamID1],
        right_index = True,
        how = 'left'
        )


    # Merge team statsDS on teamID2
    matchupNew = matchupNew.merge(
        (statsDF.reset_index('TeamID')
                .rename(columns = {'TeamID':teamID2})
                .set_index(teamID2, append = True)
                .rename(columns = dict(
                    map(lambda field: 
                        (field, '{}{}'.format(teamLabel2, field)),
                        statsDF.columns.tolist()
                        )
                    )
                )
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


     
    colCount = pd.DataFrame(
        map(lambda c: re.sub(
            '^{}(?!.*ID$)|^{}(?!.*ID$)'.format(teamLabel1, teamLabel2), 
            '', c), 
            matchupNew.columns.tolist()), 
        columns = ['field']).groupby('field').agg({'field': lambda c: len(c)})

    # Identify numeric columns to calculate deltas
    deltaCols = colCount[colCount['field'] >= 2]
    
    matchupCols = list(
        filter(lambda c: 
               matchupNew['{}{}'.format(teamLabel1, c)].dtype.hasobject == True,
               deltaCols.index.get_level_values('field')
               )
            )
    
    deltaCols = list(filter(lambda c: c not in matchupCols,
                       deltaCols.index.get_level_values('field')))
    
 
    
    
    # Calculate statistic deltas if necessary
    if calculateDelta == True:       
        for col in deltaCols:
            matchupNew.loc[:, '{}Delta'.format(col)] = (
                matchupNew.loc[:, '{}{}'.format(teamLabel1, col)]        
                - matchupNew.loc[:, '{}{}'.format(teamLabel2, col)]
                )
   
        # Update column names
        deltaCols = list(map(lambda col: '{}Delta'.format(col), deltaCols))
    
    else: deltaCols = []
    
    # Calculate matchup attributes if necessary
    if calculateMatchup == True:
        for col in list(set(matchupCols + extraMatchupCols)):
            matchupNew.loc[:, '{}Matchup'.format(col)] = pd.Series(
                createMatchupField(matchupNew, 
                                   '{}{}'.format(teamLabel1, col), 
                                   '{}{}'.format(teamLabel2, col), 
                                   sort = False)
                )
    

         # Update column names
        matchupCols = list(
            map(lambda col: '{}Matchup'.format(col), 
                list(set(matchupCols + extraMatchupCols))
                )
            )
    
    else: matchupCols = []
        
    
    
    # Compile columns to return
    returnCols = list((baseCols * returnBaseCols) 
                        + (team1Cols * returnTeamID1StatCols) 
                        + (team2Cols * returnTeamID2StatCols) 
                        + deltaCols 
                        + matchupCols
                        )
    
    
    
    return matchupNew[returnCols]
    
        




def pctIntersect(data1, data2):
    
    data1, data2 = np.array(data1), np.array(data2)
    
    upperBound = min(np.max(data1), np.max(data2))
    lowerBound = max(np.min(data1), np.min(data2))
    
    dataAgg = np.hstack((data1, data2))
    
    dataIntersect = list(filter(lambda x: (x >= lowerBound) & (x <= upperBound), data2))
    
    return len(dataIntersect) / len(dataAgg)



def independentColumnsFilter(df, excludeCols = [], includeCols = []):
    '''Filter dataframe columns down to independent columns for modeling
    
    Return list of column names of independent variables'''

    if len(includeCols) == 0:
        indCols = list(filter(lambda c: (df[c].dtype.hasobject == False)
                                    & (c.endswith('ID') == False)
                                    & (c not in excludeCols),
                        df.columns.tolist()))
    
    # Use input list
    else:
        indCols = list(filter(lambda c: (df[c].dtype.hasobject == False) 
                                    & (c.endswith('ID') == False)
                                    & (c in includeCols),
                        df.columns.tolist()))
                   
    return indCols


#%% ENVIRONMENT SETUP
## ###########################################################################


# Working Directory Dictionary
pcs = {
    'WaterBug' : {'wd':'C:\\Users\\brett\\Documents\\march_madness_ml',
                  'repo':'C:\\Users\\brett\\Documents\\march_madness_ml\\march_madness'},

    'WHQPC-L60102' : {'wd':'C:\\Users\\u00bec7\\Desktop\\personal\\march_madness_ml',
                      'repo':'C:\\Users\\u00bec7\\Desktop\\personal\\march_madness'},
                      
    'raspberrypi' : {'wd':'/home/pi/Documents/march_madness_ml',
                     'repo':'/home/pi/Documents/march_madness'},
    
    'jeebs' : {'wd':'C:\\Users\\brett\\Documents\\march_madness_ml',
                  'repo':'C:\\Users\\brett\\Documents\\march_madness_ml\\march_madness'},

}
    

# Set working directory & load functions
pc = pcs.get(socket.gethostname())

del(pcs)



# Set up environment
os.chdir(pc['repo'])
#from mm_functions import *

os.chdir(pc['wd'])
#execfile('{}\\000_mm_environment_setup.py'.format(pc['repo']))


# Initiate timing log
timer()
timeLog = []


#%% INTITIAL DATA LOAD
## ############################################################################

#execfile('{}\\010_mm_data_load.py'.format(pc['repo']))


# Read data
dataFolder = 'datasets\\2020'
dataFiles = os.listdir(dataFolder)
dataFiles.sort()

# Remove zip files
dataFiles = list(filter(lambda f: '.csv' in f, dataFiles))


#keyNames = list(map(lambda f: f[:-4], dataFiles))


keyNames = {
    'cities' : 'Cities.csv',
    'conferences' :  'Conferences.csv',
    'confTgames' :  'MConferenceTourneyGames.csv',
    'gameCities' :  'MGameCities.csv',
    'MasseyOrdinals' :  'MMasseyOrdinals.csv',
    'tGamesC' :  'MNCAATourneyCompactResults.csv',
    'tGamesD' :  'MNCAATourneyDetailedResults.csv',
    'tSeedSlots' :  'MNCAATourneySeedRoundSlots.csv',
    'tSeeds' :  'MNCAATourneySeeds.csv',
    'tSlots' :  'MNCAATourneySlots.csv',
    'rGamesC' :  'MRegularSeasonCompactResults.csv',
    'rGamesD' :  'MRegularSeasonDetailedResults.csv',
    'Seasons' :  'MSeasons.csv',
    'secTGamesC' :  'MSecondaryTourneyCompactResults.csv',
    'secTTeams' :  'MSecondaryTourneyTeams.csv',
    'teamCoaches' :  'MTeamCoaches.csv',
    'teamConferences' :  'MTeamConferences.csv',
    'teamSpellings' :  'MTeamSpellings.csv',
    'teams' :  'MTeams.csv',
    }



dataDict = {k : pd.read_csv('{}\\{}'.format(dataFolder, data), 
                            encoding = 'latin1') 
            for k, data in keyNames.items()
            }

# Log process time
logProcessTime('data load', timeLog)

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


# Log process time
logProcessTime('in game metrics', timeLog)


#%% BUILD SINGLE TEAM & MODEL DATASETS
## ############################################################################

## ORGANIZE BY TEAM VS OPPONENT INSTEAD OF WTEAM VS LTEAM
# Doubles the number of observations
#   Data shape: (m,n) -> (2m,n)
[dataDict.setdefault('{}singleTeam'.format(df),
                                    buildSingleTeam(dataDict[df]))
    for df in ('rGamesC', 'rGamesD', 'tGamesC', 'tGamesD')
    ]

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
    statCols = list(filter(lambda c: c not in ('DayNum', 'NumOT', 'FTM', 'FGM', 
                                          'FGM3', 'opponentID', 
                                          'Loc', 'TeamID', 'Season'),
                      dataDict[df + 'singleTeam'].columns))
    
    
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
    

# Log process time
logProcessTime('single team datasets', timeLog)

    
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
 
    
#%% PRINCIPLE COMPONENT ANALYSIS
## ############################################################################
    
# Perform PCA analysis on each Team Stats dataframe
#   Steps (use pipeline):
#       Scale Data
#       Calculate PCA for same number of dimensions in dataset to
#       develop contribution of each axis


toPerformPCA = True


# Base columns (carry over from 2019)
colsBase = ['Season', 'DayNum', 'WLoc', 'NumOT', 'scoreGap']

if toPerformPCA == True:
    
    teamStatsDFs = list(
            filter(lambda dfName: 
                len(re.findall('r.*TeamSeasonStats.*', dfName))>0, 
                dataDict.keys())
            )
    
    # gamesStatsDFs = list(
    #         filter(lambda dfName: 
    #             len(re.findall('r.*SeasonStatsMatchup.*', dfName))>0, 
    #             dataDict.keys())
    #         )

    
    pcaDict = {}
    

    # PCA Analysis on Team season statistics  
    for df in teamStatsDFs:

        pcaDict[df] = performPCA(
            data = dataDict[df],
            pcaExcludeCols = ['Season', 'WTeamID', 'LTeamID'],
            scaler = StandardScaler(),
            dataLabel = df,
            plotComponents = True
            )

  
    
    
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



# Log process time
logProcessTime('conference tournament champs', timeLog)


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
            list(map(lambda conf: conf if conf in ('big_east', 
                                              'big_twelve', 
                                              'acc', 
                                              'big_ten', 
                                              'sec'
                                              )
                                else 'other',
                dataDict[df]['ConfAbbrev'].values.tolist())
            ))
    
        

#%% TOURNAMENT SEED RANKS
## ############################################################################

# Tourney Seed Rank
dataDict['tSeeds'].loc[:, 'seedRank'] = (
        list(map(lambda s: float(re.findall('[0-9]+', s)[0]), 
            dataDict['tSeeds']['Seed'].values.tolist())
        ))

for df in list(map(lambda g: g + 'TeamSeasonStats', ('rGamesC', 'rGamesD'))):

    dataDict[df].loc[:, 'seedRank'] = (
            dataDict['tSeeds'].set_index(['Season', 'TeamID'])['seedRank'])


# Log process time
logProcessTime('append seed ranks', timeLog)


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


# Log process time
logProcessTime('massey ordinals', timeLog)


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


# Log process time
logProcessTime('handle missing data', timeLog)


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
strengthMetrics = list(filter(lambda metric: metric.find('Strength') >= 0, 
                         matchups.columns.tolist()))


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



# Log process time
logProcessTime('strength metrics', timeLog)


#%% STRENGTH METRIC ANALYSIS
## ############################################################################

# Create MatchUps of Tournament Games to determine which strength metrics
# Has the best performance, and which one to use for ranking teams
matchups = createMatchups(
        matchupDF = dataDict['tGamesCsingleTeam'][['Season', 'TeamID', 'opponentID', 'win']],
         statsDF = strengthDF,
         teamID1 = 'TeamID', 
         teamID2 = 'opponentID',
         teamLabel1 = 'team',
         teamLabel2 = 'opp',
         returnBaseCols = True,
         returnTeamID1StatCols = True,
         returnTeamID2StatCols = True,
         calculateDelta = True,
         calculateMatchup = False
         )

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



# Log process time
logProcessTime('strength metric analysis', timeLog)



#%% STRENGTH METRIC IMPORTANCE AND SELECTION
## ############################################################################

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
rfeCVs = {k:RFECV(v, cv = 5) for k,v in treeModels.items()}

# Train models
map(lambda tree: tree.fit(matchups.iloc[trainIndex,:], 
                          matchups.iloc[trainIndex,:].index.get_level_values('win'))
    , rfeCVs.values())


[tree.fit(matchups.iloc[trainIndex,:], 
          matchups.iloc[trainIndex,:].index.get_level_values('win'))
    for tree in rfeCVs.values()]


# Score models on train & test data
[ 
    [
        tree.score(matchups.iloc[idx,:], 
                      matchups.iloc[idx,:].index.get_level_values('win'))
        for idx in (trainIndex, testIndex)
    ]
    for tree in rfeCVs.values()]

# # of features selected for each model
[(rfeCV[0], rfeCV[1].n_features_) for rfeCV in rfeCVs.items()] 
    

# Get selected features for each model
featureImportance = pd.concat(
        list(map(lambda rfeCV:
            pd.DataFrame(
                zip(repeat(rfeCV[0], sum(rfeCV[1].support_)),
                    matchups.columns[rfeCV[1].support_],
                    rfeCV[1].estimator_.feature_importances_),
                columns = ['model', 'metric', 'importance']
                ).sort_values(['model', 'importance'], ascending = [True, False])
                , rfeCVs.items())
        ), axis = 0)



featureRank = pd.concat(
        list(map(lambda rfeCV:
            pd.DataFrame(
                zip(repeat(rfeCV[0], len(rfeCV[1].ranking_)),
                    matchups.columns,
                    rfeCV[1].ranking_),
                columns = ['model', 'metric', 'ranking']
                ).sort_values(['model', 'ranking'], ascending = [True, True])
                , rfeCVs.items())
        ), axis = 0)




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



# Log process time
logProcessTime('strength metric feature importance', timeLog)


#%% WINS AGAINST TOPN TEAMS
## ############################################################################

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
        list(map(lambda topN: 
            (matchups[matchups['oppspreadStrength'] <= topN].groupby(['Season', 'TeamID'])
                                                            .agg({'win':np.sum})),
            topNlist)),
        axis = 1))


# Fill missing values and rename columns
topNWins.fillna(0, inplace = True)
topNWins.columns = list(map(lambda topN: 'wins{}'.format(str(topN).zfill(3)), topNlist))



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
[tree.fit(matchups.iloc[trainIndex,:],
          matchups.iloc[trainIndex,:].index.get_level_values('win'))
    for tree in rfeCVs.values()]


# Score models on train & test data
[ 
   [ 
        tree.score(matchups.iloc[idx,:], 
                      matchups.iloc[idx,:].index.get_level_values('win'))
        for idx in (trainIndex, testIndex)
    ]
    for tree in rfeCVs.values()]

# # of features selected for each model
[(rfeCV[0], rfeCV[1].n_features_) for rfeCV in rfeCVs.items()]
    
 
    
    
# Get selected features for each model
featureImportanceTopN = pd.concat(
        list(map(lambda rfeCV:
            pd.DataFrame(
                zip(repeat(rfeCV[0], sum(rfeCV[1].support_)),
                    matchups.columns[rfeCV[1].support_],
                    rfeCV[1].estimator_.feature_importances_),
                columns = ['model', 'metric', 'importance']
                ).sort_values(['model', 'importance'], ascending = [True, False])
                , rfeCVs.items())
        ), axis = 0)


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
                len(list(filter(lambda delta: delta > 0, data)))/ len(data)})),
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



# Log process time
logProcessTime('wins against top N teams', timeLog)



#%% TOURNAMENT MATCHUP W/ SEASON STATS
## ############################################################################

# ADD TOURNAMENT SEED STATISTICS
#
# CALCULATE MATCHUP PAIRS FOR DUMMIES
#  
# CREATE MODEL DATASET WITH SAME COLUMN CALCULATIONS

calculateDelta = True
returnStatCols = True
calculateMatchup = True

for df in ['tGamesC', 'tGamesD']:
   

    # Reference assocated regular season data
    dataDict[df + 'statsModelData'] = createMatchups(
        matchupDF = dataDict[df + 'singleTeam'].loc[:, ['Season', 'TeamID', 'opponentID', 'win']], 
        statsDF = dataDict['{}TeamSeasonStats'.format(df.replace('t', 'r'))],
        teamID1 = 'TeamID',
        teamID2 = 'opponentID',
        teamLabel1 = 'team',
        teamLabel2 = 'opp',
        calculateDelta = calculateDelta,
        calculateMatchup = calculateMatchup,
        extraMatchupCols = ['seedRank']
        )




#%% CONFERENCE AND CONFERENCE CHAMPS MATCHUPS 
## ############################################################################


# Tournament Dataset
x = dataDict['tGamesCstatsModelData']

# Win probability based on conferences 
confDeltaStats = x.groupby(['teamconfGroups', 'teamconfChamp', 'oppconfGroups', 'oppconfChamp']).agg({'TeamID': np.count_nonzero,
                                                                           'win' : np.mean
                                                     }).rename(columns = {'TeamID': 'numGames', 'win': 'winPct'})

    
    
# Remove duplicates for same conferences
sameConfFilter = ((confDeltaStats.index.get_level_values('teamconfGroups') == confDeltaStats.index.get_level_values('oppconfGroups'))
                & (confDeltaStats.index.get_level_values('teamconfChamp') == confDeltaStats.index.get_level_values('oppconfChamp'))
                )

confDeltaStats.loc[sameConfFilter, 'numGames'] = confDeltaStats['numGames'] * 0.5



# Pivot data for heatmap
confDeltaStatsPiv = pd.pivot_table(data = confDeltaStats.reset_index(), 
                   values = ['winPct', 'numGames'], 
                   index = ['oppconfGroups', 'oppconfChamp'], 
                   columns = ['teamconfGroups', 'teamconfChamp'])



# Heat Map of win % by conference matchups & # of games
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,6))

sns.heatmap(confDeltaStatsPiv.loc[:, confDeltaStatsPiv.columns.get_level_values(None) == 'winPct'], 
            annot = True, 
            fmt='.2f',
            mask = heatMapMask(confDeltaStatsPiv.loc[:, confDeltaStatsPiv.columns.get_level_values(None) == 'winPct'], k = 1),
            square = True,
            cmap = 'RdYlGn',
            linewidths = 1, 
            linecolor = 'k',
            xticklabels = confDeltaStatsPiv.columns[confDeltaStatsPiv.columns.get_level_values(None) == 'winPct'].droplevel(None),
            ax = ax[0])

sns.heatmap(confDeltaStatsPiv.loc[:, confDeltaStatsPiv.columns.get_level_values(None) == 'winPct'], 
            annot = confDeltaStatsPiv.loc[:, confDeltaStatsPiv.columns.get_level_values(None) == 'numGames'], 
            fmt = ".0f",
            mask = heatMapMask(confDeltaStatsPiv.loc[:, confDeltaStatsPiv.columns.get_level_values(None) == 'winPct'], k = 1),
            square = True,
            cmap = 'RdYlGn',
            linewidths = 1, 
            linecolor = 'k',
            xticklabels = confDeltaStatsPiv.columns[confDeltaStatsPiv.columns.get_level_values(None) == 'winPct'].droplevel(None),
            ax = ax[1])

fig.suptitle('Win % based on Conference Matchups (1 = Conf. Champ)', fontsize = 16)
#fig.tight_layout()
fig.show()   





#%% SEED RANK MATCHUPS
## ############################################################################

# Win probability based on seed rank difference
#srDeltaStats = x.groupby(['highSeed', 'lowSeed', 'seedRankDeltaAbs']).agg({'WTeamID': np.count_nonzero,
#                                                                           'seedRankDelta' : (lambda d: (0.5 * len(d) if d.iloc[0] == 0 else len(filter(lambda g: g < 0, d))) / len(d))
#                                                     }).rename(columns = {'WTeamID': 'numGames', 'seedRankDelta': 'winPct'})


# Win probability based on seed rank difference
srDeltaStats = x.groupby(['teamseedRank', 'oppseedRank', 'seedRankDelta']).agg({'TeamID': np.count_nonzero,
                                                                           'win' : np.mean
                                                     }).rename(columns = {'TeamID': 'numGames', 'win': 'winPct'})

    
# Since each game is represented from each team's perspective, counts along the diagonal are duplicated and mean will always be 50%
sameSeedFilter = (srDeltaStats.index.get_level_values('teamseedRank') == srDeltaStats.index.get_level_values('oppseedRank'))
srDeltaStats.loc[sameSeedFilter, 'numGames'] = srDeltaStats['numGames'] * 0.5
   
   

# Heat Map of win % by seed matchups (high vs. low) & # of games
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,6))

sns.heatmap(srDeltaStats.reset_index(['teamseedRank', 'oppseedRank']).pivot(index = 'oppseedRank', columns = 'teamseedRank', values = 'winPct'), 
            annot = True, 
            fmt='.2f',
            mask = heatMapMask(srDeltaStats.reset_index(['teamseedRank', 'oppseedRank']).pivot(index = 'oppseedRank', columns = 'teamseedRank', values = 'winPct'), k=1),
            square = True,
            cmap = 'RdYlGn',
            linewidths = 1, 
            linecolor = 'k',
            ax = ax[0])

sns.heatmap(srDeltaStats.reset_index(['teamseedRank', 'oppseedRank']).pivot(index = 'oppseedRank', columns = 'teamseedRank', values = 'winPct'), 
            annot = srDeltaStats.reset_index(['teamseedRank', 'oppseedRank']).pivot(index = 'oppseedRank', columns = 'teamseedRank', values = 'numGames'), 
            fmt = ".0f",
            cmap = 'RdYlGn',
            mask = heatMapMask(srDeltaStats.reset_index(['teamseedRank', 'oppseedRank']).pivot(index = 'oppseedRank', columns = 'teamseedRank', values = 'winPct'), k=1),
            square = True,
            linewidths = 1, 
            linecolor = 'k',
            ax = ax[1])

fig.suptitle('Win % based on Seed Rank Matchups', fontsize = 16)
#fig.tight_layout()
fig.show()   



# Heat Map of win % by high seed and seed rank delta
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,6))

sns.heatmap(srDeltaStats.reset_index(['teamseedRank', 'seedRankDelta']).pivot(index = 'seedRankDelta', columns = 'teamseedRank', values = 'winPct'), 
            annot = True, 
            fmt = ".2f",
            # square = True,
            linewidths = 1, 
            linecolor = 'k',
            cmap = 'RdYlGn',
            ax = ax[0])

sns.heatmap(srDeltaStats.reset_index(['teamseedRank', 'seedRankDelta']).pivot(index = 'seedRankDelta', columns = 'teamseedRank', values = 'winPct'), 
            annot = srDeltaStats.reset_index(['teamseedRank', 'seedRankDelta']).pivot(index = 'seedRankDelta', columns = 'teamseedRank', values = 'numGames'), 
            fmt = ".0f",
#            square = True,
            cmap = 'RdYlGn',
            linewidths = 1, 
            linecolor = 'k',
            ax = ax[1])
fig.suptitle('Seed Rank Heat Map based on Rank Delta', fontsize = 16)
#fig.tight_layout()
fig.show()


# Scatter Plot of win % by seed matchups (high vs. low)
fig, ax = plt.subplots(1, figsize = (10,6))
sns.scatterplot(x = 'teamseedRank', 
                y = 'oppseedRank', 
                size = 'numGames', 
                hue = 'winPct', 
                edgecolors = 'k',
                palette = 'RdYlGn',
                marker = 'o',
#                sizes = (50,100),
                data = srDeltaStats.reset_index()[srDeltaStats.index.get_level_values('seedRankDelta') >= 0], 
                ax = ax)

# Add edges 
plt.scatter(x = 'teamseedRank', 
                 y = 'oppseedRank', 
                 s = 'numGames', 
                 c = 'winPct', 
                 cmap = 'RdYlGn',
                 edgecolors = 'k',
                 data = srDeltaStats.reset_index()[srDeltaStats.index.get_level_values('seedRankDelta') >= 0])

#fig.tight_layout()
fig.show()


# Distribution plot of Seed Rank Deltas win %
fig, ax = plt.subplots(1, figsize = (10,6))
sns.boxplot(x = 'seedRankDelta', y = 'winPct', data = srDeltaStats.reset_index(), ax = ax)
sns.swarmplot(x = 'seedRankDelta', y = 'winPct', 
              color = 'k',
              data = srDeltaStats.reset_index(), ax = ax)
ax.grid(True)
fig.suptitle('Distribution of Win % with difference rank matchups with same Delta', fontsize = 16)
fig.show()


#%% OHE MATCHUPS
## ############################################################################



######################################################
### CONFERENCE MATCHUP GROUPS FOR ONE HOT ENCODING ###
### & SEED RANK MATCHUP GROUPS FOR ONE HOT ENCODING ##
######################################################

oheDict = {}

# Thought about isolating only certain matchups for encoding, but decided to bin matchups based
# on average win % and then create a dictionary lookup to limit the number of matchups to encode
# Any conference / confchamp matchup not in dataset gets assigned to the bin of 0.5 since win probablity impact is unknown

# confDeltaStats.loc[:, 'numGamesLog'] = map(lambda x: round(x, int(np.floor(np.log10(x))*-1) - 1*(x<10)), confDeltaStats['numGames'])

# Create win % Bins
binIncrement = 0.1

confDeltaStats.loc[:, 'winPctBin'] = list(
    map(lambda w: int(100 *(round(w / binIncrement, 0) * binIncrement)), 
        confDeltaStats['winPct'])
    )

# Counts number of matchup groups and sum # of games in each bin
confDeltaStats.groupby('winPctBin').agg({'winPct': lambda x: len(x),
                                           'numGames': np.sum}).rename(columns = {'winPct':'numMatchups'})

# Store bins for OHE with predictions
oheDict['confs'] = confDeltaStats['winPctBin'].to_dict()



# Repeat process for seed rank matchups

srDeltaStats.loc[:, 'winPctBin'] = list(
    map(lambda w: int(100 *(round(w / binIncrement, 0) * binIncrement)), 
        srDeltaStats['winPct'])
    )

srDeltaStats.groupby('winPctBin').agg({'winPct': lambda x: len(x),
                                           'numGames': np.sum}).rename(columns = {'winPct':'numMatchups'})

    
oheDict['seedRanks'] = (
    srDeltaStats.reset_index('seedRankDelta')['winPctBin'].to_dict()
    )

# Prioritize matchup categories for encoding based on # of Games (support) and delta from 50% (lift)
#confDeltaStats.loc[:, 'lift'] = confDeltaStats['winPct'] / 0.5
#confDeltaStats.loc[:, 'liftsupport'] = (np.abs(confDeltaStats['winPct'] - 0.5) * confDeltaStats['numGames'])
#confDeltaStats.sort_values('liftsupport', ascending = False, inplace = True)
#confDeltaStats.loc[:, 'liftsupport_rank'] = range(confDeltaStats.shape[0])





#%% PRINCIPLE COMPONENT ANALYSIS AGAIN
## ############################################################################
    
# Perform PCA analysis on each Team Stats dataframe
#   Steps (use pipeline):
#       Scale Data
#       Calculate PCA for same number of dimensions in dataset to
#       develop contribution of each axis


toPerformPCA = True


if toPerformPCA == True:
    
    teamStatsDFs = list(
            filter(lambda dfName: 
                len(re.findall('r.*TeamSeasonStats.*', dfName))>0, 
                dataDict.keys())
            )
    
    # gamesStatsDFs = list(
    #         filter(lambda dfName: 
    #             len(re.findall('r.*SeasonStatsMatchup.*', dfName))>0, 
    #             dataDict.keys())
    #         )

    
    pcaDict = {}
    

    # PCA Analysis on Team season statistics  
    for df in teamStatsDFs:

        pcaDict[df] = performPCA(
            data = dataDict[df],
            pcaExcludeCols = ['Season', 'WTeamID', 'LTeamID'],
            scaler = StandardScaler(),
            dataLabel = df,
            plotComponents = True
            )



#%% CREATE TRAINING DATASET
## ############################################################################
        
# #############################################################################
# ############### BUILD NEW MODELING DATASET WITH NEW FEATURES ################
# #############################################################################

# STEPS 
# 1. Create Matchups (need conferences and conference champ flag)
# 2. Calculate metric deltas
# 3. Need to calculate bins for strength deltas
# 3. Calculate interactions
# 4. OHE seed rank matchups
# 5. OHE conference matchups
    

baseCols = ['Season', 'TeamID', 'opponentID']

statCols = [
    'seedRank', 
    'spreadStrengthOppStrengthRank', 
    'wins050', 
    'confChamp', 
    'confGroups'
    ]

statCols = []

modelMatchups = createMatchups(dataDict['tGamesCsingleTeam'][baseCols + ['win']],
                               statsDF = dataDict['rGamesCTeamSeasonStats'],
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
                               reindex = True)


# Create Matchups for conference + conf champ groups
modelMatchups.loc[:, 'confMatchup'] = pd.Series(
    map(lambda m: tuple(m), 
        modelMatchups[['teamconfGroups', 'teamconfChamp', 
                       'oppconfGroups', 'oppconfChamp']
                      ].values.tolist()
        )
    )

# Create Matchups for seed ranks
modelMatchups.loc[:, 'seedMatchup'] = pd.Series(
    map(lambda m: tuple(m), 
        modelMatchups[['teamseedRank', 'oppseedRank']
                      ].values.tolist()
        )
    )


# Convert matchups into win % bins
modelMatchups.loc[:, 'confMatchupBin'] = list(
    map(lambda m: oheDict['confs'].get(m, 50), 
        modelMatchups.loc[:, 'confMatchup']
        )
    )

modelMatchups.loc[:, 'seedMatchupBin'] = list(
    map(lambda m: oheDict['seedRanks'].get(m, 50), 
        modelMatchups.loc[:, 'seedMatchup']
        )
    )


#%% FEATURE IMPORTANCE & SELECTION
## ############################################################################

    
# #############################################################################
# ############### FEATURE IMPORTANCE AND FEATURE SELECTION ####################
# #############################################################################

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=1127)

model = LogisticRegression(random_state = 1127)

poly = PolynomialFeatures(degree = 1, interaction_only = True)

testTrainSplit = 0.2

modelResults, featureRankAll = list(), list()


for df in ('tGamesC', 
           #'tGamesD'
           ):

    modelResults, featureRankAll = list(), list()
    

    
    modelCols = list(
        filter(lambda c: ((modelMatchups[c].dtype.hasobject == False)
                          & (c not in baseCols + ['win'])),
               modelMatchups.columns.tolist()
               )
        )
    
    

    
    # Model Data & initial poly fit
    data = modelMatchups[modelMatchups['Season'] >= 1985]
    poly.fit(data.loc[:, modelCols])
    
    
    featureCols = list(poly.get_feature_names(modelCols))
    
    dataPoly = pd.DataFrame(poly.transform(data.loc[:, modelCols]), columns = featureCols)
    
    data = pd.concat([dataPoly, data['win']], axis = 1)
    modelData = dataPoly.merge(pd.DataFrame(data['win']), left_index = True, right_index = True)

    # Split data for analysis
    xTrain, xTest, yTrain, yTest = train_test_split(data[featureCols], 
                                                    data['win'],
                                                    test_size = testTrainSplit,
                                                    random_state = 1127)


    # Iterate through features and systematically remove features with lowest importance
    # Recalculate feature importantce and model score after each iterations
    #for nFeatures in range(len(featureCols), 1, -5):
    while len(featureCols) > 0:
        

        # update independent data with selected columns
        xTrain, xTest = xTrain[featureCols], xTest[featureCols]
        
        
        # Fit forest Model for feature importance and selection
        forest.fit(xTrain, yTrain)
        
        # Fit logistic model for predictions
        model.fit(xTrain, yTrain)
        
        
        # Calculate Model Accuracy and auc    
        modelResults.append(
                        (len(featureCols), 
                             roc_auc_score(yTest, model.predict_proba(xTest)[:,1]), 
                             accuracy_score(yTest, model.predict(xTest)),
                             roc_auc_score(yTest, forest.predict_proba(xTest)[:,1]), 
                             accuracy_score(yTest, forest.predict(xTest))
                             )
                        )
        
      
        # Get Feature Importances
        featureRank = list(
            zip(forest.feature_importances_, 
                featureCols, 
                repeat(len(featureCols), len(featureCols))
                )
            )
        
        # Sort features by importance
        featureRank.sort(reverse = True)


        featureRankAll.append(featureRank)

        # Remove lowest feature rankings
        featureCols = list(list(zip(*featureRank))[1])[:-2]



    # Aggregate all feature importance iterations
    featureRankAllDF = pd.DataFrame(
        list(chain(*featureRankAll)), 
        columns = ['importance', 'metric', 'numFeatures']
        )
    
    # Add model scores
    featureRankAllDF = featureRankAllDF.merge(
        pd.DataFrame(modelResults, 
                     columns = ['numFeatures', 
                                'aucLog', 'accLog', 
                                'aucForest', 'accForest']),
        left_on = 'numFeatures', 
        right_on = 'numFeatures'
        )

    # Add # of features for each iteration
    featureRankCount = (
        featureRankAllDF.groupby('metric')['numFeatures'].count()
        )


    
    featureRankAllPiv = pd.pivot_table(featureRankAllDF, columns = 'metric', index = 'numFeatures', values = 'importance')
    featureRankAllPiv = featureRankAllPiv.merge(pd.DataFrame(modelResults, 
                                                           columns = ['numFeatures', 
                                                                      'aucLog', 'accLog', 
                                                                      'aucForest', 'accForest']),
                                                left_index = True, right_on = 'numFeatures')
    featureRankAllPiv.set_index('numFeatures', inplace = True)
    
    featureRankAllPivT = featureRankAllPiv.transpose()


    featureRankAllPivT = featureRankAllPivT.merge(pd.DataFrame(featureRankCount), left_index = True, right_index = True, how = 'left')


    featureRankCorr = featureRankAllPiv.corr(min_periods = 5)
    featureRankCorr = featureRankCorr.loc[:, ['accLog', 'aucLog', 'accForest', 'aucForest']]    
    
   

    
    fig, ax = plt.subplots(1)
    
    #sns.barplot(zip(*featureRank)[0]/max(zip(*featureRank)[0]), zip(*featureRank)[1], ax = ax)


    #sns.lmplot(x = zip(*modelResults)[0], y = zip(*modelResults)[1])
    plt.title(df)
    plt.plot(list(zip(*modelResults))[0], list(zip(*modelResults))[1], label = 'logistic')
    plt.plot(list(zip(*modelResults))[0], list(zip(*modelResults))[3], label = 'forest')
    plt.grid()
    plt.legend()


#%% BINNING & ENCODING FEATURES
## ############################################################################
    

# One Hot Encode Matchups

# Get dummies (One Hot Encode) for Matchup Bins
# modelMatchups = pd.get_dummies(data = modelMatchups, 
#                                columns = ['confMatchupBin', 'seedMatchupBin'])


# Create bins out of spreadStrengthOppStrengthRankDelta and wins050Delta
# kBins = KBinsDiscretizer(n_bins=10, encode='onehot-dense', strategy='quantile')


# # Merge spreadStrength OHE bins to rest of model dataset
# modelMatchups = pd.concat(
#     [modelMatchups,
#      pd.DataFrame(kBins.fit_transform(
#          modelMatchups[['spreadStrengthOppStrengthRankDelta', 'wins050Delta']]),
#          columns = chain(*map(lambda b: 
#                               map(lambda bb: '{}_{}'.format(b[0], bb), 
#                                   range(b[1])
#                                   ), 
#                               zip(['spreadBin', 'win050Bin'], kBins.n_bins_)
#                               )
#                          )
#         )
#     ],
#     axis = 1
#     )

    
# Filter only bin columns for modeling
modelCols = list(
    filter(lambda col: col.find('Bin_') >= 0, 
           modelMatchups.columns
           )
    )




#%% MODEL DEVELOPMENT & GRID SEARCH
## ############################################################################

df = 'tGamesC'

modelDict = {}

modelDict[df] = {}

performPCA(modelMatchups[modelMatchups['Season'] >= 2003],
           pcaExcludeCols = ['Season', 'TeamID', 'opponentID'])

pipe = Pipeline([
    ('sScale', StandardScaler()), 
    # ('sScale', QuantileTransformer()),
    # ('sScale', MinMaxScaler()),
    ('pca',  PCA()),
    # ('poly', PolynomialFeatures(degree = 2, interaction_only = True)),
    # ('kbd', KBinsDiscretizer(n_bins = 4, encode = 'ordinal')),
    # ('fReduce', fReduce),
    # ('fReduce', PCA(n_components = 10)),
    ('mdl', LogisticRegression(random_state = 1127, max_iter = 500))
    ])



paramGrid = [
#     {'mdl' : [ExtraTreesClassifier(n_estimators = 50,
#                                   n_jobs = -1,
#                                   random_state = 1127), 
#               RandomForestClassifier(random_state = 1127,
#                                     n_estimators = 50,
#                                     n_jobs = -1,
#                                     verbose = 0),
#               GradientBoostingClassifier(n_estimators = 50,
#                                         random_state = 1127)
#               ],                        
#     'mdl__min_samples_split' : np.arange(.005, .1, .01),
#     'mdl__min_samples_leaf' : range(2, 11, 4),
# #                 'mdl__n_estimators' : [25, 100, 200]
#         },
                
            {'mdl' : [LogisticRegression(random_state = 1127)],
             'mdl__C' : list(map(lambda i: 10**i, range(-2,2)))
                },
                
            # {'mdl' : [SVC(probability = True)],
            #  'mdl__C' : map(lambda i: 10**i, range(-1,4)),
            #  'mdl__gamma' : map(lambda i: 10**i, range(-4,1))
            #     },
                
            # {'mdl' : [KNeighborsClassifier()],
            #  'mdl__n_neighbors' : range(3, 15, 2)
            #     }
            ]

# Run grid search on modeling pipeline
timer()
modelDict[df]['analysis'] = modelAnalysisPipeline(modelPipe = pipe,
                      data = modelMatchups[modelMatchups['Season'] >= 2003],
                      # indCols = [],
                      excludeCols = ['Season', 'TeamID', 'opponentID'],
                      targetCol = 'win',
                      testTrainSplit = 0.2,
                      gridSearch=True,
                      paramGrid=paramGrid,
                      scoring = 'roc_auc',
                      crossFolds = 5)

modelDict[df]['calcTime'] = timer()



#%% VISUALIZE MODEL RESULTS
## ############################################################################

# Plot Results
gridSearchResults = pd.DataFrame(modelDict[df]['analysis']['pipe'].cv_results_)

gridSearchResults['mdl'] = list(
    map(lambda m: str(m).split('(')[0], 
        gridSearchResults['param_mdl'].values.tolist())
    )



gsPlotCols = list(
    filter(lambda c: len(re.findall('^mean.*|^rank.*', c)) > 0,
           gridSearchResults.columns.tolist()
           )
    )

# Get summary for each model type and best model for each model type
mdlBests = []
for label, metric in [('mean', np.mean), ('median', np.median), ('max',np.max)]:
    
    t = gridSearchResults.groupby('mdl').agg({'mean_test_score':metric})
    t.rename(columns = {'mean_test_score':label}, inplace = True)
    mdlBests.append(t)
    
del(t)    
    
mdlBests = pd.concat(mdlBests, axis = 1)

mdlBests = (mdlBests.set_index('max', append = True)
                    .merge(gridSearchResults[['mdl', 'mean_test_score', 'param_mdl', 'params']], 
                           left_index = True, 
                           right_on = ['mdl', 'mean_test_score'], 
                           how = 'inner')
                           )
mdlBests.rename(columns = {'mean_test_score':'max'}, inplace = True)

# Make sure there's only a single value for each model type
mdlBests = mdlBests.groupby('mdl').first()    

modelDict[df]['bests'] = mdlBests
modelDict[df]['gridResults'] = gridSearchResults

#type(mdlBests['param_mdl'].iloc[0])

# Plot Results
fig, ax = plt.subplots(len(gsPlotCols))
plt.suptitle('Grid Search Results by Model Type {}'.format(df), fontsize = 24)

swPlot = True

for i, col in enumerate(gsPlotCols):
    sns.violinplot(x = 'mdl', y = col, data = gridSearchResults, ax = ax[i])    
    if swPlot == True:    
        sns.swarmplot(x = 'mdl', y = col, 
                      data = gridSearchResults, 
                      ax = ax[i], 
                      color = 'grey', 
                      size = 6)

    #ax.set_yticklabels(map(lambda v: '{:.0%}'.format(v), axs[1].get_yticks()))
    ax[i].set_ylabel(col, fontsize = 12)

    ax[i].grid()      
       
    if i == len(gsPlotCols) - 1:
        ax[i].set_xlabel('Model Type', fontsize = 12)
        ax[i].tick_params(labelsize = 12)
    else:
        ax[i].tick_params(axis = 'y', labelsize = 12)
        ax[i].tick_params(axis = 'x', which = 'both', 
                          top = 'off', bottom = 'off', 
                          labelbottom = 'off')


try: del(mdlBests, gridSearchResults, gsPlotCols, numPCASplits)
except: pass

#%% MODEL EVALUATION
## ############################################################################

#==============================================================================
# EVALUATE MODELS ON TEST DATA
#   ROC CURVES
#   AUC
#   LOG LOSS
#==============================================================================


# Plot roc curve for best params for each model type
for df in modelDict.keys():       

    # Refit pipleiine with model parameters and calculate prediciton probabilities

    # ROC Curves
    rocCurves = list(
        map(lambda params: roc_curve(modelDict[df]['analysis']['yTest'],
                                    (modelDict[df]['analysis']['pipe'].estimator.set_params(**params)
                                              .fit(modelDict[df]['analysis']['xTrain'], modelDict[df]['analysis']['yTrain'])
                                              .predict_proba(modelDict[df]['analysis']['xTest'])[:,1])),
                    modelDict[df]['bests']['params'].values.tolist())
        )


    # Append best model
    rocCurves.append(roc_curve(modelDict[df]['analysis']['yTest'],
                               modelDict[df]['analysis']['pipe'].predict_proba(modelDict[df]['analysis']['xTest'])[:,1]))

    # AUC
    rocAucs = list(
        map(lambda params: roc_auc_score(modelDict[df]['analysis']['yTest'],
                                          (modelDict[df]['analysis']['pipe'].estimator.set_params(**params)
                                              .fit(modelDict[df]['analysis']['xTrain'], modelDict[df]['analysis']['yTrain'])
                                              .predict_proba(modelDict[df]['analysis']['xTest'])[:,1])),
                    modelDict[df]['bests']['params'].values.tolist())
        )


    # Append best model
    rocAucs.append(roc_auc_score(modelDict[df]['analysis']['yTest'],
                                   modelDict[df]['analysis']['pipe'].predict_proba(modelDict[df]['analysis']['xTest'])[:,1]))

    # Log Loss
    logloss = list(
        map(lambda params: log_loss(modelDict[df]['analysis']['yTest'],
                                          (modelDict[df]['analysis']['pipe'].estimator.set_params(**params)
                                              .fit(modelDict[df]['analysis']['xTrain'], modelDict[df]['analysis']['yTrain'])
                                              .predict_proba(modelDict[df]['analysis']['xTest']))),
                    modelDict[df]['bests']['params'].values.tolist())
        )

    # Confusion Matrix
    confuseMatrix = list(
        map(lambda params: confusion_matrix(modelDict[df]['analysis']['yTest'],
                                          (modelDict[df]['analysis']['pipe'].estimator.set_params(**params)
                                              .fit(modelDict[df]['analysis']['xTrain'], modelDict[df]['analysis']['yTrain'])
                                              .predict(modelDict[df]['analysis']['xTest']))),
                    modelDict[df]['bests']['params'].values.tolist())
        )

    # Accuracy
    accuracy = list(map(lambda c: np.trace(c) / np.sum(c), confuseMatrix))


    # Plot ROC Curves
    cMap = cm.get_cmap('jet')

    fig, ax = plt.subplots(1)
    
    for i, curve in enumerate(zip(rocCurves, 
                                  modelDict[df]['bests'].index.values.tolist() + ['best Model'],
                                    rocAucs)):
        ax.plot(curve[0][0], 
                curve[0][1], 
                c = cMap(256*i//(len(rocCurves) - 1))[:3], 
                linewidth = 8,
                label = '{} {:0.2f}'.format(curve[1], curve[2]))
      
    ax.plot([0,1],[0,1], '--k', linewidth = 4, label = 'random')      
      
    plt.grid()
    plt.legend()
    ax.set_xlabel('fpr', fontsize = 24)
    ax.set_ylabel('tpr', fontsize = 24)
    ax.set_title('ROC Curve for ' + df, fontsize = 36)
    ax.tick_params(labelsize = 24)



#%% TOURNAMENT PRECITIONS ALL POSSIBILITIES
## ############################################################################

season = 2019
statsDF = dataDict['rGamesCTeamSeasonStats']


tourneyTeams = (
    statsDF[
        (statsDF.index.get_level_values('Season') == season) 
        & (statsDF['seedRank'] <= 16)
        ].index.get_level_values('TeamID').tolist()
    )



tourneyMatchups = pd.DataFrame(combinations(tourneyTeams, 2), 
                               columns = ['TeamID', 'opponentID'])
tourneyMatchups.loc[:, 'Season'] = season



tourneyMatchups = createMatchups(tourneyMatchups,
                               statsDF = statsDF,
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
                               reindex = True)


# Create Matchups for conference + conf champ groups
tourneyMatchups.loc[:, 'confMatchup'] = pd.Series(
    map(lambda m: tuple(m), 
        tourneyMatchups[['teamconfGroups', 'teamconfChamp', 
                       'oppconfGroups', 'oppconfChamp']
                      ].values.tolist()
        )
    )

# Create Matchups for seed ranks
tourneyMatchups.loc[:, 'seedMatchup'] = pd.Series(
    map(lambda m: tuple(m), 
        tourneyMatchups[['teamseedRank', 'oppseedRank']
                      ].values.tolist()
        )
    )


# Convert matchups into win % bins
tourneyMatchups.loc[:, 'confMatchupBin'] = list(
    map(lambda m: oheDict['confs'].get(m, 50), 
        tourneyMatchups.loc[:, 'confMatchup']
        )
    )

tourneyMatchups.loc[:, 'seedMatchupBin'] = list(
    map(lambda m: oheDict['seedRanks'].get(m, 50), 
        tourneyMatchups.loc[:, 'seedMatchup']
        )
    )



# Perform predictions
tourneyMatchups.loc[:, 'teamWinProb'] = (
    modelDict[df]['analysis']['pipe'].predict_proba(
        tourneyMatchups.loc[:, modelDict[df]['analysis']['independentVars']]
        )[:,1]
    )


# Add Team Names
tourneyMatchups = (
    tourneyMatchups.merge(
        pd.DataFrame(dataDict['teams'].set_index('TeamID')['TeamName']),
        left_on = 'TeamID', 
        right_index = True
        )
    )

tourneyMatchups = (
    tourneyMatchups.merge(
        pd.DataFrame(dataDict['teams'].rename(columns = {
            'TeamID':'opponentID', 
            'TeamName': 'opponentName'
            })
            .set_index('opponentID')['opponentName']),
        left_on = 'opponentID', 
        right_index = True
        )
    )


tourneyMatchups.loc[:, 'winner'] = [
    teamID if teamWinProb > 0.5 else opponentID
    for teamWinProb, teamID, opponentID in
    tourneyMatchups[['teamWinProb', 'TeamID', 'opponentID']].values.tolist()
    ]



tourneyMatchups.to_csv('{}_{}_best_model_results_all_matchups_{}.csv'.format(season, df, 
                           datetime.strftime(datetime.now(), '%Y_%m_%d')), index = False) 




#%% TOURNAMENT PREDICTIONS
## ############################################################################
  
tSlots = dataDict['tSlots']
tSeeds = dataDict['tSeeds']

bracketPredictions = tournamentPredictions(
    allPredictions = tourneyMatchups, 
    tSlots = dataDict['tSlots'], 
    tSeeds = dataDict['tSeeds']
    )


def tournamentPredictions(allPredictions, tSlots, tSeeds):
    
    tSlotsDict = {
        k: v.set_index(['StrongSeed', 'WeakSeed'])[['Slot']].to_dict('index')
        for k,v in tSlots.groupby('Season')
            }
    
    
    tSeedsDict = {
        k: dict(v[['TeamID', 'Seed']].values.tolist())
        for k,v in tSeeds.groupby('Season')
            }
    
    
    allSeasons = list(set(allPredictions['Season'].values.tolist()))
    
    allSeasonsGameCount = sum(
        [len(tSlotsDict[season]) for season in allSeasons]
        )
    
    allPredictions.loc[:, 'StrongSeed'] = None
    allPredictions.loc[:, 'WeakSeed'] = None 
    allPredictions.loc[:, 'Slot'] = None
    allPredictions.loc[:, 'complete'] = False
    
    
    while allPredictions['complete'].sum() < allSeasonsGameCount:
    
        allPredictions.loc[:, 'StrongSeed'] = [
            min(tSeedsDict[season].get(teamID),
                tSeedsDict[season].get(opponentID)
                ) 
            if slot == None else StrongSeed
            for season, teamID, opponentID, StrongSeed, slot in 
            allPredictions[['Season', 'TeamID', 'opponentID', 'StrongSeed', 'Slot']].values.tolist()
            
            ]
        
        
        allPredictions.loc[:, 'WeakSeed'] = [
            max(tSeedsDict[season].get(teamID),
                tSeedsDict[season].get(opponentID)
                ) 
            if slot == None else WeakSeed
            for season, teamID, opponentID, WeakSeed, slot in 
            allPredictions[['Season', 'TeamID', 'opponentID', 'WeakSeed', 'Slot']].values.tolist()
            ]
        
        
        
        allPredictions.loc[:, 'Slot'] = [
            tSlotsDict[season].pop((StrongSeed, WeakSeed), 
                                   {'Slot' : slot}
                                   ).get('Slot')
            for season, StrongSeed, WeakSeed, slot in 
            allPredictions[['Season', 'StrongSeed', 'WeakSeed', 'Slot']].values.tolist()
            ]
        
        
        
        
        [tSeedsDict[season].update({TeamID : slot}) 
         for season, TeamID, slot, complete in
         allPredictions[['Season', 'winner', 'Slot', 'complete']].values.tolist()
         if (slot != None) & (complete == False)
             ]
        
        
        
        allPredictions.loc[:, 'complete'] = [
            True if slot != None else False
            for slot in tourneyMatchups['Slot'].values.tolist()
            ]
    
    
    
    bracketPredictions = allPredictions[allPredictions['complete']]

    return bracketPredictions

#%%



# Year for predictions
yr = 2019

for df in modelDict.keys():
   
    allModelResults = pd.DataFrame()
    
    # Get model 
    modelBestsDict = modelDict[df]['bests'].to_dict(orient='index')
     
    # Regular Season team stast Dataframe for building modeling dataset
    teamDFname = 'rGames{}TeamSeasonStats'.format(df[-1])
    
    # Modeling columns: All numeric columns (same code as used in Grid Search)
    indCols2 = filter(lambda c: (c not in colsBase + ['ATeamID', 'BTeamID', 'winnerA'])
                                & (dataDict[df + 'modelData'][c].dtype.hasobject == False), 
                    dataDict[df + 'modelData'].columns.tolist())
    
    
    for mdl, mdlDict in modelBestsDict.items():   
        
        # Get pipeLine & set parameters
        pipe = modelDict[df]['analysis']['pipe'].estimator
        pipe.set_params(**mdlDict['params'])
        
        
        # Fit the pipeline
        pipe.fit(modelDict[df]['analysis']['xTrain'], 
                 modelDict[df]['analysis']['yTrain'])
    
        
        modelBestsDict[mdl]['bestPredictions'], modelBestsDict[mdl]['bestPredictionsClean'], modelBestsDict[mdl]['matchups'] = tourneyPredictions2(model = pipe, 
                              teamDF = dataDict[teamDFname],
                              tSeeds = dataDict['tSeeds'],
                              tSlots = dataDict['tSlots'],
                              seedDF = dataDict[df + 'SeedStats'],
                              mdlCols = indCols2,
                              yr = yr,
                              returnStatCols = True)
        
        # Add columns for dataframe and model name
        modelBestsDict[mdl]['bestPredictionsClean'].loc[:, 'df'] = df
        modelBestsDict[mdl]['bestPredictionsClean'].loc[:, 'model'] = mdl
        
        # Aggregate results
        allModelResults = pd.concat([allModelResults, 
                                     modelBestsDict[mdl]['bestPredictionsClean']],
                                    axis = 0)
        
        
        
        fName = '_'.join([str(yr),
                          'model_results',
                          df,
                          mdl, 
                          datetime.strftime(datetime.now(), '%Y_%m_%d')])
    
        modelBestsDict[mdl]['bestPredictionsClean'].to_csv(fName + '.csv', index = False, header = True)    
   

    allModelResults.to_csv('{}_all_model_results_{}_{}_2.csv'.format(yr, df, 
                           datetime.strftime(datetime.now(), '%Y_%m_%d')), index = False) 

# ============================================================================
# ================= END TOURNAMENT PRECITIONS ================================
# ============================================================================

#%% DEV
## ############################################################################

### ############################## TEAM STRENTGH METRICS ######################
### ###########################################################################

#execfile('{}\\050_mm_team_strength_metrics.py'.format(pc['repo']))

### ###########################################################################
### ##################### MAP TEAM CONFERENCES ################################
### ###########################################################################

# =============================================================================
# 
# for df in list(map(lambda g: g + 'TeamSeasonStats', ('rGamesC', 'rGamesD'))):
#     
#     dataDict[df].loc[:, 'ConfAbbrev'] = (
#             dataDict['teamConferences'].set_index(['Season', 'TeamID'])['ConfAbbrev']
#             )
#     
#     # New column with all small conferences grouped together
#     dataDict[df].loc[:, 'confGroups'] = list(
#             map(lambda conf: conf if conf in 
#                 ('big_east', 'big_twelve', 'acc', 'big_ten', 'sec')
#                 else 'other',
#                 dataDict[df]['ConfAbbrev'].values.tolist())
#             )
#     
#         
# print(timer())
# =============================================================================


