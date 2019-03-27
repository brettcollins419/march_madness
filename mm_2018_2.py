# -*- coding: utf-8 -*-
"""
Created on Mon Mar 05 20:42:49 2018

@author: brett
"""

from __future__ import division
from os.path import join
import time
import sys
import numpy as np
import pandas as pd
import string
#import sklearn as sk
#from sklearn import svm, linear_model, ensemble, neighbors, tree, naive_bayes
#from sklearn import preprocessing, model_selection

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Colormap
import seaborn as sns
import os
import re
from itertools import product, islice, chain, repeat, combinations
from datetime import datetime
import socket


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, QuantileTransformer, KBinsDiscretizer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_auc_score, accuracy_score, roc_curve
from sklearn.pipeline import Pipeline, FeatureUnion



'''Analysis steps
1. Read Data
2. Calculate end of season ranks and seed ranks
2. Merge datasets for Winning and Losing teams: conferences, 
                                                end of year median rank,
                                                seeds & seed rank
4. Engineer features:   free throw %,
                        field goal %,
                        3 point %,
                        total points,
                        score gap,
                        conference matchup,
                        seed matchup
                        

5. Aggregate regular season results to calculate team overall record & stats

6. Standardize parameters

6. Create matchup based on teams overall stats                        

7. Randomly split data, switch Winning and Losing order for one split & merge
    splits with new labels
    
8. Modeling


Expansions: encode conference / matchups


'''

#==============================================================================
# START FUNCTIONS 
#==============================================================================

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



def buildModelData2(gameDF, teamDF, 
                    indexCols = [],
                    label1 = 'A', label2 = 'B',
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



def generateGameMatchupStats2(gameDF, teamDF,
                              indexCols,
                             teamID1, teamID2, 
                             label1 = 'A', label2 = 'B',
                             labelName = 'TeamID',
                             extraMergeFields = ['Season'],
                             calculateDeltas = True,
                             returnStatCols = True,
                             deltaExcludeFields = [],
                             createMatchupFields = True,
                             matchupFields = [('confMatchup', 'ConfAbbrev'), 
                                              ('seedRankMatchup', 'seedRank')]):
    
    '''Create a new dataframe with team matchup statistics based on two
        teamIDs. 
        
        TeamDF should have index set to mergeFields.'''
    
    # Error handling for if Season not in mergeFields
    if 'Season' not in gameDF.columns.tolist():
        extraMergeFields = []
    
    
    # Set gameDF to only columns needed for indexing 
    gameDF = gameDF[indexCols]
    
    for x in [(teamID1, label1), (teamID2, label2)]:
        teamID, teamLabel = x[0], x[1]        
        mergeFields = extraMergeFields + [teamID]
        gameDFcols = gameDF.columns.tolist()
        
        # Add label to teamDF columns to match TeamID label        
        teamDFcols = map(lambda label: teamLabel + label, teamDF.columns.tolist())
        
        gameDF = gameDF.merge(teamDF, left_on = mergeFields, right_index = True)
        gameDF.columns = gameDFcols + teamDFcols
        

    # Set index for merging
    gameDF.set_index(indexCols, inplace = True)

    if calculateDeltas == True:
        dfDelta = generateMatchupDeltas(df = gameDF, 
                                        label1 = label1, 
                                        label2 = label2,
                                        labelName = labelName,
                                        excludeCols = deltaExcludeFields + [label1 + labelName,
                                                                            label2 + labelName])
        
        dfDeltaCols = dfDelta.columns.tolist()     

        gameDF = gameDF.merge(dfDelta, left_index = True, right_index = True)     

          
          
    if createMatchupFields == True:
        for field in matchupFields:        
            gameDF[field[0]] = generateMatchupField(gameDF, field[1], label1, label2)
              

    if returnStatCols == False:
        if len(matchupFields) > 0:
            matchupLabels = list(zip(*matchupFields)[0])
        else: matchupLabels = []
        
        gameDF = gameDF[dfDeltaCols + matchupLabels]

    return gameDF



def genGameMatchupswSeedStats(baseCols, 
                              gameDF, teamDF, seedDF,
                              teamID1, teamID2, 
                              label1 = 'A', label2 = 'B',
                              extraMergeFields = ['Season'],
                              calculateDeltas = True,
                              returnStatCols = True,
                              deltaExcludeFields = [],
                              createMatchupFields = True,
                              matchupFields = [('confMatchup', 'ConfAbbrev'), 
                                               ('seedRankMatchup', 'seedRank')]):
        
    
    '''Call generateGameMatchupStats function for creating initial
        matchup using team statistics, then add statistics for each
        team based on their seed rank in the tournament.

        Return DataFrame of matchups.'''                                           

    # Create initial matchup dataframe
    teamStats = generateGameMatchupStats2(gameDF = gameDF[baseCols],
                                          indexCols = baseCols,
                                          teamDF = teamDF,
                                          teamID1 = teamID1, 
                                          teamID2 = teamID2,
                                          label1 = label1, 
                                          label2 = label2,
                                          labelName = 'TeamID',
                                          extraMergeFields = extraMergeFields,
                                          calculateDeltas = calculateDeltas,
                                          returnStatCols = returnStatCols,
                                          createMatchupFields = createMatchupFields,
                                          matchupFields = matchupFields)


    # More robust method for getting seedRank matchups incase returnStatCols = False
    # Get seedRank for TeamID1      
    
    # Error handling for 'Season' in merging fields
    if 'Season' in gameDF.columns.tolist():
        leftMergeSeason = ['Season']
    else:
        leftMergeSeason = []
    
    seedStatsInput = gameDF[baseCols].merge(pd.DataFrame(teamDF['seedRank']),
                                            how = 'left', 
                                            left_on = leftMergeSeason + [teamID1],
                                            right_index = True)
    
    #seedStatsInput.reset_index(inplace = True)                                        
    seedStatsInput.rename(columns = {'seedRank':label1 + 'seedRank'}, inplace = True)
    
 
     # Get seedRank for teamID2
    seedStatsInput = seedStatsInput.merge(pd.DataFrame(teamDF['seedRank']),
                                            how = 'left', 
                                            left_on = leftMergeSeason + [teamID2],
                                            right_index = True)   

    #seedStatsInput.reset_index(inplace = True) 
    seedStatsInput.rename(columns = {'seedRank':label2 + 'seedRank'}, inplace = True)


    # Create matchup dataframe based on seed ranks (lookup seed stats )
    #seedLabels = [label1 + 'seedRank', label2 + 'seedRank']
    seedStats = generateGameMatchupStats2(gameDF = seedStatsInput,
                                          teamDF = seedDF,
                                          indexCols = baseCols,
                                          teamID1 = label1 + 'seedRank', 
                                          teamID2 = label2 + 'seedRank',
                                          labelName = 'seedRank',
                                          extraMergeFields=[],
                                          calculateDeltas = calculateDeltas,
                                          returnStatCols = returnStatCols,
                                          createMatchupFields=False,
                                          matchupFields = [],
                                          label1 = label1 + 'Seed', 
                                          label2 = label2 + 'Seed')
         
         
    # Merge results (drop seedRank label since already exists in base dataframe)
    seedStatMergeCols = filter(lambda c: c.endswith('seedRank') == False, 
                               seedStats.columns.tolist())         
       
    matchUps = teamStats.merge(seedStats[seedStatMergeCols], 
                               left_index = True, 
                               right_index = True)  

    return matchUps
    




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


#==============================================================================
# END FUNCTIONS
#==============================================================================





        

#==============================================================================
# LOAD DATA 
#==============================================================================

# Working Directory Dictionary
wds = {'WaterBug' : 'C:\\Users\\brett\\Documents\\march_madness_ml',
             'WHQPC-L60102' : 'C:\\Users\\u00bec7\\Desktop\\personal\\march_madness_ml',
             'raspberrypi' : '/home/pi/Documents/march_madness_ml'
             }

# Set working directory
os.chdir(wds.get(socket.gethostname()))


# Read data
dataFiles = os.listdir('datasets\\2019')
dataFiles.sort()

# Remove zip files
dataFiles = filter(lambda f: '.csv' in f, dataFiles)


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

dataDict = {k : pd.read_csv('datasets\\2019\\{}'.format(data)) for k, data in zip(keyNames, dataFiles)}


#==============================================================================
# INITIAL EDA AND ORGANIZATION OF GAMES DATA
#==============================================================================

# Lists of games data (C =C ompact, D = Detailed)
gamesData = ['tGamesC', 'tGamesD', 'rGamesC', 'rGamesD']



#==============================================================================
# CALCULATE ADDITIONAL STATISTICS 
#==============================================================================

# Score gap calculation
for df in gamesData:
    dataDict[df]['scoreGap'] = dataDict[df]['WScore'] - dataDict[df]['LScore']


# Detailed DF additional Stats
    if df.endswith('D'):
        for team in ['W', 'L']:
            dataDict[df][team + 'FGpct'] = dataDict[df][team + 'FGM'] / dataDict[df][team + 'FGA']
            dataDict[df][team + 'FGpct3'] = dataDict[df][team + 'FGM3'] / dataDict[df][team + 'FGA3']
            dataDict[df][team + 'FTpct'] = dataDict[df][team + 'FTM'] / dataDict[df][team + 'FTA']
            dataDict[df][team + 'Scorepct'] = (   dataDict[df][team + 'FGpct3'] * 3 
                                                + dataDict[df][team + 'FGpct'] * 2 
                                                + dataDict[df][team + 'FTpct']
                                                ) / 6
            
        for team in [('W', 'L'), ('L', 'W')]:        
            dataDict[df][team[0] + 'ORpct'] = (dataDict[df][team[0] + 'OR'] /
                                                (dataDict[df][team[0] + 'OR'] 
                                                    + dataDict[df][team[1] + 'DR']))
                                                    
            dataDict[df][team[0] + 'DRpct'] = (dataDict[df][team[0] + 'DR'] /
                                                (dataDict[df][team[0] + 'DR'] 
                                                    + dataDict[df][team[1] + 'OR']))    
    
            dataDict[df][team[0] + 'Rpct'] = ((dataDict[df][team[0] + 'DR'] 
                                            + dataDict[df][team[0] + 'OR']) /
                                                (   dataDict[df][team[0] + 'DR'] 
                                                  + dataDict[df][team[0] + 'OR']
                                                  + dataDict[df][team[1] + 'OR']
                                                  + dataDict[df][team[1] + 'DR'])) 


#==============================================================================
# IDENTIFY COLUMN TYPES AND UPDATE COLUMN SUMMARIES WITH NEW COLUMNS
#==============================================================================

# Generate dict
colSumDict = {}  
  
# Label column types
colsBase = ['Season', 'DayNum', 'WLoc', 'NumOT', 'scoreGap']   

colsWinFilter = lambda c: colsTeamFilter(c, 'W') & (c != 'WLoc')
colsLossFilter = lambda c: colsTeamFilter(c, 'L')
   

    
# Create list of unique columns in all games DataFrames
for df in gamesData:
  colSumDict[df] = generateDataFrameColumnSummaries(dataDict[df], 
                                                    returnDF=True)
                                                    
                                                    
  # Label column types
  colsWin = filter(colsWinFilter,
                   dataDict[df].columns.tolist())
  colsLoss = filter(colsLossFilter, 
                    dataDict[df].columns.tolist())

  
  for colName, colList in [('colsBase', colsBase), 
                           ('colsWin', colsWin), 
                           ('colsLoss', colsLoss)]:
      
      colSumDict[df][colName] = map(lambda c: c in colList,
                                    colSumDict[df]['colName'].values.tolist())

del(colsWin, colsLoss, colName, colList)

#==============================================================================
# CALCULATE TEAM SUMMARIES FOR REGULAR SEASON & TOURNAMENT
#==============================================================================
for df in ('rGamesC', 'rGamesD'):

    colsWinTemp = filter(colsWinFilter,
                  dataDict[df].columns.tolist())
    colsLossTemp = filter(colsLossFilter, 
                   dataDict[df].columns.tolist())
    
    # Format base/shared columns between wins & loss data
    # Remove wLoc from colsBase (object, would need to parameterize for any value)
    colsBaseTemp = filter(lambda c: (c in colsBase) & (c != 'WLoc'), 
                          dataDict[df].columns.tolist())   
    
    winDF = dataDict[df].loc[:, colsBaseTemp + colsWinTemp]
    winDF['win'] = 1

    lossDF = dataDict[df].loc[:, colsBaseTemp + colsLossTemp]
    lossDF['win'] = 0
    
    # Assign losses DayNum = 0, thus average DayNum is a reflection of when
    # teams are winning (late in the season = High number, early = low, even = mid)
#    if 'DayNum' in lossDF.columns.tolist():
#        lossDF.loc[:, 'DayNum'] = 0
    
    # Flip scoreGap for losing Team
    lossDF.loc[:, 'scoreGap'] = lossDF.loc[:, 'scoreGap'] * -1
    
    # Drop 'W' & 'L' from labels
    winDF.rename(columns=dict(zip(colsWinTemp, 
                                  map(lambda c: c[1:], 
                                      colsWinTemp))), 
                  inplace = True)
                  
    lossDF.rename(columns=dict(zip(colsLossTemp, 
                                   map(lambda c: c[1:], 
                                       colsLossTemp))), 
                   inplace = True)
    
    # Combine wins and losses data and calculate means
    aggDF = pd.concat((winDF, lossDF))
    aggDF.sort_values('DayNum', inplace = True)
    
    
    dataDict[df + 'singleTeam'] = aggDF
    
    dataDict[df + 'TeamSeasonStats'] = aggDF.groupby(['Season', 'TeamID']).mean()
    dataDict[df + 'TeamSeasonStats'].loc[:, 'last8'] =  aggDF.groupby(['Season', 'TeamID']).agg({'win': lambda games: np.mean(games[-8:])})
    
    dataDict[df + 'TeamSeasonStats'].drop(filter(lambda c: c in ('DayNum', 'NumOT', 'FTM', 'FGM', 'FGM3'), 
                                                dataDict[df + 'TeamSeasonStats'].columns.tolist()),
                                            axis = 1,
                                            inplace = True)

    
del(winDF, lossDF, aggDF, colsLossTemp, colsWinTemp, colsBaseTemp)
    

#==============================================================================
# CALCULATE SEED RANKS
#==============================================================================

# Tourney Seed Rank
dataDict['tSeeds'].loc[:, 'seedRank'] = map(lambda s: float(re.findall('[0-9]+', s)[0]), 
                                             dataDict['tSeeds']['Seed'].values.tolist())



# Get end of regular season rankings for each system
endRegSeason = 133

# Filter for data within regular season
rsRankings = dataDict['MasseyOrdinals'][dataDict['MasseyOrdinals']['RankingDayNum'] <=  endRegSeason]

# Get max rank date for regular season for each system and team
maxRankDate = rsRankings.groupby(['Season', 'TeamID', 'SystemName'])['RankingDayNum'].max()
maxRankDate.rename('maxRankDate', inplace = True)

# Merge and filter for only last rank for regular season
rsRankings = (rsRankings.set_index(['Season', 'TeamID', 'SystemName'])
                        .merge(pd.DataFrame(maxRankDate),
                               left_index = True,
                               right_index = True))
rsRankings = rsRankings[rsRankings['RankingDayNum'] == rsRankings['maxRankDate']]

# Caluate median end of season ranking and number of systems in ranking
dataDict['endSeasonRanks'] = (rsRankings.groupby(['Season', 'TeamID'])
                                        .agg({'OrdinalRank':np.median,
                                              'RankingDayNum': np.count_nonzero}))

dataDict['endSeasonRanks'].rename(columns = {'RankingDayNum':'systemsCount'},
                                  inplace = True)

# Memory Cleanup (consumes ~122 MB)
del(maxRankDate, rsRankings, dataDict['MasseyOrdinals'])



#==============================================================================
# MERGE CONFERENCES, SEEDS & RANKS TO TEAM SEASON 
#==============================================================================
# Missing Values Dict
fillDict = {'LSeed':'NA', 
            'seedRank':17,
            'OrdinalRank': 176}

oheConferences = False

for df in map(lambda g: g + 'TeamSeasonStats', gamesData):
    dataDict[df] = dataDict[df].merge(dataDict['teamConferences'].set_index(['Season', 'TeamID']), 
                                      how = 'left', 
                                      left_index = True,
                                      right_index = True)

    dataDict[df] = dataDict[df].merge(dataDict['tSeeds'].set_index(['Season', 'TeamID']),
                                      how = 'left', 
                                      left_index = True,
                                      right_index = True)
       
       
    dataDict[df] = dataDict[df].merge(pd.DataFrame(dataDict['endSeasonRanks']['OrdinalRank']),
                                      how = 'left', 
                                      left_index = True,
                                      right_index = True)

    dataDict[df].fillna(fillDict, inplace = True)

    
    # New column with all small conferences grouped together
    dataDict[df].loc[:, 'confGroups'] = map(lambda conf: conf if conf in ('big_east', 'big_twelve', 'acc', 'big_ten', 'sec')
                                            else 'other',
                                            dataDict[df]['ConfAbbrev'].values.tolist())
    
    # One Hot Encode confGroups
    if oheConferences == True:
        le, ohe = LabelEncoder(), OneHotEncoder(handle_unknown='ignore', sparse = False)
        x = ohe.fit_transform(le.fit_transform(dataDict[df]['confGroups']).reshape(-1,1))
        
        x = pd.DataFrame(x, columns = map(lambda conf: 'ohe_{}'.format(conf), 
                                                               le.classes_.tolist()))
        
        dataDict[df] = pd.concat([dataDict[df].reset_index(), x],
                                axis = 1)
        
        #dataDict[df].drop('level_0', axis = 1, inplace = True)
        dataDict[df].set_index(['Season', 'TeamID'], inplace = True)
        
        
        # Group seeds into 4's
        dataDict[df].loc[:, 'seedRankGroups'] = map(lambda rank: rank // 4, 
                                                dataDict[df]['seedRank'].values.tolist())

#==============================================================================
# CALCULATE SEED STATISTICS FOR TOURNAMENT
#==============================================================================
for df in filter(lambda n: n.startswith('t'), gamesData):


    dataDict[df + 'SeedStats'] = (dataDict[df + 'TeamSeasonStats'].reset_index()
                                              .drop(['Season', 'DayNum', 'TeamID', 'ConfAbbrev'], axis = 1)
                                              .groupby('seedRank')
                                              .mean())



#==============================================================================
# CACLUATE COLUMN SUMMARY FOR TEAM SEASON STATISTICS DATAFRAMES
#==============================================================================
# Create list of unique columns in all games DataFrames
for df in map(lambda g: g + 'TeamSeasonStats', gamesData):
    colSumDict[df] = generateDataFrameColumnSummaries(dataDict[df], returnDF=True)
                                     
                                     
  
#==============================================================================
# CREATE NEW TOURNAMENT MATCHUPS USING TEAM SEASON STATISTICS
# 
# ADD TOURNAMENT SEED STATISTICS
#
# CALCULATE MATCHUP PAIRS FOR DUMMIES
#  
# CREATE MODEL DATASET WITH SAME COLUMN CALCULATIONS
#==============================================================================

calculateDeltas = True
returnStatCols = True
createMatchupFields = True

for df in filter(lambda g: g.startswith('t'), gamesData):
   
    # Reference assocated regular season data
    regDF = 'r' + df[1:]    

    # Create tournament matchups    
    dataDict[df + 'SeasonStatsMatchup'] = generateGameMatchupStats2(indexCols = ['Season', 'DayNum', 'WTeamID', 'LTeamID'],
                                                                    gameDF = dataDict[df],
                                                                    teamDF = dataDict[regDF + 'TeamSeasonStats'],                                                                 
                                                                    teamID1 = 'WTeamID', 
                                                                    teamID2 = 'LTeamID',
                                                                    label1 = 'W', 
                                                                    label2 = 'L',
                                                                    calculateDeltas = calculateDeltas,
                                                                    returnStatCols = returnStatCols,
                                                                    createMatchupFields = createMatchupFields,
                                                                    )
      
    
    # Build initial model dataset (reorder wins and loss cols on 50% of games)                                                           
    dataDict[df + 'modelData'] = buildModelData2(gameDF = dataDict[df][['Season', 'DayNum', 'WTeamID', 'LTeamID']],
                                                 teamDF = dataDict[regDF + 'TeamSeasonStats'],
                                                 calculateMatchupStats = False
                                                    )

    # Create matchups from base model dataset
    dataDict[df + 'modelData'] = generateGameMatchupStats2(indexCols = ['Season', 'DayNum', 'ATeamID', 'BTeamID', 'winnerA'],
                                                           gameDF = dataDict[df + 'modelData'],
                                                           teamDF = dataDict[regDF + 'TeamSeasonStats'],
                                                           teamID1 = 'ATeamID', 
                                                           teamID2 = 'BTeamID',
                                                           label1 = 'A', 
                                                           label2 = 'B',
                                                           calculateDeltas = calculateDeltas,
                                                           returnStatCols = returnStatCols,
                                                           createMatchupFields = createMatchupFields)
    
    # Pull out winnerA column from index
    dataDict[df + 'modelData'].reset_index('winnerA', inplace = True)                    

    # Calculate dataframe column details
    colSumDict[df + 'SeasonStatsMatchup'] = generateDataFrameColumnSummaries(dataDict[df + 'SeasonStatsMatchup'],
                                                                             returnDF = True)
    colSumDict[df + 'modelData'] = generateDataFrameColumnSummaries(dataDict[df + 'modelData'],
                                                                    returnDF = True)

# Add matchup columns to colsBase
colsBase += ['confMatchup', 'seedRankMatchup']



#==============================================================================
# CREATE SUBSET OF REGULAR SEASON TEAM STATISTICS FOR TOURNAMENT TEAMS ONLY
#==============================================================================
#for df in ['rGamesCTeamSeasonStats', 'rGamesDTeamSeasonStats']:
#    dataDict[df + 'Tourney'] = dataDict[df][dataDict[df]['seedRank'] <= 16]
#    colSumDict[df + 'Tourney'] = generateDataFrameColumnSummaries(dataDict[df + 'Tourney'], returnDF=True)
#



#==============================================================================
# CORRELATION ANALYSIS
#==============================================================================

# Columns to exculde in correlation anaylsis:
#   All base columns excluding scoreGap
#   TeamIDs
#   All object columns 

corrExcludeFilter = lambda c: (((c not in colsBase) | 
                                (c in ('scoreGap', 'winnerA'))) 
                                    & (c.endswith('TeamID') == False))
fig, ax = plt.subplots(1)

for df in map(lambda n: n + 'modelData', 
              filter(lambda d: d.startswith('t'), gamesData)):
    
    corrColsTemp = filter(corrExcludeFilter, 
                          dataDict[df].columns.tolist())    
    
    dataDict[df + 'Corr'] = dataDict[df][corrColsTemp].corr()
    
    plotCorrHeatMap(dataDict[df + 'Corr'], 
                    plotTitle= df + ' Correlation Analysis')


    x = pd.concat([dataDict[df + 'Corr']['winnerA'].rank(pct = True),
                   dataDict[df + 'Corr']['winnerA']], axis = 1)
      
         
    x.columns = ['rank', 'corr']
    
    x = x[x.index.values != 'winnerA']

    x.sort_values('rank', inplace = True)  
    
    xScale = StandardScaler()
    
    x['scaled'] = xScale.fit_transform(x['corr'].values.reshape(-1,1))
    
    
    
    ax.plot(x['rank'], x['corr'], lw = 10, label = df)

ax.tick_params(labelsize = 18)
ax.set_xlabel('Rank', fontsize = 24)
ax.set_ylabel('correlation', fontsize = 24)
ax.legend()
    
    
fig           

del(corrColsTemp, x)


#==============================================================================
# PRINCIPLE COMPONENT ANALSYSIS OF TEAM STATS DATA
#==============================================================================
# Perform PCA analysis on each Team Stats dataframe
#   Steps (use pipeline):
#       Scale Data
#       Calculate PCA for same number of dimensions in dataset to
#       develop contribution of each axis


performPCA = True

if performPCA == True:
    
    teamStatsDFs = filter(lambda dfName: len(re.findall('t.*TeamSeasonStats.*', dfName))>0, 
                          dataDict.iterkeys())
    gamesStatsDFs = filter(lambda dfName: len(re.findall('t.*SeasonStatsMatchup.*', dfName))>0, 
                           dataDict.iterkeys())
    gamesStatsDFs = filter(lambda dfName: dfName.endswith('Corr') != True, 
                           gamesStatsDFs)
    
    labelFontSize = 20
    titleFontSize = 24
    tickFontSize = 16
    
    pcaDict = {}
    
    
    ### PCA Analysis on Team season statistics
    
    pcaExcludeCols = ['Season', 'WTeamID', 'LTeamID']
    
    for df in teamStatsDFs:
    
        # Get columns for transformation using PCA
        pcaCols = colSumDict[df]['colName'][~colSumDict[df]['isObject']]
        pcaCols = filter(lambda n: n not in pcaExcludeCols, pcaCols)
     
        pcaPipe = Pipeline([('sScale', StandardScaler()), 
                            ('pca', PCA(n_components = len(pcaCols), 
                                        random_state = 1127))])
       
        #pca = PCA(n_components = len(pcaCols), random_state = 1127)
        
        pcaDict[df] = pcaPipe.fit(dataDict[df][pcaCols])
    
    
        fig, axs = plt.subplots(1, 2)
        plt.suptitle(df + ' PCA Analysis', fontsize = 36)    
        
        
        # Determine how many labels to plot so that axis isn' cluttered
        axisLabelFreq = len(pcaCols) // 20 + 1
        xAxisLabelsMask =  map(lambda x: x % axisLabelFreq == 0, xrange(len(pcaCols)))
        xAxisLabels = dataDict[df][pcaCols].columns[xAxisLabelsMask]
        
        # Plot feature weights for each component
        sns.heatmap(pcaDict[df].named_steps['pca'].components_, 
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
        axs[0].set_title('PCA Components Feature Weights', fontsize = titleFontSize)
        axs[0].set_xlabel('Feature', fontsize = labelFontSize)
        axs[0].set_ylabel('PCA #', fontsize = labelFontSize)
           
        # Plot explained variance curve
        axs[1].plot(xrange(pcaDict[df].named_steps['pca'].n_components_), 
                    np.cumsum(pcaDict[df].named_steps['pca'].explained_variance_ratio_), 
                    '-bo', 
                    markersize = 20, 
                    linewidth = 10)
        
        # Convert y-axis to %
        axs[1].set_yticklabels(map(lambda v: '{:.0%}'.format(v), axs[1].get_yticks()))
        
        axs[1].set_title('Explained Variance vs. # of Components', 
                         fontsize = titleFontSize)
        axs[1].set_xlabel('# of Features', 
                          fontsize = labelFontSize)
        axs[1].set_ylabel('Explained Variance', 
                          fontsize = labelFontSize)
        axs[1].tick_params(labelsize = tickFontSize)
        axs[1].grid()
    
    
    
    
    ### PCA Analysis on matchup dataframes
    
    pcaExcludeCols = ['WTeamID', 'LTeamID'] + colsBase
    
    for df in gamesStatsDFs:
    
        # Get columns for transformation using PCA
        pcaCols = colSumDict[df]['colName'][~colSumDict[df]['isObject']]
        pcaCols = filter(lambda n: n not in pcaExcludeCols, pcaCols)
     
        pcaPipe = Pipeline([('sScale', StandardScaler()), 
                            ('pca', PCA(n_components = len(pcaCols), 
                                        random_state = 1127))])
       
        #pca = PCA(n_components = len(pcaCols), random_state = 1127)
        
        pcaDict[df] = pcaPipe.fit(dataDict[df][pcaCols])
    
    
        fig, axs = plt.subplots(1, 2)
        plt.suptitle(df + ' PCA Analysis', fontsize = 36)    
        
        
        # Determine how many labels to plot so that axis isn' cluttered
        axisLabelFreq = len(pcaCols) // 20 + 1
        xAxisLabelsMask =  map(lambda x: x % axisLabelFreq == 0, xrange(len(pcaCols)))
        xAxisLabels = dataDict[df][pcaCols].columns[xAxisLabelsMask]
        
        # Plot feature weights for each component
        sns.heatmap(pcaDict[df].named_steps['pca'].components_, 
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
        axs[0].set_title('PCA Components Feature Weights', fontsize = titleFontSize)
        axs[0].set_xlabel('Feature', fontsize = labelFontSize)
        axs[0].set_ylabel('PCA #', fontsize = labelFontSize)
           
        # Plot explained variance curve
        axs[1].plot(xrange(pcaDict[df].named_steps['pca'].n_components_), 
                    np.cumsum(pcaDict[df].named_steps['pca'].explained_variance_ratio_), 
                    '-bo', 
                    markersize = 20, 
                    linewidth = 10)
        
        # Convert y-axis to %
        axs[1].set_yticklabels(map(lambda v: '{:.0%}'.format(v), axs[1].get_yticks()))
        
        axs[1].set_title('Explained Variance vs. # of Components', 
                         fontsize = titleFontSize)
        axs[1].set_xlabel('# of Features', 
                          fontsize = labelFontSize)
        axs[1].set_ylabel('Explained Variance', 
                          fontsize = labelFontSize)
        axs[1].tick_params(labelsize = tickFontSize)
        axs[1].grid()
    
    
    
    
# #############################################################################
# ############### FEATURE IMPORTANCE AND FEATURE SELECTION ####################
# #############################################################################

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=1127)

model = LogisticRegression(random_state = 1127)

poly = PolynomialFeatures(degree = 2, interaction_only = True)

testTrainSplit = 0.2

modelResults, featureRankAll = list(), list()


for df in ('tGamesC', 'tGamesD'):

    modelResults, featureRankAll = list(), list()
    
    indCols2 = filter(lambda c: (c not in colsBase + ['ATeamID', 'BTeamID', 'winnerA'])
                                & (dataDict[df + 'modelData'][c].dtype.hasobject == False)
                                & c.endswith('Delta'), 
                    dataDict[df + 'modelData'].columns.tolist())
    

    
    
    # Model Data & initial poly fit
    data = dataDict[df + 'modelData'][dataDict[df + 'modelData'].index.get_level_values('Season') >= 2003]
    poly.fit(data[indCols2])
    
    
    featureCols = poly.get_feature_names(indCols2)
    
    dataPoly = pd.DataFrame(poly.transform(data[indCols2]), columns = featureCols)
    
    data = pd.concat([dataPoly, data.reset_index().loc[:,'winnerA']], axis = 1)


    # Split data for analysis
    xTrain, xTest, yTrain, yTest = train_test_split(data[featureCols], 
                                                    data['winnerA'],
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
                             roc_auc_score(yTest, model.predict(xTest)), 
                             accuracy_score(yTest, model.predict(xTest)),
                             roc_auc_score(yTest, forest.predict(xTest)), 
                             accuracy_score(yTest, forest.predict(xTest))
                             )
                        )
        
      
        # Get Feature Importances
        featureRank = zip(forest.feature_importances_, 
                          featureCols, 
                          repeat(len(featureCols), len(featureCols)))
        
        # Sort features by importanc
        featureRank.sort(reverse = True)


        featureRankAll.append(featureRank)

        # Remove lowest feature rankings
        featureCols = list(zip(*featureRank)[1][:-2])




    fig, ax = plt.subplots(1)
    
    #sns.barplot(zip(*featureRank)[0]/max(zip(*featureRank)[0]), zip(*featureRank)[1], ax = ax)


    #sns.lmplot(x = zip(*modelResults)[0], y = zip(*modelResults)[1])
    plt.title(df)
    plt.plot(zip(*modelResults)[0], zip(*modelResults)[1], label = 'logistic')
    plt.plot(zip(*modelResults)[0], zip(*modelResults)[3], label = 'forest')

    plt.legend()
# Feature Selection

#==============================================================================
# MODEL DEVELOPMENT & GRID SEARCH
#==============================================================================

modelDict = {}

for df in filter(lambda g: g.startswith('t'), gamesData):
    
    modelDict[df] = {}
    
    # Modeling columns for new pipeline
    indCols2 = filter(lambda c: (c not in colsBase + ['ATeamID', 'BTeamID', 'winnerA'])
                                & (dataDict[df + 'modelData'][c].dtype.hasobject == False), 
                    dataDict[df + 'modelData'].columns.tolist())
    
    
    # Model List
    mdlList = [ DecisionTreeClassifier(random_state = 1127), 
                RandomForestClassifier(random_state = 1127),
                LogisticRegression(random_state = 1127),
                KNeighborsClassifier(),
                SVC(random_state = 1127, probability = True)]
    
    # Configure parameter grid for pipeline
    numIndCols = len(dataDict[df + 'modelData'][indCols2].columns)
    numPCASplits = 4
    
    
  
    #fReduce = FeatureUnion([('pca', PCA()), ('kBest', SelectKBest(k = 1))])
    #fReduce = FeatureUnion([('pca', PCA()), 
                           # ('kBest', SelectKBest(k = 1))
                           # ('rfe', RFE(LogisticRegression(random_state = 1127)))
    #                        ])
   
    #fReduce = RFE(SVC(kernel="linear", random_state = 1127), n_features_to_select = 5)
    fReduce = SelectPercentile(percentile = 0.5)
    # fReduce = SelectKBest(k = 1)
 
    
    # Create pipeline of Standard Scaler, PCA reduction, and Model (default Logistic)
    pipe = Pipeline([#('sScale', StandardScaler()), 
                     #('sScale', QuantileTransformer()),
                     # ('pca',  PCA(n_components = numIndCols // 2)),
                     ('poly', PolynomialFeatures(degree = 3, interaction_only = True)),
                     #('kbd', KBinsDiscretizer(n_bins = 4, encode = 'ordinal')),
                     ('scale', StandardScaler()),
                     ('fReduce', fReduce),
                     # ('fReduce', PCA(n_components = 10)),
                     ('mdl', LogisticRegression(random_state = 1127))])
    
    
    paramGrid = [
#                {'mdl' : [DecisionTreeClassifier(random_state = 1127), 
#                          RandomForestClassifier(random_state = 1127,
#                                                 n_estimators = 100,
#                                                 n_jobs = -1,
#                                                 verbose = 0)],
#                 'mdl__min_samples_split' : np.arange(.005, .1, .01),
#                 'mdl__min_samples_leaf' : xrange(2, 11, 4)
#                    },
                    
                {'mdl' : [LogisticRegression(random_state = 1127)],
                 'mdl__C' : map(lambda i: 10**i, xrange(-1,4))
                    },
                    
                {'mdl' : [SVC(probability = True)],
                 'mdl__C' : map(lambda i: 10**i, xrange(-1,4)),
                 'mdl__gamma' : map(lambda i: 10**i, xrange(-4,1))
                    },
                    
                {'mdl' : [KNeighborsClassifier()],
                 'mdl__n_neighbors' : range(3, 10, 2)
                    }
                ]
            
    
    # Update paramGrid with other grid search parameters that apply to all models
#    map(lambda d: d.update({'fReduce__n_features_to_select' : range(1, min(1 +  numIndCols, 26), 2)}),
#        paramGrid)
#    map(lambda d: d.update({'fReduce__k' : range(1,min(10, 1 + numIndCols // 2))}),
#        paramGrid)
    map(lambda d: d.update({'fReduce__percentile' : np.arange(0.1, 0.71, 0.2)}),
        paramGrid)
#    map(lambda d: d.update({'fReduce__pca__n_components' : range(3, numIndCols, numIndCols // numPCASplits)}),
#        paramGrid)    
    
    
    

    
    
    
    
    # Run grid search on modeling pipeline
    timer()
    modelDict[df]['analysis'] = modelAnalysisPipeline(modelPipe = pipe,
                          data = dataDict[df + 'modelData'][dataDict[df + 'modelData'].index.get_level_values('Season') >= 2003],
                          indCols = indCols2,
                          targetCol = 'winnerA',
                          testTrainSplit = 0.2,
                          gridSearch=True,
                          paramGrid=paramGrid,
                          scoring = 'roc_auc')
    modelDict[df]['calcTime'] = timer()
    
    
    
    
    
    
    # Plot Results
    gridSearchResults = pd.DataFrame(modelDict[df]['analysis']['pipe'].cv_results_)
    gridSearchResults['mdl'] = map(lambda m: str(m).split('(')[0], 
                                    gridSearchResults['param_mdl'].values.tolist())
    
    
    gsPlotCols = filter(lambda c: len(re.findall('^mean.*|^rank.*', c)) > 0,
                       gridSearchResults.columns.tolist())
    
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
    
del(mdlBests, gridSearchResults, gsPlotCols, numIndCols, numPCASplits)


#==============================================================================
# ROC CURVES 
#==============================================================================


# Plot roc curve for best params for each model type
for df in filter(lambda g: g.startswith('t'), gamesData):       

    # Refit pipleiine with model parameters and calculate prediciton probabilities
    rocCurves = map(lambda params: roc_curve(modelDict[df]['analysis']['yTest'],
                                          (modelDict[df]['analysis']['pipe'].estimator.set_params(**params)
                                              .fit(modelDict[df]['analysis']['xTrain'], modelDict[df]['analysis']['yTrain'])
                                              .predict_proba(modelDict[df]['analysis']['xTest'])[:,1])),
                    modelDict[df]['bests']['params'].values.tolist())


    rocAucs = map(lambda params: roc_auc_score(modelDict[df]['analysis']['yTest'],
                                          (modelDict[df]['analysis']['pipe'].estimator.set_params(**params)
                                              .fit(modelDict[df]['analysis']['xTrain'], modelDict[df]['analysis']['yTrain'])
                                              .predict(modelDict[df]['analysis']['xTest']))),
                    modelDict[df]['bests']['params'].values.tolist())



    cMap = cm.get_cmap('jet')

    fig, ax = plt.subplots(1)
    
    for i, curve in enumerate(zip(rocCurves, 
                                  modelDict[df]['bests'].index.values.tolist(),
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









# ============================================================================
# ================= TOURNAMENT PRECITIONS ====================================
# ============================================================================

# Year for predictions
yr = 2019

for df in ('tGamesC', 
           'tGamesD'
           ):
   
    allModelResults = pd.DataFrame()
    
    # Get model 
    modelBestsDict = modelDict[df]['bests'].to_dict(orient='index')
     
    # Regular Season team stast Dataframe for building modeling dataset
    teamDFname = 'rGames{}TeamSeasonStats'.format(df[-1])
    
    # Modeling columns: All numeric columns (same code as used in Grid Search)
    indCols2 = filter(lambda c: (c not in colsBase + ['ATeamID', 'BTeamID', 'winnerA'])
                                & (dataDict[df + 'modelData'][c].dtype.hasobject == False), 
                    dataDict[df + 'modelData'].columns.tolist())
    
    
    for mdl, mdlDict in modelBestsDict.iteritems():   
        
        pipe = modelDict[df]['analysis']['pipe'].estimator
        
        pipe.set_params(**mdlDict['params'])
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

# ============================================================================
# ===================== DEV ==================================================
# ============================================================================

dataDict.keys()

# Count teams by conference
teamConfCount = dataDict['tGamesDTeamSeasonStats'].groupby('ConfAbbrev')['win'].count()
teamConfCount.sort_values(ascending = False, inplace = True)

# plot counts
fig, ax = plt.subplots(1)
sns.barplot(y = teamConfCount, x = teamConfCount.index.get_values())

# Isolate top 5 conferences and group all others
teamConfCount[:5].index.get_values().tolist()

for df in ()

help(x.sort)      
