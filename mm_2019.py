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
from win32api import GetSystemMetrics
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
from scipy.stats import ttest_ind

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, QuantileTransformer, KBinsDiscretizer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_auc_score, accuracy_score, roc_curve, log_loss
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




def heatMapMask(corrData, k = 0):
    
    ''' Create array for masking upper right corner of map.
        k is the offset from the diagonal.
            if 1 returns the diagonal
        Return array for masking'''
        
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


######### NEW METHOD FOR CREATING MATCHUPS (CLEANER AND MORE EFFIECIENT)

def createMatchups(matchupDF, 
                   statsDF, 
                   teamID1 = 'TeamID', 
                   teamID2 = 'opponentID', 
                   teamLabel1 = 'team',
                   teamLabel2 = 'opp',
                   calculateDelta = False, 
                   calculateMatchup = False, 
                   extraMatchupCols = [],
                   returnStatCols = True,
                   reindex = True):
    ''' Create dataframe game matchups using team statistics to use for modeling
        & parameter selection / performance.
        
        Options:
            Return Statistic columns for each team (returnStatCols Boolean)
            Calculate teamStatistic deltas for the matchup (calculateMatchup Boolean)
            
            Create a tuple of object columns such as conference or seeds (calculateMatchup Boolean)
                
        
        Return dataframe with same number of recors as the matchupDF.
        '''
    
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
    if calculateMatchup == True:
        for col in list(set(matchupCols + extraMatchupCols)):
            matchupNew.loc[:, '{}Matchup'.format(col)] = pd.Series(createMatchupField(matchupNew, 
                                                                                       '{}{}'.format(teamLabel1, col), 
                                                                                       '{}{}'.format(teamLabel2, col), sort = False))
            
    if returnStatCols == True:
        return matchupNew
    
    # Don't return stat cols
    else:
        deltaReturnCols = filter(lambda c: (c.endswith('Delta')) | (c.endswith('Matchup')) | (c in colCount[colCount['field'] != 2].index.get_level_values('field')),
                                                       matchupNew.columns.tolist())
        
        
        return matchupNew.loc[:, deltaReturnCols]
    

    
    return matchupNew




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
            dataDict[df].loc[:, team + 'FGpct'] = dataDict[df].loc[:, team + 'FGM'] / dataDict[df].loc[:, team + 'FGA']
            dataDict[df].loc[:, team + 'FGpct3'] = dataDict[df].loc[:, team + 'FGM3'] / dataDict[df].loc[:, team + 'FGA3']
            dataDict[df].loc[:, team + 'FTpct'] = dataDict[df].loc[:, team + 'FTM'] / dataDict[df].loc[:, team + 'FTA']
            dataDict[df].loc[:, team + 'Scorepct'] = (   dataDict[df].loc[:, team + 'FGpct3'] * 3 
                                                + dataDict[df].loc[:, team + 'FGpct'] * 2 
                                                + dataDict[df].loc[:, team + 'FTpct']
                                                ) / 6
            
        for team in [('W', 'L'), ('L', 'W')]:        
            dataDict[df].loc[:, team[0] + 'ORpct'] = (dataDict[df].loc[:, team[0] + 'OR'] /
                                                (dataDict[df].loc[:, team[0] + 'OR'] 
                                                    + dataDict[df].loc[:,team[1] + 'DR']))
                                                    
            dataDict[df].loc[:, team[0] + 'DRpct'] = (dataDict[df].loc[:, team[0] + 'DR'] /
                                                (dataDict[df].loc[:, team[0] + 'DR'] 
                                                    + dataDict[df].loc[:, team[1] + 'OR']))    
    
            dataDict[df].loc[:, team[0] + 'Rpct'] = ((dataDict[df].loc[:, team[0] + 'DR'] 
                                            + dataDict[df].loc[:, team[0] + 'OR']) /
                                                (   dataDict[df].loc[:, team[0] + 'DR'] 
                                                  + dataDict[df].loc[:, team[0] + 'OR']
                                                  + dataDict[df].loc[:, team[1] + 'OR']
                                                  + dataDict[df].loc[:, team[1] + 'DR'])) 


#==============================================================================
# IDENTIFY COLUMN TYPES AND UPDATE COLUMN SUMMARIES WITH NEW COLUMNS
#==============================================================================





# Generate dict
#colSumDict = {}  
#  
## Label column types
colsBase = ['Season', 'DayNum', 'WLoc', 'NumOT', 'scoreGap']   
#
#colsWinFilter = lambda c: colsTeamFilter(c, 'W') & (c != 'WLoc')
#colsLossFilter = lambda c: colsTeamFilter(c, 'L')
#   
#
#    
## Create list of unique columns in all games DataFrames
#for df in gamesData:
#  colSumDict[df] = generateDataFrameColumnSummaries(dataDict[df], 
#                                                    returnDF=True)
#                                                    
#                                                    
#  # Label column types
#  colsWin = filter(colsWinFilter,
#                   dataDict[df].columns.tolist())
#  colsLoss = filter(colsLossFilter, 
#                    dataDict[df].columns.tolist())
#
#  
#  for colName, colList in [('colsBase', colsBase), 
#                           ('colsWin', colsWin), 
#                           ('colsLoss', colsLoss)]:
#      
#      colSumDict[df][colName] = map(lambda c: c in colList,
#                                    colSumDict[df]['colName'].values.tolist())
#
#del(colsWin, colsLoss, colName, colList)


#==============================================================================
# BUILD SINGLE TEAM DATASETS 
#(ORGANIZE BY TEAM VS OPPONENT INSTEAD OF WTEAM VS LTEAM)
#==============================================================================

for df in ('rGamesC', 'rGamesD', 'tGamesC', 'tGamesD'):

    dataDict[df + 'singleTeam'] =  buildSingleTeam(df = dataDict[df])


#==============================================================================
# BUILD MODELING DATASETS 
#(MAKE HALF OF GAMES LOSSES
#==============================================================================

for df in ('rGamesC', 'rGamesD', 'tGamesC', 'tGamesD'):

    dataDict[df + 'ModelData'] =  buildModelData(gameDF = dataDict[df])

#==============================================================================
# CALCULATE TEAM SUMMARIES FOR REGULAR SEASON & TOURNAMENT
#==============================================================================


for df in ('rGamesC', 'rGamesD'):

 
    dataDict[df + 'singleTeam'].loc[:, 'scoreGapWin'] =  dataDict[df + 'singleTeam'].loc[:, 'scoreGap'] * dataDict[df + 'singleTeam'].loc[:, 'win']
    dataDict[df + 'singleTeam'].loc[:, 'scoreGapLoss'] = dataDict[df + 'singleTeam'].loc[:, 'scoreGap'] * dataDict[df + 'singleTeam'].loc[:, 'win'].replace({0:1, 1:0})
    
    
    statCols = filter(lambda c: c not in ('DayNum', 'NumOT', 'FTM', 'FGM', 'FGM3', 'opponentID', 'TeamID', 'Season'),
                      dataDict[df + 'singleTeam'].columns.tolist())
    
    
    # Calculate season averages for each team and store results
    dataDict[df + 'TeamSeasonStats'] = dataDict[df + 'singleTeam'].groupby(['Season', 'TeamID'])[statCols].mean()
    
    
    # Weight scoreGapWin by win %
    dataDict[df + 'TeamSeasonStats'].loc[:, 'scoreGapWinPct'] = dataDict[df + 'TeamSeasonStats'].loc[:, 'scoreGapWin'] * dataDict[df + 'TeamSeasonStats'].loc[:, 'win']
    
    # Rank teams by each season stat metrics within 
    rankDesc = filter(lambda field: field in ('pointsAllowed'), 
                      dataDict[df + 'TeamSeasonStats'].columns.tolist())

  
    # Rank teams within season for each metric and use % for all teams between 0 and 1 (higher = better)
    # Change from sequential ranker to minmax scale to avoid unrealistic spread (4/22/19)
#    statsRank = dataDict[df + 'TeamSeasonStats'].groupby('Season').rank(ascending = True, pct = True)
    statsRank = dataDict[df + 'TeamSeasonStats'].groupby('Season').apply(lambda m: (m - m.min()) / (m.max() - m.min()))
    
    # Switch fields where lower values are better
    statsRank.loc[:, rankDesc] = 1 - statsRank.loc[:, rankDesc]
    
    # Merge ranked results with orginal team season metrics
    statsRankNames = map(lambda field: '{}Rank'.format(field), 
                         statsRank.columns.tolist())
    
    
    dataDict[df + 'TeamSeasonStats'] = dataDict[df + 'TeamSeasonStats'].merge(statsRank.rename(columns = dict(zip(statsRank.columns.tolist(), statsRankNames))),
                                                                                left_index = True, right_index = True)
    
    # Calculate win % over last 8 games
    dataDict[df + 'TeamSeasonStats'].loc[:, 'last8'] =  dataDict[df + 'singleTeam'].groupby(['Season', 'TeamID']).agg({'win': lambda games: np.mean(games[-8:])})
    
    
    # Identify conference champions by win in last game of conference tournament
    dataDict['confTgames'].sort_values(['Season', 'ConfAbbrev', 'DayNum'], inplace = True)
    
    confWinners = pd.DataFrame(dataDict['confTgames'].groupby(['Season', 'ConfAbbrev'])['WTeamID'].last())
    confWinners.reset_index('ConfAbbrev', drop = True, inplace = True)
    confWinners.loc[:, 'confChamp'] = 1
    confWinners.rename(columns = {'WTeamID' : 'TeamID'}, inplace = True)
    confWinners.set_index('TeamID', append = True, inplace = True)
    
    dataDict[df + 'TeamSeasonStats'] = dataDict[df + 'TeamSeasonStats'].merge(confWinners, left_index = True, right_index = True, how = 'left')
    dataDict[df + 'TeamSeasonStats'].loc[:, 'confChamp'].fillna(0, inplace = True)
    
del(statCols, statsRank, confWinners, statsRankNames)
    

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

for df in map(lambda g: g + 'TeamSeasonStats', ('rGamesC', 'rGamesD')):
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
    
        
        
    # Group seeds into 4's
#    dataDict[df].loc[:, 'seedRankGroups'] = map(lambda rank: (rank-1) // 4, 
#                                            dataDict[df]['seedRank'].values.tolist())






# =================================================================== #
# STRENGTH OF SCHEDULE AND PERFORMANCE AGAINST STRONG TEAM METRICS #
# =================================================================== #

df = 'rGamesC'

# Create matchup stats using team season statistics
matchups = createMatchups(matchupDF = dataDict['{}singleTeam'.format(df)],
                          statsDF = dataDict['{}TeamSeasonStats'.format(df)])


#### SUBSET FOR DEVELOPMENT
#matchups = matchups[matchups['Season'] == 2019]



# Offense and defence performance: points scored and allowed against opponent compared to opponent averages
#   Invert defense metric to make a higher number better (postive = allowed fewer points in game than opponent average)
matchups.loc[:, 'offStrength'] = matchups.loc[:, 'Score'] - matchups.loc[:, 'opppointsAllowed']
matchups.loc[:, 'defStrength'] = (matchups.loc[:, 'pointsAllowed'] - matchups.loc[:, 'oppScore'])  * (-1.0)
matchups.loc[:, 'spreadStrength'] = matchups.loc[:, 'scoreGap'] + matchups.loc[:, 'oppscoreGap']



# Apply weight to offense and defense metrics by multiplying by opponent rank based on points allowed / points scored
matchups.loc[:, 'offStrengthOppDStrength'] = matchups.loc[:, 'offStrength'] * matchups.loc[:, 'opppointsAllowedRank']
matchups.loc[:, 'defStrengthOppOStrength'] = matchups.loc[:, 'defStrength'] * matchups.loc[:, 'oppScoreRank']
matchups.loc[:, 'spreadStrengthOppStrength'] = matchups.loc[:, 'spreadStrength'] * matchups.loc[:, 'oppscoreGapRank']


# Opponent Win % * if team won the game (use for calculating strength of team)
matchups.loc[:, 'TeamStrengthOppWin'] = matchups.loc[:, 'oppwin'] * matchups.loc[:, 'win']


matchups.loc[:, 'TeamStrengthOppStrength'] = matchups.loc[:, 'win'] * matchups.loc[:, 'oppscoreGapWinPct']



# Calculate team metrics for ranking

# Identify strength columns for aggregation and calculating team performance
strengthMetrics = filter(lambda metric: metric.find('Strength') >= 0, 
                         matchups.columns.tolist())


# Calculate team season means for each metric
strengthDF = matchups.groupby(['Season', 'TeamID']).agg(dict(zip(strengthMetrics, repeat(np.mean, len(strengthMetrics)))))
   


## COMMENTED OUT 4/23/19: comparison against opponents opponents doubles back onto self
# Create new matchup dataframe to compare team performance in game against opponents average in game performane
    # Example: How many points did the offense score relative to season average versus how many points opponent defense gives up relative to the team scoring average
#matchups = createMatchups(matchupDF = dataDict['{}singleTeam'.format(df)],
#                           statsDF = dataDict['{}TeamSeasonStats'.format(df)].merge(strengthDF, 
#                                                                                    left_index = True, 
#                                                                                    right_index = True)
#                                                )
#
## How many points did the offense score relative to season average versus how many points opponent defense gives up relative to the team scoring average
#matchups.loc[:, 'offStrength2'] = (matchups.loc[:, 'Score'] - matchups.loc[:, 'teamScore']) - matchups.loc[:, 'oppdefStrength']
#
## How many points did the defense give up relative to season average versus how many points opponent offsense scores relative to the team points allowed average
#matchups.loc[:, 'defStrength2'] = ((matchups.loc[:, 'pointsAllowed'] - matchups.loc[:, 'teampointsAllowed']) - matchups.loc[:, 'oppoffStrength'])  * (-1.0)
#
#
## How much did team win by relative to season average verus how much opponent typically wins by relative to the team score gap average
#matchups.loc[:, 'spreadStrength2'] = (matchups.loc[:, 'scoreGap'] - matchups.loc[:, 'teamscoreGap']) - matchups.loc[:, 'oppspreadStrength']
#
## Win * weight of average win % of their opponents win %
#matchups.loc[:, 'TeamStrengthOppWin2'] = matchups.loc[:, 'win'] * matchups.loc[:, 'oppTeamStrengthOppWin']
#
#
## Filter columns of only new metrics (all have the name "Strength" in them and end with "2")
#strengthMetrics = filter(lambda metric: len(re.findall('.*Strength.*2', metric)) > 0, 
#                         matchups.columns.tolist())
#
#strengthDF2 = matchups.groupby(['Season', 'TeamID']).agg(dict(zip(strengthMetrics, 
#                                                                  repeat(np.mean, len(strengthMetrics)))))


# Scale Data between 0 and 1 using minmax to avoid negatives and append values as '[metric]Rank'
# Commented out 4/11/19
#strengthDF2 = strengthDF2.merge(strengthDF2.groupby('Season')
#                                        .rank(pct = True)
#                                        .rename(columns = dict(map(lambda field: (field, '{}Rank'.format(field)),
#                                                                   strengthDF2.columns.tolist()))),
#                              left_index = True,
#                              right_index = True
#                              )


# Combine metrics
#strengthDF = strengthDF.merge(strengthDF2, left_index = True, right_index = True)


# Scale Data between 0 and 1 using minmax to avoid negatives and append values as '[metric]Rank'
# Change from merge to just replace scaled data (4/23/19)
strengthDF = strengthDF.groupby('Season').apply(lambda m: (m - m.min()) / (m.max() - m.min()))


#strengthDF = strengthDF.merge(strengthDF.groupby('Season')
#                                        .apply(lambda m: (m - m.min()) / (m.max() - m.min()))
##                                        .rank(pct = True)
#                                        .rename(columns = dict(map(lambda field: (field, '{}Rank'.format(field)),
#                                                                   strengthDF.columns.tolist()))),
#                              left_index = True,
#                              right_index = True)
##                              )




# Generate matchups using strength metrics and tournament games to determine power of each metric
strengthMatchupsTourney = createMatchups(matchupDF = dataDict['tGamesC'][['Season', 'WTeamID', 'LTeamID']],
                                         statsDF = strengthDF,
                                         teamID1 = 'WTeamID', 
                                         teamID2 = 'LTeamID',
                                         teamLabel1 = 'W',
                                         teamLabel2 = 'L',
                                         returnStatCols = True,
                                         calculateDelta = True,
                                         calculateMatchup = False)
 
    

# Generate matchups using strength metrics from Tournament Model dataset with 50/50 split
sMatchupsTmodel = createMatchups(matchupDF = dataDict['tGamesCModelData'][['Season', 'WTeamID', 'LTeamID', 'win']],
                                         statsDF = strengthDF,
                                         teamID1 = 'WTeamID', 
                                         teamID2 = 'LTeamID',
                                         teamLabel1 = 'W',
                                         teamLabel2 = 'L',
                                         returnStatCols = True,
                                         calculateDelta = True,
                                         calculateMatchup = False)


    
# Calculate Deltas for independent offense vs defense metrics
# Commented out 4/23/19
#offenseVsDefense = [('offStrengthOppDStrength', 'defStrengthOppOStrength'),
#                    ('offStrength', 'defStrength'),
#                    ('offStrengthOppDStrength', 'defStrengthOppOStrength')[::-1],
#                    ('offStrength', 'defStrength')[::-1]]
#
#
#for team, opp in offenseVsDefense:
#    newCol = '{}{}{}Delta'.format(team[:3], opp[:3], team[3:])
#    strengthMatchupsTourney.loc[:, newCol] = (strengthMatchupsTourney.loc[:, 'W{}'.format(team)] 
#                                                - strengthMatchupsTourney.loc[:, 'L{}'.format(opp)])
#
##    strengthMatchupsTS.loc[:, newCol] = (strengthMatchupsTS.loc[:, 'team{}'.format(team)] 
##                                                - strengthMatchupsTS.loc[:, 'opp{}'.format(opp)])
#
#    sMatchupsTmodel.loc[:, newCol] = (sMatchupsTmodel.loc[:, 'W{}'.format(team)] 
#                                                - sMatchupsTmodel.loc[:, 'L{}'.format(opp)])
#




# Only delta columns for plotting    
deltaFilter = filter(lambda metric: metric.endswith('Delta'), strengthMatchupsTourney.columns)
deltaFilterTModel = filter(lambda metric: metric.endswith('Delta'), sMatchupsTmodel.columns)   

# Calculate metric stats based on if delta is positive it results in a win
strengthMatchupsTourneyResults = pd.concat([strengthMatchupsTourney.groupby('Season')[deltaFilter].agg(lambda d: len(filter(lambda g: g > 0, d)) / len(filter(lambda g: g != 0, d))).max(),
                                            strengthMatchupsTourney.groupby('Season')[deltaFilter].agg(lambda d: len(filter(lambda g: g > 0, d)) / len(filter(lambda g: g != 0, d))).min(),
                                            strengthMatchupsTourney.groupby('Season')[deltaFilter].agg(lambda d: len(filter(lambda g: g > 0, d)) / len(filter(lambda g: g != 0, d))).mean(),
                                            strengthMatchupsTourney[deltaFilter].agg(lambda d: len(filter(lambda g: g > 0, d)) / len(filter(lambda g: g != 0, d)))
                                            ], axis = 1)
strengthMatchupsTourneyResults.rename(columns = {0:'season_max', 1:'season_min', 2:'season_mean', 3:'overall_mean'}, inplace = True)



fig, ax = plt.subplots(1, figsize = (0.9*GetSystemMetrics(0)//96, 0.8*GetSystemMetrics(1)//96))
#sns.boxplot(x='variable', y='value', data = pd.melt(strengthMatchupsTourney[deltaFilter]))
sns.boxplot(x='value', y='variable', hue = 'win', data = pd.melt(sMatchupsTmodel[deltaFilterTModel + ['win']], id_vars = 'win'), orient = 'h')
fig.tight_layout()
ax.grid(True)
fig.show()


# Plot distributions of winning and losing results
nRows = int(np.ceil(len(deltaFilterTModel)**0.5))
nCols = int(np.ceil(len(deltaFilterTModel)/nRows))


fig, ax = plt.subplots(nrows = nRows, ncols = nCols, figsize = (0.9*GetSystemMetrics(0)//96, 0.8*GetSystemMetrics(1)//96))
for i, metric in enumerate(deltaFilterTModel):
    sns.distplot(sMatchupsTmodel[metric][sMatchupsTmodel['win'] == 1], hist = False, ax = ax[i//nCols, i%nCols], kde_kws={"shade": True}, label = 'win')
    sns.distplot(sMatchupsTmodel[metric][sMatchupsTmodel['win'] == 0], hist = False, ax = ax[i//nCols, i%nCols], kde_kws={"shade": True}, label = 'loss')
    ax[i//nCols, i%nCols].grid(True)
    ax[i//nCols, i%nCols].legend()
    
fig.tight_layout()
fig.show()



# Plot distributions of winning and losing results metric deltas
nRows = int(np.ceil(len(deltaFilter)**0.5))
nCols = int(np.ceil(len(deltaFilter)/nRows))

fig, ax = plt.subplots(nrows = nRows, ncols = nCols, figsize = (0.9*GetSystemMetrics(0)//96, 0.8*GetSystemMetrics(1)//96))
for i, metric in enumerate(deltaFilter):
    sns.distplot(strengthMatchupsTourney[metric], hist = True, ax = ax[i//nCols, i%nCols], kde_kws={"shade": True}, label = 'delta')
    ax[i//nCols, i%nCols].grid(True)
    ax[i//nCols, i%nCols].legend()
    
fig.tight_layout()
fig.show()

# Plot distributions of winning and losing results metric deltas on Single Plot
fig, ax = plt.subplots(1, figsize = (0.9*GetSystemMetrics(0)//96, 0.8*GetSystemMetrics(1)//96))
for i, metric in enumerate(deltaFilter):
    sns.distplot(strengthMatchupsTourney[metric], hist = False, kde_kws={"shade": True}, ax = ax, label = metric)
ax.grid(True)
ax.legend()
ax.set_xlabel('Metric Delta')
    
fig.tight_layout()
fig.show()

# Plot all actual metrics for winning and losing team as distrbutions
#nRows = int(np.ceil(len(filter(lambda metric: not(metric.startswith('offdef') | metric.startswith('defoff')), deltaFilter))**0.5))
#nCols = int(np.ceil(len(filter(lambda metric: not(metric.startswith('offdef') | metric.startswith('defoff')), deltaFilter))/nRows))
#
#fig, ax = plt.subplots(nrows = nRows, ncols = nCols, figsize = (0.9*GetSystemMetrics(0)//96, 0.8*GetSystemMetrics(1)//96))
#for i, metric in enumerate(filter(lambda metric: not(metric.startswith('offdef') | metric.startswith('defoff')), deltaFilter)):
#    sns.distplot(strengthMatchupsTourney['W{}'.format(metric.replace('Delta', ''))], hist = False, ax = ax[i//nCols, i%nCols], kde_kws={"shade": True}, label = 'win')
#    sns.distplot(strengthMatchupsTourney['L{}'.format(metric.replace('Delta', ''))], hist = False, ax = ax[i//nCols, i%nCols], kde_kws={"shade": True}, label = 'loss')
#    ax[i//nCols, i%nCols].grid(True)
#    ax[i//nRows, i%nCols].legend()
#    
#fig.tight_layout()
#fig.show()




# Plot all actual metrics for winning and losing team as scatterplot
nRows = int(np.ceil(len(filter(lambda metric: not(metric.startswith('offdef') | metric.startswith('defoff')), deltaFilter))**0.5))
nCols = int(np.ceil(len(filter(lambda metric: not(metric.startswith('offdef') | metric.startswith('defoff')), deltaFilter))/nRows))

fig, ax = plt.subplots(nrows = nRows, ncols = nCols, figsize = (0.9*GetSystemMetrics(0)//96, 0.8*GetSystemMetrics(1)//96))
for i, metric in enumerate(filter(lambda metric: not(metric.startswith('offdef') | metric.startswith('defoff')), deltaFilter)):
    sns.scatterplot(strengthMatchupsTourney['W{}'.format(metric.replace('Delta', ''))], 
                    strengthMatchupsTourney['L{}'.format(metric.replace('Delta', ''))], 
                    ax = ax[i//nCols, i%nCols]
                    )
    ax[i//nCols, i%nCols].grid(True)
    ax[i//nRows, i%nCols].legend()
    
fig.tight_layout()
fig.show()


# Plot all actual metrics for winning team and metric delta as scatterplot
nRows = int(np.ceil(len(filter(lambda metric: not(metric.startswith('offdef') | metric.startswith('defoff')), deltaFilter))**0.5))
nCols = int(np.ceil(len(filter(lambda metric: not(metric.startswith('offdef') | metric.startswith('defoff')), deltaFilter))/nRows))

fig, ax = plt.subplots(nrows = nRows, ncols = nCols, figsize = (0.9*GetSystemMetrics(0)//96, 0.8*GetSystemMetrics(1)//96))
for i, metric in enumerate(filter(lambda metric: not(metric.startswith('offdef') | metric.startswith('defoff')), deltaFilter)):
    sns.scatterplot(strengthMatchupsTourney['W{}'.format(metric.replace('Delta', ''))], 
                    strengthMatchupsTourney[metric], 
                    ax = ax[i//nCols, i%nCols]
                    )
    ax[i//nCols, i%nCols].grid(True)
    ax[i//nRows, i%nCols].legend()
    
fig.tight_layout()
fig.show()


# Plot all actual metrics for winning team and metric delta as box and whisker plot
nRows = int(np.ceil(len(deltaFilter)**0.5))
nCols = int(np.ceil(len(deltaFilter)/nRows))

fig, ax = plt.subplots(nrows = nRows, ncols = nCols, figsize = (0.9*GetSystemMetrics(0)//96, 0.8*GetSystemMetrics(1)//96))
for i, metric in enumerate(deltaFilter):
    sns.boxplot(strengthMatchupsTourney['W{}'.format(metric.replace('Delta', ''))].round(1), 
                    strengthMatchupsTourney[metric], 
                    ax = ax[i//nCols, i%nCols]
                    )
    ax[i//nCols, i%nCols].grid(True)
    ax[i//nRows, i%nCols].legend()
    
fig.tight_layout()
fig.show()


# Create new interaction metrics: team metric * delta metric
interactionMetrics = map(lambda m: ('W{}'.format(m.replace('Delta', '')), m), deltaFilter)
#interactionMetrics = list(product(filter(lambda c: (c.startswith('W') & (c != 'WTeamID')), sMatchupsTmodel.columns),
#                                  filter(lambda c: c.endswith('Delta'), sMatchupsTmodel.columns)))

# Iterate through interactions
deltaInterFilter = []
for w, d in interactionMetrics:
    strengthMatchupsTourney.loc[:, '{}DeltaInter'.format(w)] = sMatchupsTmodel[w] * sMatchupsTmodel[d]
    

deltaInterFilter = map(lambda m: '{}DeltaInter'.format(m[0]), interactionMetrics)


# Plot distributions of winning and losing results metric deltas on Single Plot
fig, ax = plt.subplots(1, figsize = (0.9*GetSystemMetrics(0)//96, 0.8*GetSystemMetrics(1)//96))
for i, metric in enumerate(deltaInterFilter):
    sns.distplot(strengthMatchupsTourney[metric], hist = False, kde_kws={"shade": True}, ax = ax, label = metric)
ax.grid(True)
ax.legend()
ax.set_xlabel('Metric Delta')
    
fig.tight_layout()
fig.show()



# Calculate metric stats based on if delta is positive it results in a win
strengthMatchupsTourneyResults = pd.concat([strengthMatchupsTourney.groupby('Season')[deltaFilter + deltaInterFilter].agg(lambda d: len(filter(lambda g: g > 0, d)) / len(filter(lambda g: g != 0, d))).max(),
                                            strengthMatchupsTourney.groupby('Season')[deltaFilter + deltaInterFilter].agg(lambda d: len(filter(lambda g: g > 0, d)) / len(filter(lambda g: g != 0, d))).min(),
                                            strengthMatchupsTourney.groupby('Season')[deltaFilter + deltaInterFilter].agg(lambda d: len(filter(lambda g: g > 0, d)) / len(filter(lambda g: g != 0, d))).mean(),
                                            strengthMatchupsTourney[deltaFilter + deltaInterFilter].agg(lambda d: len(filter(lambda g: g > 0, d)) / len(filter(lambda g: g != 0, d)))
                                            ], axis = 1)
strengthMatchupsTourneyResults.rename(columns = {0:'season_max', 1:'season_min', 2:'season_mean', 3:'overall_mean'}, inplace = True)





# ###### DEV Create bins out of strength delta metrics and analyze win rate by bin
kBins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')

strengthBins = pd.DataFrame(kBins.fit_transform(sMatchupsTmodel[deltaFilterTModel]),
                 columns = sMatchupsTmodel[deltaFilterTModel].columns)

strengthBins = strengthBins.merge(pd.DataFrame(sMatchupsTmodel.loc[:, 'win']), 
                                  left_index = True, 
                                  right_index = True)

strengthBinsMelt = pd.melt(strengthBins, id_vars = 'win', var_name = 'metric', value_name = 'bin')

strengthBinsWins = (strengthBinsMelt.groupby(['metric', 'bin'])
                                    .agg({'win':np.mean})
                                    .rename(columns = {'win':'winPct'})
                                    .reset_index('bin')
                                    )

# Perform linear regression on all metrics to determine which one has the strongest relationship
lm = LinearRegression()

strengthBinsLM = []
for metric in strengthBinsWins.groupby('metric').groups.keys():
    lm.fit(strengthBinsWins.groupby('metric').get_group(metric)['bin'].values.reshape(-1, 1),
           strengthBinsWins.groupby('metric').get_group(metric)['winPct'])
    
    strengthBinsLM.append((metric,
                           lm.score(strengthBinsWins.groupby('metric').get_group(metric)['bin'].values.reshape(-1, 1),
                                    strengthBinsWins.groupby('metric').get_group(metric)['winPct']), 
                            lm.coef_[0], 
                            lm.intercept_))
    
strengthBinsLM = pd.DataFrame(strengthBinsLM, columns = ['metric', 'r2', 'coef', 'intercept']).set_index('metric')




# Plot heat map of metric bins
fig, ax = plt.subplots(1, figsize = (0.9*GetSystemMetrics(0)//96, 0.8*GetSystemMetrics(1)//96))
fig.suptitle('Strength Metric Bins Win %', fontsize = 16)
sns.heatmap(strengthBinsWins.pivot(columns = 'bin', values = 'winPct'), 
            center = 0.5, 
            annot = True, 
            linewidths = 1,
            linecolor = 'k',
            cmap = 'RdYlGn', 
            ax = ax)
fig.tight_layout()
fig.show()


# PLot line plot of metric bins
fig, ax = plt.subplots(1, figsize = (0.9*GetSystemMetrics(0)//96, 0.8*GetSystemMetrics(1)//96))
sns.lineplot(x = 'bin', y = 'winPct', data = strengthBinsWins.reset_index(), hue = 'metric', ax = ax, marker = 'o')
ax.grid(True)

fig.tight_layout()
fig.show()




# Boxplot of results
#fig, ax = plt.subplots(1, figsize = (10, 6))
#sns.boxplot(x = 'accuracy', 
#            y = 'metric', 
#            orient = 'h', 
#            data = pd.melt(strengthMatchupsTourney.groupby('Season')[deltaFilter]
#                                                    .agg(lambda d: len(filter(lambda g: g > 0, d)) / len(filter(lambda g: g != 0, d))), 
#                            var_name = 'metric', 
#                            value_name = 'accuracy'), 
#            ax = ax)
#            
#ax.grid(True)
#fig.tight_layout()



###########################################
#### CALCULATE WINS AGAINST TOP N TEAMS ###
###########################################

# Create matchups of regular season games with spreadStrengthOppStrength for ranking
matchups2 = createMatchups(matchupDF = dataDict['rGamesCsingleTeam'][['Season', 'TeamID', 'opponentID', 'win']],
                           statsDF = pd.DataFrame(strengthDF.loc[:, 'spreadStrengthOppStrength'].groupby('Season').rank(ascending = False)),                                                                 
                           teamID1 = 'TeamID', 
                           teamID2 = 'opponentID',
                           teamLabel1 = 'team', 
                           teamLabel2 = 'opp',
                           calculateDelta = False,
                           returnStatCols = True,
                           calculateMatchup = False,
                           )


# use 'spreadStrengthOppStrengthDelta' as teamstrength since it highest correlation with Wins in the Tournament
# Find best metric for # of wins agains topN teams
topNlist = range(10, 301, 10)

for topN in topNlist:
    strengthDF = strengthDF.merge((pd.DataFrame(matchups2[matchups2['oppspreadStrengthOppStrength'] <= topN]
                                                                                      .groupby(['Season', 'TeamID'])
                                                                                      .agg({'win':np.sum}))
                                                                            .rename(columns = {'win':'wins{}'.format(str(topN).zfill(3))})
                                                                            ),
                                                                left_index = True,
                                                                right_index = True,
                                                                how = 'left'
                                                                )

    
# Fill teams with no wins against top teams with 0
strengthDF.fillna(0, inplace = True)


# Create tournament matchups with just wins against TopN fields
# Generate matchups using strength metrics and tournament games to determine power of each metric

winCols = filter(lambda metric: metric.startswith('wins'), strengthDF.columns)
strengthMatchupsTourney = createMatchups(matchupDF = dataDict['tGamesC'][['Season', 'WTeamID', 'LTeamID']],
                                         statsDF = strengthDF[winCols],
                                         teamID1 = 'WTeamID', 
                                         teamID2 = 'LTeamID',
                                         teamLabel1 = 'W',
                                         teamLabel2 = 'L',
                                         returnStatCols = True,
                                         calculateDelta = True,
                                         calculateMatchup = False)



sMatchupsTmodel = createMatchups(matchupDF = dataDict['tGamesCModelData'][['Season', 'WTeamID', 'LTeamID', 'win']],
                                         statsDF = strengthDF[winCols],
                                         teamID1 = 'WTeamID', 
                                         teamID2 = 'LTeamID',
                                         teamLabel1 = 'W',
                                         teamLabel2 = 'L',
                                         returnStatCols = True,
                                         calculateDelta = True,
                                         calculateMatchup = False)

# Only delta columns for plotting    
deltaFilter = filter(lambda metric: metric.endswith('Delta'), strengthMatchupsTourney.columns)


# Box plot of win against Top N results
#fig, ax = plt.subplots(1, figsize = (10, 6))
#sns.boxplot(y='variable', x='value', orient = 'h', data = pd.melt(strengthMatchupsTourney[deltaFilter]), ax = ax)
#ax.grid(True)
#fig.tight_layout()
#fig.show()

# Calculate win % for all games where there is a difference
strengthMatchupsTourneyResultsTopN = pd.concat([strengthMatchupsTourney.groupby('Season')[deltaFilter].agg(lambda d: len(filter(lambda g: g > 0, d)) / len(filter(lambda g: g != 0, d))).max(),
                                            strengthMatchupsTourney.groupby('Season')[deltaFilter].agg(lambda d: len(filter(lambda g: g > 0, d)) / len(filter(lambda g: g != 0, d))).min(),
                                            strengthMatchupsTourney.groupby('Season')[deltaFilter].agg(lambda d: len(filter(lambda g: g > 0, d)) / len(filter(lambda g: g != 0, d))).mean(),
                                            strengthMatchupsTourney[deltaFilter].agg(lambda d: len(filter(lambda g: g > 0, d)) / len(filter(lambda g: g != 0, d)))
                                            ], axis = 1)
strengthMatchupsTourneyResultsTopN.rename(columns = {0:'season_max', 
                                                     1:'season_min', 
                                                     2:'season_mean', 
                                                     3:'overall_mean'}, 
                                            inplace = True)



# Plot topN results
plotWins = map(lambda s: (int(re.findall('[0-9]+', s)[0]), s), 
                        strengthMatchupsTourneyResultsTopN.index.get_values())

plotWins.sort()
plotWins = zip(*plotWins)[1]

fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (0.9*GetSystemMetrics(0)//96, 0.8*GetSystemMetrics(1)//96))
sns.barplot(x='overall_mean', y = 'index', data = strengthMatchupsTourneyResultsTopN.reset_index(), order = plotWins, ax = ax[0])
sns.lineplot(x = map(lambda s: float(re.findall('[0-9]+', s)[0]), 
                        strengthMatchupsTourneyResultsTopN.index.get_values()),
                y = strengthMatchupsTourneyResultsTopN.loc[:, 'overall_mean'], ax = ax[1], marker = 'o')

ax[0].grid(True)
ax[1].grid(True)
fig.tight_layout()   
fig.show() 
  


# Create bins for wins aganst TopN
kBins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')

strengthBinsTopN = pd.DataFrame(kBins.fit_transform(sMatchupsTmodel[deltaFilter]),
                 columns = sMatchupsTmodel[deltaFilter].columns)

strengthBinsTopN = strengthBinsTopN.merge(pd.DataFrame(sMatchupsTmodel.loc[:, 'win']), 
                                  left_index = True, 
                                  right_index = True)

strengthBinsTopNMelt = pd.melt(strengthBinsTopN, id_vars = 'win', var_name = 'metric', value_name = 'bin')

strengthBinsTopNWins = (strengthBinsTopNMelt.groupby(['metric', 'bin'])
                                    .agg({'win':np.mean})
                                    .rename(columns = {'win':'winPct'})
                                    .reset_index('bin')
                                    )

# Perform linear regression on all metrics to determine which one has the strongest relationship
lm = LinearRegression()

strengthBinsTopNLM = []
for metric in strengthBinsTopNWins.groupby('metric').groups.keys():
    lm.fit(strengthBinsTopNWins.groupby('metric').get_group(metric)['bin'].values.reshape(-1, 1),
           strengthBinsTopNWins.groupby('metric').get_group(metric)['winPct'])
    
    strengthBinsTopNLM.append((metric,
                           lm.score(strengthBinsTopNWins.groupby('metric').get_group(metric)['bin'].values.reshape(-1, 1),
                                    strengthBinsTopNWins.groupby('metric').get_group(metric)['winPct']), 
                            lm.coef_[0], 
                            lm.intercept_))
    
strengthBinsTopNLM = pd.DataFrame(strengthBinsTopNLM, columns = ['metric', 'r2', 'coef', 'intercept']).set_index('metric')


# Plot heat map of metric bins
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (0.9*GetSystemMetrics(0)//96, 0.8*GetSystemMetrics(1)//96))
sns.heatmap(strengthBinsTopNWins.pivot(columns = 'bin', values = 'winPct'), center = 0.5, cmap = 'RdYlGn', annot = True, ax = ax)
fig.tight_layout()
fig.show()

# PLot line plot of metric bins
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (0.9*GetSystemMetrics(0)//96, 0.8*GetSystemMetrics(1)//96))
sns.lineplot(x = 'bin', y = 'winPct', data = strengthBinsTopNWins.reset_index(), hue = 'metric', ax = ax, marker = 'o')
ax.grid(True)

fig.tight_layout()
fig.show()

# lmplot
#g = sns.lmplot(x = 'bin', y = 'winPct', data = strengthBinsTopNWins.reset_index(), hue = 'metric')



### MERGE FINAL METRICS INTO FINAL TEAM SEASON STATS

# Merge team strength and wins against top 50 teams metrics to team metrics
dataDict['{}TeamSeasonStats'.format('rGamesC')] = dataDict['{}TeamSeasonStats'.format('rGamesC')].merge(strengthDF.loc[:, ['spreadStrengthOppStrength', 'spreadStrengthOppStrengthRank', 'wins050']], left_index = True, right_index = True)  
dataDict['{}TeamSeasonStats'.format('rGamesD')] = dataDict['{}TeamSeasonStats'.format('rGamesD')].merge(strengthDF.loc[:, ['spreadStrengthOppStrength', 'spreadStrengthOppStrengthRank', 'wins050']], left_index = True, right_index = True)  
   



# DEV - Matplotlib changing figure size
#plt.get_fignums()
#plt.get_figlabels()
#plt.figure(15).tight_layout()
#plt.figure(15).set_size_inches(10,6)

#==============================================================================
# CACLUATE COLUMN SUMMARY FOR TEAM SEASON STATISTICS DATAFRAMES
#==============================================================================
# Create list of unique columns in all games DataFrames
#for df in map(lambda g: g + 'TeamSeasonStats', ('rGamesD', 'rGamesC')):
#    colSumDict[df] = generateDataFrameColumnSummaries(dataDict[df], returnDF=True)
#                                     
#                                     
  
#==============================================================================
# CREATE NEW TOURNAMENT MATCHUPS USING TEAM SEASON STATISTICS
# 
# ADD TOURNAMENT SEED STATISTICS
#
# CALCULATE MATCHUP PAIRS FOR DUMMIES
#  
# CREATE MODEL DATASET WITH SAME COLUMN CALCULATIONS
#==============================================================================

calculateDelta = True
returnStatCols = True
calculateMatchup = True

for df in filter(lambda g: g.startswith('t'), gamesData):
   
    # Reference assocated regular season data
#    dataDict[df + 'statsModelData'] = createMatchups(matchupDF = dataDict[df + 'ModelData'].loc[:, ['Season', 'WTeamID', 'LTeamID', 'win']], 
#                                                statsDF = dataDict['{}TeamSeasonStats'.format(df.replace('t', 'r'))],
#                                                teamID1 = 'WTeamID',
#                                                teamID2 = 'LTeamID',
#                                                teamLabel1 = 'W',
#                                                teamLabel2 = 'L',
#                                                returnStatCols = returnStatCols,
#                                                calculateDelta = calculateDelta,
#                                                calculateMatchup = calculateMatchup,
#                                                extraMatchupCols = ['seedRank'])



    # Reference assocated regular season data
#    dataDict[df + 'statsModelData'] = createMatchups(matchupDF = dataDict[df].loc[:, ['Season', 'WTeamID', 'LTeamID']], 
#                                                statsDF = dataDict['{}TeamSeasonStats'.format(df.replace('t', 'r'))],
#                                                teamID1 = 'WTeamID',
#                                                teamID2 = 'LTeamID',
#                                                teamLabel1 = 'W',
#                                                teamLabel2 = 'L',
#                                                returnStatCols = returnStatCols,
#                                                calculateDelta = calculateDelta,
#                                                calculateMatchup = calculateMatchup,
#                                                extraMatchupCols = ['seedRank'])


    # Reference assocated regular season data
    dataDict[df + 'statsModelData'] = createMatchups(matchupDF = dataDict[df + 'singleTeam'].loc[:, ['Season', 'TeamID', 'opponentID', 'win']], 
                                                statsDF = dataDict['{}TeamSeasonStats'.format(df.replace('t', 'r'))],
                                                teamID1 = 'TeamID',
                                                teamID2 = 'opponentID',
                                                teamLabel1 = 'team',
                                                teamLabel2 = 'opp',
                                                returnStatCols = returnStatCols,
                                                calculateDelta = calculateDelta,
                                                calculateMatchup = calculateMatchup,
                                                extraMatchupCols = ['seedRank'])




# #############################################################################
# ###################### TOURNAMENT MODEL DATA EDA  ###########################
# #############################################################################


# Tournament Dataset
x = dataDict['tGamesCstatsModelData']

#x.loc[:, 'highSeed'] = np.min(x[['WseedRank', 'LseedRank']], axis = 1)
#x.loc[:, 'lowSeed'] = np.max(x[['WseedRank', 'LseedRank']], axis = 1)
#x.loc[:, 'seedRankDeltaAbs'] = x.loc[:, 'lowSeed'] - x.loc[:, 'highSeed']



# #################################################################
# ####### CONFERENCE AND CONFERENCE CHAMPS MATCHUPS ###############
# #################################################################


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






# ##########################################
# ####### SEED RANK MATCHUPS ###############
# ##########################################

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

confDeltaStats.loc[:, 'winPctBin'] = map(lambda w: int(100 *(round(w / binIncrement, 0) * binIncrement)), confDeltaStats['winPct'])

# Counts number of matchup groups and sum # of games in each bin
confDeltaStats.groupby('winPctBin').agg({'winPct': lambda x: len(x),
                                           'numGames': np.sum}).rename(columns = {'winPct':'numMatchups'})

# Store bins for OHE with predictions
oheDict['confs'] = confDeltaStats['winPctBin'].to_dict()



# Repeat process for seed rank matchups
srDeltaStats.loc[:, 'winPctBin'] = map(lambda w: int(100 *(round(w / binIncrement, 0) * binIncrement)), srDeltaStats['winPct'])

srDeltaStats.groupby('winPctBin').agg({'winPct': lambda x: len(x),
                                           'numGames': np.sum}).rename(columns = {'winPct':'numMatchups'})

    
oheDict['seedRanks'] = srDeltaStats.reset_index('seedRankDelta')['winPctBin'].to_dict()

# Prioritize matchup categories for encoding based on # of Games (support) and delta from 50% (lift)
#confDeltaStats.loc[:, 'lift'] = confDeltaStats['winPct'] / 0.5
#confDeltaStats.loc[:, 'liftsupport'] = (np.abs(confDeltaStats['winPct'] - 0.5) * confDeltaStats['numGames'])
#confDeltaStats.sort_values('liftsupport', ascending = False, inplace = True)
#confDeltaStats.loc[:, 'liftsupport_rank'] = range(confDeltaStats.shape[0])




    







#==============================================================================
# CORRELATION ANALYSIS
#==============================================================================

# Columns to exculde in correlation anaylsis:
#   All base columns excluding scoreGap
#   TeamIDs
#   All object columns 


performCorrelation = False

if performCorrelation == True:
    
    corrExcludeFilter = lambda c: (((c not in colsBase) | 
                                    (c in ('scoreGap', 'win'))) 
                                        & (c.endswith('ID') == False))
    
    
    
    dataDict[df + 'modelData'][c].dtype.hasobject == False
    
    fig, ax = plt.subplots(1)
    
    for df in map(lambda n: n + 'modelData', 
                  filter(lambda d: d.startswith('t'), gamesData)):
        
        corrColsTemp = filter(corrExcludeFilter, 
                              dataDict[df].columns.tolist())    
        
        dataDict[df + 'Corr'] = dataDict[df][corrColsTemp].corr()
        
        plotCorrHeatMap(dataDict[df + 'Corr'], 
                        plotTitle= df + ' Correlation Analysis')
    
    
        x = pd.concat([dataDict[df + 'Corr']['win'].rank(pct = True),
                       dataDict[df + 'Corr']['win']], axis = 1)
          
             
        x.columns = ['rank', 'corr']
        
        x = x[x.index.values != 'win']
    
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


performPCA = False

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
#        pcaCols = colSumDict[df]['colName'][~colSumDict[df]['isObject']]
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
        #pcaCols = colSumDict[df]['colName'][~colSumDict[df]['isObject']]
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
statCols = ['seedRank', 'spreadStrengthOppStrengthRank', 'wins050', 'confChamp', 'confGroups']
statCols = []

modelMatchups = createMatchups(dataDict['tGamesCsingleTeam'][baseCols + ['win']],
                               statsDF = dataDict['rGamesCTeamSeasonStats'],
                               teamID1 = 'TeamID',
                               teamID2 = 'opponentID',
                               teamLabel1 = 'team',
                               teamLabel2 = 'opp',
                               returnStatCols = True,
                               calculateDelta = True,
                               calculateMatchup = False)


modelMatchups.loc[:, 'confMatchup'] = pd.Series(map(lambda m: tuple(m), 
                                                 modelMatchups[['teamconfGroups', 'teamconfChamp', 'oppconfGroups', 'oppconfChamp']].values.tolist()))

modelMatchups.loc[:, 'seedMatchup'] = pd.Series(map(lambda m: tuple(m), 
                                                 modelMatchups[['teamseedRank', 'oppseedRank']].values.tolist()))


    
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
    

    
    modelCols = filter(lambda c: (modelMatchups[c].dtype.hasobject == False)
                                & (c not in baseCols + ['win'])
                              #  | (c == 'win')
                       , modelMatchups.columns.tolist())
    
    

    
    # Model Data & initial poly fit
    data = modelMatchups[modelMatchups['Season'] >= 1985]
    poly.fit(data.loc[:, modelCols])
    
    
    featureCols = poly.get_feature_names(modelCols)
    
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
        featureRank = zip(forest.feature_importances_, 
                          featureCols, 
                          repeat(len(featureCols), len(featureCols)))
        
        # Sort features by importanc
        featureRank.sort(reverse = True)


        featureRankAll.append(featureRank)

        # Remove lowest feature rankings
        featureCols = list(zip(*featureRank)[1][:-2])



   
    featureRankAllDF = pd.DataFrame(list(chain(*featureRankAll)), columns = ['importance', 'metric', 'numFeatures'])
    featureRankAllDF = featureRankAllDF.merge(pd.DataFrame(modelResults, 
                                                           columns = ['numFeatures', 
                                                                      'aucLog', 'accLog', 
                                                                      'aucForest', 'accForest']),
                                                left_on = 'numFeatures', right_on = 'numFeatures')

    featureRankCount = featureRankAllDF.groupby('metric')['numFeatures'].count()


    
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
    plt.plot(zip(*modelResults)[0], zip(*modelResults)[1], label = 'logistic')
    plt.plot(zip(*modelResults)[0], zip(*modelResults)[3], label = 'forest')
    plt.grid()
    plt.legend()




# =============================================================================
# BINNING AND ENCODING FEATURES FOR MODELING
# =============================================================================
    

# One Hot Encode Matchups

# Get bins
modelMatchups.loc[:, 'confMatchupBin'] = map(lambda m: oheDict['confs'].get(m, 50), modelMatchups.loc[:, 'confMatchup'])
modelMatchups.loc[:, 'seedMatchupBin'] = map(lambda m: oheDict['seedRanks'].get(m, 50), modelMatchups.loc[:, 'seedMatchup'])

# Get dummies (One Hot Encode) for Matchup Bins
modelMatchups = pd.get_dummies(data = modelMatchups, columns = ['confMatchupBin', 'seedMatchupBin'])


# Create bins out of spreadStrengthOppStrengthRankDelta and wins050Delta
kBins = KBinsDiscretizer(n_bins=10, encode='onehot-dense', strategy='quantile')


# Merge spreadStrength OHE bins to rest of model dataset
modelMatchups = pd.concat([modelMatchups,
                           pd.DataFrame(kBins.fit_transform(modelMatchups[['spreadStrengthOppStrengthRankDelta', 'wins050Delta']]),
                                        columns = chain(*map(lambda b: map(lambda bb: '{}_{}'.format(b[0], bb), 
                                                                                    range(b[1])), 
                                                                zip(['spreadBin', 'win050Bin'], kBins.n_bins_))))],
                            axis = 1)

    
# Filter only bin columns for modeling
modelCols = filter(lambda col: col.find('Bin_') >= 0, modelMatchups.columns)   


# ============================================================================
# DEV 4/21/19
# LOGISTIC CV MODELING
# ============================================================================

from sklearn.linear_model import LogisticRegressionCV

logCV = LogisticRegressionCV(cv = 5, 
                             scoring = 'neg_log_loss', 
                             max_iter=100, 
#                             solver = 'liblinear',
                             solver = 'lbfgs'
                             )

xTrain, yTrain, xTest, yTest = testTrainSplit()


xTrain, xTest, yTrain, yTest = train_test_split(modelMatchups[modelCols], 
                                                modelMatchups['win'],
                                                test_size = 0.2)

season = 2018
xTrain = modelMatchups[(modelMatchups['Season'] != season)][modelCols]
yTrain = modelMatchups[(modelMatchups['Season'] != season)]['win']
xTest = modelMatchups[(modelMatchups['Season'] == season)][modelCols]
yTest = modelMatchups[(modelMatchups['Season'] == season)]['win']

logCV.fit(xTrain, yTrain)
logCV.score(xTest, yTest)

x = modelAnalysis(model = logCV,
                  data = modelMatchups,
                  targetCol = 'win',
                  indCols = modelCols)

def modelAnalysis(model, data = [], 
                  targetCol = None, 
                  indCols = None, 
                  testTrainDataList = [], 
                  testTrainSplit = 0.2):


#==============================================================================
# MODEL DEVELOPMENT & GRID SEARCH
#==============================================================================

modelDict = {}

for df in ('tGamesC', 
#           'tGamesD'
           ):
    
    modelDict[df] = {}
    
    # Modeling columns for new pipeline
#    indCols2 = filter(lambda c: (dataDict[df + 'modelData'][c].dtype.hasobject == False)
#                                & ((c.find('Delta') >= 0) & ((c.find('Rank') >= 0) | (c.find('wins50') >= 0)))
#                              #  | (c == 'win')
#                       , dataDict[df + 'modelData'].columns.tolist())
    
    
    # Model List
#    mdlList = [ ExtraTreesClassifier(n_estimators = 50, random_state = 1127), 
#                RandomForestClassifier(n_estimators = 50, random_state = 1127),
#                LogisticRegression(random_state = 1127),
#                KNeighborsClassifier(),
#                SVC(random_state = 1127, probability = True)]
    
    # Configure parameter grid for pipeline
    numIndCols = len(modelCols)
    numPCASplits = 4
    
    
  
    #fReduce = FeatureUnion([('pca', PCA()), ('kBest', SelectKBest(k = 1))])
    #fReduce = FeatureUnion([('pca', PCA()), 
                           # ('kBest', SelectKBest(k = 1))
                           # ('rfe', RFE(LogisticRegression(random_state = 1127)))
    #                        ])
   
    #fReduce = RFE(SVC(kernel="linear", random_state = 1127), n_features_to_select = 5)
    #fReduce = SelectPercentile(percentile = 0.5)
    fReduce = SelectKBest(k = 1)
 
    
    # Create pipeline of Standard Scaler, PCA reduction, and Model (default Logistic)
    pipe = Pipeline([#('sScale', StandardScaler()), 
                     #('sScale', QuantileTransformer()),
#                     ('scale', MinMaxScaler()),
#                     ('pca',  PCA(n_components = numIndCols // 2)),
#                     ('poly', PolynomialFeatures(degree = 2, interaction_only = True)),
                     #('kbd', KBinsDiscretizer(n_bins = 4, encode = 'ordinal')),
#                     ('fReduce', fReduce),
                     # ('fReduce', PCA(n_components = 10)),
                     ('mdl', LogisticRegression(random_state = 1127))])
    
    
    paramGrid = [
#                {'mdl' : [ExtraTreesClassifier(n_estimators = 50,
#                                               n_jobs = -1,
#                                               random_state = 1127), 
#                          RandomForestClassifier(random_state = 1127,
#                                                 n_estimators = 50,
#                                                 n_jobs = -1,
#                                                 verbose = 0),
#                          GradientBoostingClassifier(n_estimators = 50,
#                                                     random_state = 1127)
#                          ],                        
#                 'mdl__min_samples_split' : np.arange(.005, .1, .01),
#                 'mdl__min_samples_leaf' : xrange(2, 11, 4),
##                 'mdl__n_estimators' : [25, 100, 200]
#                    },
                    
                {'mdl' : [LogisticRegression(random_state = 1127)],
                 'mdl__C' : map(lambda i: 10**i, xrange(-1,4))
                    },
                    
                {'mdl' : [SVC(probability = True)],
                 'mdl__C' : map(lambda i: 10**i, xrange(-1,4)),
                 'mdl__gamma' : map(lambda i: 10**i, xrange(-4,1))
                    },
                    
                {'mdl' : [KNeighborsClassifier()],
                 'mdl__n_neighbors' : range(3, 15, 2)
                    }
                ]
            
    
    # Update paramGrid with other grid search parameters that apply to all models
#    map(lambda d: d.update({'fReduce__n_features_to_select' : range(1, min(1 +  numIndCols, 26), 2)}),
#        paramGrid)
#    map(lambda d: d.update({'fReduce__k' : range(1,min(20, 1 + numIndCols // 2))}), paramGrid)
#    map(lambda d: d.update({'fReduce__percentile' : np.arange(0.01, 0.21, 0.04)}),  paramGrid)
#    map(lambda d: d.update({'pca__n_components' : range(3, numIndCols, numIndCols // numPCASplits)}), paramGrid)    
    
    
    

    
    
    
    
    # Run grid search on modeling pipeline
    timer()
    modelDict[df]['analysis'] = modelAnalysisPipeline(modelPipe = pipe,
                          data = modelMatchups,
                          indCols = modelCols,
                          targetCol = 'win',
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
# EVALUATE MODELS ON TEST DATA
#   ROC CURVES
#   AUC
#   LOG LOSS
#==============================================================================


# Plot roc curve for best params for each model type
for df in modelDict.iterkeys():       

    # Refit pipleiine with model parameters and calculate prediciton probabilities

    # ROC Curves
    rocCurves = map(lambda params: roc_curve(modelDict[df]['analysis']['yTest'],
                                          (modelDict[df]['analysis']['pipe'].estimator.set_params(**params)
                                              .fit(modelDict[df]['analysis']['xTrain'], modelDict[df]['analysis']['yTrain'])
                                              .predict_proba(modelDict[df]['analysis']['xTest'])[:,1])),
                    modelDict[df]['bests']['params'].values.tolist())


    # Append best model
    rocCurves.append(roc_curve(modelDict[df]['analysis']['yTest'],
                               modelDict[df]['analysis']['pipe'].predict_proba(modelDict[df]['analysis']['xTest'])[:,1]))

    # AUC
    rocAucs = map(lambda params: roc_auc_score(modelDict[df]['analysis']['yTest'],
                                          (modelDict[df]['analysis']['pipe'].estimator.set_params(**params)
                                              .fit(modelDict[df]['analysis']['xTrain'], modelDict[df]['analysis']['yTrain'])
                                              .predict_proba(modelDict[df]['analysis']['xTest'])[:,1])),
                    modelDict[df]['bests']['params'].values.tolist())


    # Append best model
    rocAucs.append(roc_auc_score(modelDict[df]['analysis']['yTest'],
                                   modelDict[df]['analysis']['pipe'].predict_proba(modelDict[df]['analysis']['xTest'])[:,1]))

    # Log Loss
    logloss = map(lambda params: log_loss(modelDict[df]['analysis']['yTest'],
                                          (modelDict[df]['analysis']['pipe'].estimator.set_params(**params)
                                              .fit(modelDict[df]['analysis']['xTrain'], modelDict[df]['analysis']['yTrain'])
                                              .predict_proba(modelDict[df]['analysis']['xTest']))),
                    modelDict[df]['bests']['params'].values.tolist())

    # Confusion Matrix
    confuseMatrix = map(lambda params: confusion_matrix(modelDict[df]['analysis']['yTest'],
                                          (modelDict[df]['analysis']['pipe'].estimator.set_params(**params)
                                              .fit(modelDict[df]['analysis']['xTrain'], modelDict[df]['analysis']['yTrain'])
                                              .predict(modelDict[df]['analysis']['xTest']))),
                    modelDict[df]['bests']['params'].values.tolist())

    # Accuracy
    accuracy = map(lambda c: np.trace(c) / np.sum(c), confuseMatrix)


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


help(combinations)





# ============================================================================
# ================= TOURNAMENT PRECITIONS ====================================
# ============================================================================

# Year for predictions
yr = 2019

for df in modelDict.iterkeys():
   
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
# ================= TOURNAMENT PRECITIONS ALL POSSIBILITIES===================
# ============================================================================

season = 2019
x = dataDict['rGamesCTeamSeasonStats'][(dataDict['rGamesCTeamSeasonStats'].index.get_level_values('Season') == season) & (dataDict['rGamesCTeamSeasonStats']['seedRank'] <= 16)].index.get_level_values('TeamID').tolist()

tourneyMatchups = pd.DataFrame(combinations(x, 2), columns = ['TeamID', 'opponentID'])
tourneyMatchups.loc[:, 'Season'] = season

tourneyMatchups = createMatchups(matchupDF = tourneyMatchups, 
                                                statsDF = dataDict['rGamesCTeamSeasonStats'],
                                                returnStatCols = False,
                                                calculateDelta = True,
                                                calculateMatchup = True,
                                                extraMatchupCols = ['seedRank'])


tourneyMatchups.loc[:, 'teamWinProb'] = modelDict[df]['analysis']['pipe'].predict_proba(tourneyMatchups.loc[:, indCols2])[:,1]


tourneyMatchups = tourneyMatchups.merge(pd.DataFrame(dataDict['teams'].set_index('TeamID')['TeamName']),
                                                            left_on = 'TeamID', right_index = True)
tourneyMatchups = tourneyMatchups.merge(pd.DataFrame(dataDict['teams'].rename(columns = {'TeamID':'opponentID', 'TeamName': 'opponentName'})
                                                                                             .set_index('opponentID')['opponentName']),
                                                                left_on = 'opponentID', right_index = True)

tourneyMatchups.to_csv('{}_{}_best_model_results_all_matchups_{}.csv'.format(season, df, 
                           datetime.strftime(datetime.now(), '%Y_%m_%d')), index = False) 



# ============================================================================
# ================= END TOURNAMENT PRECITIONS ALL POSSIBILITIES===================
# ============================================================================


# ============================================================================
# ===================== DEV ==================================================
# ============================================================================

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
