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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from itertools import product, islice, chain, repeat
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline



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


def buildModelData(df):
    '''Randomnly split games data frames in half and swap columns for creating
    model datasets.
        
    Return dataframe of same shape with plus winnerA boolean column.'''

    #filter only numeric columns
    colNames = df.columns.tolist()
    #colNames = filter(lambda c: df[c].dtype.hasobject == False, colNames)
    
    wCols = filter(lambda col: (len(re.findall('^W.*', col))>0) & (col != 'WLoc'), colNames)
    lCols = filter(lambda col: len(re.findall('^L.*', col))>0, colNames)
    baseCols = filter(lambda col: (len(re.findall('^[^L^W].*', col))>0) | (col == 'WLoc'), colNames)
    deltaCols = filter(lambda col: (len(re.findall('.*Delta.*', col)) > 0) | (col == 'scoreGap'), colNames)

    # Rename columns
    aCols = map(lambda col: 'A' + col[1:], wCols)    
    bCols = map(lambda col: 'B' + col[1:], wCols)    
    
    # Split data
    a, b = train_test_split(df, test_size = 0.5, random_state = 1127)
    
    # Reorder dataframes (flip winners and losers for b)
    a = a[baseCols + wCols + lCols]
    b = b[baseCols + lCols + wCols]
    
    # Assign classifaction for if Team A wins    
    a['winnerA'] = 1
    b['winnerA'] = 0

    # Inverse deltas for b since order is reversed 
    b[deltaCols] = b[deltaCols].applymap(lambda x: x * (-1.0))
    
    # Rename columns and stack dataframes
    a.columns = baseCols + aCols + bCols + ['winnerA']
    b.columns = baseCols + aCols + bCols + ['winnerA']
    
    mdlData = pd.concat([a,b], axis = 0)

    return mdlData


def buildModelDataFold(df):
    '''Randomnly split games data frames in half and swap columns for creating
    model datasets.
        
    Return dataframe of same shape with plus winnerA boolean column.'''

    colNames = dataDict[df].columns.tolist()
    wCols = filter(lambda col: (len(re.findall('^W.*', col))>0) & (col != 'WLoc'), colNames)
    lCols = filter(lambda col: len(re.findall('^L.*', col))>0, colNames)
    baseCols = filter(lambda col: (len(re.findall('^[^L^W].*', col))>0) | (col == 'WLoc'), colNames)
    deltaCols = filter(lambda col: (len(re.findall('.*Delta.*', c)) > 0) | (col == 'scoreGap'), colNames)
    
    aCols = map(lambda col: 'A' + col[1:], wCols)    
    bCols = map(lambda col: 'B' + col[1:], wCols)    
    
    #a, b = train_test_split(dataDict[df], test_size = 0.5, random_state = 1127)
    
    # Reorder dataframes (flip winners and losers for b)
    a = dataDict[df][baseCols + wCols + lCols]
    b = dataDict[df][baseCols + lCols + wCols]
    
    # Assign classifaction for if Team A wins    
    a['winnerA'] = 1
    b['winnerA'] = 0

    # Inverse deltas for b since order is reversed
   
    b[deltaCols] = b[deltaCols].applymap(lambda x: x * (-1.0))
    
    # Rename columns and stack dataframes
    a.columns = baseCols + aCols + bCols + ['winnerA']
    b.columns = baseCols + aCols + bCols + ['winnerA']
    
    mdlData = pd.concat([a,b], axis = 0)
    
    mdlData.index = range(len(mdlData))

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



def generateGameMatchupStats(gameDF, teamDF, 
                             teamID1, teamID2, 
                             label1 = 'A', label2 = 'B',
                             extraMergeFields = ['Season'],
                             deltaFields = ['seedRank', 'OrdinalRank'],
                             createMatchupFields = True,
                             matchupFields = [('confMatchup', 'ConfAbbrev'), 
                                              ('seedRankMatchup', 'seedRank')]):
    
    '''Create a new dataframe with team matchup statistics based on two
        teamIDs. 
        
        TeamDF should have index set to mergeFields.'''
    
    for x in [(teamID1, label1), (teamID2, label2)]:
        teamID, teamLabel = x[0], x[1]        
        mergeFields = extraMergeFields + [teamID]
        gameDFcols = gameDF.columns.tolist()
        
        # Add label to teamDF columns to match TeamID label        
        teamDFcols = map(lambda label: teamLabel + label, teamDF.columns.tolist())
        
        gameDF = gameDF.merge(teamDF, left_on = mergeFields, right_index = True)
        gameDF.columns = gameDFcols + teamDFcols
        
    if createMatchupFields == True:
        
        for field in matchupFields:        
            gameDF[field[0]] = generateMatchupField(gameDF, field[1], label1, label2)
    
    if len(deltaFields)>0:
        for field in deltaFields:
            gameDF[field + 'Delta'] = gameDF[label1 + field] - gameDF[label2 + field]
            
    return gameDF



def modelAnalysis(model, data = [], targetCol = None, indCols = None, testTrainDataList = [], testTrainSplit = 0.2):
    
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
    auc = roc_auc_score(yTest, predictions)
    accuracy = accuracy_score(yTest, predictions)
    
    return predictions, predProbs, auc, accuracy



def modelAnalysisPipeline(modelPipe, data = [], 
                          targetCol = None, 
                          indCols = None, 
                          testTrainDataList = [], 
                          testTrainSplit = 0.2,
                          gridSearch = False,
                          paramGrid = None):


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
    
    if gridSearch == True:
        modelPipe = GridSearchCV(modelPipe, paramGrid)
    
    
    modelPipe.fit(xTrain, yTrain)
    predictions = modelPipe.predict(xTest)
    predProbs = np.max(modelPipe.predict_proba(xTest), axis = 1)
    auc = roc_auc_score(yTest, predictions)
    accuracy = accuracy_score(yTest, predictions)
    
    return modelPipe, predictions, predProbs, auc, accuracy



def generateTeamLookupDict(teamDF, yrFilter=True, yr=2018):
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


#==============================================================================
# END FUNCTIONS
#==============================================================================

#location = 'dsw'
location = 'home'

if location.strip().lower() == 'home':
  # Home setup
  dataFolder = 'C:\\Users\\brett\\Documents\\march_madness_ml\\datasets\\2018\\'
  os.chdir('C:\\Users\\brett\\Documents\\march_madness_ml\\')
  
else:
  # DSW setup
  dataFolder = '/home/cdsw/data/'
  %matplotlib inline

        

#==============================================================================
# LOAD DATA 
#==============================================================================

# Read data
dataFiles = os.listdir(dataFolder)
dataFiles.sort()

# Remove zip files
dataFiles = filter(lambda f: '.csv' in f, dataFiles)



#==============================================================================
# rsGamesD = pd.read_csv(dataFolder + 'RegularSeasonDetailedResults.csv')
# tGamesD = pd.read_csv(dataFolder + 'NCAATourneyDetailedResults.csv')
# rsGamesD = pd.read_csv(dataFolder + 'RegularSeasonDetailedResults.csv')
# tGamesD = pd.read_csv(dataFolder + 'NCAATourneyDetailedResults.csv')
# conferences = pd.read_csv(dataFolder + 'Conferences.csv')
# teamConf = pd.read_csv(dataFolder + 'TeamConferences.csv')
# seasons =  pd.read_csv(dataFolder + 'Seasons.csv')
# teams = pd.read_csv(dataFolder + 'Teams.csv') 
#==============================================================================

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
             'teams',
             'teamSpellings'
             ]

dataDict = {k[0]: pd.read_csv(dataFolder + k[1]) for k in zip(keyNames, dataFiles)}


#==============================================================================
# INITIAL EDA AND ORGANIZATION OF GAMES DATA
#==============================================================================

# Lists of games data (C =C ompact, D = Detailed)
gamesData = ['tGamesC', 'tGamesD', 'rGamesC', 'rGamesD']
gamesDataC = ['tGamesC', 'rGamesC']
gamesDataD = ['tGamesD', 'rGamesD']



#for df in ['rGamesC', 'rGamesD']:
#    dataDict[df].info()
#    dataDict[df].describe()



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

# Create list of unique columns in all games DataFrames
#colSummary = [generateDataFrameColumnSummaries(dataDict[df]) for df in gamesData]

#colSummary = pd.DataFrame(list(set(list(chain(*colSummary)))),
#                          columns = ['colName', 'colDataType', 'isObject'])

#colSummary = colSummary.sort_values(by = 'colName')


# Generate dict
colSumDict = {}  
  
# Label column types
colsBase = ['Season', 'DayNum', 'WLoc', 'NumOT', 'scoreGap']   

colsWinFilter = lambda c: c.startswith('W') & (c != 'WLoc')
colsLossFilter = lambda c: c.startswith('L')
       
    
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



#==============================================================================
# CALCULATE TEAM SUMMARIES FOR REGULAR SEASON & TOURNAMENT
#==============================================================================
for df in gamesData:

    colsWinTemp = filter(colsWinFilter,
                  dataDict[df].columns.tolist())
    colsLossTemp = filter(colsLossFilter, 
                   dataDict[df].columns.tolist())

    # Split wins and loss data (specific for each dataframe)
    #colsWinTemp = filter(lambda c: c in colsWin, dataDict[df].columns.tolist())
    #colsLossTemp = filter(lambda c: c in colsLoss, dataDict[df].columns.tolist())
    
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
    if 'DayNum' in lossDF.columns.tolist():
        lossDF['DayNum'] = 0
    
    # Flip scoreGap for losing Team
    lossDF['scoreGap'] = lossDF['scoreGap'] * -1
    
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
    
    dataDict[df + 'TeamSeasonStats'] = aggDF.groupby(['Season', 'TeamID']).mean()
    dataDict[df + 'TeamSeasonStats']['games'] = aggDF.groupby(['Season', 'TeamID'])['win'].count()
    
del(winDF, lossDF, aggDF, colsLossTemp, colsWinTemp, colsBaseTemp)
    


#==============================================================================
# CALCULATE SEED RANKS
#==============================================================================

# Tourney Seed Rank
dataDict['tSeeds']['seedRank'] = map(lambda s: float(re.findall('[0-9]+', s)[0]), 
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
#==============================================================================

# Align Complete and Detailed Datasets for compact and detailed datasets
gamesDataT = filter(lambda g: g.startswith('t'), gamesData)
gamesDataR = filter(lambda g: g.startswith('r'), gamesData)
gamesDataT.sort()
gamesDataR.sort()


for df, dfM in zip(gamesDataT, map(lambda n: n + 'TeamSeasonStats', gamesDataR)):

    # Base dataframe for merging
    dataDict[df + 'SeasonStatsMatchup'] = dataDict[df][colsBase + ['WTeamID', 'LTeamID']]
    
    for t in ('W', 'L'):
        # Rename columns before merging
        renameDict = dict(zip(dataDict[dfM].columns.tolist(),
                          map(lambda n: t + n, dataDict[dfM].columns.tolist())))
            
        
        # Merge DataFrames
        dataDict[df + 'SeasonStatsMatchup'] = dataDict[df + 'SeasonStatsMatchup'].merge(dataDict[dfM].rename(columns = renameDict),
                                                                                        how = 'left',
                                                                                        left_on = ['Season', t + 'TeamID'], 
                                                                                        right_index = True)

 
        # ADD OVERALL SEED AVERAGE METRICS TO TOURNAMENT DATA                                                                                       
        if df.startswith('t'):

            # Relabel Seed data
            seedNameDict = dict(zip(dataDict[df + 'SeedStats'].columns.tolist(),
                                    map(lambda n: t + 'Seed' + n, 
                                        dataDict[df + 'SeedStats'].columns.tolist())))
            
            dataDict[df + 'SeasonStatsMatchup'] = (
                dataDict[df + 'SeasonStatsMatchup'].merge(dataDict[df + 'SeedStats'].rename(columns = seedNameDict),
                                                          left_on = t + 'seedRank', 
                                                          right_index = True)
                                                )

     # Calculate matchup pairs
    for matchup in [('confMatchup', 'ConfAbbrev'), ('seedRankMatchup', 'seedRank')]:
        dataDict[df + 'SeasonStatsMatchup'][matchup[0]] = generateMatchupField(df = dataDict[df + 'SeasonStatsMatchup'], 
                                                                               matchupName = matchup[1], 
                                                                                label1 = 'W', 
                                                                                label2 = 'L')

    # Cacluate column stats
    colSumDict[df + 'SeasonStatsMatchup'] = generateDataFrameColumnSummaries(dataDict[df + 'SeasonStatsMatchup'], returnDF=True)


    # Generate new dataframe with deltas between winning and losing teams
    numericCols =  colSumDict[df + 'SeasonStatsMatchup'][~colSumDict[df + 'SeasonStatsMatchup']['isObject']]['colName'].values.tolist()  
    colsWinTemp = filter(lambda c: colsWinFilter(c) & (c != 'WTeamID'),
                  numericCols)
    colsLossTemp = filter(lambda c: colsLossFilter(c) & (c != 'LTeamID'), 
                   numericCols)

    # Base columns (all other columns)
    colsBaseTemp = filter(lambda c: c not in colsWinTemp + colsLossTemp, dataDict[df + 'SeasonStatsMatchup'].columns.tolist())    
    
    
    # Merge delta calculations with base dataframe
    dataDict[df + 'SeasonStatsMatchupDeltas'] = pd.concat(
                                                    [dataDict[df + 'SeasonStatsMatchup'][colsBaseTemp],
                                                     pd.DataFrame(zip(*[(dataDict[df + 'SeasonStatsMatchup'][colWin] 
                                                                         - dataDict[df + 'SeasonStatsMatchup'][colLoss]) 
                                                                         for colWin, colLoss in zip(colsWinTemp, colsLossTemp)]),
                                           columns = map(lambda colName: colName[1:] + 'Delta',
                                                         colsWinTemp))],
                                                         axis = 1)

    colSumDict[df + 'SeasonStatsMatchupDeltas'] = generateDataFrameColumnSummaries(dataDict[df + 'SeasonStatsMatchupDeltas'], returnDF=True)


del(renameDict, seedNameDict, colsLossTemp, colsWinTemp, numericCols)

# Add matchup columns to colsBase
colsBase += ['confMatchup', 'seedRankMatchup']



#==============================================================================
# CREATE SUBSET OF REGULAR SEASON TEAM STATISTICS FOR TOURNAMENT TEAMS ONLY
#==============================================================================
for df in ['rGamesCTeamSeasonStats', 'rGamesDTeamSeasonStats']:
    dataDict[df + 'Tourney'] = dataDict[df][dataDict[df]['seedRank'] <= 16]
    colSumDict[df + 'Tourney'] = generateDataFrameColumnSummaries(dataDict[df + 'Tourney'], returnDF=True)


#==============================================================================
# CORRELATION ANALYSIS
#==============================================================================

# Columns to exculde in correlation anaylsis:
#   All base columns excluding scoreGap
#   TeamIDs
#   All object columns 

corrExcludeCols = filter(lambda c: c != 'scoreGap', colsBase + ['WTeamID', 'LTeamID'])


for df in map(lambda n: n + 'SeasonStatsMatchupDeltas', gamesDataT):
    corrColsTemp = filter(lambda c: c not in corrExcludeCols, dataDict[df].columns.tolist())    
    dataDict[df + 'Corr'] = dataDict[df][corrColsTemp].corr()
    
    plotCorrHeatMap(dataDict[df + 'Corr'], 
                    plotTitle= df + ' Correlation Analysis')





#==============================================================================
# PRINCIPLE COMPONENT ANALSYSIS OF TEAM STATS DATA
#==============================================================================
# Perform PCA analysis on each Team Stats dataframe
#   Steps (use pipeline):
#       Scale Data
#       Calculate PCA for same number of dimensions in dataset to
#       develop contribution of each axis


teamStatsDFs = filter(lambda dfName: len(re.findall('.*TeamSeasonStats.*', dfName))>0, 
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
    
    # Plot feature weights for each component
    sns.heatmap(pcaDict[df].named_steps['pca'].components_, 
                square = False, 
                cmap = 'coolwarm', 
                ax=axs[0], 
                #annot=True,
                xticklabels = axisLabelFreq,
                yticklabels = axisLabelFreq)
    
    # Add feature lables & format plot    
    axs[0].set_xticklabels(dataDict[df][pcaCols].columns.tolist(), 
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
    
    # Plot feature weights for each component
    sns.heatmap(pcaDict[df].named_steps['pca'].components_, 
                square = False, 
                cmap = 'coolwarm', 
                ax=axs[0], 
                #annot=True,
                xticklabels = axisLabelFreq,
                yticklabels = axisLabelFreq)
    
    # Add feature lables & format plot    
    axs[0].set_xticklabels(dataDict[df][pcaCols].columns.tolist(), 
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





#==============================================================================
# BUILD MODELING DATASETS
#==============================================================================
# Randomly split data and assign a win/loss column
# Invert delta calculations for 50% that was flipped (loss observations)

for df in gamesStatsDFs:
    dataDict[df + 'Mdl'] = buildModelData(dataDict[df])

#==============================================================================
# MODEL DEVELOPMENT & GRID SEARCH
#==============================================================================



# Model List
mdlList = [ DecisionTreeClassifier(random_state = 1127) 
            RandomForestClassifier(random_state = 1127),
            LogisticRegression(random_state = 1127),
            KNeighborsClassifier(),
            SVC(random_state = 1127, probability = True)]


paramGrid = [{'fReduce__n_components' : range(3, len(dataDict[df + 'Mdl'][indCols].columns),
                                              len(dataDict[df + 'Mdl'][indCols].columns) // 4),
                'mdl' : [DecisionTreeClassifier(random_state = 1127), 
                         RandomForestClassifier(random_state = 1127,
                                                         n_estimators = 100,
                                                         n_jobs = -1,
                                                         verbose = 0)],
                'mdl__min_samples_split' : map(lambda c: c/20, range(1, 5)),
                'mdl__min_samples_leaf' : xrange(2, 10, 4)
                
                },
                
            {'fReduce__n_components' : range(3, len(dataDict[df + 'Mdl'][indCols].columns),
                                              len(dataDict[df + 'Mdl'][indCols].columns) // 4),
                'mdl' : [LogisticRegression(random_state = 1127)],
                'mdl__C' : map(lambda i: 10**i, xrange(-1,3))
                },
                
            {'fReduce__n_components' : range(3, len(dataDict[df + 'Mdl'][indCols].columns),
                                              len(dataDict[df + 'Mdl'][indCols].columns) // 4),
                'mdl' : [SVC(probability = True)],
                'mdl__C' : map(lambda i: 10**i, xrange(-1,3)),
                'mdl__gamma' : map(lambda i: 10**i, xrange(-3,1))
                },
                
            {'fReduce__n_components' : range(3, len(dataDict[df + 'Mdl'][indCols].columns),
                                              len(dataDict[df + 'Mdl'][indCols].columns) // 4),
                'mdl' : [KNeighborsClassifier()],
                'mdl__n_neighbors' : range(3, 10, 2)
                
                }]


pipe = Pipeline([('sScale', StandardScaler()), 
                 ('fReduce', PCA(n_components = 10)),
                 ('mdl', LogisticRegression(random_state = 1127))])


indCols = filter(lambda c: (c not in colsBase + ['ATeamID', 'BTeamID', 'winnerA'])
                            & (dataDict[df + 'Mdl'][c].dtype.hasobject == False), 
                dataDict[df + 'Mdl'].columns.tolist())

timer()
pipe, predictions, predProbs, auc, accuracy = modelAnalysisPipeline(modelPipe = pipe,
                      data = dataDict[df + 'Mdl'],
                      indCols = indCols,
                      targetCol = 'winnerA',
                      testTrainSplit = 0.2,
                      gridSearch=True,
                      paramGrid=paramGrid)
x = timer()


# Plot Results
gridSearchResults = pd.DataFrame(pipe.cv_results_)
gridSearchResults['mdl'] = map(lambda m: str(m).split('(')[0], gridSearchResults['param_mdl'].values.tolist())

fig, ax = plt.subplots(1)

ax = sns.swarmplot(x = 'mdl', y = 'mean_test_score', data = gridSearchResults)
ax.tick_params(labelsize = 20)
plt.legend()
sns.boxplot()

pipe.named_steps['mdl']

x = pipe.cv_results

grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid)

mdlDict = {'dTree' : {'model' : DecisionTreeClassifier(random_state = 1127),
                      'gridParams' : {'min_samples_split' : np.arange(.05, .21, .05),
                                      'min_samples_leaf' : xrange(2, 10, 4)
                                     }
                     },
                                      
           'rForest' : {'model' : RandomForestClassifier(random_state = 1127,
                                                         n_estimators = 100,
                                                         n_jobs = -1,
                                                         verbose = 0),
                        'gridParams' : {'min_samples_split' : np.arange(.05, .21, .05),
                                        'min_samples_leaf' : xrange(2, 10, 4),
                                        }
                        },
                                        
            'logR' : {'model': LogisticRegression(random_state = 1127),
                      'gridParams' : {'C': map(lambda i: 10**i, xrange(-1,3))}
                      },
            'knn' : {'model': KNeighborsClassifier(),
                     'gridParams' : {'n_neighbors' : xrange(3, 9, 2)}
                     },
            'nb' : {'model': GaussianNB(),
                    'gridParams' : {}
                     },
            'svc' : {'model': SVC(probability = True),
                     'gridParams' : {'C': map(lambda i: 10**i, xrange(-1,3)),
                                     'gamma' : map(lambda i: 10**i, xrange(-3,1))}
                     }
            }












help(PCA)

help(LogisticRegression)

#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
# # # # # # 
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================


dir

dir(ax)


help(pca)


pcaDict = {}

plt.figure()
cMap = plt.cm.jet


pcaList = ['allDDataTeamScaled', 'allDDataTTeamScaled', 
           'allCDataTeamScaled', 'allCDataTTeamScaled']

for i, df in enumerate(pcaList):
    
    pcaDict[df] = map(lambda n: pcaVarCheck(n, dataDict[df].drop(['Seed', 'ConfAbbrev'], 
                                                                 axis = 1)), 
                      xrange(2, dataDict[df].drop(['Seed', 'ConfAbbrev'], 
                                                                 axis = 1).shape[1], 1))

    plt.plot(xrange(2, dataDict[df].drop(['Seed', 'ConfAbbrev'], 
                                                                 axis = 1).shape[1], 1), 
             zip(*pcaDict[df])[1], 
             c = cMap(int(((i/(len(pcaList)-1))*256))), 
             marker = 'o', label = df)
    
    plt.xlabel('# of Components', fontsize = 20)
    plt.ylabel('% Vairance Explained', fontsize = 20)
    plt.title('Principal Component Analysis', fontsize = 24)
    
    plt.legend(fontsize = 20)
    plt.grid(True)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)


#==============================================================================
# CORRELATION ANALYSIS
#==============================================================================
  
dataDColumns = dataDict['tGamesD'].columns.tolist()
dataCColumns = dataDict['tGamesC'].columns.tolist()

# Remove undesired columns for plotting
nonPlotCols = ['LTeamID', 'WTeamID', 
               'Season', 'WLoc', 
               'WConfAbbrev', 'LConfAbbrev', 
               'WSeed', 'LSeed',
               'seedRankMatchup', 'confMatchup',
               'WGames', 'LGames']

loserColumns = filter(lambda col: col.startswith('W') == False, dataDColumns)
loserColumns = filter(lambda col: (len(re.findall('^[^W].*', col))>0) & (col not in nonPlotCols), dataDColumns)
winnerColumns = filter(lambda col: (col not in loserColumns) & (col not in nonPlotCols), dataDColumns)


# Correlation Dictionary for detailed datasets
corrDictD = {df : dataDict[df][winnerColumns + loserColumns].corr() for df in gamesDataD} 

# Get unique correlations
for df in gamesDataD:
    corrDictD[df + 'Melt'] = pd.melt(corrDictD[df].reset_index(), 
                                    id_vars = 'index', 
                                    var_name = 'index2', 
                                    value_name = 'corr')
    
    corrDictD[df + 'Melt']['attPair'] = zip(corrDictD[df + 'Melt']['index'].values.tolist(), 
                                            corrDictD[df + 'Melt']['index2'].values.tolist())
    
    corrDictD[df + 'Melt']['attPair'] = corrDictD[df + 'Melt']['attPair'].map(lambda p: list(p))                                        
    corrDictD[df + 'Melt']['attPair'].map(lambda p: p.sort())
    corrDictD[df + 'Melt']['match'] = corrDictD[df + 'Melt'].apply(lambda row: ((row['index'] == row['attPair'][0]) 
                                                                                & (row['index2'] == row['attPair'][1])), 
                                                                    axis = 1)
    #corrDictD[df + 'MeltUnq'] = corrDictD[df + 'Melt'].groupby(['attPair', 'corr'])['index'].count()
    corrDictD[df + 'Melt']['diag'] = corrDictD[df + 'Melt']['attPair'].map(lambda p: p[0] == p[1])
    corrDictD[df + 'MeltUnq'] = corrDictD[df + 'Melt'][((corrDictD[df + 'Melt']['diag'] == False)
                                                            & (corrDictD[df + 'Melt']['match'] == True))][['index', 'index2', 'corr']]
    # Isolate correlations </> 0.5                                                    
    corrDictD[df + 'MeltUnqSig'] = corrDictD[df + 'MeltUnq'][corrDictD[df + 'MeltUnq']['corr'].map(lambda c: np.abs(c) >= 0.5)]
    del(corrDictD[df + 'Melt'])
    
    # Sorted correlation for scoreGap
    corrDictD[df + 'MeltScoreGap'] = corrDictD[df + 'MeltUnq'][corrDictD[df + 'MeltUnq']['index2'] == 'scoreGap']
    corrDictD[df + 'MeltScoreGap'].sort_values(by = ['corr'], inplace = True)
    
    # Plot histogram of correlations
    sns.distplot(corrDictD[df + 'MeltUnq']['corr'], bins = 20, kde=True)


# Plot data for highest correlations (+ & -)
topN = 3
for df in gamesDataD:
    jPlotList = (corrDictD[df + 'MeltScoreGap'][['index', 'index2']].head(topN).values.tolist()
                    + corrDictD[df + 'MeltScoreGap'][['index', 'index2']].tail(topN).values.tolist())
    
    if df == 'tGamesD':
        for pair in jPlotList:
            sns.jointplot(x = pair[0], y = pair[1], data = dataDict[df], kind = 'reg')  
    



#==============================================================================
# CALCULATE TEAM SEASON STATISTICS
#==============================================================================

# Aggregate metrics for team statisitcs

winnerDCols = [  'Season',
                 'DayNum',
                 'WTeamID',
                 'WScore',
                 'WFGM',
                 'WFGA',
                 'WFGM3',
                 'WFGA3',
                 'WFTM',
                 'WFTA',
                 'WOR',
                 'WDR',
                 'WAst',
                 'WTO',
                 'WStl',
                 'WBlk',
                 'WPF',
                 'WseedRank',
                 'WOrdinalRank',
                 'WGames',
                 'scoreGap',
                 'seedDelta',
                 'ordinalRankDelta',
                 'WFGpct',
                 'WFGpct3',
                 'WFTpct',
                 'WScorepct',
                 'WORpct',
                 'WDRpct',
                 'WRpct']
                 
loserDCols = [  'Season',
                 'DayNum',
                 'LTeamID',
                 'LScore',
                 'LFGM',
                 'LFGA',
                 'LFGM3',
                 'LFGA3',
                 'LFTM',
                 'LFTA',
                 'LOR',
                 'LDR',
                 'LAst',
                 'LTO',
                 'LStl',
                 'LBlk',
                 'LPF',
                 'LseedRank',
                 'LOrdinalRank',
                 'LGames',
                 'scoreGap',
                 'seedDelta',
                 'ordinalRankDelta',
                 'LFGpct',
                 'LFGpct3',
                 'LFTpct',
                 'LScorepct',
                 'LORpct',
                 'LDRpct',
                 'LRpct']

winnerCCols = [  'Season',
                 'DayNum',
                 'WTeamID',
                 'WScore',
                 'WseedRank',
                 'WOrdinalRank',
                 'WGames',
                 'scoreGap',
                 'seedDelta',
                 'ordinalRankDelta'
                 ]
                 
loserCCols = [   'Season',
                 'DayNum',
                 'LTeamID',
                 'LScore',
                 'LseedRank',
                 'LOrdinalRank',
                 'LGames',
                 'scoreGap',
                 'seedDelta',
                 'ordinalRankDelta',
                 ]             


dataDict['winnerDData'] = dataDict['rGamesD'][winnerDCols]
dataDict['loserDData'] = dataDict['rGamesD'][loserDCols]
dataDict['winnerCData'] = dataDict['rGamesC'][winnerCCols]
dataDict['loserCData'] = dataDict['rGamesC'][loserCCols]

# Reverse deltas for losing data
dataDict['loserDData'][['scoreGap', 'seedDelta', 'ordinalRankDelta']] = dataDict['loserDData'][['scoreGap', 'seedDelta', 'ordinalRankDelta']].applymap(lambda x: x * -1)
dataDict['loserCData'][['scoreGap', 'seedDelta', 'ordinalRankDelta']] = dataDict['loserCData'][['scoreGap', 'seedDelta', 'ordinalRankDelta']].applymap(lambda x: x * -1)


# Relabel columns for aggregating wins and losses data by team and season
for df in ['winnerDData', 'loserDData']:
    
    dataDict[df].columns = [ 'Season',
                             'DayNum',
                             'TeamID',
                             'Score',
                             'FGM',
                             'FGA',
                             'FGM3',
                             'FGA3',
                             'FTM',
                             'FTA',
                             'OR',
                             'DR',
                             'Ast',
                             'TO',
                             'Stl',
                             'Blk',
                             'PF',
                             'seedRank',
                             'OrdinalRank',
                             'Games',
                             'scoreGap',
                             'seedDelta',
                             'ordinalRankDelta',
                             'FGpct',
                             'FGpct3',
                             'FTpct',
                             'Scorepct',
                             'ORpct',
                             'DRpct',
                             'Rpct']


for df in ['winnerCData', 'loserCData']:
    
    dataDict[df].columns = [ 'Season',
                             'DayNum',
                             'TeamID',
                             'Score',
                             'seedRank',
                             'OrdinalRank',
                             'Games',
                             'scoreGap',
                             'seedDelta',
                             'ordinalRankDelta',
                             ]


# Combine wins and losses and aggregate to team level and then scale metrics

# Scaler Dictionary
scaleDict = {'scaleD': StandardScaler(), 
             'scaleC' : StandardScaler(),
             'scaleDTourney' : StandardScaler(),
             'scaleCTourney' : StandardScaler()
             }


# Concat winning and losing data for team average statistics for the season
# for both Detailed and compact datasets
for x in ['D', 'C']:
    dataDict['all' + x + 'Data'] = pd.concat([dataDict['winner' + x + 'Data'], 
                                             dataDict['loser' + x + 'Data']], 
                                            axis = 0)
    
    # Calculate mean statistics for each team & season                                        
    dataDict['all' + x + 'DataTeam'] = dataDict['all' + x + 'Data'].groupby(['Season', 'TeamID']).mean()
    
    # Convert games to winning percentage    
    dataDict['all' + x + 'DataTeam']['Games'] = dataDict['all' + x + 'DataTeam']['Games'].map(lambda x: 0.5*x + 0.5)
    dataDict['all' + x + 'DataTeam'].rename(columns = {'Games':'winPct'}, inplace = True)

    # Count # of games & merge with mean datagrame
    counts = dataDict['all' + x + 'Data'].groupby(['Season', 'TeamID'])['DayNum'].count()
    counts.rename('gameCount', inplace = True)
    dataDict['all' + x + 'DataTeam'] = dataDict['all' + x + 'DataTeam'].merge(pd.DataFrame(counts), 
                                                                              left_index = True, 
                                                                              right_index = True)
    

        
    
    # Isoloate tournament teams
    dataDict['all' + x + 'DataTTeam'] = dataDict['all' + x + 'DataTeam'][dataDict['all' + x + 'DataTeam']['seedRank'] <= 16]

    # Scale data based on all teams
    dataDict['all' + x + 'DataTeamScaled'] = pd.DataFrame(scaleDict['scale' + x].fit_transform(dataDict['all' + x + 'DataTeam']),
                                                          index = dataDict['all' + x + 'DataTeam'].index,
                                                          columns = dataDict['all' + x + 'DataTeam'].columns)                
      
    # Scale data based on tournament teams only  
    dataDict['all' + x + 'DataTTeamScaled'] = pd.DataFrame(scaleDict['scale' + x].fit_transform(dataDict['all' + x + 'DataTTeam']),
                                                          index = dataDict['all' + x + 'DataTTeam'].index,
                                                          columns = dataDict['all' + x + 'DataTTeam'].columns) 
    
    # Merge conferences & seed data
    for df in ['all' + x + 'DataTeam', 
               'all' + x + 'DataTTeam',
               'all' + x + 'DataTeamScaled',
               'all' + x + 'DataTTeamScaled']:
                   
            
        dataDict[df] = (dataDict[df].merge(pd.DataFrame(
                                                dataDict['tSeeds'].set_index(['Season', 'TeamID'])['Seed']),
                                                  how = 'left', 
                                                  left_index = True,
                                                  right_index = True)
                                            )                                                                    
        
        dataDict[df] = (dataDict[df].merge(dataDict['teamConferences'].set_index(['Season', 'TeamID']), 
                                            how = 'left', 
                                            left_index = True,
                                            right_index = True)
                                              )
    
    
    # Delete building datafarmes
    del(dataDict['winner' + x + 'Data'], dataDict['loser' + x + 'Data'])                                                                          



#==============================================================================
# PCA ON TEAM STATISTICS
#==============================================================================



pcaDict = {}

plt.figure()
cMap = plt.cm.jet


pcaList = ['allDDataTeamScaled', 'allDDataTTeamScaled', 
           'allCDataTeamScaled', 'allCDataTTeamScaled']

for i, df in enumerate(pcaList):
    
    pcaDict[df] = map(lambda n: pcaVarCheck(n, dataDict[df].drop(['Seed', 'ConfAbbrev'], 
                                                                 axis = 1)), 
                      xrange(2, dataDict[df].drop(['Seed', 'ConfAbbrev'], 
                                                                 axis = 1).shape[1], 1))

    plt.plot(xrange(2, dataDict[df].drop(['Seed', 'ConfAbbrev'], 
                                                                 axis = 1).shape[1], 1), 
             zip(*pcaDict[df])[1], 
             c = cMap(int(((i/(len(pcaList)-1))*256))), 
             marker = 'o', label = df)
    
    plt.xlabel('# of Components', fontsize = 20)
    plt.ylabel('% Vairance Explained', fontsize = 20)
    plt.title('Principal Component Analysis', fontsize = 24)
    
    plt.legend(fontsize = 20)
    plt.grid(True)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)




# Create new matchups with team stats (one with all data and one with just tournament data)
baseDFcols = ['Season', 'DayNum', 'WTeamID', 'LTeamID', 'scoreGap']
for df in ['tGames', 'rGames']:
    for label in ['C', 'D']:
        for datatype in ['DataTeam', 'DataTeamScaled']:
             dataDict[df + label + datatype] = generateGameMatchupStats(gameDF = dataDict[df + label][baseDFcols],
                                                                        teamDF = dataDict['all' + label + datatype],
                                                                        teamID1 = 'WTeamID',
                                                                        teamID2 = 'LTeamID',
                                                                        label1 = 'W',
                                                                        label2 = 'L',
                                                                        extraMergeFields = ['Season'],
                                                                        createMatchupFields = True,
                                                                        matchupFields = [('confMatchup', 'ConfAbbrev'), 
                                                                                         ('seedRankMatchup', 'seedRank')])
                                                                                         
             if (df == 'tGames') & (datatype == 'DataTeamScaled'):
                dataDict[df + label + 'DataTTeamScaled'] = generateGameMatchupStats(gameDF = dataDict[df + label][baseDFcols],
                                                                        teamDF = dataDict['all' + label + 'DataTTeamScaled'],
                                                                        teamID1 = 'WTeamID',
                                                                        teamID2 = 'LTeamID',
                                                                        label1 = 'W',
                                                                        label2 = 'L',
                                                                        extraMergeFields = ['Season'],
                                                                        createMatchupFields = True,
                                                                        matchupFields = [('confMatchup', 'ConfAbbrev'), 
                                                                                         ('seedRankMatchup', 'seedRank')])
                                      

# Create new matchups with team stats (one with all data and one with just tournament data)
#==============================================================================
# for df in ['tGames', 'rGames']:
#     for label in ['C', 'D']:
#         for datatype in ['DataTeam', 'DataTeamScaled']:
# 
#             
# 
#             # Filter selected base columns
#             joinDF = dataDict[df + label][['Season', 'DayNum', 
#                                            'WTeamID', 'LTeamID',
#                                            'scoreGap',
#                                            'WConfAbbrev', 'LConfAbbrev',
#                                            'confMatchup', 'seedRankMatchup', 
#                                            'seedDelta', 'ordinalRankDelta']]
#             
#             joinDFT = joinDF.copy()
# 
#             for t in ['W', 'L']:     
#     
#                 joinDFCols = joinDF.columns.tolist()          
#                 
#                 # Combine base columns with team stats
#                 joinDF = joinDF.merge(dataDict['all' + label + datatype], 
#                                       left_on = ['Season', t + 'TeamID'],
#                                       right_index = True)
#                                   
#                 colNames = dataDict['all' + label + datatype].columns.tolist()
#                 colNames = map(lambda name: t + name, colNames)
#             
#                 joinDF.columns = joinDFCols + colNames
#         
#                 # Create special dataframe of tournament data with data
#                 #   scaled with tournament teams only
#                 if (df == 'tGames') & (datatype == 'DataTeamScaled'):
#                     joinDFTCols = joinDFT.columns.tolist()                     
#                     joinDFT = joinDFT.merge(dataDict['all' + label + 'DataTTeamScaled'], 
#                                             left_on = ['Season', t + 'TeamID'],
#                                             right_index = True)
#                                             
#                     joinDFT.columns = joinDFTCols + colNames           
#                     
#                     
#         
#             dataDict[df + label + datatype] = joinDF
#             if (df == 'tGames') & (datatype == 'DataTeamScaled'):
#                 dataDict[df + label + 'DataTTeamScaled'] = joinDFT
#==============================================================================
    

# Randomnly split games data frames in half and swap columns for creating
# model datasets
matchupDFsList = list(product(['tGames', 'rGames'], 
                              ['C', 'D'], 
                              ['DataTeam', 'DataTeamScaled']))
                              
matchupDFsList = map(lambda l: reduce(lambda x,y: x+y, l), matchupDFsList) 
matchupDFsList += ['tGamesCDataTTeamScaled', 'tGamesDDataTTeamScaled']

for df in matchupDFsList:
    dataDict[df + 'Model'] = buildModelData(df)     
    dataDict[df + 'ModelFold'] = buildModelDataFold(df)   
     
# Perform pca on new matchup modeling for dimensionality reduction
pcaExcludeCols = ['ATeamID', 'BTeamID',
                   'AConfAbbrev', 'BConfAbbrev',
                   'ASeed', 'BSeed',
                  'Season', 'DayNum', 
                  'confMatchup', 'seedRankMatchup',
                  'scoreGap', 'winnerA']  

pcaList = filter(lambda name: len(re.findall('.*Scaled.*', name))>0, 
                 map(lambda name2: name2 + 'ModelFold', matchupDFsList))

pcaList = filter(lambda name: name.endswith('ScaledModelFold')==False, 
                 map(lambda name2: name2 + 'ModelFold', matchupDFsList))

'a'.endswith

plt.figure()
cMap = plt.cm.jet

for i, df in enumerate(pcaList):
    pcaCols = filter(lambda col: col not in pcaExcludeCols, dataDict[df].columns.tolist())
    
    pcaDict[df] = map(lambda n: pcaVarCheck(n, dataDict[df][pcaCols]), 
                      xrange(2, len(pcaCols), 2))

    plt.plot(xrange(2, len(pcaCols), 2), 
             zip(*pcaDict[df])[1], 
             c = cMap(int(((i/(len(pcaList)-1))*256))), 
             marker = 'o', 
             label = df)
             
    plt.xlabel('# of Components', fontsize = 20)
    plt.ylabel('% Vairance Explained', fontsize = 20)
    plt.title('Principal Component Analysis of Model Matchups', fontsize = 24)
    plt.legend(fontsize = 20)
    plt.grid(True)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    

## Based on the loop above 99%+ of the total variance can be explained with
## 4 pca dimensions

# transform scaled data to pca dimensions
for df in pcaList:

    scaleDict[df] = StandardScaler()    
    
    pcaCols = filter(lambda col: col not in pcaExcludeCols, 
                     dataDict[df].columns.tolist())    
    
    
    dataTrans = pcaDict[df][1][0].transform(dataDict[df][pcaCols])
    
    dataTransS = scaleDict[df].fit_transform(dataTrans)  
    
    dataStack = pd.concat((dataDict[df][pcaExcludeCols], 
                           pd.DataFrame(dataTrans)), 
                            axis = 1, 
                            ignore_index = True)  
     
    dataStackS = pd.concat((dataDict[df][pcaExcludeCols], 
                           pd.DataFrame(dataTransS)), 
                            axis = 1, 
                            ignore_index = True)                        
                            
    dataStack.columns = pcaExcludeCols + ['pca0', 'pca1', 'pca2', 'pca3']
    dataStackS.columns = pcaExcludeCols + ['pca0', 'pca1', 'pca2', 'pca3']
    dataDict[df + 'PCA'] = dataStack
    dataDict[df + 'PCAS'] = dataStackS
  
 




#x = dataDict[df + 'TeamStats' + label]['LTeamID'].head()



mdlDict = {#'dTree' : {'model' : DecisionTreeClassifier(),
            #          'gridParams' : {'min_samples_split' : xrange(2, 20, 2),
             #                         }
             #         },
                                      
           'rForest' : {'model' : RandomForestClassifier(random_state = 1127,
                                                         n_estimators = 100,
                                                         n_jobs = -1,
                                                         verbose = 0),
                        'gridParams' : {'min_samples_split' : xrange(2, 20, 4),
                                        'min_samples_leaf' : xrange(2, 10, 4),
                                        'n_estimators' : [50, 100]}
                        },
                                        
            'logR' : {'model': LogisticRegression(random_state = 1127),
                      'gridParams' : {'C': map(lambda i: 10**i, xrange(-1,3))}
                      },
            'knn' : {'model': KNeighborsClassifier(),
                     'gridParams' : {'n_neighbors' : xrange(3, 9, 2)}
                     },
            'nb' : {'model': GaussianNB(),
                    'gridParams' : {}
                     },
            'svc' : {'model': SVC(probability = True),
                     'gridParams' : {'C': map(lambda i: 10**i, xrange(-1,3)),
                                     'gamma' : map(lambda i: 10**i, xrange(-3,1))}
                     }
            }





modelDataList = filter(lambda dfName: ((len(re.findall('^t.*TT.*ScaledModel.*', dfName))>0) 
                                        | (len(re.findall('^t.*PCAS', dfName))>0)), dataDict.keys())


modelDataList = filter(lambda dfName: ((len(re.findall('^t.*TT.*ScaledModel.*', dfName))>0) 
                                        ), dataDict.keys())
                                        
trainDataList =  filter(lambda dfName: ((len(re.findall('^r.*Team.*ScaledModel.*', dfName))>0) 
                                        ), dataDict.keys())                                       
modelDataList.sort()
trainDataList.sort()

mdlExcludeCols = ['ATeamID', 'BTeamID', 'AConfAbbrev', 'BConfAbbrev',
                  'Season', 'DayNum', 'confMatchup', 'seedRankMatchup',
                  'scoreGap', 'winnerA', 'ASeed', 'BSeed']

#df = modelDataList[1]
 
#timeList = []
mdlSum = []
timer()



for df in modelDataList:
    mdlCols = filter(lambda col: col not in mdlExcludeCols, dataDict[df].columns.tolist())
    
#==============================================================================
#     xTrain, xTest, yTrain, yTest = train_test_split(dataDict[df][mdlCols], 
#                                                 dataDict[df]['winnerA'],
#                                                 test_size = 0.2,
#                                                 train_size = 0.8)
#==============================================================================
       
    gridSearch = False                              
    for mdl in mdlDict.iterkeys():
        mdlDict[mdl][df]={} 
        
        if gridSearch == True:
            mdlDict[mdl][df]['model'] = GridSearchCV(estimator = mdlDict[mdl]['model'],
                                                     param_grid = mdlDict[mdl]['gridParams'],
                                                     refit = True,
                                                     verbose = 2)
        
        else: mdlDict[mdl][df]['model'] = mdlDict[mdl]['model']
            
        (mdlDict[mdl][df]['predictions'],
         mdlDict[mdl][df]['predictProbas'],
         mdlDict[mdl][df]['auc'],
         mdlDict[mdl][df]['accuracy']) = modelAnalysis(model = mdlDict[mdl][df]['model'],
                                                                    data = dataDict[df],
                                                                    targetCol = 'winnerA',
                                                                    indCols = mdlCols,
                                                                    testTrainSplit = 0.2) 
             
        mdlSum.append((df, mdl, mdlDict[mdl][df]['auc'], mdlDict[mdl][df]['accuracy'], timer()))
        
        
mdlSum = pd.DataFrame(mdlSum, columns = ['df', 'model', 'auc', 'accuracy', 'calcTime'])       
 
mdlSum = mdlSum.sort_values(by = 'accuracy', ascending = False) 
 
bestModel = mdlSum[mdlSum['accuracy'] == mdlSum['accuracy'].max()].values.tolist()[0]    

# Get best model for each model type
bestModelType = mdlSum.groupby('model')['accuracy'].max()
bestModelType = mdlSum.merge(pd.DataFrame(bestModelType, 
                          columns = ['maxAccuracy']), 
            left_on = 'model', 
            right_index = True)
bestModelType = bestModelType[bestModelType['accuracy'] == bestModelType['maxAccuracy']]


 
#==============================================================================
# ### 2018 PREDICTIONS ### 
#==============================================================================
 
mdlCols = filter(lambda col: col not in mdlExcludeCols, 
                 dataDict[bestModel[0]].columns.tolist())

model = mdlDict[bestModel[1]][bestModel[0]]['model']

yr = 2018

tSeeds = dataDict['tSeeds'][dataDict['tSeeds']['Season'] == yr][['Seed', 'TeamID']]
tSlots = dataDict['tSlots'][dataDict['tSlots']['Season'] == yr][['Slot', 'StrongSeed', 'WeakSeed']]
seedDict = dict(tSeeds.values.tolist())
resultProbDict = {}


# Get correct dataframe with team stats for modeling
teamDF = 'all' + bestModel[0][6:bestModel[0].find('Model')] 
teamDF = dataDict[teamDF][dataDict[teamDF]['Season'] == yr]
teamDF.drop('Season', inplace=True, axis = 1)




tSlots['rndWinner'], tSlots['winProb'] = 'x', 0


while len(filter(lambda result: result == 'x', tSlots['rndWinner'].values.tolist())) > 0:
    
    for seed in ['Strong', 'Weak']:
        tSlots[seed + 'Team'] = tSlots[seed + 'Seed'].map(lambda t: seedDict.get(t, 'x'))
    
    # Need to error handle for all numeric indexes (last round)    
    slotMatchUps = tSlots[((tSlots['StrongTeam'].map(str) != 'x')
                             & (tSlots['WeakTeam'].map(str) != 'x')
                             & (tSlots['winProb'] == 0))][['Slot', 'StrongTeam', 'WeakTeam']]
    
    
    slotMatchUps2 = generateGameMatchupStats(gameDF = slotMatchUps, 
                                             teamDF = teamDF, 
                                            teamID1='StrongTeam', 
                                            teamID2 = 'WeakTeam',
                                            label1 = 'A',
                                            label2 = 'B',
                                            extraMergeFields=[],
                                            createMatchupFields=True,
                                            deltaFields=['seedRank', 'OrdinalRank'], 
                                            matchupFields=[('confMatchup', 'ConfAbbrev'), 
                                                           ('seedRankMatchup', 'seedRank')])
    
    slotMatchUps['rndWinner'] = model.predict(slotMatchUps2[mdlCols])
    slotMatchUps['winProb'] = np.max(model.predict_proba(slotMatchUps2[mdlCols]), axis = 1)
    
    # Assign TeamID to roundWinner (1 = StrongSeed/Team, 0 = WeakTeam/Seed)
    slotMatchUps['rndWinner'] = slotMatchUps.apply(lambda game: game['StrongTeam'] if game['rndWinner'] == 1 else game['WeakTeam'], 
                                                    axis = 1)
    
    # Convert results to dictionary and update base dictionaries with new results
    winnerDict = slotMatchUps.set_index('Slot').to_dict()
    seedDict.update(winnerDict['rndWinner'])
    resultProbDict.update(winnerDict['winProb'])
    
    
    #tSlots['rndWinner'] = tSlots.apply(lambda game: winnerDict['rndWinner'].get(game, tSlots['rndWinner']), axis = 1)
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

    tSlots.rename(columns = {'Seed' : team + 'Seed'}, inplace = True)

# Order Results
tSlots = tSlots.sort_values('Slot')
tSlotsClean = tSlots[['Slot', 'StrongTeamSeed', 'WeakTeamSeed', 'rndWinnerSeed', 
                      'StrongTeamName', 'WeakTeamName', 'rndWinnerName', 
                      'winProb']]


mdlCols = filter(lambda col: col not in mdlExcludeCols, 
                 dataDict[bestModel[0]].columns.tolist())

model = mdlDict[bestModel[1]][bestModel[0]]['model']

yr = 2018

# Get correct dataframe with team stats for modeling
teamDFname = 'all' + bestModel[0][6:bestModel[0].find('Model')] 
teamDF = dataDict[teamDF][dataDict[teamDF]['Season'] == yr]
tSeeds = dataDict['tSeeds'][dataDict['tSeeds']['Season'] == yr][['Seed', 'TeamID']]
tSlots = dataDict['tSlots'][dataDict['tSlots']['Season'] == yr][['Slot', 'StrongSeed', 'WeakSeed']]


tPredict, tPredictClean = tourneyPredictions(model = model, 
                          teamDF = dataDict[teamDFname],
                          tSeeds = dataDict['tSeeds'],
                          tSlots = dataDict['tSlots'],
                          mdlCols = mdlCols,
                          yr = 2018)


for mdl, df in bestModelType[['model', 'df']].values.tolist():   
    mdlCols = filter(lambda col: col not in mdlExcludeCols, 
                 dataDict[df].columns.tolist()) 
    
    teamDFname = 'all' + df[6:df.find('Model')]    
    
    mdlDict[mdl]['bestPredictions'], mdlDict[mdl]['bestPredictionsClean'] = tourneyPredictions(model = mdlDict[mdl]['model'], 
                          teamDF = dataDict[teamDFname],
                          tSeeds = dataDict['tSeeds'],
                          tSlots = dataDict['tSlots'],
                          mdlCols = mdlCols,
                          yr = 2018)
    
    fName = '_'.join(['2018_model_results',
                      mdl,
                      df, 
                      datetime.strftime(datetime.now(), '%Y_%m_%d')])

    mdlDict[mdl]['bestPredictionsClean'].to_csv(fName + '.csv', index = False, header = True)    
    


tOldResults = generateOldTourneyResults(tSeeds = dataDict['tSeeds'],
                                        tSlots = dataDict['tSlots'],
                                        tGames = dataDict['tGamesC'],
                                        yr = 2017)

tSeeds, tSlots, tGames, yr = dataDict['tSeeds'], dataDict['tSlots'], dataDict['tGamesC'], 2017




def tourneyPredictions(model, teamDF, tSeeds, tSlots, mdlCols, yr = 2018):
    
    if 'Season' in tSeeds.columns.tolist():
        tSeeds = tSeeds[tSeeds['Season'] == yr][['Seed', 'TeamID']]
        #tSeeds = tSeeds.drop('Season', inplace = True, axis = 1)
    else: tSeeds = tSeeds[['Seed', 'TeamID']]
    
    if 'Season' in tSlots.columns.tolist():
        tSlots = tSlots[tSlots['Season'] == yr][['Slot', 'StrongSeed', 'WeakSeed']]
        #tSlots = tSlots.drop('Season', inplace = True, axis = 1)
    else: tSlots = tSlots[['Slot', 'StrongSeed', 'WeakSeed']]    
    
    if 'Season' in teamDF.columns.tolist():
        teamDF = teamDF[teamDF['Season'] == yr]
        teamDF.drop('Season', inplace = True, axis = 1)

    seedDict = dict(tSeeds.values.tolist())
    resultProbDict = {}
    
    tSlots['rndWinner'], tSlots['winProb'] = 'x', 0
    
    
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
        slotMatchUps2 = generateGameMatchupStats(gameDF = slotMatchUps, 
                                                 teamDF = teamDF, 
                                                teamID1='StrongTeam', 
                                                teamID2 = 'WeakTeam',
                                                label1 = 'A',
                                                label2 = 'B',
                                                extraMergeFields=[],
                                                createMatchupFields=True,
                                                deltaFields=['seedRank', 'OrdinalRank'], 
                                                matchupFields=[('confMatchup', 'ConfAbbrev'), 
                                                               ('seedRankMatchup', 'seedRank')])
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
    
        tSlots.rename(columns = {'Seed' : team + 'Seed'}, inplace = True)
    
    # Order & clean results 
    tSlots = tSlots.sort_values('Slot')
    tSlotsClean = tSlots[['Slot', 'StrongTeamSeed', 'WeakTeamSeed', 'rndWinnerSeed', 
                          'StrongTeamName', 'WeakTeamName', 'rndWinnerName', 
                          'winProb']]

    return tSlots, tSlotsClean





fName = '_'.join(['2018_model_results',
                  bestModel[0],
                  bestModel[1], 
                  datetime.strftime(datetime.now(), '%Y_%m_%d')])

tPredictClean.to_csv(fName + '.csv', index = False, header = True)




#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
# # # # # # DEV
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================


#==============================================================================
# for dfs in zip(trainDataList, modelDataList):
# 
#     dfTr, df = dfs[0], dfs[1]    
#     
#     mdlCols = filter(lambda col: col not in mdlExcludeCols, dataDict[df].columns.tolist())
# 
# 
# 
#     xTrain, yTrain = dataDict[dfTr][mdlCols], dataDict[dfTr]['winnerA']
#     xTest, yTest = dataDict[df][mdlCols], dataDict[df]['winnerA']
# 
# 
#     gridSearch = False                              
#     for mdl in mdlDict.iterkeys():
#         mdlDict[mdl][df]={} 
#         
#         if gridSearch == True:
#             mdlDict[mdl][df]['model'] = GridSearchCV(estimator = mdlDict[mdl]['model'],
#                                                      param_grid = mdlDict[mdl]['gridParams'],
#                                                      refit = True,
#                                                      verbose = 2)
#         
#         else: mdlDict[mdl][df]['model'] = mdlDict[mdl]['model']
#             
#         (mdlDict[mdl][df]['predictions'],
#          mdlDict[mdl][df]['predictProbas'],
#          mdlDict[mdl][df]['auc'],
#          mdlDict[mdl][df]['accuracy']) = modelAnalysis(model = mdlDict[mdl][df]['model'],
#                                                        testTrainDataList=[xTrain, xTest, yTrain, yTest]) 
#              
#         mdlSum.append((df, mdl, mdlDict[mdl][df]['auc'], mdlDict[mdl][df]['accuracy'], timer(), 'regSeasonTrain'))
# 
#==============================================================================









#==============================================================================
# 
# 
# sns.heatmap(dataDict['tGamesD'][winnerColumns + loserColumns].corr(),cmap='coolwarm')
# 
# for col in winnerColumns + loserColumns:
#     sns.jointplot(x = col, y = 'scoreGap', data = dataDict['tGamesD'])
# 
# sns.pairplot(dataDict['tGamesD'], 
#              hue = 'Season', 
#              palette='rainbow', 
#              x_vars = loserColumns, 
#              y_vars = winnerColumns)
#              
#==============================================================================
             
dataDict['tGamesD'].head()


'abc'.startswith('L')
               
        