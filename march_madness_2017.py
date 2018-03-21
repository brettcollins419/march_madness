# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:30:37 2017

@author: u00bec7
"""

from scipy import stats
import os
from os.path import join
import time
import numpy as np
import pandas as pd
import string
import sklearn as sk
from sklearn import svm, linear_model, ensemble, neighbors, tree, naive_bayes, metrics
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel




def gamesSplit(data):
    '''Split winner and loser results and concat to calculate team statistics'''
    
    # Collect wins
    winners = data[dColRemaining + dColNamesWin]
    winners.columns = dColRemaining + dColNamesGeneric

    # Collect losses
    losers = data[dColRemaining + dColNamesLoss]
    losers.columns = dColRemaining + dColNamesGeneric
    
    # Fill opponent stats
    for metric in [label for label in dColNamesGeneric if label not in ['team', 'gap', 'loc']]:
        winners[metric + 'Allow'] = data['L' + metric]
        losers[metric + 'Allow'] = data['W' + metric]
    
    # Fill opponent team
    winners['opponent'] = data['Lteam']
    losers['opponent'] = data['Wteam']

    # Assign outcome to each group (win = 1, loss = 0)
    winners['outcome'] = 1
    losers['outcome'] = 0
    
    # Combine winner & loser dataframes
    teamResults = pd.concat([winners, losers], 
                            ignore_index = True, 
                            axis = 0, 
                            copy = True)   
    
    # Calculate game number
    teamResults['gameNumber'] = teamResults.groupby(['Season', 'team']).cumcount() + 1 
    
    return teamResults


def appendSeeds(data, seeds):
    '''Append seed data to datasets''' 
    
    data = data.merge(seeds, 
                      how = 'left',
                      left_on = ['Season', 'team'], 
                      right_index = True)
        
    
    #data[['Seed']].fillna('No', inplace = True)
    #data[['seedNumber']].fillna(999, inplace = True)    
    
    if 'opponent' in data.columns.tolist(): 
    
        data = data.merge(seeds,
                          how = 'left',
                          left_on = ['Season', 'opponent'],
                          right_index = True)
        
        
        data.rename(columns = {'Seed_x':'Seed',
                               'Seed_y': 'opponentSeed',
                               'seedNumber_x':'seedNumber',
                               'seedNumber_y':'oppSeedNumber'},
                    inplace = True)

    
        data.fillna(False, inplace = True)
    return data


def WeightedAvgMetric(data, weight, metric):
    '''Calculate weighted average of a metric using a provided weights.'''
   
   
    wa = data.apply(lambda x: np.dot(x[weight].astype(np.float), 
                                     x[metric].astype(np.float)) / x[weight].sum())
    
    return wa



def teamRecords(gameResults):
    '''Aggregate team results for a given season'''
    
    ### SEASON AVERAGE TEAM STATISTICS ###
    metrics = gameResults.columns.tolist()
    
    summaryMetrics = [x for x in metrics if x not in ['opponent', 
                                                      'loc',
                                                      'Season',
                                                      'team']]  
    
    
    teamSum = gameResults[['Season','team'] + summaryMetrics].groupby(['Season', 
                                                                       'team']) 
    
    # Calculate season average metrics
    records = teamSum.mean()
    
    
    # records['winPctWeighted'] = WeightedAvgMetric(teamSum, 'gameNumber', 'outcome')
    
    for metric in summaryMetrics:
        records[metric + 'Weighted'] = WeightedAvgMetric(teamSum, 'gameNumber', metric)
    
    
    records.fillna(0.0, inplace = True)    
    
    ### WINS & LOSSES ###
    
    # Rename win percentage column
    records.rename(columns = {'outcome':'winPct'}, inplace = True)
    
    # Calculate total games played
    games = pd.DataFrame(teamSum['outcome'].count())
    games.columns = ['totalGames']
    
    # Calculate total wins
    wins = pd.DataFrame(teamSum['outcome'].sum())
    wins.columns = ['wins']
    
    games = games.merge(wins, 
                        how = 'left', 
                        left_index = True, 
                        right_index = True)
                        
    # Count losses                   
    games['losses'] = games['totalGames'] - games['wins']
    
    
    records = records.merge(games, 
                            how = 'left', 
                            left_index = True, 
                            right_index = True)
    
    records.reset_index(inplace = True)    
    
    return records


def addTeamStats(teamStats, games):
    '''Generate dataframe with all games, game stats 
    and team season stats appended.'''
    
    teamStatsCols = teamStats.columns.tolist()
    
    matchup =  games[['Season', 'team', 
                      'opponent', 'outcome', 
                      'gap', 'Daynum']] 
                      
    matchup.rename(columns = {'gap':'gapGame',
                              'Daynum':'DaynumGame'}, 
                              inplace = True)
    
    # Append stats for base team
    gameswTeamStats = matchup.merge(teamStats, 
                                    how = 'left',
                                    left_on = ['Season', 'team'],
                                    right_on = ['Season', 'team'])
   
                                       
    # Append stats for opponent
    gameswTeamStats = gameswTeamStats.merge(teamStats,
                                           how = 'left',
                                           left_on = ['Season', 'opponent'],
                                           right_on = ['Season', 'team'],
                                           suffixes = ('', 'Opp'))
    
    del(gameswTeamStats['teamOpp'])                                                                              
    
    deltaCols = [stat for stat in teamStatsCols if stat not in ['Season',
                                                                'team',
                                                                'totalGames',
                                                                'Seed',
                                                                'Daynum',
                                                                'SeedOpp']]
                                                                
    for stat in deltaCols:
        gameswTeamStats[stat + 'Delta'] = (gameswTeamStats[stat]
                                            - gameswTeamStats[stat + 'Opp'])                                                       
    
    
    
    #gameswTeamStats['seedDelta'] = gameswTeamStats['seedNumber'] - gameswTeamStats['seedNumberOpp']
    
    return gameswTeamStats
  
                                        
def scaleMetrics(resultsDF, indexList, statsList, scaler):
    '''Create scaled dataframe using Standard Scale (metrics normalized
    based on mean and standard deviation - Z-Score)'''

    if scaler == 'standard':    
        sScaler = sk.preprocessing.StandardScaler()
    else:
        sScaler = sk.preprocessing.MinMaxScaler()
    
    
    scaledArray = sScaler.fit_transform(resultsDF[statsList])

    scaledDF  = pd.DataFrame(scaledArray, columns = statsList)

    scaledDF = resultsDF[indexList].merge(scaledDF, 
                                          how = 'left', 
                                          left_index = True, 
                                          right_index = True)


    return scaledArray, scaledDF
 

def featureExtraction(stats, outcome, statsList):

    clf = ExtraTreesClassifier()
    clf = clf.fit(gameResultsScaleArrayT, gameResultsOutcomeT)
    clf.feature_importances_  
    
    
    statsRanked = zip(clf.feature_importances_, statsList)
    statsRanked.sort(reverse = True)
    statsRanked = pd.DataFrame(statsRanked, columns = ['rank', 'metric'])   
    
    model = SelectFromModel(clf, prefit=True)
    statsModeling = model.transform(stats)  
    
    selectedFeatures = model.get_support()
    
    selectedFeatures = np.array(statsList)[selectedFeatures].tolist()
    
    return statsRanked, statsModeling, selectedFeatures


#####################
### END FUNCTIONS ###
#####################

os.chdir('C:\\Users\\brett\\Documents\\march_madness_ml\\')
#os.chdir('C:\\Users\\u00bec7\\Desktop\\personal\\MM_Challenge\\')

loc = 'C:\\Users\\brett\\Documents\\march_madness_ml\\datasets'
#loc = 'C:\\Users\\u00bec7\\Desktop\\personal\\MM_Challenge\\datasets'


# Detailed results dtypes
dColTypes = {'Season' : np.int,
             'Daynum' : np.float64,
             'Wteam' : np.int,
             'Wscore' : np.float64,
             'Lteam' : np.int,
             'Lscore' : np.float64,
             'Wloc' : str,
             'Numot' : np.float64,
             'Wfgm' : np.float64,
             'Wfga' : np.float64,
             'Wfgm3' : np.float64,
             'Wfga3' : np.float64,
             'Wftm' : np.float64,
             'Wfta' : np.float64,
             'Wor' : np.float64,
             'Wdr' : np.float64,
             'Wast' : np.float64,
             'Wto' : np.float64,
             'Wstl' : np.float64,
             'Wblk' : np.float64,
             'Wpf' : np.float64,
             'Lfgm' : np.float64,
             'Lfga' : np.float64,
             'Lfgm3' : np.float64,
             'Lfga3' : np.float64,
             'Lftm' : np.float64,
             'Lfta' : np.float64,
             'Lor' : np.float64,
             'Ldr' : np.float64,
             'Last' : np.float64,
             'Lto' : np.float64,
             'Lstl' : np.float64,
             'Lblk' : np.float64,
             'Lpf' : np.float64}


# Read datasets
slots = pd.read_csv(join(loc, 'TourneySlots.csv'))
seeds = pd.read_csv(join(loc, 'TourneySeeds.csv'))
teams = pd.read_csv(join(loc, 'Teams.csv'))
cResultsT = pd.read_csv(join(loc, 'TourneyCompactResults.csv'))
dResultsT = pd.read_csv(join(loc, 'TourneyDetailedResults.csv'), dtype = dColTypes)
dResultsR = pd.read_csv(join(loc, 'RegularSeasonDetailedResults.csv'), dtype = dColTypes)
cResultsR = pd.read_csv(join(loc, 'RegularSeasonCompactResults.csv'))

# Apply tournament seeds as integers
seeds['seedNumber'] = seeds['Seed'].apply(lambda x: int(x.strip(string.letters)))
seeds.set_index(['Season', 'Team'], inplace = True)

# Calculate ratios
ratios = [('Wfgm', 'Wfga', 'Wfgp'), 
          ('Wfgm3', 'Wfga3', 'Wfgp3'), 
          ('Wftm', 'Wfta', 'Wftp'),
          ('Lfgm', 'Lfga', 'Lfgp'), 
          ('Lfgm3', 'Lfga3', 'Lfgp3'), 
          ('Lftm', 'Lfta', 'Lftp')]
          
for makes, attempts, ratio in ratios:
    dResultsR[ratio] = dResultsR[makes] / dResultsR[attempts]
    dResultsT[ratio] = dResultsT[makes] / dResultsT[attempts]


   

dResultsR.fillna(0.0, inplace = True)
dResultsT.fillna(0.0, inplace = True)


# Calculate score gap for winning and losing team and losing location
for data in [dResultsR, dResultsT]:
    data['Wgap'] = data['Wscore'] - data['Lscore']
    data['Lgap'] = data['Wgap'] * (-1.0)
    data['Lloc'] = data['Wloc'].replace({'H':'A', 'A':'H', 'N':'N'}, inplace = True)



# Get column names from detailed results
dColNames = dResultsR.columns.tolist()
dColNamesWin = [label for label in dColNames if (string.find(label, 'W') > -1)]
dColNamesLoss = [label for label in dColNames if (string.find(label, 'L') > -1)]
dColRemaining = [label for label in dColNames if (string.find(label, 'L') == -1) 
                                                    & (string.find(label, 'W') == -1)]

# Sort column names for merging
dColNamesLoss.sort()
dColNamesWin.sort()

dColNamesGeneric = [x.replace('W', '') for x in dColNamesWin]
  

# Split team results for team stats  
teamResultsR = gamesSplit(dResultsR)
teamResultsT = gamesSplit(dResultsT)


# Generate team record & statistics
teamRecordsR = teamRecords(teamResultsR)
teamRecordsT = teamRecords(teamResultsT)

# Append tournament seeds to tournament data
teamResultsR = appendSeeds(data = teamResultsR, seeds = seeds)
teamResultsT = appendSeeds(data = teamResultsT, seeds = seeds)

teamRecordsR = appendSeeds(data = teamRecordsR, seeds = seeds)
teamRecordsT = appendSeeds(data = teamRecordsT, seeds = seeds)



gameResultsTeamStatsR['seedNumberDelta'].fillna(False)
    
    
# Team match ups (teams stats vs opponent stats & stat deltas)
# Regular season games
gameResultsTeamStatsR = addTeamStats(teamRecordsR, teamResultsR)

#Tournament Games (use regular season stats)
gameResultsTeamStatsT = addTeamStats(teamRecordsR, teamResultsT)




# Normalize statistics
gameStatsIndex = ['Season',
                  'DaynumGame',
                  'team',
                  'opponent',
                  'gapGame',
                  'outcome',
                  'Seed',
                  'seedNumber',
                  'SeedOpp',
                  'seedNumberOpp']

gameStatsExcludeList = ['wins',
                        'losses',
                        'winsOpp',
                        'lossesOpp',
                        'winsDelta',
                        'lossesDelta',
                        'totalGames',
                        'totalGamesOpp']

gameStatsList = [stat for stat in 
                 gameResultsTeamStatsR.columns.tolist() 
                 if stat not in gameStatsIndex + gameStatsExcludeList]



# Create minmax scaled datasets
gameResultsMMArrayT, gameResultsScaleMMT = scaleMetrics(gameResultsTeamStatsT, 
                                                        gameStatsIndex, 
                                                        gameStatsList, 
                                                        'standard')

# Round scaled statistics for grouping
gameResultsScaleMMT[gameStatsList] = (gameResultsScaleMMT[gameStatsList] * 20).round(0) / 20

gameResultsWinProbT = gameResultsScaleMMT.copy()


# Calculate the probably of winning based on every individual metric
probDict = {}

for m in gameStatsList:
    gb =  gameResultsScaleMMT[[m, 'outcome']].groupby(by = [m])  
    
    df = pd.DataFrame(gb.sum() / gb.count())
    df.reset_index(inplace = True)   
    
    probDict[m] = [df[m].tolist(), df['outcome'].tolist()]
    
    gameResultsWinProbT[m].replace(df[m].tolist(), 
                                    df['outcome'].tolist(), 
                                    inplace = True)
    
    
    
   
  

 
# Created normalized datasets
gameResultsScaleArrayR, gameResultsScaleR = scaleMetrics(gameResultsTeamStatsR.drop(['seedNumber', 
                                                                                     'seedNumberOpp', 
                                                                                     'seedNumberDelta'], axis = 1), 
                                                         gameStatsIndex, 
                                                         gameStatsList, 
                                                         'minmax') 
                                                         
gameResultsScaleArrayT, gameResultsScaleT = scaleMetrics(gameResultsTeamStatsT, 
                                                         gameStatsIndex, 
                                                         gameStatsList, 
                                                         'minmax') 

gameResultsOutcomeR = np.array(gameResultsTeamStatsR['outcome']) 
gameResultsOutcomeT = np.array(gameResultsTeamStatsT['outcome'])     

# Perform feature selection prioriziation   
statsRankedT, statsModelT, modelFeaturesT = featureExtraction(gameResultsScaleArrayT, gameResultsOutcomeT, gameStatsList)
statsRankedR, statsModelR, modelFeaturesR = featureExtraction(gameResultsScaleArrayR, gameResultsOutcomeR, gameStatsList)


x = gameResultsTeamStatsR.isnull().sum()


#plotting
plt.figure()
plt.bar(np.arange(len(statsRankedT))[:50], np.array(statsRankedT['rank'][:50]))
plt.xticks(np.arange(len(statsRankedT))[:50], np.array(statsRankedT['metric'][:50]))
plt.show()

plt.figure()
plt.scatter(np.array(statsRankedT['rank']), np.array(statsRankedR['rank']))
plt.show()

statsRankedT['rank'].cumsum().plot()
(statsRankedT['rank'] / statsRankedT['rank'].max()).plot()



gameStatsList = statsRankedT['metric'].tolist()[:16]
   
gameResultsWinProbT['overallProb'] = (gameResultsWinProbT[gameStatsList].applymap(
                                        lambda x: (2.0 / (1+np.exp(-2*(x - 0.5)))) - 1.0).sum(axis = 1) 
                                        / 
                                        gameResultsWinProbT[gameStatsList].applymap(
                                        lambda x: abs((2.0 / (1+np.exp(-2*(x - 0.5)))) - 1.0)).sum(axis = 1))
                                        
                                        
gameResultsWinProbT['overallProb'] = (gameResultsWinProbT[gameStatsList].applymap(
                                        lambda x: (x * ((x-0.5)**8)/0.25)).sum(axis = 1) 
                                        / 
                                        gameResultsWinProbT[gameStatsList].applymap(
                                        lambda x: ((x-0.5)**8)/0.25).sum(axis = 1))                                        


gameResultsWinProbT['predictedOutcome'] = gameResultsWinProbT['overallProb'].apply(lambda x: abs(np.ceil(x)))
gameResultsWinProbT['predictedOutcome'] = gameResultsWinProbT['overallProb'].round(0)

gameResultsWinProbT['predictedOutcome'][gameResultsWinProbT['outcome'] == 1].sum() / gameResultsWinProbT['predictedOutcome'][gameResultsWinProbT['outcome'] == 1].count()



#############################
### MAKE 2017 PREDICTIONS ###
#############################

slots2017 = slots[['Slot', 'Strongseed', 'Weakseed']][slots['Season']==2017]

slots2017['winningSeed'] = 'blank'
slots2017['winner'] = 0
slots2017['strongTeam'] = 0
slots2017['weakTeam'] = 0
slots2017['overallProb'] = 0.0


teamRecords2017 = teamRecordsR[(teamRecordsR['Season']==2017) & (teamRecordsR['seedNumber'] > 0)]
tourneyTeams = teamRecords2017.copy()


statsList = [x for x in tourneyTeams.columns.tolist() if x not in ['totalGames', 'wins', 'losses']]
deltaList = [x for x in statsList if x not in ['Season', 'team', 'Seed']] 
indexList = ['Slot', 'Strongseed', 'Weakseed', 'team', 'teamOpp']

totalGames = len(slots2017)
predictedGames = 0

while predictedGames < totalGames:
    
    roundMatchUp = slots2017[['Slot', 
                              'Strongseed', 
                              'Weakseed']].merge(tourneyTeams[statsList], 
                                                 left_on = ['Strongseed'], 
                                                 right_on = ['Seed'], 
                                                 how = 'inner')
    
    roundMatchUp = roundMatchUp.merge(tourneyTeams[statsList], 
                                      left_on = ['Weakseed'], 
                                      right_on = ['Seed'], 
                                      how = 'inner', 
                                      suffixes = ['', 'Opp'])
    
    slots2017 = slots2017.merge(tourneyTeams[['Seed', 'team']], 
                                left_on = ['Strongseed'], 
                                right_on = ['Seed'], 
                                how = 'left', 
                                suffixes = ['', 'Temp'])
                                
    slots2017['team'].fillna(0, inplace = True)
    slots2017['strongTeam'] = slots2017['strongTeam'] + slots2017['team'] * (slots2017['strongTeam'] == 0)

    del(slots2017['Seed'])
    del(slots2017['team'])
    
    slots2017 = slots2017.merge(tourneyTeams[['Seed', 'team']], left_on = ['Weakseed'], right_on = ['Seed'], how = 'left', suffixes = ['', 'Temp'])
    slots2017['team'].fillna(0, inplace = True)
    slots2017['weakTeam'] = slots2017['weakTeam'] + slots2017['team'] *(slots2017['weakTeam'] == 0)
    del(slots2017['Seed'])
    del(slots2017['team'])
    
    
    for stat in deltaList:
        roundMatchUp[stat + 'Delta'] = (roundMatchUp[stat]
                                            - roundMatchUp[stat + 'Opp'])
    
    roundMatchUpScale = pd.DataFrame(mmScale.transform(roundMatchUp[gameStatsList]), columns = gameStatsList)
    
    roundMatchUpScale = (roundMatchUpScale * 20).round(0) / 20
    roundMatchUpScale = roundMatchUpScale.apply(lambda x: np.clip(x, 0.0, 1.0))
    
    roundMatchUpScale = roundMatchUp[indexList].merge(roundMatchUpScale, left_index = True, right_index = True)
    
    for stat in gameStatsList:
        roundMatchUpScale[stat].replace(probDict[stat][0], probDict[stat][1], inplace = True)
        
    
    roundMatchUpScale['overallProb'] = (roundMatchUpScale[gameStatsList].applymap(
                                            lambda x: (x * ((x-0.5)**8)/0.25)).sum(axis = 1) 
                                            / 
                                            roundMatchUpScale[gameStatsList].applymap(
                                            lambda x: ((x-0.5)**8)/0.25).sum(axis = 1))  
    
    roundMatchUpScale['predictedOutcome'] = roundMatchUpScale['overallProb'].round(0)   
    
    roundMatchUpScale['winningSeed'] = (roundMatchUpScale['predictedOutcome'] == 1) * roundMatchUpScale['Strongseed'] + (roundMatchUpScale['predictedOutcome'] == 0) * roundMatchUpScale['Weakseed']
    roundMatchUpScale['winner'] = (roundMatchUpScale['predictedOutcome'] == 1) * roundMatchUpScale['team'] + (roundMatchUpScale['predictedOutcome'] == 0) * roundMatchUpScale['teamOpp']
    roundMatchUpScale['loser'] = (roundMatchUpScale['predictedOutcome'] == 0) * roundMatchUpScale['team'] + (roundMatchUpScale['predictedOutcome'] == 1) * roundMatchUpScale['teamOpp']
    
    
    nextRound = roundMatchUpScale[['Slot', 'winningSeed', 'winner', 'overallProb']]
    slots2017 = slots2017.merge(nextRound, left_on = 'Slot', right_on = 'Slot', how = 'left', suffixes = ['', 'Temp'])
    
    slots2017['winningSeedTemp'].fillna('blank', inplace = True) 
    slots2017['winnerTemp'].fillna(0.0, inplace = True)
    slots2017['overallProbTemp'].fillna(0.0, inplace = True) 
    
    slots2017['overallProb'] = slots2017['overallProb'] + slots2017['overallProbTemp']
    
    slots2017['winner'] = slots2017['winner'] + slots2017['winnerTemp']
    slots2017['winningSeed'] = slots2017['winningSeed'] * (slots2017['winningSeedTemp'] == 'blank') + slots2017['winningSeedTemp'] * (slots2017['winningSeedTemp'] != 'blank')
    
    del(slots2017['winnerTemp'])  
    del(slots2017['winningSeedTemp']) 
    del(slots2017['overallProbTemp'])
    
    
    tourneyTeams = tourneyTeams[statsList].merge(pd.DataFrame(roundMatchUpScale['loser']), left_on = 'team', right_on = 'loser', how = 'left')
    tourneyTeams = tourneyTeams.merge(pd.DataFrame(roundMatchUpScale[['winner', 'Slot']]), left_on = 'team', right_on = 'winner', how = 'left')
    tourneyTeams = tourneyTeams[tourneyTeams['loser'].isnull()]
    tourneyTeams['Slot'].fillna('None', inplace = True)
    
    tourneyTeams['Seed'] = ((tourneyTeams['Seed'] * (tourneyTeams['Slot'] == 'None')) + (tourneyTeams['Slot'] * (tourneyTeams['Slot'] != 'None')))

    predictedGames += len(roundMatchUp)

                             
slots2017wNames = slots2017.copy()

slots2017wNames.replace(teams['Team_Id'].tolist(), teams['Team_Name'].tolist(), inplace = True)                                       
  
  
slots2017wNames.to_csv(loc + '2017 predictions.csv', index = False)


############################
### END 2017 PREDICTIONS ###
############################




# Create training and test sets using tournament data
train, test, train_target, test_target = train_test_split(statsModelT, gameResultsOutcomeT, test_size = 0.2, train_size = 0.8, random_state = 11)

trainSelect = np.random.randint(0, len(statsModelR), 5000)
train = statsModelR[trainSelect]
train_target = gameResultsOutcomeR[trainSelect]

sScaler = sk.preprocessing.StandardScaler()
sScaler.fit(gameResultsTeamStatsR[modelFeaturesR])

test = sScaler.transform(gameResultsTeamStatsT[modelFeaturesR])
test_target = gameResultsOutcomeT




#######################
###  CREATE MODELS  ###
#######################

calc_time = {}

# knn
knn3 = neighbors.KNeighborsClassifier(3)
knn5 = neighbors.KNeighborsClassifier(5)
knn9 = neighbors.KNeighborsClassifier(9)
knn15 = neighbors.KNeighborsClassifier(15)
knn25 = neighbors.KNeighborsClassifier(25)

''''knn3':knn3,
                         'knn5':knn5, 
                         'knn9':knn9,
                         'knn15':knn15,
                         'knn25':knn25,'''

# Decision Tree
tree_mod = tree.DecisionTreeClassifier(max_depth = 5)

# Random Forest
rf = ensemble.RandomForestClassifier(max_depth = 5, n_estimators = 10, max_features = 1)

# Naive Bayes
nb = naive_bayes.GaussianNB()

# Create model dictionary
classification_models = {'tree_mod':tree_mod, 
                         'rf':rf, 
                         'nb':nb}
                         

####################
###  FIT MODELS  ###
####################

for model_name, model in classification_models.iteritems():
    model.fit(train, train_target)
    
########################
###  PREDICT VALUES  ###
########################

# Create prediction dictionary
classification_predictions = {}
classification_probas = {}

for model_name, model in classification_models.iteritems():
    st = time.time()
    classification_predictions[model_name] = model.predict(test)
    classification_probas[model_name] = model.predict_proba(test)
    calc_time[model_name] = time.time() - st
    
#########################
###  EVALUATE MODELS  ###
#########################

# Create validation metrics dictionary
classification_validation = {}

for model, pred in classification_predictions.iteritems():
    classification_validation[model] = {}
    classification_validation[model]['confusion_matrix'] = metrics.confusion_matrix(test_target, pred)
    classification_validation[model]['auc'] = metrics.roc_auc_score(test_target, pred)
    classification_validation[model]['accuracy'] = metrics.accuracy_score(test_target, pred)

for model, proba in classification_probas.iteritems():
    classification_validation[model]['fpr'], classification_validation[model]['tpr'], classification_validation[model]['thresholds'] = metrics.roc_curve(test_target, proba[:, 1])
    
print classification_validation

# Plot ROC curves
plt.figure()

for model in classification_validation:
    plt.plot(classification_validation[model]['fpr'], classification_validation[model]['tpr'], label = 'ROC curve for %s (area = %0.2f)' % (model, classification_validation[model]['auc']))

plt.plot([0, 1], [0, 1], '--', color = (0.6, 0.6, 0.6), label = 'Random Guess')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.title('ROC Curves')
plt.show()




##############################
##############################
### 2016 Predicitons #########
##############################
##############################

# Reset index to make joins
seeds.reset_index(inplace = True)


# Create models
tourney_models = {'tree_mod':tree_mod, 
                  'rf':rf, 
                  'nb':nb}

# Create prediction dictionary
tourney_predictions = {}
tourney_probas = {}
ss_16i = {}
model_results = {}


# create new column names for comparisons
s_cols = {}
w_cols = {}

cols = team_rAll.columns.tolist()

for col in cols:
    s_cols[col] = 'T_' + col
    w_cols[col] = 'O_' + col


# Create brackets
slot_16 = slots[slots['Season'] == 2017]
seed_16 = seeds[['Seed', 'Team']][seeds['Season'] == 2017]

ss_16 = slot_16.merge(seed_16, 
                      left_on = ['Strongseed'], 
                      right_on = ['Seed'],
                      how = 'left')
                      
ss_16.drop('Seed', axis = 1, inplace = True)
ss_16.rename(columns = {'Team':'STeam'}, inplace = True)

ss_16 = ss_16.merge(seed_16, 
                      left_on = ['Weakseed'], 
                      right_on = ['Seed'],
                      how = 'left')
                      
ss_16.drop('Seed', axis = 1, inplace = True)
ss_16.rename(columns = {'Team':'WTeam'}, inplace = True)

ss_16.fillna(0, inplace = True)

ss_eval_names = ss_16.columns.tolist()

ss_16['evaluated'] = False
 

for model_name, model in tourney_models.items():    
    # set dictionary and list to store results    
    i = 0
    ss_16i[model_name] = pd.DataFrame.copy(ss_16)
    Rounds = []
    
    # While loop to iterate through all the rounds until a winner is found
    while ss_16i[model_name].evaluated.all() == False:        
        Rounds.append([i])
        Rounds[i] = {}  
        
        # Collect games ready to evaluate
        ss_eval = ss_16i[model_name][(ss_16i[model_name].STeam > 1) \
                            & (ss_16i[model_name].WTeam > 1) \
                            & (ss_16i[model_name].evaluated == False)]
        
        # append team record info & align column names            
        ss_eval = ss_eval.merge(team_rAll, 
                                left_on = ['Season', 'STeam'],
                                right_index = True,
                                how = 'inner')
        
        ss_eval.rename(columns = s_cols, inplace = True)
        
        ss_eval = ss_eval.merge(team_rAll, 
                                left_on = ['Season', 'WTeam'],
                                right_index = True,
                                how = 'inner')
                                
        ss_eval.rename(columns = w_cols, inplace = True)        
        
        # check off games evaludated
        ss_16i[model_name]['evaluated'] = ((ss_16i[model_name].STeam > 1) 
                                            & (ss_16i[model_name].WTeam > 1))
        
        # Make predictions and probabilities
        tourney_predictions[model_name] = model.predict(ss_eval[test_atts])
        tourney_probas[model_name] = model.predict_proba(ss_eval[test_atts])
        Rounds[i][model_name + '_predictions'] = tourney_predictions[model_name]
        Rounds[i][model_name + '_probas'] = tourney_probas[model_name]
        
        # 1 = First team win, 0 = 2nd team wins - use to mask los
        winner = pd.DataFrame(ss_eval.STeam.multiply(
                              tourney_predictions[model_name]).add(
                              ss_eval.WTeam.multiply(
                              1-tourney_predictions[model_name])))
                              
        winner = winner.merge(pd.DataFrame(ss_eval.Slot), 
                              left_index = True, 
                              right_index = True)
                    
        winner.columns = ['Team', 'new_seed']
        
        
        # Store round winners
        Rounds[i][model_name + '_winners'] = pd.DataFrame(winner.Team).merge(teams, 
                                                                             left_on = 'Team',
                                                                             right_on = 'Team_Id')
        model_results[model_name] = Rounds
        
        # Align teams with new matchup
        ss_STeam = ss_16i[model_name].merge(winner, 
                                           left_on = 'Strongseed', 
                                           right_on = 'new_seed', 
                                           how = 'left')
        
                       
        ss_WTeam = ss_16i[model_name].merge(winner, 
                                           left_on = 'Weakseed', 
                                           right_on = 'new_seed', 
                                           how = 'left')
                               
        ss_STeam.fillna(0, inplace = True)
        ss_WTeam.fillna(0, inplace = True)
        
        ss_16i[model_name].STeam = ss_16i[model_name].STeam.add(ss_STeam.Team)        
        ss_16i[model_name].WTeam = ss_16i[model_name].WTeam.add(ss_WTeam.Team)
             
        i += 1


# Write winners from all models to a single dataframe
winners = {}

for key in model_results:
    winners[key] = pd.DataFrame ()
    for i in range(len(model_results[key])):
        winners[key] = pd.concat([winners[key], model_results[key][i][key +'_winners']['Team_Name']])
        
winners_df = pd.concat(winners, axis = 1)

winners_df.to_csv('\\predictions\\winners.csv')



gameResultsTeamStatsR.to_csv('generated_data\\gameResultsTeamStatsR.csv', index = False, headers = True)
gameResultsTeamStatsT.to_csv('generated_data\\gameResultsTeamStatsT.csv', index = False, headers = True)
teamRecordsR.to_csv('generated_data\\teamRecordsR.csv', index = False, headers = True)





