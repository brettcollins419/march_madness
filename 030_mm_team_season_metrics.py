# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:24:58 2019

@author: U00BEC7
"""

### ###########################################################################
### ############# REGULAR SEASON TEAM STATISTIC SUMMARY #######################
### ###########################################################################

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
    
### ###########################################################################
### ################### CONFERENCE TOURNAMENT CHAMPIONS #######################
### ###########################################################################

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
    
