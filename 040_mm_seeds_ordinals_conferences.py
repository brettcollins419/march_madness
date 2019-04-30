# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:12:17 2019

@author: U00BEC7
"""

### ###########################################################################
### ################# TOURNAMENT SEED RANKS ###################################
### ###########################################################################

# Tourney Seed Rank
dataDict['tSeeds'].loc[:, 'seedRank'] = (
        map(lambda s: float(re.findall('[0-9]+', s)[0]), 
            dataDict['tSeeds']['Seed'].values.tolist())
        )

for df in map(lambda g: g + 'TeamSeasonStats', ('rGamesC', 'rGamesD')):

#    dataDict[df] = dataDict[df].merge(dataDict['tSeeds'].set_index(['Season', 'TeamID']),
#                                      how = 'left', 
#                                      left_index = True,
#                                      right_index = True)


    dataDict[df].loc[:, 'seedRank'] = (
            dataDict['tSeeds'].set_index(['Season', 'TeamID'])['seedRank'])

### ###########################################################################
### ############ END OF SEASON MASSEY ORDINAL RANKS ###########################
### ###########################################################################

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


### ###########################################################################
### ################### SCALED TEAM SEASON STATISTICS## #######################
### ###########################################################################    
  
for df in ('rGamesC', 'rGamesD'):    
    
    # Rank teams by each season stat metrics within 
    rankDesc = filter(lambda field: field in ('pointsAllowed', 'TO', 'PF'), 
                      dataDict[df + 'TeamSeasonStats'].columns.tolist())

  
    # Rank teams within season for each metric and use % for all teams between 0 and 1 (higher = better)
    # Change from sequential ranker to minmax scale to avoid unrealistic spread (4/22/19)
    dataDict[df+'TeamSeasonStatsRank'] = (dataDict[df + 'TeamSeasonStats']
                                            .groupby('Season')
                                            .apply(lambda m: (m - m.min()) 
                                                    / (m.max() - m.min()))
                                            )
    
    # Switch fields where lower values are better
    dataDict[df+'TeamSeasonStatsRank'].loc[:, rankDesc] = (
            1 - dataDict[df+'TeamSeasonStatsRank'][rankDesc]
            )
    


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
    
        
  
    
### ###########################################################################
### ############## HANDLE MISSING DATA ########################################
### ###########################################################################


# Missing Values Dict
fillDict = {'LSeed':'NA', 
            'seedRank':17,
            'OrdinalRank': 176}

    
for df in map(lambda g: g + 'TeamSeasonStats', ('rGamesC', 'rGamesD')):
    dataDict[df].fillna(fillDict, inplace = True)
    
    
    
    
# Memory Cleanup (consumes ~122 MB)
del(maxRankDate, rsRankings, dataDict['MasseyOrdinals'], 
    fillDict, endRegSeason, df, rankDesc)