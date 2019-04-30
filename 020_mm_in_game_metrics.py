# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:26:47 2019

@author: U00BEC7
"""

### ###########################################################################
### ############### ADDITIONAL IN GAME METRICS ################################
### ###########################################################################

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
