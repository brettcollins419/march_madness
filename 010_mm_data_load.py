# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 14:07:04 2019

@author: U00BEC7
"""


### ###########################################################################
### ################## INITIAL DATA LOAD ######################################
### ###########################################################################


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

dataDict = {k : pd.read_csv('datasets\\2019\\{}'.format(data)) 
            for k, data in zip(keyNames, dataFiles)}
