# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 20:14:35 2016

@author: brett
"""

import sqlite3
import os
from os.path import join
import time
import numpy as np
import pandas as pd
import re

# loc = 'C:\\Users\\brett\\Documents\\mm_2016_ml'
loc = 'C:\Users\u00bec7\Desktop\MM_Challenge\datasets'
db = 'database.sqlite'

# Read datasets
t_slots = pd.read_csv(join(loc, 'TourneySlots.csv'))
t_seeds = pd.read_csv(join(loc, 'TourneySeeds.csv'))
t_cResults = pd.read_csv(join(loc, 'TourneyCompactResults.csv'))
teams = pd.read_csv(join(loc, 'Teams.csv'))

t_cResults = t_cResults.merge(t_seeds, left_on = ['Season', 'Wteam'], right_on = ['Season', 'Team'])
t_cResults = t_cResults.merge(t_seeds, left_on = ['Season', 'Lteam'], right_on = ['Season', 'Team'])
t_cResults.drop(['Team_y', 'Team_x'], axis = 1, inplace = True)
t_cResults.rename(columns = {'Seed_x':'Wseed', 'Seed_y':'Lseed'}, inplace = True)

t_cResults = pd.merge(t_cResults, t_slots['Slot'], left_on = t_cResults[['Season', 'Wseed', 'Lseed']], right_on = [['Season', 'Strong]])

t_slots2 = t_slots.merge(t_slots, left_on = ['Season', 'Slot'], right_on = ['Season', 'Strongseed'], how = 'left')
t_slots2.drop(['])


conn = sqlite3.connect(os.path.join(loc, db))




query = '''SELECT
	ss.Season,
	ss.Slot,
	ss.Strongseed,
	ws.Weakseed,
	ss.team AS strongteam,
	ws.team AS weakteam
FROM (
	SELECT 
		tslot.Season,
		tslot.Slot,
		tslot.Strongseed,
		tseed.team
	FROM TourneySlots AS tslot
	INNER JOIN TourneySeeds AS tseed
		ON tslot.Strongseed = tseed.seed
			AND tslot.Season = tseed.season
	) AS ss
		
LEFT JOIN (
	SELECT 
		tslot.Season,
		tslot.Slot,
		tslot.Weakseed,
		tseed.team
	FROM TourneySlots AS tslot
	INNER JOIN TourneySeeds AS tseed
		ON tslot.Weakseed = tseed.seed
			AND tslot.Season = tseed.season
	) AS ws
	
	ON ss.Season = ws.Season
		AND ss.Slot = ws.Slot'''

d = pd.read_sql(query, conn)

