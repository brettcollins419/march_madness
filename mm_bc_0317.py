# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 20:14:35 2016

@author: brett
"""

from os.path import join
import time
import numpy as np
import pandas as pd
import string
import sklearn as sk
from sklearn import svm, linear_model, ensemble, neighbors, tree, naive_bayes, metrics
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


loc = 'C:\\Users\\brett\\Documents\\march_madness_ml\\datasets'
#loc = 'C:\Users\u00bec7\Desktop\MM_Challenge\datasets'

# Read datasets
t_slots = pd.read_csv(join(loc, 'TourneySlots.csv'))
t_seeds = pd.read_csv(join(loc, 'TourneySeeds.csv'))
teams = pd.read_csv(join(loc, 'Teams.csv'))
t_cResults = pd.read_csv(join(loc, 'TourneyCompactResults.csv'))
t_dResults = pd.read_csv(join(loc, 'TourneyDetailedResults.csv'))
r_dResults = pd.read_csv(join(loc, 'RegularSeasonDetailedResults.csv'))
r_cResults = pd.read_csv(join(loc, 'RegularSeasonCompactResults.csv'))

# Apply tournament seeds as integers
t_seeds['seed_int'] = t_seeds['Seed'].apply(lambda x: int(x.strip(string.letters)))
t_seeds.set_index(keys = ['Season', 'Team'], inplace = True)


#################################################
# Calculate additional metrics for regular season
#################################################

# Field Goal %
# 3-pt %
# Free throw %
r_dResults['Wfgp'] = r_dResults.Wfgm / r_dResults.Wfga
r_dResults['Wfg3p'] = r_dResults.Wfgm3 / r_dResults.Wfga3
r_dResults['Wftp'] = r_dResults.Wftm / r_dResults.Wfta

r_dResults['Lfgp'] = r_dResults.Lfgm / r_dResults.Lfga
r_dResults['Lfg3p'] = r_dResults.Lfgm3 / r_dResults.Lfga3
r_dResults['Lftp'] = r_dResults.Lftm / r_dResults.Lfta


# Winning & Losing point margin/gap
r_dResults['gap'] = r_dResults.Wscore - r_dResults.Lscore


#####################################################################
### Calculate regular season team records and performance metrics ###
#####################################################################

# Column Names for records table
dCol = r_dResults.columns.tolist()
dCol_win = dCol[0:4] + dCol[8:21] + dCol[34:37] + [dCol[40]]
dCol_loss = dCol[0:2] + dCol[4:6] + dCol[21:34] + dCol[37:]

# Create groups by team and season
r_winT = r_dResults[dCol_win].groupby(by = ['Season', 'Wteam'])
r_loseT = r_dResults[dCol_loss].groupby(by = ['Season', 'Lteam'])

# count wins and losses by team - rename to join and aggregate results
team_rWins = pd.DataFrame(r_winT.Wscore.count())
team_rWins.index.rename(['Season', 'Team'], inplace = True)
team_rWins.columns = ['Wins']

team_rLoss = pd.DataFrame(r_loseT.Lscore.count())
team_rLoss.index.rename(['Season', 'Team'], inplace = True)
team_rLoss.columns = ['Losses']


# join wins and losses.  Fill NA's with 0
team_rAll = pd.merge(team_rWins, 
                     team_rLoss, 
                     left_index = True, 
                     right_index = True, 
                     how = 'outer')

team_rAll.fillna(0, inplace = True)


# Calculate win percentage
team_rAll['win_pct'] = team_rAll.Wins.divide(
                       team_rAll.Wins.add(team_rAll.Losses))


# average winning stats
team_rWins_stats = r_winT.mean()
team_rWins_stats.rename(columns = {'Daynum':'WDaynum', 
                                   'gap':'Wgap'}, 
                                   inplace = True)


# average losing stats
team_rLoss_stats = r_loseT.mean()
team_rLoss_stats.rename(columns = {'Daynum':'LDaynum', 
                                   'gap':'Lgap'}, 
                                   inplace = True)

# Change 'Lgap' to negative to reflect the loss
team_rLoss_stats.Lgap = team_rLoss_stats.Lgap.multiply(-1)


# Calculate weighted average of win/loss stats 
# and aggregate for overall season stats
wa_win_stats = team_rWins_stats.multiply(
               team_rAll.win_pct, axis = 'index')
               
wa_loss_stats = team_rLoss_stats.multiply(
                team_rAll.win_pct.sub(1).multiply(-1), axis = 'index')

# Fill nan values for teams with 0 wins/losses              
wa_win_stats.fillna(0, inplace = True)
wa_loss_stats.fillna(0, inplace = True)
                
# Drop Daynum column for overall stats
wa_win_stats.drop(labels = ['WDaynum'], axis = 1, inplace = True)
wa_loss_stats.drop(labels = ['LDaynum'], axis = 1, inplace = True)

# Rename column names for merge
dCol_all = ['A' + dCol_win[i][1:] for i in range(3, len(dCol_win))]
dCol_all[17] = 'Agap'

wa_win_stats.columns = dCol_all
wa_loss_stats.columns = dCol_all


# Aggregate wins and loss stats and combine with record dataframe
# Delete temp tables
wa_all_stats = wa_win_stats.add(wa_loss_stats, axis = 'index')

team_rAll = team_rAll.merge(wa_all_stats, 
                            left_index = True,
                            right_index = True,
                            how = 'inner')


# Combine winning and losing metrics with overall team metrics
team_rAll = team_rAll.merge(team_rWins_stats,
                            left_index = True,
                            right_index = True,
                            how = 'left')

team_rAll = team_rAll.merge(team_rLoss_stats,
                            left_index = True,
                            right_index = True,
                            how = 'left')

team_rAll.fillna(0, inplace = True)

# Append tournament seed 
# & fill any non tournament qualifying team with #20 seed
team_rAll = team_rAll.merge(t_seeds, 
                            left_index = True, 
                            right_index = True,
                            how = 'left')
                            
team_rAll.fillna(20, inplace = True)
team_rAll.drop('Seed', axis = 1, inplace = True)


                          
del(wa_win_stats, wa_loss_stats, wa_all_stats)
del(team_rLoss, team_rLoss_stats, team_rWins, team_rWins_stats)

###################################################
### Calculate additional metrics for tournament ###
###################################################

# Field Goal %
# 3-pt %
# Free throw %
t_dResults['Wfgp'] = t_dResults.Wfgm / t_dResults.Wfga
t_dResults['Wfg3p'] = t_dResults.Wfgm3 / t_dResults.Wfga3
t_dResults['Wftp'] = t_dResults.Wftm / t_dResults.Wfta

t_dResults['Lfgp'] = t_dResults.Lfgm / t_dResults.Lfga
t_dResults['Lfg3p'] = t_dResults.Lfgm3 / t_dResults.Lfga3
t_dResults['Lftp'] = t_dResults.Lftm / t_dResults.Lfta


# Winning & Losing point margin/gap
t_dResults['gap'] = t_dResults.Wscore - t_dResults.Lscore


#################################################################
### Calculate tournament team records and performance metrics ###
#################################################################

# Column Names for records table
dCol = t_dResults.columns.tolist()
dCol_win = dCol[0:4] + dCol[8:21] + dCol[34:37] + [dCol[40]]
dCol_loss = dCol[0:2] + dCol[4:6] + dCol[21:34] + dCol[37:]

# Create groups by team and season
t_winT = t_dResults[dCol_win].groupby(by = ['Season', 'Wteam'])
t_loseT = t_dResults[dCol_loss].groupby(by = ['Season', 'Lteam'])

# count wins and losses by team - rename to join and aggregate results
team_tWins = pd.DataFrame(t_winT.Wscore.count())
team_tWins.index.rename(['Season', 'Team'], inplace = True)
team_tWins.columns = ['Wins']

team_tLoss = pd.DataFrame(t_loseT.Lscore.count())
team_tLoss.index.rename(['Season', 'Team'], inplace = True)
team_tLoss.columns = ['Losses']


# join wins and losses.  Fill NA's with 0
team_tAll = pd.merge(team_tWins, 
                     team_tLoss, 
                     left_index = True, 
                     right_index = True, 
                     how = 'outer')

team_tAll.fillna(0, inplace = True)


# Calculate win percentage
team_tAll['win_pct'] = team_tAll.Wins.divide(
                       team_tAll.Wins.add(team_tAll.Losses))


# average winning stats
team_tWins_stats = t_winT.mean()
team_tWins_stats.rename(columns = {'Daynum':'WDaynum', 
                                   'gap':'Wgap'}, 
                                   inplace = True)


# average losing stats
team_tLoss_stats = t_loseT.mean()
team_tLoss_stats.rename(columns = {'Daynum':'LDaynum', 
                                   'gap':'Lgap'}, 
                                   inplace = True)

# Change 'Lgap' to negative to reflect the loss
team_tLoss_stats.Lgap = team_tLoss_stats.Lgap.multiply(-1)


# Calculate weighted average of win/loss stats 
# and aggregate for overall season stats
wa_win_stats = team_tWins_stats.multiply(
               team_tAll.win_pct, axis = 'index')
               
wa_loss_stats = team_tLoss_stats.multiply(
                team_tAll.win_pct.sub(1).multiply(-1), axis = 'index')

# Fill nan values for teams with 0 wins/losses              
wa_win_stats.fillna(0, inplace = True)
wa_loss_stats.fillna(0, inplace = True)
                
# Drop Daynum column for overall stats
wa_win_stats.drop(labels = ['WDaynum'], axis = 1, inplace = True)
wa_loss_stats.drop(labels = ['LDaynum'], axis = 1, inplace = True)

# Rename column names for merge
dCol_all = ['A' + dCol_win[i][1:] for i in range(3, len(dCol_win))]
dCol_all[17] = 'Agap'

wa_win_stats.columns = dCol_all
wa_loss_stats.columns = dCol_all


# Aggregate wins and loss stats and combine with record dataframe
# Delete temp tables
wa_all_stats = wa_win_stats.add(wa_loss_stats, axis = 'index')

team_tAll = team_tAll.merge(wa_all_stats, 
                            left_index = True,
                            right_index = True,
                            how = 'inner')


# Combine winning and losing metrics with overall team metrics
team_tAll = team_tAll.merge(team_tWins_stats,
                            left_index = True,
                            right_index = True,
                            how = 'left')

team_tAll = team_tAll.merge(team_tLoss_stats, 
                            left_index = True, 
                            right_index = True,
                            how = 'left')

team_tAll.fillna(0, inplace = True)

# Append tournament seed 
# & fill any non tournament qualifying team with #20 seed
team_tAll = team_tAll.merge(t_seeds, 
                            left_index = True, 
                            right_index = True,
                            how = 'left')
                            
team_tAll.fillna(20, inplace = True)
team_tAll.drop('Seed', axis = 1, inplace = True)
                          
del(wa_win_stats, wa_loss_stats, wa_all_stats)
del(team_tLoss, team_tLoss_stats, team_tWins, team_tWins_stats)

######################################################################
### Create regular season dataframe with Team vs. Opponent metrics ### 
### for modeling #####################################################
######################################################################

r_wins = r_dResults[['Season', 'Daynum', 'Wteam', 'Lteam']]
r_loss = r_dResults[['Season', 'Daynum', 'Lteam', 'Wteam']]

# create 1 for every win and 0 for every loss and combine dataframes
# Will evaluate Team vs. oppennt (first vs. second team in model)
w = pd.DataFrame(pd.Series([1 for i in range(0, len(r_wins))], name = 'W_L'))
l = pd.DataFrame(pd.Series([0 for i in range(0, len(r_wins))], name = 'W_L'))

r_wins = r_wins.merge(pd.DataFrame(w), left_index = True, right_index = True)
r_loss = r_loss.merge(pd.DataFrame(l), left_index = True, right_index = True)

r_wins.rename(columns = {'Wteam':'Team', 'Lteam':'Opponent'}, inplace = True)
r_loss.rename(columns = {'Lteam':'Team', 'Wteam':'Opponent'}, inplace = True)

r_loss.set_index(keys = [range(len(r_loss), 2 * len(r_loss))], inplace = True)

r_games = pd.concat([r_wins, r_loss])

# create new column names for comparisons
team_cols = {}
opp_cols = {}

cols = team_rAll.columns.tolist()

for col in cols:
    team_cols[col] = 'T_' + col
    opp_cols[col] = 'O_' + col

# Add metrics for Team and reassign column names
r_games = r_games.merge(team_rAll, 
                        left_on = ['Season', 'Team'], 
                        right_index = True)
  
r_games.rename(columns = team_cols, inplace = True)

# Add metrics for Opponent and reassign column names
r_games = r_games.merge(team_rAll, 
                        left_on = ['Season', 'Opponent'], 
                        right_index = True) 
                        
r_games.rename(columns = opp_cols, inplace = True)                   

r_games.set_index(keys = ['Season', 'Team', 'Opponent'], inplace = True)
   
del(r_wins, r_loss, w, l, cols, opp_cols, team_cols)


##################################################################
### Create tournament dataframe with Team vs. Opponent metrics ### 
### for modeling #################################################
##################################################################

t_wins = t_dResults[['Season', 'Daynum', 'Wteam', 'Lteam']]
t_loss = t_dResults[['Season', 'Daynum', 'Lteam', 'Wteam']]

# create 1 for every win and 0 for every loss and combine dataframes
# Will evaluate Team vs. oppennt (first vs. second team in model)
w = pd.DataFrame(pd.Series([1 for i in range(0, len(t_wins))], name = 'W_L'))
l = pd.DataFrame(pd.Series([0 for i in range(0, len(t_wins))], name = 'W_L'))

t_wins = t_wins.merge(pd.DataFrame(w), left_index = True, right_index = True)
t_loss = t_loss.merge(pd.DataFrame(l), left_index = True, right_index = True)

t_wins.rename(columns = {'Wteam':'Team', 'Lteam':'Opponent'}, inplace = True)
t_loss.rename(columns = {'Lteam':'Team', 'Wteam':'Opponent'}, inplace = True)

t_loss.set_index(keys = [range(len(t_loss), 2 * len(t_loss))], inplace = True)

t_games = pd.concat([t_wins, t_loss])

# create new column names for comparisons
team_cols = {}
opp_cols = {}

cols = team_rAll.columns.tolist()

for col in cols:
    team_cols[col] = 'T_' + col
    opp_cols[col] = 'O_' + col

# Add regular season metrics for Team and reassign column names
t_games = t_games.merge(team_rAll, 
                        left_on = ['Season', 'Team'], 
                        right_index = True)
  
t_games.rename(columns = team_cols, inplace = True)

# Add metrics for Opponent and reassign column names
t_games = t_games.merge(team_rAll, 
                        left_on = ['Season', 'Opponent'], 
                        right_index = True) 
                        
t_games.rename(columns = opp_cols, inplace = True)                   

t_games.set_index(keys = ['Season', 'Team', 'Opponent'], inplace = True)
   
del(t_wins, t_loss, w, l, cols, opp_cols, team_cols)


####################################
### Calculate bracket statistics ###
####################################

seed_stats = t_games.groupby(by = ['T_seed_int', 'O_seed_int'])

seed_win = seed_stats.W_L.sum()
seed_matches = seed_stats.W_L.count()
seed_wpct = seed_win / seed_matches

seed_records = pd.concat([seed_matches, seed_win, seed_wpct], axis = 1)
seed_records.columns = ['matches', 'wins', 'win_pct']
non_first_rnd = seed_records [seed_records.matches != 52]


##################################
### Generate dataset for model ###
##################################

# Select attributes to model around
atts = r_games.columns.tolist()
test_atts = atts[2:8] + [atts[15]] + atts[19:24] + atts[38:42] + atts[57:62] + \
            atts[62:68] + [atts[75]] + atts[79:84] + atts[98:102] + atts[117:]

# test_atts = atts[2:]

# Use regular season team records as training set and tournament 
train = r_games
train_target = train.W_L
train = train[test_atts]

test = t_games[test_atts]
test_target = t_games.W_L

######################
### Normalize data ###
######################

# Preprocess using min-max normalizing between 0 and 1
mmScale = preprocessing.MinMaxScaler()
pp_t_games = mmScale.fit_transform(t_games[test_atts])

# Standardize data to z-score
pp_t_games = preprocessing.scale(t_games[test_atts])

# Other method to preprocess to z-score and then apply to another data set
# zScale = preprocessing.StandardScaler()
# pp_t_games = zScale.fit_transform(t_games[test_atts])



# Create training and test sets using tournament data
train, test, train_target, test_target = train_test_split(pp_t_games, t_games.W_L, test_size = 0.2, random_state = 11)


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


# Decision Tree
tree_mod = tree.DecisionTreeClassifier(max_depth = 5)

# Random Forest
rf = ensemble.RandomForestClassifier(max_depth = 5, n_estimators = 10, max_features = 1)

# Naive Bayes
nb = naive_bayes.GaussianNB()

''''knn3':knn3,
                         'knn5':knn5, 
                         'knn9':knn9,
                         'knn15':knn15,
                         'knn25':knn25,'''

# Create model dictionary
classification_models = {
                         'tree_mod':tree_mod, 
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
t_seeds.reset_index(inplace = True)


''''knn3':knn3,
                  'knn5':knn5, 
                  'knn9':knn9,
                  'knn15':knn15,
                  'knn25':knn25,'''

# Create models
tourney_models = {
                  'tree_mod':tree_mod, 
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
slot_16 = t_slots[t_slots.Season == 2017]
seed_16 = t_seeds[['Seed', 'Team']][t_seeds.Season ==2017]

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

winners_df.to_csv('C:\\Users\\brett\\Documents\\mm_2016_ml\\tournament_train_winners.csv')