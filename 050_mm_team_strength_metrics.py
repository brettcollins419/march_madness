# -*- coding: utf-8 -*-
"""
Created on Fri May 03 12:58:18 2019

@author: u00bec7
"""


#%% STRENGTH METRICS
## ############################################################################



df = 'rGamesC'

# Create matchup stats using team season statistics
matchups = createMatchups(matchupDF = dataDict['{}singleTeam'.format(df)],
                          statsDF = dataDict['{}TeamSeasonStats'.format(df)])


#### SUBSET FOR DEVELOPMENT
#matchups = matchups[matchups['Season'] == 2019]



# Offense and defence performance: 
#   points scored and allowed against opponent compared to opponent averages
#   Invert defense metric to make a higher number better (postive = allowed fewer points in game than opponent average)
matchups.loc[:, 'offStrength'] = (
        matchups['Score'] - matchups['opppointsAllowed']
        )

matchups.loc[:, 'defStrength'] = (
        matchups['pointsAllowed'] - matchups['oppScore']
        )  * (-1.0)

# How much did opponent win/lose by compared to their average
matchups.loc[:, 'spreadStrength'] = (
        matchups['scoreGap'] - (matchups['oppscoreGap'] * -1.0)
        )


# How much did team win/lose by versus how much they should have
#   based on averages
matchups.loc[:, 'spreadStrength2'] = (
        matchups['scoreGap'] 
            - (matchups['teamScore'] - matchups['opppointsAllowed'])
        )


# Win weighted by opponent win%
matchups.loc[:, 'teamStrength'] = (
        matchups['win'] * matchups['oppwin']
        )


# Opponent strength Metrics
matchups.loc[:, 'oppStrength'] = (
        (matchups['oppscoreGap'] * matchups['oppwin'])
        )

#matchups.loc[:, 'oppStrengthWin'] = (
#        (matchups['oppscoreGapWin'] * matchups['oppwin'])
#        )

# Win weighted by opponent strength
matchups.loc[:, 'teamStrength2'] = (
        matchups['win'] * matchups['oppStrength']
        )




# Weighted metrics (higher is better)
matchups.loc[:, 'offStrengthRatio'] = (
        matchups['offStrength'] * (1/ matchups['opppointsAllowed'])
        )


matchups.loc[:, 'defStrengthRatio'] = (
        matchups['defStrength'] * matchups['oppScore']
        )


matchups.loc[:, 'spreadStrengthRatio'] = (
        matchups['spreadStrength'] * matchups['oppscoreGap']
        )


matchups.loc[:, 'spreadStrengthRatio2'] = (
        matchups['spreadStrength2'] * matchups['oppscoreGap']
        )







# Identify strength columns for aggregation and calculating team performance
strengthMetrics = filter(lambda metric: metric.find('Strength') >= 0, 
                         matchups.columns.tolist())


# Strength metrics incorporating game outcome
# Opponent Win % * if team won the game (use for calculating strength of team)

#for metric in strengthMetrics + ['oppwin']:
#    matchups.loc[:, '{}Win'.format(metric)] = (
#            matchups[metric] * matchups['win']
#            )


# Identify strength columns for aggregation and calculating team performance
strengthMetrics = filter(lambda metric: metric.find('Strength') >= 0, 
                         matchups.columns.tolist())


# Calculate team season means for each metric
strengthDF = (matchups.groupby(['Season', 'TeamID'])
                      .agg(dict(zip(strengthMetrics, 
                                    repeat(np.mean, len(strengthMetrics))))
                            )
                )
   
   
# Plot heat map of strength metric correlations
fig, ax = plt.subplots(nrows = 1, ncols = 1, 
                       figsize = (0.9*GetSystemMetrics(0)//96, 
                                  0.8*GetSystemMetrics(1)//96))

sns.heatmap(strengthDF.corr(),
            annot = True, 
            fmt='.2f',
            mask = heatMapMask(strengthDF.corr()),
#            square = True,
            cmap = 'RdYlGn',
            linewidths = 1, 
            linecolor = 'k',
            ax = ax)
fig.tight_layout(rect=[0,0,1,0.97])
fig.suptitle('Correlation of Strength Metrics', fontsize = 20)
fig.show()
            

# Scale Data between 0 and 1 using minmax to avoid negatives and append values as '[metric]Rank'
# Change from merge to just replace scaled data (4/23/19)
strengthDF = (strengthDF.groupby('Season')
                        .apply(lambda m: (m - m.min()) / (m.max() - m.min()))
                        )


# Principle Component Analysis
pcaStrength = PCA(strengthDF.shape[1])
pcaStrength.fit(strengthDF)


np.cumsum(pcaStrength.explained_variance_ratio_)


#%% STRENGTH METRIC ANALYSIS

# Create MatchUps of Tournament Games to determine which strength metrics
# Has the best performance, and which one to use for ranking teams
matchups = createMatchups(matchupDF = dataDict['tGamesCsingleTeam'][['Season', 'TeamID', 'opponentID', 'win']],
                                         statsDF = strengthDF,
                                         teamID1 = 'TeamID', 
                                         teamID2 = 'opponentID',
                                         teamLabel1 = 'team',
                                         teamLabel2 = 'opp',
                                         returnBaseCols = True,
                                         returnTeamID1StatCols = True,
                                         returnTeamID2StatCols = True,
                                         calculateDelta = True,
                                         calculateMatchup = False)

matchups.set_index(['Season', 'TeamID', 'opponentID', 'win'], inplace = True)






nRows = int(np.ceil(len(strengthDF.columns)**0.5))
nCols = int(np.ceil(len(strengthDF.columns)/nRows))

# Plot distributions of tourament team strength metrics vs. winning team strength metrics
fig, ax = plt.subplots(nrows = nRows, 
                       ncols = nCols, 
                       figsize = (0.9*GetSystemMetrics(0)//96, 
                                  0.8*GetSystemMetrics(1)//96))

for i, metric in enumerate(strengthDF.columns):
    sns.distplot(matchups.groupby(['Season', 'TeamID'])['team{}'.format(metric)].mean(), 
                 hist = True, 
                 ax = ax[i//nCols, i%nCols], 
                 kde_kws={"shade": True}, 
                 label = 'unique')
    
    sns.distplot(matchups['team{}'.format(metric)][matchups.index.get_level_values('win') == 1], 
                 hist = True, 
                 ax = ax[i//nCols, i%nCols], 
                 kde_kws={"shade": True}, 
                 label = 'win')
   
    ax[i//nCols, i%nCols].grid(True)
    ax[i//nCols, i%nCols].legend()
 
    
if len(ax.flatten()) > len(strengthDF.columns):
    for i in range(len(strengthDF.columns), len(ax.flatten())):
        ax.flatten()[i].axis('off')   
        
fig.tight_layout(rect=[0,0,1,0.97])
fig.suptitle('Strength Metrics of Teams and Winning Teams', fontsize = 20)
fig.show()




# Plot heat maps of win % by metric bins

# Desired # of Bins
mBins = 10


fig1, ax1 = plt.subplots(nrows = nRows, 
                       ncols = nCols, 
                       figsize = (0.9*GetSystemMetrics(0)//96, 
                                  0.8*GetSystemMetrics(1)//96))

fig2, ax2 = plt.subplots(nrows = nRows, 
                       ncols = nCols, 
                       figsize = (0.9*GetSystemMetrics(0)//96, 
                                  0.8*GetSystemMetrics(1)//96))

for i, metric in enumerate(strengthDF.columns):
    
    heatMapData = pd.pivot_table(matchups.reset_index(['win', 'TeamID'])
                                        .applymap(lambda p: 
                                            round(p*mBins, 0)/ mBins),
                index = 'opp{}'.format(metric),
                columns = 'team{}'.format(metric),
                values = ['win', 'TeamID'],
                aggfunc = {'win':np.mean, 'TeamID':len}
                )
                                
    
    sns.heatmap(heatMapData.loc[:, heatMapData.columns.get_level_values(0) == 'win'], 
                annot = True, 
                fmt='.2f',
                mask = heatMapMask(heatMapData.loc[:, heatMapData.columns.get_level_values(0) == 'win'], k = -1, corner = 'lower_left'),
#                square = True,
                cmap = 'RdYlGn',
                linewidths = 1, 
                linecolor = 'k',
                ax = ax1[i//nCols, i%nCols]
                )
    ax1[i//nCols, i%nCols].invert_yaxis()
    ax1[i//nCols, i%nCols].invert_xaxis()
    ax1[i//nCols, i%nCols].set_xticklabels(
            heatMapData.loc[:, heatMapData.columns.get_level_values(0) == 'TeamID'].columns.droplevel(0)
            )
  
    sns.heatmap(heatMapData.loc[:, heatMapData.columns.get_level_values(0) == 'win'], 
                annot = heatMapData.loc[:, heatMapData.columns.get_level_values(0) == 'TeamID'], 
                fmt='.0f',
                mask = heatMapMask(heatMapData.loc[:, heatMapData.columns.get_level_values(0) == 'win'], k = -1, corner = 'lower_left'),
#                square = True,
                cmap = 'RdYlGn',
                linewidths = 1, 
                linecolor = 'k',
                ax = ax2[i//nCols, i%nCols]
                )
    ax2[i//nCols, i%nCols].invert_yaxis()
    ax2[i//nCols, i%nCols].invert_xaxis()
    ax2[i//nCols, i%nCols].set_xticklabels(
            heatMapData.loc[:, heatMapData.columns.get_level_values(0) == 'TeamID'].columns.droplevel(0)
            )
    
for ax in (ax1, ax2):
    if len(ax.flatten()) > len(strengthDF.columns):
        for i in range(len(strengthDF.columns), len(ax.flatten())):
            ax.flatten()[i].axis('off')
    
 


fig1.tight_layout(rect=[0,0,1,0.97])
fig1.suptitle('Win % for Metric Bins {}'.format(mBins), fontsize = 20)
fig1.show()

fig2.tight_layout(rect=[0,0,1,0.97])
fig2.suptitle('# of Games for Metric Bins {}'.format(mBins), fontsize = 20)
fig2.show()




#%% STRENGTH METRIC IMPORTANCE AND SELECTION


# Models for getting feature importance
treeModels = {'gb': GradientBoostingClassifier(random_state = 1127, 
                                               n_estimators = 20),
              'et': ExtraTreesClassifier(random_state = 1127, 
                                         n_estimators = 20),
              'rf': RandomForestClassifier(random_state = 1127,
                                           n_estimators = 20)}

trainIndex, testIndex = train_test_split(range(matchups.shape[0]), 
                                         test_size = 0.2)

# Create recursive feature selection models for each treeModel
rfeCVs = {k:RFECV(v, cv = 5) for k,v in treeModels.iteritems()}

# Train models
map(lambda tree: tree.fit(matchups.iloc[trainIndex,:], 
                          matchups.iloc[trainIndex,:].index.get_level_values('win'))
    , rfeCVs.itervalues())


# Score models on train & test data
map(lambda tree: 
    map(lambda idx: 
        rfecvGB.score(matchups.iloc[idx,:], 
                      matchups.iloc[idx,:].index.get_level_values('win')),
        (trainIndex, testIndex)
    )
    , rfeCVs.itervalues())

# # of features selected for each model
map(lambda rfeCV: 
    (rfeCV[0], rfeCV[1].n_features_)
    , rfeCVs.iteritems())    
    

# Get selected features for each model
featureImportance = pd.concat(
        map(lambda rfeCV:
            pd.DataFrame(
                zip(repeat(rfeCV[0], sum(rfeCV[1].support_)),
                    matchups.columns[rfeCV[1].support_],
                    rfeCV[1].estimator_.feature_importances_),
                columns = ['model', 'metric', 'importance']
                ).sort_values(['model', 'importance'], ascending = [True, False])
                , rfeCVs.iteritems())
        , axis = 0)


# Aggregate Feature Importance Metrics 
featureImportanceAgg = (featureImportance.groupby('metric')
                                         .agg({'importance':np.sum,
                                               'model':len})
                        ).sort_values('importance', ascending = False)    
 
    
#%% Wins Against TopN Teams

# use 'spreadStrengthDelta' as teamstrength since it highest correlation with Wins in the Tournament
# Find best metric for # of wins agains topN teams
topNlist = range(10, 301, 10)


matchups = createMatchups(matchupDF = dataDict['rGamesCsingleTeam'][['Season', 'TeamID', 'opponentID', 'win']],
                          statsDF = strengthDF[['spreadStrength', 'teamStrength2']].groupby('Season').rank(ascending = False, method = 'min'),
                          teamID1 = 'TeamID', 
                          teamID2 = 'opponentID',
                          teamLabel1 = 'team',
                          teamLabel2 = 'opp',
                          returnBaseCols = True,
                          returnTeamID1StatCols = False,
                          returnTeamID2StatCols = True,
                          calculateDelta = False,
                          calculateMatchup = False)


topNWins = (pd.concat(
        map(lambda topN: 
            (matchups[matchups['oppspreadStrength'] <= topN].groupby(['Season', 'TeamID'])
                                                            .agg({'win':np.sum})),
            topNlist),
        axis = 1))


# Fill missing values and rename columns
topNWins.fillna(0, inplace = True)
topNWins.columns = map(lambda topN: 'wins{}'.format(str(topN).zfill(3)), topNlist)



# Create tournament matchups with wins against topN counts
matchups = createMatchups(matchupDF = dataDict['tGamesCsingleTeam'][['Season', 'TeamID', 'opponentID', 'win']],
                          statsDF = topNWins,
                          teamID1 = 'TeamID', 
                          teamID2 = 'opponentID',
                          teamLabel1 = 'team',
                          teamLabel2 = 'opp',
                          returnBaseCols = True,
                          returnTeamID1StatCols = False,
                          returnTeamID2StatCols = False,
                          calculateDelta = True,
                          calculateMatchup = False)


matchups.set_index(['Season', 'TeamID', 'opponentID', 'win'], inplace = True)


# Fit and score feature selection
rfecvGB.fit(matchups, matchups.index.get_level_values('win'))
rfecvGB.score(matchups, matchups.index.get_level_values('win'))

# Selected Feature Importances:
featureImportanceTopN = pd.DataFrame(
        zip(matchups.columns[rfecvGB.support_],
            rfecvGB.estimator_.feature_importances_),
        columns = ['metric', 'importance']
        ).sort_values('importance', ascending = False)


featureImportanceTopNRank = pd.DataFrame(
        zip(matchups.columns,
            rfecvGB.ranking_),
        columns = ['metric', 'importance']
        ).sort_values('importance', ascending = True)



fig, ax = plt.subplots(1, 
                       figsize = (0.9*GetSystemMetrics(0)//96, 
                                  0.8*GetSystemMetrics(1)//96))

sns.barplot(x = 'metric', 
            y = 'importance', 
            data = featureImportanceTopNRank.sort_values('metric'), 
            ax = ax)

ax.tick_params(axis='x', rotation=90)



ax2 = ax.twinx()
plt.plot((pd.melt(matchups[matchups.index.get_level_values('win') == 1])
            .groupby('variable')
            .agg({'value': lambda data: 
                len(filter(lambda delta: delta > 0, data))/ len(data)})),
        'go--', linewidth=2, markersize=12)

ax2.grid()
ax2.set_ylabel('Win %')
fig.tight_layout(rect=[0,0,1,0.97])
fig.suptitle('Wins Against Top N Teams Feature Rank', fontsize = 20)
fig.show()