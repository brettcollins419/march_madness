# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:25:45 2019

@author: U00BEC7
"""


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




def heatMapMask(corrData, k = 0, corner = 'upper_right'):
    
    ''' Create array for masking upper right or lower left corner of map.
        k is the offset from the diagonal.
            if 1 returns the diagonal
        Return array for masking'''
        
    # Mask lower left
    if corner == 'lower_left':
        mask = np.zeros_like(corrData) 
        mask[np.tril_indices_from(arr = mask, k = k)] = True

    # Mask upper right
    else:
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
                   returnTeamID1StatCols = True,
                   returnTeamID2StatCols = True,
                   returnBaseCols = True,
                   reindex = True):
    ''' Create dataframe game matchups using team statistics to use for modeling
        & parameter selection / performance.
        
        Options:
            Return Statistic columns for each team (returnStatCols Boolean)
            Calculate teamStatistic deltas for the matchup (calculateMatchup Boolean)
            
            Create a tuple of object columns such as conference or seeds (calculateMatchup Boolean)
                
        
        Return dataframe with same number of recors as the matchupDF.
        '''
    baseCols = matchupDF.columns.tolist()
    team1Cols = map(lambda field: '{}{}'.format(teamLabel1, field), statsDF.columns.tolist())
    team2Cols = map(lambda field: '{}{}'.format(teamLabel2, field), statsDF.columns.tolist())

    
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
   
        # Update column names
        deltaCols = map(lambda col: '{}Delta'.format(col), deltaCols)
    
    else: deltaCols = []
    
    # Calculate matchup attributes if necessary
    if calculateMatchup == True:
        for col in list(set(matchupCols + extraMatchupCols)):
            matchupNew.loc[:, '{}Matchup'.format(col)] = pd.Series(createMatchupField(matchupNew, 
                                                                                       '{}{}'.format(teamLabel1, col), 
                                                                                       '{}{}'.format(teamLabel2, col), sort = False))
    

         # Update column names
        matchupCols = map(lambda col: '{}Delta'.format(col), list(set(matchupCols + extraMatchupCols)))   
    
    else: matchupCols = []
        
    
    
    # Compile columns to return
    returnCols = list((baseCols * returnBaseCols) 
                        + (team1Cols * returnTeamID1StatCols) 
                        + (team2Cols * returnTeamID2StatCols) 
                        + deltaCols 
                        + matchupCols
                        )
    
    
    
    return matchupNew[returnCols]
    
        
#    if returnStatCols == True:
#        return matchupNew
#    
#    # Don't return stat cols
#    else:
#        deltaReturnCols = filter(lambda c: (c.endswith('Delta')) | (c.endswith('Matchup')) | (c in colCount[colCount['field'] != 2].index.get_level_values('field')),
#                                                       matchupNew.columns.tolist())
#        
#        
#        return matchupNew.loc[:, deltaReturnCols]
#    
#
#    
#    return matchupNew




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

print 'Functions Loaded'

