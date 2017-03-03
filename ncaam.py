import urllib2
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.externals import joblib
from datetime import datetime
import itertools
from fuzzywuzzy import process
#############################
# Functions to clean-up data:
#############################

#Fix minutes from string 'xx:xx' to float xx.x
def fixTime(timeStr):
    '''
    INPUT: String with a Player's game-time in Min:Sec
    OUTPUT: Float decimal value of minutes played
    We also need to check for the string "DNP" which
        signifies "Did Not Play" at set those values
        to zero.
    '''
    if timeStr == 'DNP':
        return 0.0
    else:
        strSplt = timeStr.split(":")
        return int(strSplt[0]) + 1.0*int(strSplt[1])/60

def fixDate(dateStr):
    '''
    INPUT: String with date as YYYY-MM-DD
    OUTPUT: datetime object
    '''
    return datetime.strptime(dateStr, '%Y-%m-%d')

#Fix 'MATCH' and 'TEAM' columns to remove '\n'
def remNewline(inputStr):
    '''
    INPUT: string
    OUTPUT: string
    Clean up and remove newline chars from strings
    '''
    return re.sub(r'\n', '', inputStr)

#We need to split the shots-made vs shots-attempted into sperate columns
def shotsMade(shotStr):
    '''
    INPUT: string with "shotsMade-shotsAttempted" format
    OUTPUT: integer with just shotsMade
    '''
    if type(shotStr) == str:
        shots = shotStr.split('-')
        return int(shots[0])

def shotsAttmp(shotStr):
    '''
    INPUT: string with "shotsMade-shotsAttempted" format
    OUTPUT: integer with just shotsAttempted
    '''
    if type(shotStr) == str:
        shots = shotStr.split('-')
        return int(shots[1])

def shotPct(shotStr):
    '''
    INPUT: string with "shotsMade-shotsAttempted" format
    OUTPUT: Float with percentage of shots made
    '''
    if type(shotStr) == str:
        shots = shotStr.split('-')
        if int(shots[1]) == 0:
            return 0
        else:
            return (1.0 *float(shots[0]) / float(shots[1]))

def misterClean(df):
    '''
    INPUT: pandas dataframe with player stats
    OUTPUT: cleaned dataframe
    '''
    #We can use apply to run fixTime() along the whole 'MIN' column:
    if type(df['MIN'].iloc[0]) == str:
        df['MIN'] = df['MIN'].apply(fixTime)
    if type(df['DATE'].iloc[0]) == str:
        df['DATE'] = df['DATE'].apply(fixDate)

    #Sometimes is '3GM-A' and sometimes is '3PM-A'??
    if '3GM-A' in df.columns:
        df['3Pct'] = df['3GM-A'].apply(shotPct)
        del df['3GM-A']
    if '3PM-A' in df.columns:
        df['3Pct'] = df['3PM-A'].apply(shotPct)
        del df['3PM-A']

    df['FGPct'] = df['FGM-A'].apply(shotPct)
    del df['FGM-A']
    df['FTPct'] = df['FTM-A'].apply(shotPct)
    del df['FTM-A']

    del df['Unnamed: 0']

    #remove '\n' from team names and match strings:
    df['MATCH'] = df['MATCH'].apply(remNewline)
    df['TEAM'] = df['TEAM'].apply(remNewline)
    return df

def splitSeason(df):
    '''
    INPUT: pandas dataframe with player stats
    OUTPUT: 2 dataframes split between regular season and March Madness games.
    This will be our train/test split for each season
        '''
    TourneyDates = {'2007': '2007-03-11',
                    '2008': '2008-03-18',
                    '2009': '2009-03-17',
                    '2010': '2010-03-16',
                    '2011': '2011-03-15',
                    '2012': '2012-03-13',
                    '2013': '2013-03-19',
                    '2014': '2014-03-18',
                    '2015': '2015-03-17',
                    '2016': '2016-03-15',
                    '2017': '2017-03-14'}
    if type(df['DATE'].iloc[0]) == str:
        tourney_year = datetime.strptime(max(df['DATE']), '%Y-%m-%d').year
    else:
        tourney_year = max(df['DATE']).year
    df_reg = df[df['DATE'] < TourneyDates[str(tourney_year)]]
    df_tourney = df[df['DATE'] >= TourneyDates[str(tourney_year)]]
    return df_reg, df_tourney

def getAllTeamGames(team_name, df):
    '''
    INPUT: string, pandas DF - valid team name, game stats DF
    OUTPUT: pandas DF with subset of stats for team in the
        given season
    '''
    return df[df['TEAM'] == team_name]

def getGameByDate(date, df):
    '''
    INPUT: string, pandas DF - date string for games, game stats DF
    OUTPUT: pandas DF with subset of games played on the given date
    '''
    return df[df['DATE'] == date]

def getPointSpread(df, keepNan=False):
    '''
    INPUT: pandas dataframe, Boolean if a None should be filled in where Player level stats aren't available for both teams in that game.
    OUTPUT: list of point spreads (margins of victory) for each matchup in the dataframe where player stats are available for both teams.

    We want to be able to calculate the margin of victory for all games in a given dataframe.
    '''
    #Get all unique matchups from the dataframe: <<<Need to fix for repeat matchups!
    pt_spread = []
    dates = df['DATE'].unique()
    for date in dates:
        df_date = df[df['DATE']==date]
        unq_mtch = df_date['MATCH'].unique()
        for idx in range(len(unq_mtch)):
            match_results = df_date[df_date['MATCH'] == unq_mtch[idx]].groupby(['TEAM'], sort=False)['PTS'].sum()
            if len(match_results) == 2:
                pt_spread.append(match_results[0] - match_results[1])
            elif keepNan:
                pt_spread.append(None)
    return pt_spread

def getTeamAsst(df):
    '''
    INPUT: pandas dataframe
    OUTPUT: tuple with delta of assists between teams for each matchup in the dataframe where player stats are available for both teams.
    '''
    #Get all unique matchups from the dataframe:
    unq_mtch = df['MATCH'].unique()
    team_assts = []
    for idx in range(len(unq_mtch)):
        match_assts = df[df['MATCH'] == unq_mtch[idx]].groupby(['TEAM'], sort=False)['A'].sum()
        if len(match_assts) == 2:
            team_assts.append(match_assts[0] - match_assts[1])
        else:
            team_assts.append(None)
    return team_assts

def bench_warmers(df):
    t1 = datetime.now()
    df_out = pd.DataFrame(columns = df.columns)
    teams = df['TEAM'].unique()
    roster_dict = {}
    for team in teams:
        roster = df[df['TEAM'] == team]['PLAYER'].unique()
        roster_dict[team] = roster

    matches = df['MATCH'].unique()
    for match in matches:
        # match = 'Wofford vs. Stanford'
        gameDF = df[df['MATCH'] == match]
        if len(gameDF['TEAM'].unique()) == 2:
            #Let's find out who sat this game out:
            team1 = gameDF['TEAM'].unique()[0]
            t1_roster = roster_dict[team1]
            t1_player = np.array(gameDF[gameDF['TEAM'] == team1]['PLAYER'])
            didnt_play1 = t1_roster[np.where([x not in t1_player for x in t1_roster])]
            #Now make a new DF with zeros for numeric stats:
            didnt_playDF1 = pd.DataFrame(columns=gameDF.columns)
            didnt_playDF1['PLAYER'] = didnt_play1
            didnt_playDF1['MATCH'] = match
            didnt_playDF1['DATE'] = gameDF['DATE'].iloc[0]
            didnt_playDF1['TEAM'] = team1
            didnt_playDF1.replace(np.NAN, 0, inplace=True)
            #repeat for team2:
            team2 = gameDF['TEAM'].unique()[1]
            t2_roster = roster_dict[team2]
            t2_player = np.array(gameDF[gameDF['TEAM'] == team2]['PLAYER'])
            didnt_play2 = t2_roster[np.where([x not in t2_player for x in t2_roster])]
            #Now make a new DF with zeros for numeric stats:
            didnt_playDF2 = pd.DataFrame(columns=gameDF.columns)
            didnt_playDF2['PLAYER'] = didnt_play2
            didnt_playDF2['MATCH'] = match
            didnt_playDF2['DATE'] = gameDF['DATE'].iloc[0]
            didnt_playDF2['TEAM'] = team2
            didnt_playDF2.replace(np.NAN, 0, inplace=True)
            gameDF = gameDF.append([didnt_playDF1, didnt_playDF2])
            df_out = df_out.append(gameDF)
    t2 = datetime.now()
    t_del = t2 - t1
    run_time = divmod(t_del.total_seconds(), 60)
    print run_time
    return df_out

def NNreshape(df, numPlayers = 15):
    '''
    INPUT: pandas dataframe, int with number of players per team to keep as NN input.
    OUTPUT: pandas dataframe where we have selected the top 5 players
        from each team in the matchup (by minutes played) and reshaped to have one long row with palyer stats stacked horizontally instead of vertically. And just the numeric columns.
    This will be our input shape for our Neural Net. Right now, this is pretty slow...vectorize maybe?
    '''
    t1 = datetime.now()
    df_out = pd.DataFrame()
    dates = df['DATE'].unique()
    for date in dates:
        df_date = df[df['DATE']==date]
        unq_mtch = df_date['MATCH'].unique()
        for idx in range(0, len(unq_mtch), 1):
            # print idx
            match = df_date[df_date['MATCH'] == unq_mtch[idx]]
            row_match = []
            if len(match['TEAM'].unique()) == 2:
                for team in match['TEAM'].unique():
                    # print team
                    top5 = match[match['TEAM'] == team].sort('MIN', ascending=False)#.iloc[:5]
                    #pad out to 15 players regaurdless of number of player stats for the game in question
                    if top5.shape[0] < numPlayers:
                        pad = np.zeros(shape=(numPlayers - top5.shape[0],top5.shape[1]))
                        pad = pd.DataFrame(pad)
                    else:
                        top5 = top5.iloc[:numPlayers]
                    row_match.extend(top5.stack().values)
                    if top5.shape[0] < numPlayers:
                        row_match.extend(pad.stack().values)
                df_out = df_out.append(pd.DataFrame([row_match]))
    t2 = datetime.now()
    t_del = t2 - t1
    run_time = divmod(t_del.total_seconds(), 60)
    print "RUNtime: ", run_time
    return df_out._get_numeric_data()

def getRollingAvg(df, num=5):
    '''
    INPUT: pandas dataframe of game stats, int with number of games to use for rolling avg.
    OUTPUT: pandas dataframe
    We want to replace game performance data with that player's rolling average from the last 'num' games.
    '''
    df_out = df.copy()
    for i in range(len(df['PLAYER'].unique())):
        plyr = df['PLAYER'].unique()[i]
        plyr_df = df[df['PLAYER'] == plyr].rolling(window=num).mean()
        idx = plyr_df.index
        df_out.ix[idx] = plyr_df.ix[idx]
    return df_out

def getPlayerAvg(df):
    '''
    INPUT: pandas dataframe of boxscore stats
    OUTPUT: dataframe with each player's average stats for the season
    '''
    df_out = pd.DataFrame()
    teams = df['TEAM'].unique()
    for idx in range(0, len(teams), 1):
        players_df = df[df['TEAM'] == teams[idx]].groupby(['PLAYER'], sort=False).mean()
        players_df['TEAM'] = [teams[idx]]*len(players_df)
        players_df['PLAYER'] = players_df.index
        df_out = df_out.append(players_df)
    return df_out

def replaceTourneyStats(df_tourney, df_player, round_=False):
    '''
    INPUT: pandas dataframe with game stats from tournament results, pandas DF with season avgs for each player, and Boolean of whether or not to round off stats to integer values
    OUTPUT: pandas dataframe with player stats replaced with their season average.
    We want to hide player performance during the tournament during testing of our model. We will sub in that player's season avg to replace their actual performance for each game.
    '''
    t1 = datetime.now()
    df_out = df_tourney.copy()
    indices = df_tourney.index
    play_ind = df_player.index
    matches = []
    players = []
    # dates = []
    for idx in range(0, len(df_tourney), 1):
        team = df_tourney.ix[indices[idx]]['TEAM']
        player = df_tourney.ix[indices[idx]]['PLAYER']
        date = df_tourney.ix[indices[idx]]['DATE']
        players.append(player)
        matches.append(df_tourney.ix[indices[idx]]['MATCH'])
        if player in play_ind:
            df_out.ix[indices[idx]] = df_player[df_player['TEAM'] == team].loc[player]
        else:
            print "ooops... ", player
    df_out['MATCH'] = matches
    df_out['PLAYER'] = players
    # print len(matches), len(players), len(dates)
    # df_out['DATE'] = dates
    t2 = datetime.now()
    t_del = t2 - t1
    run_time = divmod(t_del.total_seconds(), 60)
    print "replace stats RUNtime: ", run_time
    return df_out


def getAllPreds(team_list, df_player, model):
    '''
    INPUT: list of teams to compare, a pandas DF with player averages for the season, and a pickled model to use for predictions.
    OUTPUT: dict with key as team-pair, and value as predicted point spreads.
    We want to compute all the predicted point spreads for all teams in the input -- this will be a square matrix (NxN for N teams)
    ####
    Can check some pairs with:
        first5pairs = {k: mydict[k] for k in mydict.keys()[:5]}
    Find max with:
        max(mydict.iterkeys(), key=(lambda key: mydict[key])
    ####
    '''
    numPlayers = 7
    dict_out = dict()
    np_out = np.zeros(shape = (len(team_list), len(team_list)))
    team_combos = list(itertools.combinations(team_list, 2))
    for team in team_combos:
        if team[0] not in df['TEAM'].unique():
            print team[0]
        if team[1] not in df['TEAM'].unique():
            print team[1]
        NNarray = []
        df_team1 = df_player[df_player['TEAM'] == team[0]].sort('MIN', ascending=False)
        df_team2 = df_player[df_player['TEAM'] == team[1]].sort('MIN', ascending=False)

        if df_team1.shape[0] < numPlayers:
            pad = np.zeros(shape=(numPlayers - df_team1.shape[0], df_team1.shape[1]))
            pad = pd.DataFrame(pad)
        else:
            df_team1 = df_team1.iloc[:numPlayers]
        NNarray.extend(df_team1.stack().values)
        if df_team1.shape[0] < numPlayers:
            NNarray.extend(pad.stack().values)
        #Repeat for second team:
        if df_team2.shape[0] < numPlayers:
            pad = np.zeros(shape=(numPlayers - df_team2.shape[0], df_team2.shape[1]))
            pad = pd.DataFrame(pad)
        else:
            df_team2 = df_team2.iloc[:numPlayers]
        NNarray.extend(df_team2.stack().values)
        if df_team2.shape[0] < numPlayers:
            NNarray.extend(pad.stack().values)
        df_out = pd.DataFrame([NNarray])
        dict_out[team] = model.predict(df_out._get_numeric_data())
    return dict_out

def getTourneyTeams(year):
    '''
    INPUT: string of year for tournament
    OUTPUT: list of strings with teams that qualified that year
    '''
    tourney_teams = {'2016': \
    ['Kansas', 'Villanova', 'Miami-Florida', 'California', 'Maryland', 'Arizona', 'Iowa', 'Colorado', 'Connecticut', 'Temple', 'Vanderbilt', 'Wichita St.', 'South Dakota State', 'Hawaii', 'Buffalo', 'N.C. Asheville', 'Austin Peay', 'Oregon', 'Oklahoma', 'Texas A&M', 'Duke', 'Baylor', 'Texas', 'Oregon St.', "St. Joseph's", 'Cincinnati', 'VCU', 'Northern Iowa', 'Yale', 'NC-Wilmington', 'Green Bay', 'Cal. State - Bakersfield', 'Holy Cross', 'Southern', 'North Carolina', 'Xavier', 'West Virginia', 'Kentucky', 'Indiana', 'Notre Dame', 'Wisconsin', 'Southern California', 'Providence', 'Pittsburgh', 'Michigan', 'Tulsa', 'Chattanooga', 'Stony Brook', 'Stephen F. Austin', 'Weber St.', 'Florida Gulf Coast', 'Fairleigh Dickinson', 'Virginia', 'Michigan St', 'Utah', 'Iowa St.', 'Purdue', 'Seton Hall', 'Dayton', 'Texas Tech', 'Butler', 'Syracuse', 'Gonzaga', 'Arkansas-Little Rock', 'Iona', 'Fresno St.', 'Middle Tennessee St.', 'Hampton']}
    return tourney_teams[year]

def predLookup(matchup, teamlist, predDict):
    '''
    INPUT: tuple of both team names in the desired matchup
    OUTPUT: dcitionary result of that matchup
    We need to check if the matchup is defined in the getAllPreds dict,
        and return the prediction if it is. If it isn't there we also need to check with team order reversed. We will use 'fuzzywuzzy' partial string matching to find the closest team names if neither order is found.
    Remember that if we reverse the order of teams, we have to flip the sign of the predicted point spread!
    '''
    if matchup in predDict:
        return predDict[matchup]
    elif (matchup[1], matchup[0]) in predDict:
        return -1. * predDict[(matchup[1], matchup[0])]
    else:
        team1 = process.extractOne(matchup[0], teamlist)[0]
        team2 = process.extractOne(matchup[1], teamlist)[0]
        if (team1, team2) in predDict:
            return predDict[(team1, team2)]
        elif (team2, team1) in predDict:
            return -1. * predDict[(team2, team1)]
        else:
            return "Sorry! We can't find those teams..."

#############################################################
# Things to do:
#--------------
#   X  -Reset shots made/attempted to shooting percentages
#   X  -Scale features by season. So all features are [0-1]
#   X  -Replace game stats with player avgs or performance preds.
#        Otherwise we are looking at the stats of the game we are predicting on.
#   X  -Trim out unnecessary features from reshaping in getStarters()
#   X  -Set up Neural Net
#         This will take output from getStarters() and use getPointSpread() as the target.
#     -Set up KNN / Recommender to look for most similar matchups
#     -Set up Train/Test splits
#         how many seasons to look at?
#   X      how to seperate tournament games from regular season?
#     --Make tests?
#############################################################

if __name__ == '__main__':
    df = pd.read_csv('data/2012-2013_gamedata.csv')
    df16 = pd.read_csv('data/2015-2016_gamedata.csv')
    df15 = pd.read_csv('data/2014-2015_gamedata.csv')
    df14 = pd.read_csv('data/2013-2014_gamedata.csv')

    # df = df.append([df14, df15, df16])

    ###############
    #Data clean-up:
    ###############
    df = misterClean(df)

    df = bench_warmers(df)#about 12 mins per season#
    df.reset_index(inplace=True)
    del df['index']
    #Split regular season data to train and tournement data to test:
    df_reg, df_tourney = splitSeason(df)


    ##########################
    #Let's switch to using Rolling Avg's
    #   This will avoid leakage issues caused by using what should be unavailable game data
    ##########################

    # print "Getting Rolling Averages..."
    # df_avg = getRollingAvg(df, num=3)
    #
    # df_avg_reg, df_avg_tourney = splitSeason(df_avg)
    # # #We get a bunch of NaNs from the rolling average, so let's drop 'em':
    # df_avg_reg = df_avg_reg.dropna()
    # df_avg_tourney = df_avg_tourney.dropna()


    print "Replace tournament game data with each player's season avgs..."
    df_player = getPlayerAvg(df_reg)
    df_tourney_season = replaceTourneyStats(df_tourney, df_player)#about 2 mins  per season#
    #replace NaNs in 'DATE' column:
    # df_tourney_season.replace(np.NAN, 'date', inplace=True)
    df_tourney_season['DATE'] = df_tourney['DATE']

    ############################
    #Reshape DF for input to NN:
    ############################

    print "Reshaping data to conform to NN input..."
    X_reg = NNreshape(df_reg, numPlayers=7)#about 30-45 mins per season...ouch!#
    X_tourney = NNreshape(df_tourney_season, numPlayers=7)

    # y = getPointSpread(df, keepNan=False)
    y_reg = getPointSpread(df_reg, keepNan = False)
    y_tourney = getPointSpread(df_tourney, keepNan=False)

    #NaN Hunt:
    #inds = pd.isnull(X_reg).any(1).nonzero()[0]


    import numpy as np
    import pandas as pd
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import confusion_matrix

    print "Conerting dataframes to numpy arrays for NN..."
    X2_reg = X_reg.values
    X2_tourney = X_tourney.values
    y2_reg = np.array(y_reg)
    y2_tourney = np.array(y_tourney)

    X2_reg = np.nan_to_num(X2_reg)
    X2_tourney = np.nan_to_num(X2_tourney)

    #NaN Hunt:
    #np.isfinite(X).all()
    #np.isfinite(X.sum())


    # X2_reg.dump("data/RegNNarraySeason.pkl")
    # X2_tourney.dump("data/TourneyNNarraySeason.pkl")
    # pickle.dump(y2_reg, open( 'data/RegTargetListSeason.pkl', "wb" ))
    # pickle.dump(y2_tourney, open( 'data/TourneyTargetListSeason.pkl', "wb" ))

    ########
    #Read pickles if necessary:
    ########
    # X2_reg.load("data/RegNNarray.pkl")
    # X2_tourney.load("data/TourneyNNarray.pkl")
    # y2_reg = pickle.load('data/RegTargetList.pkl')
    # y2_tourney = pickle.load('data/TourneyTargetList.pkl')


    # fix random seed for reproducibility
    seed = 23
    np.random.seed(seed)

    # X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.18, random_state=seed)

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(200, input_dim=X2_reg.shape[1], init='uniform', activation='softsign'))
    model.add(Dense(75, input_dim=X2_reg.shape[1], init='uniform', activation='softsign'))
    model.add(Dense(10, input_dim=X2_reg.shape[1], init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


print "Start Neural Net training..."
# evaluate model with standardized dataset
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=200, batch_size=30, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=3, random_state=seed)
results = cross_val_score(pipeline, X2_reg, y2_reg, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

pipeline.fit(X2_reg, y2_reg)
y_pred = pipeline.predict(X2_tourney)
mse = ((np.round(y_pred) - y2_tourney)**2).mean()
print '\n'*3
print '='*30
print "MSE:", mse

print np.sqrt(results)
print np.sqrt(results).mean()

win_loss = 1.*(np.sign(y_pred) == np.sign(y2_tourney)).sum()/len(y_pred)
print "Win / Loss correct call pct: ", win_loss
#Pickle the model:
with open('data/Model.?e, f)
with open('data/Model.pkl') as f:
    model2 = pickle.load(f)

#######
#Plotting predictions vs. actual results:
#######
# plt.style.use('ggplot')
# col = [np.sign(y_pred[x]) != np.sign(y2_tourney[x]) for x in range(len(y_pred))]
# xrng = np.arange(-40,40,0.1)
# yrng=xrng
# plt.plot(xrng, yrng, 'r-', alpha=0.6)
# plt.scatter(y_pred, y2_tourney, c=col, alpha=0.5)


teams2016 = getTourneyTeams('2016')
preds2016 = getAllPreds(teams2016, df_player, pipeline)

with open('data/teamList2016','w') as f:
    pickle.dump(teams2016,f)

with open('data/predDict2016','w') as f:
    pickle.dump(preds2016,f)
#Heteroskedasticity?
#EDA!!
#get data product running...Flask / shiny?
#Website / README.md / blog?
#output matrix of 351 x 351 teams and pred outcomes
