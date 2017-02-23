import urllib2
from bs4 import BeautifulSoup
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.externals import joblib
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
    #Get all unique matchups from the dataframe:
    unq_mtch = df['MATCH'].unique()
    pt_spread = []
    for idx in range(len(unq_mtch)):
        match_results = df[df['MATCH'] == unq_mtch[idx]].groupby(['TEAM'], sort=False)['PTS'].sum()
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

def NNreshape(df):
    '''
    INPUT: pandas dataframe
    OUTPUT: pandas dataframe where we have selected the top 5 players
        from each team in the matchup (by minutes played) and reshaped to have one long row with palyer stats stacked horizontally instead of vertically. And just the numeric columns.
    This will be our input shape for our Neural Net. Right now, this is pretty slow...vectorize maybe?
    '''
    df_out = pd.DataFrame()
    unq_mtch = df['MATCH'].unique()
    for idx in range(0, len(unq_mtch), 1):
        # print idx
        match = df[df['MATCH'] == unq_mtch[idx]]
        row_match = []
        if len(match['TEAM'].unique()) == 2:
            for team in match['TEAM'].unique():
                # print team
                top5 = match[match['TEAM'] == team].sort('MIN', ascending=False)[:5]
                row_match.extend(top5.stack().values)
            df_out = df_out.append(pd.DataFrame([row_match]))
    return df_out._get_numeric_data()#.dropna()

def getRollingAvg(df, num=5):
    '''
    INPUT: pandas dataframe of game stats, int with number of games to use for rolling avg.
    OUTPUT:
    We want to replace game performance data with that player's rolling average from the last 'num' games.
    '''
    df_out = df.copy()
    for i in range(len(df['PLAYER'].unique())):
        plyr = df['PLAYER'].unique()[i]
        plyr_df = df[df['PLAYER'] == plyr].rolling(window=num).mean()
        idx = plyr_df.index
        df_out.ix[idx] = plyr_df.ix[idx]
    return df_out



# We can get team averages / max / mode of the features in the matrix
#   and use those to plot against pt_spread and look for
#   relationships...

#############################################################
# Things to do:
#--------------
#   X  -Reset shots made/attempted to shooting percentages
#     -Scale features by season. So all features are [0-1]
#     -Replace game stats with player avgs or performance preds.
#        Otherwise we are looking at the stats of the game we are predicting on.
#   X  -Trim out unnecessary features from reshaping in getStarters()
#     -Set up Neural Net
#         This will take output from getStarters() and use getPointSpread() as the target.
#     -Set up KNN / Recommender to look for most similar matchups
#     -Set up Train/Test splits
#         how many seasons to look at?
#         how to seperate tournament games from regular season?
#     --Make tests?
#############################################################

if __name__ == '__main__':
    df = pd.read_csv('data/2015-2016_gamedata.csv')

    ###############
    #Data clean-up:
    ###############
    #We can use apply to run fixTime() along the whole 'MIN' column:
    df['MIN'] = df['MIN'].apply(fixTime)

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

    #remove '\n' from team names and match strings:
    df['MATCH'] = df['MATCH'].apply(remNewline)
    df['TEAM'] = df['TEAM'].apply(remNewline)

    print df.head()

    #Split regular season data to train and tournement data to test:
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
    tourney_year = '2016'
    df_reg = df[df['DATE'] < TourneyDates[tourney_year]]
    df_tourney = df[df['DATE'] >= TourneyDates[tourney_year]]

    ##########################
    #Let's switch to using Rolling Avg's
    #   This will avoid leakage issues caused by using what should be unavailable game data
    ##########################
    print "Getting Rolling Averages..."
    df_avg = getRollingAvg(df, num=3)


    print "Splitting data to regular season and tournement..."
    df_avg_reg = df_avg[df_avg['DATE'] < TourneyDates[tourney_year]]
    df_avg_tourney = df_avg[df_avg['DATE'] >= TourneyDates[tourney_year]]
    #We get a bunch of NaNs from the rolling average, so let's drop 'em':
    df_avg_reg = df_avg_reg.dropna()
    df_avg_tourney = df_avg_tourney.dropna()

    ############################
    #Reshape DF for input to NN:
    ############################

    print "Reshaping data to conform to NN input..."
    X_reg = NNreshape(df_avg_reg)
    y_reg = getPointSpread(df_avg_reg, keepNan=False)

    X_tourney = NNreshape(df_avg_tourney)
    y_tourney = getPointSpread(df_avg_tourney, keepNan=False)

    #NaN Hunt:
    #inds = pd.isnull(X_reg).any(1).nonzero()[0]


    ########################################
    #Scale DF with sklearn's MinMaxScaler():
    # http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
    ########################################
    # min_max_scaler = MinMaxScaler()
    #Need to get Train/Test split running...
    # X_train_minmax = min_max_scaler.fit_transform(X_train)
    # X_test_minmax = min_max_scaler.transform(X_test)


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
    X2_reg.dump("data/RegNNarray.pkl")
    X2_tourney.dump("data/TourneyNNarray.pkl")
    pickle.dump(y2_reg, open( 'data/RegTargetList.pkl', "wb" ))
    pickle.dump(y2_tourney, open( 'data/TourneyTargetList.pkl', "wb" ))

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
        model.add(Dense(400, input_dim=X2_reg.shape[1], init='uniform', activation='tanh'))
        model.add(Dense(100, input_dim=X2_reg.shape[1], init='uniform', activation='tanh'))
        model.add(Dense(15, input_dim=X2_reg.shape[1], init='uniform', activation='relu'))
        model.add(Dense(1, init='uniform'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model


    print "Start Neural Net training..."
    # evaluate model with standardized dataset
    np.random.seed(seed)
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=200, batch_size=100, verbose=1)))
    pipeline = Pipeline(estimators)
    # kfold = KFold(n_splits=5, random_state=seed)
    results = cross_val_score(pipeline, X2_reg, y2_reg)
    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    pipeline.fit(X2_reg, y2_reg)
    y_pred = pipeline.predict(X2_tourney)
    mse = ((np.round(y_pred) - y2_tourney)**2).mean()
    print '\n'*3
    print '='*30
    print "MSE:", mse

    #Pickle the model:
    model_pkl = pickle.dumps(pipeline)
    #model2 = pickle.loads(model_pkl)


    #######
    #Plotting predictions vs. actual results:
    #######
    # plt.style.use('ggplot')
    # col = [np.sign(y_pred[x]) != np.sign(y2_tourney[x]) for x in range(len(y_pred))]
    # xrng = np.arange(-40,40,0.1)
    # yrng=xrng
    # plt.plot(xrng, yrng, 'r-', alpha=0.6)
    # plt.scatter(y_pred, y2_tourney, c=col, alpha=0.5)
    # plt.show()
