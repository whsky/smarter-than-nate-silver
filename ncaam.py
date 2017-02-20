import urllib2
from bs4 import BeautifulSoup
import pandas as pd
import re

data = []
years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
table_pages = 25
main_page1 = 'http://basketball.realgm.com/ncaa/stats/'
main_page2 = '/Averages/Qualified/All/Season/All/points/desc/'

for year in years:
    for page in xrange(1, table_pages + 1):
        url = main_page1 + str(year) + main_page2 + str(page)
        content = urllib2.urlopen(url)
        soup = BeautifulSoup(content, 'lxml')
        tables = soup.find('table', class_='tablesaw compact')

        if tables != None:
            #print page
            #print year
            table_body = tables.find('tbody')
            rows = table_body.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                cols = [ele.text.strip() for ele in cols]
                data.append([ele for ele in cols if ele])

df = pd.DataFrame(data, columns = ['Rank', 'Player', 'Team', 'GP', 'MPG',
    'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM','FTA', 'FT%', 'TOV',
    'PF', 'ORB', 'DRB', 'RPG', 'APG', 'SPG', 'BPG', 'PPG'])

#df.to_csv('PlayerStats-10years.csv', sep=',')
print df.head()

#looks like some of the values are coming in as strings, but we can still
#   manipulate the values
gp = [int(item) for item in df['GP'].tolist()]
print max(gp)


###
#   Past Tournement records from cbssports.com
###
import urllib2
from bs4 import BeautifulSoup
import pandas as pd
import re

years2 = [2007, 2008, 2009, 2010, 2011, 2012]

comb_res = []
main_page3 = 'http://www.cbssports.com/collegebasketball/ncaa-tournament/history/yearbyyear/'
for year in years2:
    tourney = []
    url = main_page3 + str(year)
    content = urllib2.urlopen(url)
    soup = BeautifulSoup(content, 'lxml')
    table = soup.find_all('table', class_='data')

    for idx in xrange(len(table)):
        table_body = table[idx].find('tbody')
        rows = table_body.find_all('tr', class_='row2') #Need to clean this up -- 'row2' used throughout page to define a dark row in any table
        for row in rows:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols]
            tourney.append([ele for ele in cols if ele])


    for idx in xrange(1, len(tourney), 1):
        res = ''.join(str(e) for e in tourney[idx]) #Need to fix this get a unicode error about bad chars -- need to use .encode('utf-8')?
        #Need to remove team rank if in string (i.e. 'No. 2')
        res = res.split('\n')
        res = [re.sub(r'No. \d+ ', '', s) for s in res]
        res_team = [[filter(str.isalpha, s) for s in item.split(',')] for item in res]
        res_score = [[filter(str.isdigit, s) for s in item.split(',')] for item in res]


        #Need to clean up res_team and res_score
        #   empty strings '' and ['']
        #   'OT'
        #   region names 'East'
        res_team = [s for s in res_team if s!=['']]
        res_team = [[s for s in item if s!='' and s!='OT'] for item in res_team]
        res_team = [s for s in res_team if len(s) > 1]
        res_score = [s for s in res_score if s!=['']]
        res_score = [[int(s) for s in item if s!=''] for item in res_score]
        comb_res.append(zip(res_team, res_score))

#Need to cycle through all rounds and then years 2007-2016



#########
# Wiki pages:
#########
years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
results_wiki = []
for year in years:
    url_base = "https://en.wikipedia.org/wiki/"
    url_supp="_NCAA_Men's_Division_I_Basketball_Tournament"
    url2 = url_base + str(year) + url_supp

    req2 = requests.get(url2)
    soup2 = BeautifulSoup(req2.text, 'html.parser')
    tabs = soup2.find_all('table', attrs = { 'style' : "border-style:none;font-size:90%;margin:1em 2em 1em 1em;border-collapse:separate;border-spacing:0" })

    ##Main tournament bracket tables start at len(tabs)-5 and go to len(tabs)
    tab_results = []
    brack_tabs = range(len(tabs)-5,len(tabs),1)
    for idx in brack_tabs:
        teams = tabs[idx].find_all('td', attrs = {'style' : "background-color:#f9f9f9;border-color:#aaa;border-style:solid;border-top-width:1px;border-left-width:1px;border-right-width:1px;border-bottom-width:1px;padding:0 2px"})
        teams_list = [ele.text for ele in teams]

        scores = tabs[idx].find_all('td', attrs = {'style' : "text-align:center;border-color:#aaa;border-style:solid;border-top-width:1px;border-left-width:1px;border-right-width:1px;border-bottom-width:1px;background-color:#f9f9f9"})
        scores_list = [int(re.sub(r'[^\d.]+', '', ele.text)) for ele in scores]

        tab_results.append(zip(teams_list, scores_list))

    results_wiki.append(tab_results)


###################
# Clean up data
##################

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

#We can use apply to run it along the whole 'MIN' column:
df['MIN'] = df['MIN'].apply(fixTime)


#Fix "MATCH" and "TEAM" columns to remove '\n'
def remNewline(inputStr):
    '''
    INPUT: string
    OUTPUT: string
    Clean up and remove newline chars from strings
    '''
    return re.sub(r'\n', '', inputStr)

df['MATCH'] = df['MATCH'].apply(remNewline)
df['TEAM'] = df['TEAM'].apply(remNewline)

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

df['3GM'] = df['3GM-A'].apply(shotsMade)
df['3GA'] = df['3GM-A'].apply(shotsAttmp)
#Sometimes is '3GM-A' and sometimes is '3PM-A'??
df['3GM'] = df['3PM-A'].apply(shotsMade)
df['3GA'] = df['3PM-A'].apply(shotsAttmp)
del df['3PM-A']

#Picking one team's game out:
gm1 = df[(df['DATE'] == '2015-11-13') & (df['TEAM'] == 'Vermont\n')]


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

def getPointSpread(df):
    '''
    INPUT: pandas dataframe
    OUTPUT: point spreads (margins of victory) for each matchup in the dataframe where player stats are available for both teams.
    We want to be able to calculate the margin of victory for a game.
    '''
    #Get all unique matchups from the dataframe:
    unq_mtch = df['MATCH'].unique()
    pt_spread = []
    for idx in range(len(unq_mtch)):
        match_results = df[df['MATCH'] == unq_mtch[idx]].groupby(['TEAM'], sort=False)['PTS'].sum()
        if len(match_results) == 2:
            pt_spread.append(match_results[0] - match_results[1])
        else:
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

# We can get team averages / max / mode of the features in the matrix
#   and use those to plot against pt_spread and look for
#   relationships...
