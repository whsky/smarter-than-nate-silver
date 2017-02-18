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


#####
#http://sportsdata.wfmz.com/
####
#   Try to get all season game stats
######
import pands as pd
from datetime import datetime, timedelta

years = [2010, 2011, 2012, 2013, 2014, 2015, 2016]
for year in years:
    start,end = datetime(year, 11, 1), datetime(year + 1, 4, 30)
    dates = [start + timedelta(days=i) for i in range((end-start).days+1)]

    df = pd.DataFrame()
    for dte in dates:
        print dte
        yr = str(dte.year)
        game_day = '{:02d}'.format(dte.month)+'{:02d}'.format(dte.day)
        if dte.month >= 11:
            season = str(dte.year) + '-' + str(dte.year + 1)
        elif dte.month <= 5:
            season = str(dte.year - 1) + '-' + str(dte.year)

        base_url = 'http://sportsdata.wfmz.com'
        url3 = base_url + "/sports-scores/College-Basketball-Scores-Matchups.aspx?Year={0}&Period={1}&CurrentSeason={2}".format(yr, game_day, season)
        req3 = requests.get(url3)
        soup3 = BeautifulSoup(req3.text, 'html.parser')
        # boxscore = soup3.find_all('a', href=re.compile('^/basketball/ncaab-boxscores.aspx'))
        boxscore = soup3.find_all('a', href=re.compile('/boxscore'))
        if len(boxscores)==0:
            break
        print '#', len(boxscore), ' games scraped.'

        for box in boxscore:
            box_link = base_url + box['href']
            #Need to repeat for each boxscore link in the page
            req_box = requests.get(box_link)
            soup_box = BeautifulSoup(req_box.text, 'html.parser')
            teams = soup_box.select('td.sdi-datacell > strong')
            if len(teams)==0:
                break
            team_names = [teams[0].text, teams[2].text]
            score = soup_box('span', attrs={'class':"sdi-font-highlight-colour"})
            score1 = int(score[1].text)
            score2 = int(score[2].text)
            #Need to repeat for both teams boxscore table:
            for tble in range(1,3,1):
                data_table = soup_box.find_all('table')
                rows = data_table[tble].find_all('tr')
                team_results = []
                for row in rows:
                    cols = row.find_all('td')
                    cols = [ele.text.strip() for ele in cols]
                    team_results.append([ele for ele in cols if ele])
                df2 = pd.DataFrame(team_results[2:len(team_results)-2], columns=team_results[1])
                df2.insert(0, 'TEAM', [team_names[tble-1] for idx in range(len(df2))])
                df2.insert(1, 'MATCH', [team_names[0] + ' vs. ' + team_names[1] for idx in range(len(df2))])
                df2.insert(2, 'DATE', [dte for idx in range(len(df2))])
                df = df.append(df2)
    print "Writing ", str(season), " game data to csv"
    filepath = '/Users/siannaccone/Documents/galvanize/project/' + str(season) + '_gamedata.csv'

    df = df.replace(r'[^\x00-\x7F]', '', regex=True)
    df.to_csv(filepath)

###################
# Clean up data
##################

#Fix minutes from string 'xx:xx' to int xx.x
def fixTime(timeStr):
    if timeStr == 'DNP':
        return 0.0
    else:
        strSplt = timeStr.split(":")
        return int(strSplt[0]) + 1.0*int(strSplt[1])/60

#We can use apply to run it along the whole 'MIN' column:
df['MIN'] = df['MIN'].apply(fixTime)


#Fix "MATCH" and "TEAM" columns to remove '\n'
def remNewline(inputStr):
    return re.sub(r'\n', '', inputStr)

df['MATCH'] = df['MATCH'].apply(remNewline)
df['TEAM'] = df['TEAM'].apply(remNewline)

#We need to split the shots-made vs shots-attempted into sperate columns
def shotsMade(shotStr):
    if type(shotStr) == str:
        shots = shotStr.split('-')
        return int(shots[0])

def shotsAttmp(shotStr):
    if type(shotStr) == str:
        shots = shotStr.split('-')
        return int(shots[1])

df['3GM'] = df['3GM-A'].apply(shotsMade)
df['3GA'] = df['3GM-A'].apply(shotsAttmp)

df.drop['3GM-A']
