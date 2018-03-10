##################################################
#   Feb. 2017
#
# Webscraping to get all NCAA div1 basketball
#   boxscores.
# http://sportsdata.wfmz.com/ has boxscores for each
#   game, but only listed one day at a time. :/
#
# Have no fear! scrapey.py to the rescue!  :D
##################################################

import pandas as pd
from datetime import datetime, timedelta
import urllib2
import requests
from bs4 import BeautifulSoup
import re

def grab_boxscores(years, directory, verbose_=True):
    '''
    INPUT: list of years to scrape, string of directory where to save csv files, and Boolean of if you want screen outputs of dates and number of games.
    OUTPUT: one csv file for each season's boxscores for each year in list
    '''
    for year in years:
        #The season runs from November to April each year, so we need
        #   to be able to step through each day in the given season:
        start,end = datetime(year, 11, 1), datetime(year + 1, 4, 30)
        dates = [start + timedelta(days=i) for i in range((end-start).days+1)]

        df = pd.DataFrame()
        for dte in dates:
            if verbose_:
                print dte
            #Now we want to start building the webaddress using the dates:
            yr = str(dte.year)
            game_day = '{:02d}'.format(dte.month)+'{:02d}'.format(dte.day)
            if dte.month >= 11:
                season = str(dte.year) + '-' + str(dte.year + 1)
            elif dte.month <= 5:
                season = str(dte.year - 1) + '-' + str(dte.year)

            base_url = 'http://sportsdata.wfmz.com'
            url3 = base_url + "/sports-scores/College-Basketball-Scores-Matchups.aspx?Year={0}&Period={1}&CurrentSeason={2}".format(yr, game_day, season)
            # print url3
            req3 = requests.get(url3)
            soup3 = BeautifulSoup(req3.text, 'html.parser')
            # boxscore = soup3.find_all('a', href=re.compile('^/basketball/ncaab-boxscores.aspx'))

            #Now we grab out all the links to each boxscore page for the day in question:
            boxscore = soup3.find_all('a', href=re.compile('/boxscore'))
            if len(boxscore)==0:
                continue
            if verbose_:
                print '#', len(boxscore), ' games scraped.'

            for box in boxscore:
                box_link = base_url + box['href']
                #Need to repeat for each boxscore link in the page
                req_box = requests.get(box_link)
                soup_box = BeautifulSoup(req_box.text, 'html.parser')
                #Start grabbing data out of page tables
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
                    #Dump results to a dataframe:
                    df2 = pd.DataFrame(team_results[2:len(team_results)-2], columns=team_results[1])
                    df2.insert(0, 'TEAM', [team_names[tble-1] for idx in range(len(df2))])
                    df2.insert(1, 'MATCH', [team_names[0] + ' vs. ' + team_names[1] for idx in range(len(df2))])
                    df2.insert(2, 'DATE', [dte for idx in range(len(df2))])
                    df = df.append(df2)
        if verbose_:
            print "Writing ", str(season), " game data to csv"
        filepath = directory + str(season) + '_gamedata.csv'
        #We need to strip out any crazy chars to be able to write out our csv file (accents, umlauts, etc...):
        df = df.replace(r'[^\x00-\x7F]', '', regex=True)
        df.to_csv(filepath)
