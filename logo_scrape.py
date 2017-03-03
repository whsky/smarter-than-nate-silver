##################################################
#   Feb. 2017
#
# Webscraping to get all NCAA div1 college logos.
# How can we possibly get logos for all the college teams?!
#   logo_scrape.py will get 'em for you!
##################################################

import pandas as pd
from datetime import datetime, timedelta
import urllib2
from bs4 import BeautifulSoup
import requests
import urllib2
import re


url = "http://www.foxsports.com/college-basketball/teams"
req = requests.get(url)
soup = BeautifulSoup(req.text, 'html.parser')

logos = soup.find_all('img', attrs = {'class': "wisfb_logoImage"})
team_divs = soup.find_all('div', attrs = {'class': "wisbb_fullTeamStacked"})
team = [x.select('span')[0].text for x in team_divs]

for i, logo in enumerate(logos):
    png_link = logo["src"].split("src=")[-1]
    download_img = urllib2.urlopen(png_link).read()
    with open('static/img/%s.png'%team[i], 'wb') as f:
        f.write(download_img)
