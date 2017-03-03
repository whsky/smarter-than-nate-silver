#############################################################################
# Steve Iannaccone
# March 2017
#   Flask app for running NCAA March Madness predictions
#
# used this blog post as inspiration:
#   https://medium.com/@amirziai/a-flask-api-for-serving-scikit-learn-models-c8bcdaa41daa#.q8gdx7oln
#############################################################################
from flask import Flask, request, url_for, render_template, request
import cPickle as pickle
import numpy as np
from fuzzywuzzy import process

app = Flask(__name__)

# Form page to submit text
@app.route('/')
def index():
    return render_template('index.html')

# def team_submission():
#     # in this form, method = 'POST' means that data entered into
#     # the 'user_input' variable will be sent to the /word_counter routing
#     # block when the 'Enter text' submit button is pressed
#     return '''
#         <form action="/predict" method='POST' >
#             <input type="text" name="user_input1" />
#             <p> vs. </p>
#             <input type="text" name="user_input2" />
#             <input type="submit" value = 'Enter teams'/>
#         </form>
#         '''


# My word counter app
# this routing block accepts data in variable 'user_input' from the form
# and then processes it on the server and returns word counts to the browser
@app.route('/predict', methods=['POST'])
def pnt_predicter():
    team1 = str(request.form['user_input1'])
    team2 = str(request.form['user_input2'])

    team1 = process.extractOne(team1, teamlist)[0]
    team2 = process.extractOne(team2, teamlist)[0]
    if (team1, team2) in predDict:
        spread = int(np.round(predDict[(team1, team2)]))
    elif (team2, team1) in predDict:
        spread = int(np.round(-1. * predDict[(team2, team1)]))
    else:
        spread = "Sorry! We can't find those teams..."


    team1_img = 'img/TeamLogos/{0}'.format(process.extractOne(team1, all_team_imgs)[0])
    team1_img_url = url_for('static', filename=team1_img)

    team2_img = 'img/TeamLogos/{0}'.format(process.extractOne(team2, all_team_imgs)[0])
    team2_img_url = url_for('static', filename=team2_img)
    # make html that gives us a button to go back to the home page
    go_to_home_html = '''
        <img src={0}></img>
        <img src={1}></img>
        <form action="/" >
            <input type="submit" value = "Enter new matchup"/>
        </form>
    '''

    # return page.format(team1, team2, spread) + go_to_home_html.format(team1_img_url, team2_img_url)
    return render_template('predict.html', team1 = team1, team2 = team2, point_spread=spread, team1_img=team1_img, team2_img=team2_img)

if __name__ == '__main__':
    # with open('data/ModelMar2.pkl') as f:
    #     model = pickle.load(f)
    all_team_imgs = pickle.load(open('data/all_team_imgs', 'rb'))
    teamlist = pickle.load(open('data/teamList2016', 'rb'))
    predDict = pickle.load(open('data/predDict2016', 'rb'))
    app.run(host='0.0.0.0', port=8080, debug=True)
