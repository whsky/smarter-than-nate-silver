Smarter Than Nate Silver
========================
[SmarterThanNateSilver.com](http://smarterthannatesilver.com:8080) - website is live!
##Making NCAA March Madness predictions

**It's about that time of year again!**

Time to fill out a bracket and pick the best college basketball team in America.

_But, how should we pick who wins?_

Tons of very smart people spend an inordinate amount of time trying to answer exactly this question. Nate Silver's website [FiveThirthyEight.com](https://fivethirtyeight.com/sports/) does a good job of showing probabilities of any team winning the tournament.

_So why the beef with Nate, dog?_

Most of his predictions _(as well as others')_ rely heavily on stats like BPI, Power Ratings, and Elo scores which are themselves based mostly on the ranking of the team going into the tournament. These rank-based stats take many names, sometimes it's "Strength of Schedule", or "Opponent Strength". But those stats skip over the fact that they have not defined what determines highly ranked team to begin with. So, models built on this information lean toward picking a winning team based on who is the better seed. You can see it this effect based on who [FiveThirtyEight](https://fivethirtyeight.com/sports/) has picked to be in the Final Four from the last couple of tournaments:

|**Year**|**Team**|**Seed**|**Made Final Four**
-----|-----|-----|:-----:|-----
2016|Kansas|1|No
||UNC|1|Yes
||Mich. St.|2|No
||Oklahoma|2|No
2015|Kentucky|1|Yes
||Villanova|1|No
||Duke|1|Yes (won championship)
||Arizona|2|No
2014|Florida|1|Yes
||Arizona|1|No
||Loisville|4|No
||Mich. St.|4|No


Hmmmm...that seems to be a lot of No.1 seeds. Perhaps they are seeded higher because they are the better team, so we _should_ use this number, right?

Not exactly, ranking and placing teams in the opening round of the tournament is far more complicated, and even worse, dependent on a panel to decide where each team should rank.

##A Rank Agnostic Approach

Why not just ignore the rankings, and mine through historical data to evaluate team performance based solely on player-level data?

We can then use this game data to predict the outcome of future matches based on the patterns seen in the historical data. Machine Learning techniques like Neural Nets do a great job of finding these type of patterns. So let's build a Multilayer Perceptron to take player data from regular season game data to predict the margin of victory _(the point-spread)_ of post-season games like the March Madness tournament.

This data was scraped for every game that had complete boxscores for the last decade using my `scrapey.py` file in this repo. Also, csv files for player data of each season are available in the `data` folder.

###What is a point-spread?

Point-spreads were introduced by [Charles K. McNeil](https://en.wikipedia.org/wiki/Charles_K._McNeil) _(who later became JFK's math teacher)_ as an attempt to balance out betting between both outcomes. For example, if **Team 1** is a strong favorite to beat **Team 2**, you would expect that a majority of the bets will be for **Team 1** to win. This is a huge disadvantage to whomever is taking the bet. If **Team 1** wins, they have a massive amount of payouts to make. But, if we make the bet over whether **Team 1** will win _by more than an certain number of points_ rather than just if they will win. We now can move this handicap to persuade bets to take one side or the other and thus keep an even number of bets on both sides.

_Wait, why would the people taking the bet want an even number on both sides?_

Great question, because they want to minimize their exposure to risk. With balanced betting they can trust that regardless of the outcome, there is no risk of one side having to pay out more than the other. Don't worry though, they still get a piece of every bet they see, so they still make money.

##Neural Nets

Neural Nets _(NN)_ adapt to the data being feed into them. They are adapting to what they see in relation to the desired outcome. This adaptation of the model is why these methods are referred to as Machine Learning. The model changes a neuron's weighted input that minimizes the error between the model output and desired output.

To build the Multilayer Perceptron NN I used Keras and followed a lot of what Jason Brownlee blogged about on [Machine Learning Mastery](http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/). Because we are concerned with predicting point_spreads, this is a regression model. I used a deep network in the hopes that it would allow the NN to find more interactive patterns in the player stats, like an underlying synergy or "team-i-ness". I tested just under 100 different parameter tweaks (e.g. number of hidden layers, width of input layer, activation function, and epoch/batch sizes). Using the following model I achieved the lowest RMSE and highest percentange of correct calls on game winner:

```python
def baseline_model():
    model = Sequential()
    model.add(Dense(200, input_dim=X2_reg.shape[1], init='uniform',
        activation='softsign'))
    model.add(Dense(75, input_dim=X2_reg.shape[1], init='uniform',
        activation='softsign'))
    model.add(Dense(10, input_dim=X2_reg.shape[1], init='uniform',
        activation='relu'))
    model.add(Dense(1, init='uniform'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
```
