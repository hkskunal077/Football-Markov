import numpy as np
import pandas as pd
import os

cwd = os.getcwd()
print(cwd)
files = os.listdir(cwd)
print(files)

#Reading .csv files containing information about season 11-12 of EPL
all_raw_data_12 = pd.read_csv(r'C:\Users\kushk\Desktop\Projects For Semester 5\2011-12.csv')
raw_data_12 = all_raw_data_12[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR',
                               'HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_12.shape)
print(raw_data_12.head(), raw_data_12.tail())
print(raw_data_12[['FTR']])

playing_stat = pd.concat([raw_data_12], ignore_index = True)
seasons = [raw_data_12]
print(playing_stat.head())
number = playing_stat.shape[0]
print(number)
#We consider the data for this season in general.

#Feature Extraction for using Logit Regression.
table = pd.DataFrame(columns = ('Team','HGS','AGS','HAS','AAS',
                                'HGC','AGC','HDS','ADS','FTAG','FTHG'))
avg_home_scored = playing_stat.FTHG.sum()/number
avg_away_scored = playing_stat.FTAG.sum()/number
avg_home_conceded = avg_away_scored
avg_away_conceded = avg_home_scored

res_home = playing_stat.groupby('HomeTeam')
res_away = playing_stat.groupby('AwayTeam')
all_teams_list = list(res_home.groups.keys())
print("All Teams List\n", all_teams_list)

table.Team = list(res_home.groups.keys())
table.HGS = res_home.FTHG.sum().values
table.HGC = res_home.FTAG.sum().values
table.AGS = res_away.FTAG.sum().values
table.AGC = res_away.FTHG.sum().values

table.HAS = (table.HGS / 19.0) / avg_home_scored
table.AAS = (table.AGS / 19.0) / avg_away_scored
table.HDS = (table.HGC / 19.0) / avg_home_conceded
table.ADS = (table.AGC / 19.0) / avg_away_conceded

feature_table = playing_stat.iloc[:,:23]
feature_table = feature_table[['HomeTeam','AwayTeam','FTR','HST','AST','HC','AC']]

#Home Attacking Strength(HAS), Home Defensive Strength(HDS), Away Attacking Strength(AAS), Away Defensive Strength(ADS)
f_HAS = []
f_HDS = []
f_AAS = []
f_ADS = []
for index,row in feature_table.iterrows():
    f_HAS.append(table[table['Team'] == row['HomeTeam']]['HAS'].values[0])
    f_HDS.append(table[table['Team'] == row['HomeTeam']]['HDS'].values[0])
    f_AAS.append(table[table['Team'] == row['AwayTeam']]['AAS'].values[0])
    f_ADS.append(table[table['Team'] == row['AwayTeam']]['ADS'].values[0])

feature_table['HAS'] = f_HAS
feature_table['HDS'] = f_HDS
feature_table['AAS'] = f_AAS
feature_table['ADS'] = f_ADS
print(feature_table)
#This data gets us the Home and Away, Attacking and Defending Strength
#Which we will use to calculate probabilties.

n_matches = len(playing_stat)
average_home_goals = sum(playing_stat['FTHG'])/n_matches
average_away_goals = sum(playing_stat['FTAG'])/n_matches
average_home_points = (3*sum(playing_stat['FTR'] == 'H') + sum(playing_stat['FTR'] == 'D'))/n_matches
average_away_points = (3*sum(playing_stat['FTR'] == 'A') + sum(playing_stat['FTR'] == 'D'))/n_matches
print("Aveage Home Goals",average_home_goals)
print("Average Away Goals",average_away_goals)
print("Average Home Points",average_home_points)
print("Average Away Points",average_away_points)

#Since all the parameters like
#Forward Passing, Free-Kicks impact performance
x_train_home = raw_data_12[['FTHG']]
y_train_home = raw_data_12[['FTR']] 

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train_home, y_train_home)

x_test = raw_data_12[['FTAG']]
y_pred = classifier.predict(x_test)
print(y_pred.shape)
print(y_pred)


## Now representing the data as markov chain.
import matplotlib.pyplot as plt
import numpy as np

class MarkovChain():
    def __init__(self, transition_prob):
        self.transition_prob = transition_prob
        self.states = list(transition_prob.keys())

    def next_state(self, current_state):
        p = [self.transition_prob[current_state][next_state] for next_state in self.states]
        p = np.array(p)
        p/=p.sum() #Normalise
        return np.random.choice(
            self.states, p = p , replace = False)

    def generate_states(self, current_state, number = 5):
        future_states = []
        for i in range(number):
            future_states.append(self.next_state(current_state))
        return future_states


transition_prob = {'X0':{'X0': 0.732,'X1': 0.165,'X2':0.016 ,'X3':0.029 ,'X4':0.0130 ,'X5':0.004 ,'X6':0.019 ,'X7': 0.006,'X8': 0.0006,'X9': 0.0075,'X10': 0.0029},
                   'X1':{'X0': 0.308,'X1': 0.4659,'X2':0.1152 ,'X3':0.0184 ,'X4':0.0143 ,'X5':0.0109 ,'X6': 0.0157,'X7':0.0072 ,'X8':0.0023 ,'X9':0.0258 ,'X10': 0.0119},
                   'X2':{'X0': 0.2579,'X1': 0.2063,'X2': 0.2440,'X3':0.0203 ,'X4':0.0179 ,'X5':0.0428 ,'X6':0.0242 ,'X7':0.0027 ,'X8': 0.0027,'X9': 0.0811,'X10': 0.0786},
                   'X3':{'X0': 0.7379,'X1':0.1827 ,'X2':0.0132 ,'X3':0.0183 ,'X4': 0.0105,'X5':0.0039 ,'X6':0.0202 ,'X7':0.0085 ,'X8':0 ,'X9':0.0027 ,'X10':0.0015},
                   'X4':{'X0': 0.2841,'X1':0.5255 ,'X2': 0.1021,'X3':0.0154 ,'X4':0.0126 ,'X5':0.0147 ,'X6':0.0147 ,'X7':0.0084 ,'X8':0.0042 ,'X9':0.0140 ,'X10':0.0042},
                   'X5':{'X0': 0.1118,'X1':0.3291 ,'X2':0.4284 ,'X3': 0.0136,'X4':0.0136 ,'X5':0.0240 ,'X6':0.0261 ,'X7':0.0021 ,'X8':0.0073,'X9':0.0188 ,'X10':0.0219},
                   'X6':{'X0': 0.6070,'X1': 0.2279,'X2':0.0564 ,'X3':0.0244 ,'X4':0.0120 ,'X5':0.0099 ,'X6':0.0259 ,'X7': 0.0049,'X8':0.0019 ,'X9':0.0209 ,'X10':0.0069  },
                   'X7':{'X0':0.2541 ,'X1':0.3398 ,'X2':0.1684 ,'X3':0.0135 ,'X4':0.0181 ,'X5':0.0270 ,'X6': 0.0225,'X7': 0.0060,'X8': 0,'X9':0.0962 ,'X10':0.0300},
                   'X8':{'X0':0.2538 ,'X1': 0.1769,'X2':0.1461 ,'X3':0.0153 ,'X4':0.0153 ,'X5':0.0461 ,'X6': 0.0153,'X7': 0,'X8': 0,'X9':0.1307 ,'X10':0.1461},
                   'X9':{'X0': 0.6359,'X1': 0.2495,'X2':0.0074 ,'X3':0.0358 ,'X4':0.0189 ,'X5': 0.0019,'X6': 0.0303,'X7':0.0134 ,'X8':0.0004 ,'X9':0.0044 ,'X10': 0.0009},
                   'X10':{'X0': 0.2137,'X1':0.1882 ,'X2':0.2832 ,'X3':0.0229 ,'X4':0.0245 ,'X5': 0.0254,'X6':0.0432 ,'X7':0.0025 ,'X8':0.0016 ,'X9':0.1153 ,'X10': 0.0585}}
    
football_game = MarkovChain(transition_prob = transition_prob)
print("The probabililty of moving from state of Defending to free kick in Attack",football_game.transition_prob['X2']['X7'])
print("The probability of moving from state of Central to Attacking Free Kick",football_game.transition_prob['X0']['X7'])
print("The probability of moving from state of Attacking to Corner", football_game.transition_prob['X2']['X10'])
print("The probability of moving from state of Defending Free Kick to Goal Kick", football_game.transition_prob['X8']['X9'])
print("The probability of moving from Throw In Central to Attacking Pass", football_game.transition_prob['X3']['X2'])
print("The next possible jumpt from Atacking Throw:", football_game.next_state(current_state = 'X7'))
print("The next possible jumpt from Corner:", football_game.next_state(current_state = 'X10'))
print("\nCreating Random Markov Chain, this is one instance of what Markov Chain could look like if we start from Goal Kick\n")
print(football_game.generate_states(current_state = 'X9', number = 10))
print("\nAnother Possible Markov Chain from a Defensive Pass could be\n")
print(football_game.generate_states(current_state = 'X2', number = 10))




