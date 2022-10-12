import pandas as pd 
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import bhtsne

result_01 = pd.read_csv('game_result0.csv')

for i in range(1,30):
    a = pd.read_csv('data_set/game_result{}.csv'.format(i))
    result_01 = pd.concat((result_01,a)).reset_index(drop=True)
    
result_01 = result_01.drop(['試合日','節','K/O時刻','入場者数','インターネット中継・TV放送'],axis=1)
#result_01 = result_01.drop(result_01.index[[9294]],axis=0).reset_index()

result_01 = result_01.drop(result_01.index[22853:])
result_01 = result_01.drop(result_01.index[6597])

result_01 = result_01[result_01['スコア'] != '中止']

def get_Hscore(x):
    return int(x.split('-')[0])

def get_Ascore(x):
    return int(x.split('-')[1].split('(')[0])

result_01['HScore'] = result_01['スコア'].map(lambda x: get_Hscore(x))
result_01['AScore'] = result_01['スコア'].map(lambda x: get_Ascore(x))

result_01.rename(columns={'大会':'Seed','年度':'Year','ホーム':'Home','アウェイ':'Away','スタジアム':'Place'},inplace=True)
result_01 = result_01.reset_index()

result_01 = result_01.drop('スコア',axis=1)

result_01['HResult'] = ''
result_01['AResult'] = ''

result_01['HScore'] = result_01['HScore'].map(lambda x: int(x))
result_01['AScore'] = result_01['AScore'].map(lambda x: int(x))

for i in range(len(result_01)):
    if result_01['HScore'][i] > result_01['AScore'][i]:
        result_01['HResult'][i] = 'w'
    elif result_01['HScore'][i] == result_01['AScore'][i]:
        result_01['HResult'][i] ='d'
    elif result_01['HScore'][i] < result_01['AScore'][i]:
        result_01['HResult'][i] ='l'


    if result_01['AScore'][i] > result_01['HScore'][i]:
        result_01['AResult'][i] ='w'
    elif result_01['AScore'][i] == result_01['HScore'][i]:
        result_01['AResult'][i] ='d'
    elif result_01['AScore'][i] < result_01['HScore'][i]:
        result_01['AResult'][i] ='l'
        

home_result = result_01[['Year','Home','HScore','AScore','HResult','Place']]
away_result = result_01[['Year','Away','AScore','HScore','AResult','Place']]

home_result.rename(columns={'Home':'Team','HResult':'Result','HScore':'Score','AScore':'Lost'},inplace=True)
away_result.rename(columns={'Away':'Team','AResult':'Result','AScore':'Score','HScore':'Lost'},inplace=True)
home_result['Home'] = 1
away_result['Home'] = 0

result = pd.concat((home_result,away_result)).reset_index(drop=True)

year_score = result.groupby(['Year','Team'])['Score'].sum().reset_index()
year_lost = result.groupby(['Year','Team'])['Lost'].sum().reset_index()

#############################

teams = list(result_01.Home.unique())

result_01['HScoreAME'] = result_01.groupby(['Home'])['HScore'].rolling(5).mean().reset_index().drop(['level_1','Home'],axis=1)
result_01['AScoreAME'] = result_01.groupby(['Away'])['AScore'].rolling(5).mean().reset_index().drop(['level_1','Away'],axis=1)

result_01['HLostAME'] = result_01.groupby(['Home'])['AScore'].rolling(5).mean().reset_index().drop(['level_1','Home'],axis=1)
result_01['ALostAME'] = result_01.groupby(['Away'])['HScore'].rolling(5).mean().reset_index().drop(['level_1','Away'],axis=1)

result_01 = result_01.drop([8280,8282,9034])

result_01 = result_01.dropna()

Hscore_sum = result_01.groupby(['Year','Seed','Home'])['HScore'].sum().reset_index()
Hlost_sum = result_01.groupby(['Year','Seed','Home'])['AScore'].sum().reset_index()

Ascore_sum = result_01.groupby(['Year','Seed','Away'])['AScore'].sum().reset_index()
Alost_sum = result_01.groupby(['Year','Seed','Away'])['HScore'].sum().reset_index()

result_01 = result_01.drop('HScore',axis=1)
result_01 = result_01.drop('AScore',axis=1)
result_01 = pd.merge(result_01,Hscore_sum,left_on=['Year','Seed','Home'],right_on=['Year','Seed','Home'],how='left')
result_01 = pd.merge(result_01,Hlost_sum,left_on=['Year','Seed','Home'],right_on=['Year','Seed','Home'],how='left')
result_01.rename(columns={'HScore':'HScore_sum','AScore':'HLost_sum'},inplace=True)

result_01 = pd.merge(result_01,Ascore_sum,left_on=['Year','Seed','Away'],right_on=['Year','Seed','Away'],how='left')
result_01 = pd.merge(result_01,Alost_sum,left_on=['Year','Seed','Away'],right_on=['Year','Seed','Away'],how='left')
result_01.rename(columns={'HScore':'ALost_sum','AScore':'AScore_sum'},inplace=True)

result_01 = result_01.drop(['index','Place'],axis=1)
result_01 = result_01.drop('AResult',axis=1)

def results(x):
    if x == 'w':
        return 2
    elif x == 'l':
        return 0
    elif x == 'd':
        return 1
result_01['HResult'] = result_01['HResult'].map(lambda x: results(x))

result_01['HSeed'] = result_01['HScore_sum'] - result_01['HLost_sum']
result_01['ASeed'] = result_01['AScore_sum'] - result_01['ALost_sum']

result_01['HSeed'] = result_01.groupby(['Year','Seed','Home'])['HSeed'].rank().reset_index().drop('index',axis=1)

result_01['ASeed'] = result_01.groupby(['Year','Seed','Away'])['ASeed'].rank().reset_index().drop('index',axis=1)
result_00 = result_01.copy()

result_00['ASeed'] = result_00.groupby(['Away','Seed'])['ASeed'].shift().reset_index().drop('index',axis=1)
result_00['HSeed'] = result_00.groupby(['Home','Seed'])['HSeed'].shift().reset_index().drop('index',axis=1)


result_00 =result_00.dropna()

bin_edges = [-float('inf'),3.5,5.5,float('inf')]
binned = pd.cut(result_01.HSeed,bin_edges,labels=False)
result_01['HSeed_Class'] = binned

bin_edges = [-float('inf'),3.5,5.5,float('inf')]
binned = pd.cut(result_00.HSeed,bin_edges,labels=False)
result_00['HSeed_Class'] = binned

bin_edges = [-float('inf'),3.5,5.5,float('inf')]
binned = pd.cut(result_00.ASeed,bin_edges,labels=False)
result_00['ASeed_Class'] = binned

result_00 = result_00.drop('Seed',axis=1)
result_00['HFreq'] = 1
result_00['AFreq'] = 1


AFreq = result_00.groupby(['Year','Away'])['AFreq'].sum().reset_index()
HFreq = result_00.groupby(['Year','Home'])['HFreq'].sum().reset_index()

result_00 = result_00.drop(['HFreq','AFreq'],axis=1)

result_00 = pd.merge(result_00,HFreq,left_on=['Year','Home'],right_on=['Year','Home'],how='left')
result_00 = pd.merge(result_00,AFreq,left_on=['Year','Away'],right_on=['Year','Away'],how='left')

result_00['Lost_diff'] = result_00['HLostAME'] - result_00['ALostAME']


result_00['Seed_diff'] = result_00['HSeed'] - result_00['ASeed']


result_00['Score_diff'] = result_00['HScoreAME'] - result_00['AScoreAME']

result_00 = result_00.drop(['Year','Home','Away'],axis=1)

X0 = result_00.drop(['HResult'],axis=1)
y0 = result_00['HResult']


result_01['HFreq'] = 1
result_01['AFreq'] = 1


AFreq = result_01.groupby(['Year','Away'])['AFreq'].sum().reset_index()
HFreq = result_01.groupby(['Year','Home'])['HFreq'].sum().reset_index()

result_01 = result_01.drop(['HFreq','AFreq'],axis=1)

result_01 = pd.merge(result_01,HFreq,left_on=['Year','Home'],right_on=['Year','Home'],how='left')
result_01 = pd.merge(result_01,AFreq,left_on=['Year','Away'],right_on=['Year','Away'],how='left')

result_01['Lost_diff'] = result_01['HLostAME'] - result_01['ALostAME']


result_01['Seed_diff'] = result_01['HSeed'] - result_01['ASeed']


result_01['Score_diff'] = result_01['HScoreAME'] - result_01['AScoreAME']

result_01 = result_01.drop(['Year'],axis=1)

X = result_01.drop(['HResult'],axis=1)
y = result_01['HResult']


print(f'ここからコピペしてチームを選んでね⇒{set(result_01.Home) | set(result_01.Away)}')
teamA = input('ホームチームを入力してね：')
teamB = input('アウェーチームを入力してね：')


x_A = X[X['Home'] == '{}'.format(teamA)][:][-1:]
x_A = x_A.reindex(columns=['Home','HScoreAME','HLostAME','HScore_sum','HLost_sum','HSeed','HSeed_Class','HFreq']).reset_index()

x_B = X[X['Away'] == '{}'.format(teamB)][:][-1:]
x_B = x_B.reindex(columns=['Away','AScoreAME','ALostAME','AScore_sum','ALost_sum','ASeed','ASeed_Class','AFreq']).reset_index()

x_A = x_A.drop('index',axis=1)
x_B = x_B.drop('index',axis=1)


x_A['AScoreAME'] = x_B['AScoreAME']
x_A['ALostAME'] = x_B['ALostAME']
x_A['AScore_sum'] = x_B['AScore_sum']
x_A['ALost_sum'] = x_B['ALost_sum']
x_A['ASeed'] = x_B['ASeed']
x_A['ASeed_Class'] = x_B['ASeed_Class']
x_A['AFreq'] = x_B['AFreq']

x_A = x_A.fillna(0)

x_A['Lost_diff'] = x_A['HLostAME'] - x_A['ALostAME']


x_A['Seed_diff'] = x_A['HSeed'] - x_A['ASeed']


x_A['Score_diff'] = x_A['HScoreAME'] - x_A['AScoreAME']

x_A = x_A.drop('Home',axis=1)

X = X.drop(['Seed','Home','Away'],axis=1)

x_A = x_A.reindex(columns=['HScoreAME', 'AScoreAME', 'HLostAME', 'ALostAME', 'HScore_sum',
       'HLost_sum', 'AScore_sum', 'ALost_sum', 'HSeed', 'ASeed', 'HSeed_Class',
       'HFreq', 'AFreq', 'Lost_diff', 'Seed_diff', 'Score_diff'])

tr_x,te_x,tr_y,te_y = train_test_split(X,y,test_size=0.2,shuffle=True)
dtrain = xgb.DMatrix(tr_x, label=tr_y)
dtest = xgb.DMatrix(te_x,label=te_y)
xgb_params = {
    'objective':'multi:softprob',
    'num_class':3,
    'eval_metric':'mlogloss'
}
evals = [(dtrain,'train'),(dtest,'eval')]
evals_result ={}

bst = xgb.train(xgb_params,
               dtrain,
               num_boost_round=100,
               early_stopping_rounds=10,
               evals=evals,
               evals_result=evals_result)

y_pred = pd.DataFrame(bst.predict(dtest))
te_y = pd.DataFrame(te_y)
y_pred['HResult'] = te_y['HResult']

dtest2 = xgb.DMatrix(x_A)
bst = xgb.train(xgb_params,
               dtrain,
               num_boost_round=100,
               early_stopping_rounds=10,
               evals=evals,
               evals_result=evals_result)

y_pred = bst.predict(dtest2)


y_pred = pd.DataFrame(y_pred)

