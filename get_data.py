import pandas as pd 

l = []

years = list(range(1992,2023))

for year in years:
    #url = 'https://data.j-league.or.jp/SFMS01/search?competition_years={}&tv_relay_station_name='.format(year)
    url = 'https://data.j-league.or.jp/SFMS01/search?competition_years={}&competition_frame_ids=1&competition_frame_ids=11&competition_frame_ids=2&tv_relay_station_name='.format(year)
    dfs = pd.read_html(url)
    l.append(pd.DataFrame(dfs[0]))

for i in range(31):
    l[i].to_csv('game_result{}.csv'.format(i),index=False,encoding='utf-8-sig')