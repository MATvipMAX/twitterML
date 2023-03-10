import pandas as pd


DIR = 'C:/Users/Bright/Downloads/'


con = pd.read_csv(DIR + 'LIWC2015 Results (final_replace_control0220.csv).csv', encoding='utf8')
dep = pd.read_csv(DIR + 'LIWC2015 Results (final_replace_depress0220.csv).csv', encoding='utf8')

con = con[['A', 'B', 'i',	'we', 'you', 'shehe', 'they', 'ipron']].copy()
dep = dep[['A', 'B', 'i',	'we', 'you', 'shehe', 'they', 'ipron']].copy()

con.columns = ['tweetid', 'userid', 'i',	'we', 'you', 'shehe', 'they', 'ipron']
dep.columns = ['tweetid', 'userid', 'i',	'we', 'you', 'shehe', 'they', 'ipron']

combine = pd.concat([con, dep])

combine = combine.reset_index(drop=True)

combine.to_csv('../data/LIWC2015-pron.csv', encoding='utf8', index=False)
