import os
import pandas as pd 
#import modin.pandas as pd
from tqdm import tqdm
import datetime
from p_tqdm import p_map
from functools import partial
import gc

lk_path = '/mnt/data0/lkyin/'

# ------------------- processing commits ---------------------- 
print('reading commits...')
df_data = pd.read_csv(lk_path+'commits_final.csv')
df_data = df_data.loc[df_data.date.str.len()==19]
df_data['week'] = df_data['date'].apply(lambda x: pd.Period(x, freq='W-MON'))
print('grouping by project...')
df = dict(tuple(df_data.groupby(df_data['project_name'])))
to_path = './weekly_data/commits/'
if not os.path.exists(to_path):
	os.makedirs(to_path)

print('grouping by period...')
for project in tqdm(df):
	weekly_df_dict = dict(tuple(df[project].groupby(df[project]['week'])))
	weekint = 0
	for week in weekly_df_dict:
		weekly_df = weekly_df_dict[week]
		weekly_df = weekly_df[weekly_df['dealised_author_full_name'].notna()]
		weekint+=1
		if weekly_df.empty: continue
		weekstr = str(weekint).zfill(3)
		file_path = to_path + '{}__{}.csv'.format(project, weekstr)
		weekly_df.to_csv(file_path, index=False)
	gc.collect()
print('Commits Done.')
