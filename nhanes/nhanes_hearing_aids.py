#%%
import pandas as pd
import os.path as op
import sys
sys.path.append('./utils/')

from nhanes_utils import pull_nhanes, merge_datasets

#%% Download nhanes data example
outdir = '../data/nhanes_data'
year = 2015
sub_dirs = ['Demographics'] #'Examination', 'Questionnaire', 'Laboratory', 'Dietary',

df_dems = pull_nhanes(2015, outdir=None, sub_dirs=sub_dirs)

#%%
data_files = [ 'AUQ_I', 'AUX_I', 'DEMO_I']
indir = outdir
df = merge_datasets(indir=indir, data_files=data_files)


col_list = ['AUQ191', 'AUQ054', 'RIAGENDR', 'RIDAGEYR', 'AUD148',
            'AUXU1K1R', 'AUXU500R', 'AUXU2KR', 'AUXU3KR', 'AUXU4KR',
            'AUXU6KR', 'AUXU8KR', 'AUXU1K1L', 'AUXU500L', 'AUXU2KL', 
            'AUXU3KL', 'AUXU4KL', 'AUXU6KL', 'AUXU8KL']

cur_df = df[col_list]


#%% drop uniformative columns (i.e. columns with mostly nans)
df['hearing_aid'] = [3 if np.isnan(val) else val for val in df['AUD148']]
nan_list = [False if df[col].isna().sum() > 9000 else True for col in df]


clean_df = df.loc[: , nan_list]
#%%
clean_df.dropna(inplace=True)
#%% Determine probability of possessing a hearing aid
import matplotlib.pyplot as plt
import seaborn as sns
from boruta import BorutaPy
import numpy as np

from sklearn.ensemble import RandomForestClassifier

y = clean_df['hearing_aid'] == 1
X = df.loc[:, df.columns != 'hearing_aid']

#select features
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# let's initialize Boruta
feat_selector = BorutaPy(
    verbose=2,
    estimator=model,
    n_estimators='auto',
    max_iter=200,
    random_state=42069,
)

feat_selector.fit(np.array(X), np.array(y))
# %%
