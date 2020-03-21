# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:49:37 2020

@author: Marko Pejić
"""


#%% Imports
import pandas as pd

#%% Classes
class YTVideoText:
    def __init__(self, df_row):
        self.title = df_row.title
        self.description = df_row.description
        self.category_id = df_row.category_id
        
    def to_dict(self):
        return {
            'title': self.title,
            'description': self.description,
            'category_id': self.category_id
        }
    
    
#%% Data Loading
df = pd.read_csv('../../YTKaggleData/USvideos.csv')
df.head()

videos = dict()

# Load every single video just one time (in this Kaggle dataset there are multiple samples of same video)
for index, row in df.iterrows():
    if row.title not in videos:
        videos[row.title] = YTVideoText(row)

print('Length of original dataset: {}'.format(df.shape[0]))        
print('Number of unique videos: {}'.format(len(videos)))

#%% Create dataframe of generated dictinary
video_text_df = pd.DataFrame([video.to_dict() for video in videos.values()])
video_text_df.head()

set(video_text_df['category_id'])

# Remove videos with category_id == 43 (just 4 videos, probably some mistake)
video_text_df = video_text_df[video_text_df['category_id'] != 43]
print('Length of processed dataset: {}'.format(len(video_text_df)))

#%% Add category_name feature into dataframe
category_df = pd.read_csv('../../YTKaggleData/Categories.csv')

def map_category_id_to_name(category_id):
    return category_df[category_df['category_id'] == category_id]['category_name'].values[0]

video_text_df['category_name'] = video_text_df['category_id'].apply(lambda var: map_category_id_to_name(var))
video_text_df.head()

#%% Export dataframe into pickle file
video_text_df.to_pickle('../US_trending_kaggle.pkl')
