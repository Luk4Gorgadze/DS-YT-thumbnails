#!/usr/bin/env python
# coding: utf-8

import math

import isodate
import numpy as np
import pandas as pd
import seaborn as sns
from googleapiclient.discovery import build

api_key = 'AIzaSyDwzGJkjNRBNkVQUbdNoPaV82r9WkH5gO4'
channel_ids = [
    'UCXuqSBlHAE6Xw-yeJA0Tunw',  # LTT
]

youtube = build('youtube', 'v3', developerKey=api_key)


def get_channel_stats(youtube, channel_ids):
    all_data = []
    request = youtube.channels().list(
        part='snippet,contentDetails,statistics',
        id=','.join(channel_ids),
    )
    response = request.execute()

    for i in range(len(response['items'])):
        data = dict(
            Channel_name=response['items'][i]['snippet']['title'],
            Subscribers=response['items'][i]['statistics']['subscriberCount'],
            Views=response['items'][i]['statistics']['viewCount'],
            Total_videos=response['items'][i]['statistics']['videoCount'],
            playlist_id=response['items'][i]['contentDetails']
            ['relatedPlaylists']['uploads'],
        )
        all_data.append(data)

    return all_data


channel_statistics = get_channel_stats(youtube, channel_ids)

channel_data = pd.DataFrame(channel_statistics)

channel_data['Subscribers'] = pd.to_numeric(channel_data['Subscribers'])
channel_data['Views'] = pd.to_numeric(channel_data['Views'])
channel_data['Total_videos'] = pd.to_numeric(channel_data['Total_videos'])
channel_data['Click_rate'] = np.log1p(channel_data['Views']
                                     ) / np.log1p(channel_data['Subscribers'])
channel_data.dtypes

sns.set(rc={'figure.figsize': (10, 8)})
ax = sns.barplot(x='Channel_name', y='Subscribers', data=channel_data)
ax = sns.barplot(x='Channel_name', y='Views', data=channel_data)
ax = sns.barplot(x='Channel_name', y='Total_videos', data=channel_data)

playlist_id = channel_data.loc[0].playlist_id


def get_video_ids(youtube, playlist_id):
    request = youtube.playlistItems().list(
        part='contentDetails',
        playlistId=playlist_id,
        maxResults=50,
    )
    response = request.execute()

    video_ids = []
    for i in range(len(response['items'])):
        video_ids.append(response['items'][i]['contentDetails']['videoId'])

    next_page_token = response.get('nextPageToken')
    more_pages = True

    while more_pages:
        if next_page_token is None:
            more_pages = False
        else:
            request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token,
            )
            response = request.execute()

            for i in range(len(response['items'])):
                video_ids.append(
                    response['items'][i]['contentDetails']['videoId']
                )

            next_page_token = response.get('nextPageToken')

    return video_ids


video_ids = get_video_ids(youtube, playlist_id)


def get_video_details(youtube, video_ids):
    all_video_stats = []

    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part='snippet,statistics,contentDetails',
            id=','.join(video_ids[i:i + 50]),
        )
        response = request.execute()

        for video in response['items']:
            duration_seconds = isodate.parse_duration(
                video['contentDetails'].get('duration', None)
            ).total_seconds()

            video_stats = dict(
                ID=video['id'],
                Title=video['snippet']['title'],
                Thumbnail=video['snippet']['thumbnails']['default']['url'],
                Language=video['snippet']['defaultLanguage'],
                Duration_minutes=math.ceil(duration_seconds / 60),
                Likes=video['statistics']['likeCount'],
                Views=video['statistics']['viewCount'],
                Published_date=video['snippet']['publishedAt'],
            )
            all_video_stats.append(video_stats)

    return all_video_stats


video_ids = video_ids[:50]
video_details = get_video_details(youtube, video_ids)

video_data = pd.DataFrame(video_details)

video_data['Published_date'] = pd.to_datetime(
    video_data['Published_date']
).dt.date
video_data['Views'] = pd.to_numeric(video_data['Views'])
video_data['Likes'] = pd.to_numeric(video_data['Likes'])
video_data['Duration_minutes'] = pd.to_numeric(video_data['Duration_minutes'])
video_data['Click_rate'] = np.log1p(video_data['Views']) / np.log1p(
    channel_data['Subscribers'].iloc[0]
)

video_data.to_csv('channel_videos.csv', index=False)
