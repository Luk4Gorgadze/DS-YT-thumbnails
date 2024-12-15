#!/usr/bin/env python
# coding: utf-8

import math
import os

import isodate
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from googleapiclient.discovery import build

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv('YOUTUBE_API_KEY')

CHANNEL_IDS = [
    'UCXuqSBlHAE6Xw-yeJA0Tunw',  # LTT
]


def build_youtube_service(api_key):
    """Build the YouTube service client."""
    return build('youtube', 'v3', developerKey=api_key)


def get_channel_stats(youtube, channel_ids):
    """Fetch channel statistics for the given channel IDs."""
    all_data = []
    try:
        request = youtube.channels().list(
            part='snippet,contentDetails,statistics',
            id=','.join(channel_ids),
        )
        response = request.execute()

        for item in response['items']:
            data = {
                'Channel_name':
                    item['snippet']['title'],
                'Subscribers':
                    item['statistics']['subscriberCount'],
                'Views':
                    item['statistics']['viewCount'],
                'Total_videos':
                    item['statistics']['videoCount'],
                'playlist_id':
                    item['contentDetails']['relatedPlaylists']['uploads'],
            }
            all_data.append(data)
    except Exception as e:
        print(f"Error fetching channel stats: {e}")

    return all_data


def get_video_ids(youtube, playlist_id):
    """Retrieve video IDs from the specified playlist."""
    video_ids = []
    next_page_token = None

    while True:
        try:
            request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token,
            )
            response = request.execute()

            video_ids.extend(
                item['contentDetails']['videoId'] for item in response['items']
            )
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        except Exception as e:
            print(f"Error fetching video IDs: {e}")
            break

    return video_ids


def get_video_details(youtube, video_ids):
    """Fetch details for the specified video IDs."""
    all_video_stats = []

    for i in range(0, len(video_ids), 50):
        try:
            request = youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=','.join(video_ids[i:i + 50]),
            )
            response = request.execute()

            for video in response['items']:
                duration_seconds = isodate.parse_duration(
                    video['contentDetails'].get('duration', None)
                ).total_seconds()
                video_stats = {
                    'ID':
                        video['id'],
                    'Title':
                        video['snippet']['title'],
                    'Thumbnail':
                        video['snippet']['thumbnails']['default']['url'],
                    'Language':
                        video['snippet'].get('defaultLanguage', None),
                    'Duration_minutes':
                        math.ceil(duration_seconds / 60),
                    'Likes':
                        video['statistics'].get('likeCount', None),
                    'Views':
                        video['statistics'].get('viewCount', None),
                    'Published_date':
                        video['snippet']['publishedAt'],
                }
                all_video_stats.append(video_stats)
        except Exception as e:
            print(f"Error fetching video details: {e}")

    return all_video_stats


def main():
    """Main function to execute the analysis."""
    youtube = build_youtube_service(API_KEY)

    # Get channel statistics
    channel_statistics = get_channel_stats(youtube, CHANNEL_IDS)
    channel_data = pd.DataFrame(channel_statistics)

    # Convert data types
    channel_data['Subscribers'] = pd.to_numeric(channel_data['Subscribers'])
    channel_data['Views'] = pd.to_numeric(channel_data['Views'])
    channel_data['Total_videos'] = pd.to_numeric(channel_data['Total_videos'])

    # Calculate Click_rate
    channel_data['Click_rate'] = np.log1p(channel_data['Views']) / np.log1p(
        channel_data['Subscribers']
    )

    # Get video IDs
    playlist_id = channel_data.loc[0, 'playlist_id']
    video_ids = get_video_ids(youtube, playlist_id)

    # Get video details
    video_details = get_video_details(youtube, video_ids)
    video_data = pd.DataFrame(video_details)

    # Process video data
    video_data['Published_date'] = pd.to_datetime(
        video_data['Published_date']
    ).dt.date
    video_data['Views'] = pd.to_numeric(video_data['Views'])
    video_data['Likes'] = pd.to_numeric(video_data['Likes'])
    video_data['Duration_minutes'] = pd.to_numeric(
        video_data['Duration_minutes']
    )

    # Calculate Click_rate for each video
    video_data['Click_rate'] = np.log1p(video_data['Views']) / np.log1p(
        channel_data['Subscribers'].iloc[0]
    )

    # Save to CSV
    video_data.to_csv('storage/channel_videos.csv', index=False)


if __name__ == "__main__":
    main()
