import math
import os

import isodate
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
API_KEY = os.getenv('YOUTUBE_API_KEY')

CHANNEL_IDS = [
    'UCXuqSBlHAE6Xw-yeJA0Tunw',  # LTT
    'UCX6OQ3DkcsbYNE6H8uQQuVA',  # MR Beast
    'UCBJycsmduvYEL83R_U4JriQ',  # mkbhd
    'UCokIq0eihRhkiqClyV0WCdw',  # KrisKrohn
    'UChIs72whgZI9w6d6FhwGGHA',  # GamersNexus
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
                'Channel_id':
                    item['id'],
                'Channel_name':
                    item['snippet']['title'],
                'Subscribers':
                    item['statistics']['subscriberCount'],
                'Views':
                    item['statistics']['viewCount'],
                'Total_videos':
                    item['statistics']['videoCount'],
                'Playlist_id':
                    item['contentDetails']['relatedPlaylists']['uploads'],
            }
            all_data.append(data)
    except Exception as e:
        print(f"Error fetching channel stats: {e}")

    return all_data


def get_video_ids(youtube, playlist_id, limit=5):
    """Retrieve video IDs from the specified playlist."""
    video_ids = []
    next_page_token = None

    while len(video_ids) < limit:
        request = youtube.playlistItems().list(
            part='contentDetails',
            playlistId=playlist_id,
            maxResults=min(50, limit - len(video_ids)),
            pageToken=next_page_token,
        )
        response = request.execute()

        video_ids.extend(
            item['contentDetails']['videoId'] for item in response['items']
        )

        next_page_token = response.get('nextPageToken')
        if not next_page_token or len(video_ids) >= limit:
            break

    return video_ids[:limit]


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
                        get_best_thumbnail(video['snippet']['thumbnails']),
                    'Language':
                        get_supported_language(video),
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


def get_best_thumbnail(thumbnails):
    """Select the best available thumbnail in order of resolution."""
    return (
        thumbnails.get('maxres', {}).get('url') or
        thumbnails.get('standard', {}).get('url') or
        thumbnails.get('high', {}).get('url') or
        thumbnails.get('medium', {}).get('url') or
        thumbnails.get('default', {}).get('url')
    )


def get_supported_language(video):
    default_language = video['snippet'].get('defaultLanguage', None)

    return default_language if default_language else video['snippet'].get(
        'defaultAudioLanguage', None
    )


def get_channel_video_data(youtube, channel_data):
    """Fetch video details for each channel"""
    all_details = []

    for _, channel in channel_data.iterrows():
        video_ids = get_video_ids(youtube, channel['Playlist_id'])

        video_details = get_video_details(youtube, video_ids)
        video_data = pd.DataFrame(video_details)

        video_data.insert(0, 'Subscribers', channel['Subscribers'])
        video_data.insert(0, 'Channel_name', channel['Channel_name'])
        video_data.insert(0, 'Channel_id', channel['Channel_id'])

        video_data['Published_date'] = pd.to_datetime(
            video_data['Published_date']
        ).dt.date
        video_data['Views'] = pd.to_numeric(video_data['Views'])
        video_data['Likes'] = pd.to_numeric(video_data['Likes'])
        video_data['Duration_minutes'] = pd.to_numeric(
            video_data['Duration_minutes']
        )

        video_data['Click_rate'] = np.log1p(video_data['Views']) / np.log1p(
            channel['Subscribers']
        )

        all_details.append(video_data)

    combined_video_data = pd.concat(all_details, ignore_index=True)

    return combined_video_data


def main():
    """Main function to execute the analysis."""
    youtube = build_youtube_service(API_KEY)

    channel_statistics = get_channel_stats(youtube, CHANNEL_IDS)
    channel_data = pd.DataFrame(channel_statistics)

    channel_data['Subscribers'] = pd.to_numeric(channel_data['Subscribers'])
    channel_data['Views'] = pd.to_numeric(channel_data['Views'])
    channel_data['Total_videos'] = pd.to_numeric(channel_data['Total_videos'])

    channel_video_data = get_channel_video_data(youtube, channel_data)

    channel_video_data.to_csv('storage/channel_videos.csv', index=False)


if __name__ == "__main__":
    main()
