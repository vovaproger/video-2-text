import json
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import isodate

# Replace with your API key and the channel ID
API_KEY = 'AIzaSyCusAp2X-7HcDvqP9c-kxJkYFt5Vx7uTnI'
CHANNEL_ID = 'UClHVl2N3jPEbkNJVx-ItQIQ'

def get_most_popular_video_ids(channel_id, max_results=100):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    video_ids_new = []

    try:
        # Retrieve the most popular videos from the channel
        next_page_token = None
        while len(video_ids_new) < max_results:
            search_response = youtube.search().list(
                part='snippet',
                channelId=channel_id,
                maxResults=50,
                order='viewCount',  # Order by view count to get the most popular videos
                pageToken=next_page_token,
                type='video'
            ).execute()
            
            if 'items' not in search_response:
                break

            video_ids = [item['id']['videoId'] for item in search_response['items']]
            video_details_response = youtube.videos().list(
                part='contentDetails',
                id=','.join(video_ids)
            ).execute()

            #filtrating videos by duration lenght (more than 2 minutes)

            for item in video_details_response['items']:
                if isodate.parse_duration(item['contentDetails']['duration']).total_seconds() > 120:
                    video_ids_new.append({
                        'videoId': item['id'],
                    })
                    if len(video_ids_new) >= max_results:
                        break

            next_page_token = search_response.get('nextPageToken')
            if not next_page_token:
                break
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred: {e.content.decode('utf-8')}")

    return video_ids_new

def save_ids_to_json(ids, filename='video_ids.json'):
    with open(filename, 'w') as file:
        json.dump(ids, file, indent=4)

if __name__ == "__main__":
    video_links = get_most_popular_video_ids(CHANNEL_ID, max_results=120)
    if video_links:
        save_ids_to_json(video_links)
        print(f"Saved {len(video_links)} IDs to video_ids.json")
    else:
        print("No video ids found.")
