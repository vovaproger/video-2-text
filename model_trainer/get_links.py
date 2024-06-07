import json
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import isodate

# Replace with your API key and the channel ID
API_KEY = 'AIzaSyCusAp2X-7HcDvqP9c-kxJkYFt5Vx7uTnI'
CHANNEL_ID = 'UClHVl2N3jPEbkNJVx-ItQIQ'

def get_most_popular_video_links(channel_id, max_results=100):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    video_links = []

    try:
        # Retrieve the most popular videos from the channel
        next_page_token = None
        while len(video_links) < max_results:
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

            for item in video_details_response['items']:
                if isodate.parse_duration(item['contentDetails']['duration']).total_seconds() > 60:
                    video_links.append(f"https://www.youtube.com/watch?v={item['id']}")
                    if len(video_links) >= max_results:
                        break

            next_page_token = search_response.get('nextPageToken')
            if not next_page_token:
                break
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred: {e.content.decode('utf-8')}")

    return video_links

def save_links_to_json(links, filename='video_links.json'):
    with open(filename, 'w') as file:
        json.dump(links, file, indent=4)

if __name__ == "__main__":
    video_links = get_most_popular_video_links(CHANNEL_ID, max_results=120)
    if video_links:
        save_links_to_json(video_links)
        print(f"Saved {len(video_links)} video links to video_links.json")
    else:
        print("No video links found.")
