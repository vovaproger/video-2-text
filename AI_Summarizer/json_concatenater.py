from googleapiclient.discovery import build
import json

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

API_KEY = 'AIzaSyCusAp2X-7HcDvqP9c-kxJkYFt5Vx7uTnI'
youtube = build('youtube', 'v3', developerKey=API_KEY)

path_ids = f'/Users/uladzimircharniauski/Documents/GetBetterHead/video_ids.json'
path_text = f'/Users/uladzimircharniauski/Documents/GetBetterHead/youtube_texts.json'

def concatenater():

    json_list = []

    f_ids = json.load(open(path_ids))
    f_text = json.load(open(path_text))
    for id, text_item in zip(f_ids, f_text):
        video_name = get_video_title(id["videoId"])
        json_list.append({
            'name': video_name,
            'context': text_item
        })

    return json_list

def get_video_title(video_id):
    request = youtube.videos().list(
        part="snippet",
        id=video_id
    )
    response = request.execute()

    if "items" in response and response["items"]:
        video_title = response["items"][0]["snippet"]["title"]
        return video_title
    else:
        return "Video not found"
    
def save_to_json(video_list, filename='text_database.json'):
     with open(filename, 'w') as file:
        json.dump(video_list, file, indent=4)

if __name__ == "__main__":
    list = concatenater()
    save_to_json(list)
    print("Database created")