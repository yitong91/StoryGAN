import json
import argparse
from collections import defaultdict

def extract_story_info(src_path, dst_path):
    with open(src_path) as src_file:
        annotations = json.load(src_file)['annotations']
        stories = defaultdict(list)
        for annotation in annotations:
            data = annotation[0]
            story_id, storylet_id, image_id = \
                data["story_id"], data["storylet_id"], data["photo_flickr_id"]
            stories[story_id].append({"storylet_id": storylet_id, "image_id": image_id})

    with open(dst_path, 'w') as dst_file:
        json.dump(stories, dst_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', action='store', type=str, default='train.story-in-sequence.json')
    parser.add_argument('--dst', action='store', type=str, default='train.stories.json')
    args = parser.parse_args()

    extract_story_info(args.src, args.dst)