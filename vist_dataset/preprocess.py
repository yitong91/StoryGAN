import json
import argparse

import numpy as np
import nltk

from collections import defaultdict
from nltk.tokenize import word_tokenize


class AnnotationPreprocessing():
    def __init__(self, src_path):
        self.src_path = src_path
        with open(self.src_path) as src_file:
            self.src_json = json.load(src_file)

    def extract_stories(self):
        annotations = self.src_json['annotations']
        stories = defaultdict(list)
        for annotation in annotations:
            data = annotation[0]
            story_id, storylet_id, image_id = \
                data["story_id"], data["storylet_id"], data["photo_flickr_id"]
            stories[story_id].append({"storylet_id": storylet_id, "image_id": image_id})
        return stories

    def arr_to_json(self, arr, dst_path):
        with open(dst_path, 'w') as dst_file:
            json.dump(arr, dst_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', dest='src_path', type=str, default='train.story-in-sequence.json')
    parser.add_argument('--dst', dest='dsc_path', type=str, default='train.stories.json')
    args = parser.parse_args()

    prep = AnnotationPreprocessing(args.src_path)
    stories = prep.extract_stories()
    prep.arr_to_json(stories, args.dsc_path)
