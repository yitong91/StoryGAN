import os
import csv
import shutil

import numpy as np


def raw_to_book_info(raw_csv_path, new_csv_path):
    raw_csv_file = open(raw_csv_path, 'r')
    reader = csv.reader(raw_csv_file)
    next(reader)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + '/' + new_csv_path, 'w') as new_csv_file:
        writer = csv.writer(new_csv_file)
        prev_book = 0
        page_idx = 0
        for row in reader:
            img_name = row[2]         
            try:
                curr_book = int(img_name.split('-')[0])
            except Exception as err:
                print('{}:{}'.format(err, row))
                raise
            if prev_book != curr_book:
                page_idx = 0
            _, ext = os.path.splitext(img_name)
            desc = " ".join([s if s.strip() else s for s in row[8].replace('\n', ' ').splitlines(True)])       
            real_img_name = '{}_{}{}'.format(curr_book, page_idx, ext)
            writer.writerow([curr_book, page_idx, desc, real_img_name])

            page_idx += 1
            prev_book = curr_book            
    raw_csv_file.close()

def rows_to_stories(csv_path, new_csv_path, img_dir, video_len=4):
    '''
    From csv file format of [curr_book, page_idx, desc, real_img_name]
    To csv file format of [story_id, scene_id, img_key_name, desc, real_img_name]
    '''
    csv_file = open(csv_path, 'r')
    reader = csv.reader(csv_file)
    dir_path = os.path.dirname(os.path.realpath(__file__))

    prev_book, curr_book = 0, 0
    story_id = 0
    scene_id = 0

    with open(new_csv_path, 'w') as new_csv_file:    
        writer = csv.writer(new_csv_file)
        while True:
            row = next(reader, None)
            if not row:
                break

            curr_book, desc, real_img_name = row[0], row[2], row[3]

            # Sanity check for images
            if not os.path.exists(os.path.join(img_dir, real_img_name)):
                if os.path.exists(os.path.join(img_dir, real_img_name.replace('.jpg', '.png'))):
                    real_img_name = real_img_name.replace('.jpg', '.png')
                elif os.path.exists(os.path.join(img_dir, real_img_name.replace('.jpg', '.gif'))):
                    real_img_name = real_img_name.replace('.jpg', '.gif')
                else:
                    continue

            ext = os.path.splitext(real_img_name)[1]
            img_key_name = '{}_{}{}'.format(story_id, scene_id, ext)

            # Group image-annotation pairs
            if prev_book != curr_book:
                scenes = [ [story_id, 0, '{}_{}{}'.format(story_id, 0, ext), desc, real_img_name] ]
                scene_id = 1 
                prev_book = curr_book
                continue
            scenes.append( [story_id, scene_id, img_key_name, desc, real_img_name] )                         
            scene_id += 1               
            if scene_id == video_len:
                for i in range(video_len):
                    writer.writerow(scenes[i])
                scene_id = 0
                story_id += 1                     
                scenes = []
            prev_book = curr_book
    csv_file.close()

def filter_images(src_img_dir, dst_img_dir, csv_path):
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            story_id, scene_id, img_name = row[0], row[1], row[3]
            _, ext = os.path.splitext(img_name)

            src_img_path = os.path.join(src_img_dir, img_name)
            dst_img_name = "{}_{}{}".format(story_id, scene_id, ext)            
            dst_img_path = os.path.join(dst_img_dir, dst_img_name)

            try:
                shutil.copy2(src_img_path, dst_img_path)
            except FileNotFoundError as err:
                print(err, 'src: ', img_name, '/ dst:', dst_img_name)


# Uncomment to use the functions above.
#raw_to_book_info('mini_guten_dataset/images_filtered.csv', 'mini_guten_dataset/books.csv')
# rows_to_stories('mini_guten_dataset/books.csv',
#                 'mini_guten_dataset/stories_paired.csv',
#                 '/Users/eunjeesung/work/miniGutenStories/images',
#                 4)
# filter_images('mini_guten_dataset/images',
#             'mini_guten_dataset/images_grouped',
#             'mini_guten_dataset/stories_paired.csv')