from argparse import ArgumentParser
from pathlib import Path

import click
from glob import glob
import imageio
import numpy as np
import os
import pandas as pd

from preprocessing.preprocessing_utils import bb_intersection_over_union, join, crop_bbox_from_frames, save

REF_FPS = 25


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--iou_with_initial", type=float, default=0.25, help="The minimal allowed iou with inital bbox")
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")
    parser.add_argument("--increase", default=0.1, type=float, help='Increase bbox by this amount')
    parser.add_argument("--min_frames", default=64, type=int, help='Mimimal number of frames')
    parser.add_argument("--max_frames", default=1024, type=int, help='Maximal number of frames')
    parser.add_argument("--min_size", default=256, type=int, help='Minimal allowed size')
    parser.add_argument("--format", default='.mp4', help='Store format (.png, .mp4)')

    parser.add_argument("--annotations_folder", default='txt', help='Path to utterance annotations')

    parser.add_argument("--chunk_folder", default='chunks', help="Path to folder with video chunks")
    parser.add_argument("--bbox_folder", default='bbox', help="Path to folder with bboxes")
    parser.add_argument("--out_folder", default='vox-png', help='Folder for processed dataset')
    parser.add_argument("--chunks_metadata", default='vox-metadata.csv', help='File with metadata')

    parser.add_argument("--data_range", default=(0, 10000), type=lambda x: tuple(map(int, x.split('-'))),
                        help="Range of ids for processing")

    return parser.parse_args()


@click.command()
@click.option('--parquet_dir', prompt='Path to parquet directory', help='Path to parquet directory.')
@click.option('--video_dir', prompt='Path to video directory', help='Path to video directory.')
def main(parquet_dir, video_dir):
    args = parse_args()
    parquets = glob(f"{parquet_dir}/*.parquet")
    for pq in parquets:
        crop_video(pq, video_dir, args)



def crop_video(pq, video_path, args):
    # utterance = video_path.split('#')[1]
    #bbox_path = os.path.join(args.bbox_folder, os.path.basename(video_path)[:-4] + '.txt')
    reader = imageio.get_reader(video_path)

    chunk_start = float(video_path.split('#')[2].split('-')[0])

    d = pd.read_parquet(pq)
    video_count = 0
    initial_bbox = None
    start = 0
    tube_bbox = None
    frame_list = []
    chunks_data = []
    frame_number = 0
    video = ""
    try:
        for row in d.itertuples():

            #get frame, bbox, and video from row
            bbox = row[0]
            frame_number = row[3]
            if video != row[2]:
                video = row[2]
                reader = imageio.get_reader(os.path.join(video_path, video))

            reader.set_image_index(frame_number - 1)
            frame = reader.get_next_data()[:, :, ::-1]

            if initial_bbox is None:
                initial_bbox = bbox
                start = frame_number
                tube_bbox = bbox

            if bb_intersection_over_union(initial_bbox, bbox) < args.iou_with_initial or len(
                    frame_list) >= args.max_frames:
                chunks_data += store(frame_list, tube_bbox, Path(video).with_suffix(''), start, frame_number, video_count,
                                     chunk_start,
                                     args)
                video_count += 1
                initial_bbox = bbox
                start = frame_number
                tube_bbox = bbox
                frame_list = []
            tube_bbox = join(tube_bbox, bbox)
            frame_list.append(frame)
    except IndexError as e:
        None

    chunks_data += store(frame_list, tube_bbox, Path(video).with_suffix(''), start, frame_number + 1, video_count, chunk_start,
                         args)

    return chunks_data


def store(frame_list, tube_bbox, video, start, end, video_count, chunk_start, args):
    out, final_bbox = crop_bbox_from_frames(frame_list, tube_bbox, min_frames=args.min_frames,
                                            image_shape=args.image_shape, min_size=args.min_size,
                                            increase_area=args.increase)
    if out is None:
        return []

    start += round(chunk_start * REF_FPS)
    end += round(chunk_start * REF_FPS)
    name = f"{video}#{str(video_count).zfill(3)}.mp4"
    partition = 'train' #'test' if person_id in TEST_PERSONS else 'train'
    save(os.path.join(args.out_folder, partition, name), out, args.format)
    return [{'bbox': '-'.join(map(str, final_bbox)), 'start': start, 'end': end, 'fps': REF_FPS,
             'video': video, 'height': frame_list[0].shape[0],
             'width': frame_list[0].shape[1], 'partition': partition}]


if __name__ == '__main__':
    main()