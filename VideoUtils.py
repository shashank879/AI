import cv2
import re
from datetime import datetime, timedelta

def time_in_range(start, end, x):
    """Return true if x is in the range [start, end]"""
    if start <= end:
        return start <= x <= end
    else:
        return False

def mm_ss_exclusions(file_path):
    """Exclude ranges annotated in a file. """

    regex = '(\d+:\d+) - (\d+:\d+)'
    time_format = '%M:%S'
    zero = datetime.strptime('00:00', time_format)
    ranges = []
    with open(file_path) as infile:
        for line in infile:
            (st_time, en_time) = re.findall(regex, line)[0]
            st_time = datetime.strptime(st_time, time_format) - zero
            en_time = datetime.strptime(en_time, time_format) - zero
            ranges.append([st_time.total_seconds()*1000, en_time.total_seconds()*1000])

    return ranges

class VideoLoader:
    """Class to aid loading of video frames in batches as training, test and validation data"""

    def __init__(self, vid_path, batch_size=20, exclude_ranges=None):
        self.vid_path = vid_path
        self.batch_size = batch_size
        self.vidcap = None
        self.exclude_ranges = exclude_ranges
        self.range_to_check = 0

    def __enter__(self):
        print("Opening video")
        self.vidcap = cv2.VideoCapture(self.vid_path)
        self.range_to_check = 0
        print("Video opened")

    def __exit__(self, exception_type, exception_value, traceback):
        print("Closing video")
        self.vidcap.release()

    def fetch_next_batch(self):
        """Returns the next batch of frames with the specified batch_size."""

        frames = []
        i = 0
        while self.vidcap.isOpened() and i < self.batch_size:
            curr_time = self.vidcap.get(cv2.CAP_PROP_POS_MSEC)
            next_range = self.exclude_ranges[self.range_to_check]

            # Check if we are going to check the correct range
            if self.range_to_check < len(self.exclude_ranges) and \
               curr_time > next_range[1]:
                self.range_to_check += 1
                continue

            _, frame = self.vidcap.read()

            # Check if this section is to be skipped, if so clear all we have in this batch,
            # this will happen till we are in a valid range and
            # the batch does not intersect an invalid range at all
            if time_in_range(next_range[0], next_range[1], curr_time):
                frames = []
                i = 0
                print("Skipping frame...", curr_time)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            cv2.imshow('frames', gray)
            i += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        return frames

exclude_ranges = mm_ss_exclusions('./data/traffic_junction/Annotations_cleaned.txt')
vl = VideoLoader(vid_path='./data/traffic_junction/traffic-junction.avi', exclude_ranges=exclude_ranges)

with vl:
    while True:
        batch = vl.fetch_next_batch()
        if len(batch) is 0:
            break