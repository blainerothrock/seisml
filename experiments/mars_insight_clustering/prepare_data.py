import csv, json, os, sys, time
import requests
from time import sleep
import gin
import concurrent.futures

import obspy

from obspy import read
from tqdm import tqdm

from obspy.core.event import read_events
from obspy import read_inventory
from datetime import datetime, timedelta
from obspy.core import UTCDateTime

def split_stream(file, raw_dir, starttime, endtime, save_dir, stride, length):

    stream = obspy.read(os.path.join(raw_dir, file))
    stream = stream.merge(method=0, fill_value=0)

    # only include streams with 3 channels
    if len(stream) != 3:
        return

    # common start
    starts = [t.stats.starttime for t in stream]

    max_start = max(starts)
    if max_start > endtime:
        return

    start = max_start if max_start >= starttime else starttime

    # common end
    ends = [t.stats.endtime for t in stream]

    max_end = max(ends)
    if max_end < starttime:
        return

    end = max_end if max_end < endtime else endtime

    tmp_start = start
    tmp_end = start + 12
    while tmp_end <= end:
        traces = []
        for trace in stream:
            tmp_t = trace.slice(tmp_start, tmp_end)
            traces.append(tmp_t)

        new_st = obspy.core.Stream(traces)
        file_path = os.path.join(save_dir, f'{str(tmp_start).replace(":", "-")}.mseed')
        new_st.write(file_path, format='MSEED')

        tmp_start = tmp_start + stride
        tmp_end = tmp_start + length

@gin.configurable()
def split_data(raw_dir, save_dir, stride=3, length=12, starttime=None, endtime=None):
    raw_dir = os.path.expanduser(raw_dir)
    save_dir = os.path.expanduser(save_dir)

    starttime = obspy.UTCDateTime(starttime)
    endtime = obspy.UTCDateTime(endtime)

    file_names = tuple(filter(lambda f: os.path.splitext(f)[1] == '.mseed', os.listdir(raw_dir)))
    print(f'num of files: {len(file_names)}')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pbar = tqdm(total=len(file_names))

    with concurrent.futures.ProcessPoolExecutor() as executor:

        futures_to_file = {
            executor.submit(
                split_stream,
                file,
                raw_dir,
                starttime,
                endtime,
                save_dir,
                stride,
                length
            ): file for file in file_names
        }

        for _ in concurrent.futures.as_completed(futures_to_file):
            pbar.update(1)

    pbar.close()


if __name__ == '__main__':
    gin.parse_config_file('config_prepare.gin')
    split_data()
