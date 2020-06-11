import csv, json, os, sys
import requests
from time import sleep

import obspy
from obspy import read
from tqdm import tqdm

from obspy.core.event import read_events
from obspy import read_inventory
from datetime import datetime, timedelta
from obspy.core import UTCDateTime

def split_data(raw_dir, save_dir, stride=3, length=12, starttime=None, endtime=None):
    file_names = list(filter(lambda f: os.path.splitext(f)[1] == '.mseed', os.listdir(raw_dir)))
    print('num of files: %i' % len(file_names))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    streams = []
    for file in file_names:
        stream = obspy.read(os.path.join(raw_dir, file))
        stream = stream.merge(method=0, fill_value=0)

        include = True
        for trace in stream:
            if trace.stats.starttime < starttime:
                include = False
            if trace.stats.endtime > endtime:
                include = False

        if not include:
            continue

        # only include streams with 3 channels
        if len(stream) != 3:
            continue

        streams.append(stream)

    for i, stream in enumerate(streams):
        print('------')
        print('processing {} of {}'.format(i+1, len(streams)))
        print(stream)

        # common start
        starts = [t.stats.starttime for t in stream]
        start = max(starts)

        # common end
        ends = [t.stats.endtime for t in stream]
        end = min(ends)

        pbar = tqdm(total=int((end-start)/3))

        tmp_start = start
        tmp_end = start + 12
        while tmp_end <= end:
            traces = []
            for trace in stream:
                tmp_t = trace.slice(tmp_start, tmp_end)
                traces.append(tmp_t)

            new_st = obspy.core.Stream(traces)
            file_path = os.path.join(save_dir, '{}.mseed'.format(str(tmp_start).replace(':', '-')))
            new_st.write(file_path, format='MSEED')

            tmp_start = tmp_start + stride
            tmp_end = tmp_start + length
            pbar.update(1)

        pbar.close()


if __name__ == '__main__':
    split_data(
        raw_dir=os.path.expanduser('~/.seisml/mars/all_BH/raw'),
        save_dir=os.path.expanduser('~/.seisml/mars/all_BH/prepared_12-3_oct-dec'),
        stride=3,
        length=12,
        starttime=obspy.UTCDateTime('2019-10-01T00:00:00'),
        endtime=obspy.UTCDateTime('2019-12-31T23:59:59')
    )
