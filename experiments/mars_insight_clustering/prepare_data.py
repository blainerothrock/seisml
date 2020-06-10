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


DATA_PATH = os.path.expanduser('~/dev/seisml/experiments/mars_insight_clustering/.data/')

file_names = list(filter(lambda f: os.path.splitext(f)[1] == '.mseed', os.listdir(DATA_PATH + 'raw/')))
print('num of files: %i' % len(file_names))

streams = []
for file in file_names[]:
    stream = obspy.read(os.path.join(DATA_PATH + 'raw/', file)).merge(method=1, fill_value='interpolate')
    if len(stream) != 3:
        print('not 3')
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
        file_name = '.data/prepared/{}.mseed'.format(str(tmp_start).replace(':', '-'))
        try:
            new_st.write(file_name, format='MSEED')
        except:
            pass

        tmp_start = tmp_start + 3
        tmp_end = tmp_start + 12
        pbar.update(1)

    pbar.close()