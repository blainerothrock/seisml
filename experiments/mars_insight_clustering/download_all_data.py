import csv, json, os, sys
import requests
from time import sleep

import obspy
from obspy import read

from obspy.core.event import read_events
from obspy import read_inventory
from datetime import datetime, timedelta
from obspy.core import UTCDateTime


DATA_PATH = os.path.expanduser('~/dev/seisml/experiments/mars_insight_clustering/.data/')


def download_data_availability(data_path=DATA_PATH):
    try:
        os.makedirs(data_path)
    except FileExistsError:
        pass

    payload = {
        'option': 'com_ajax',
        'data': 'ELYSE',
        'format': 'json',
        'module': 'seis_data_available'
    }
    r = requests.post('https://www.seis-insight.eu/en/science/seis-data/seis-data-availability', payload)
    with open(os.path.join(data_path, 'data_availability.json'), 'wb') as f:
        f.write(r.content)

def split_availability(path):
    with open(os.path.join(path, 'data_availability.json'), 'r') as f:
        raw_ava = json.load(f)

    ava = []
    for t in raw_ava['data']:
        if t['network'] == 'XB' and t['location'] == '02' and t['channel'] == 'BHU':
            ava.append(t)

    with open(os.path.join(path, 'catelog.json'), 'w') as f:
        json.dump(ava, f)

    return ava


def download_mseed(event, channel='BH?', data_path='.'):
    try:
        os.makedirs(data_path)
    except FileExistsError:
        pass

    payload = {
        'network': event['network'],
        'station': event['station'],
        'startTime': event['startTime'],
        'endTime': event['endTime'],
        'location': event['location'],
        'channel': channel
    }

    req = requests.get('http://ws.ipgp.fr/fdsnws/dataselect/1/query', params=payload)
    file_name = '-'.join(
        [event['network'], event['station'], event['location'], event['startTime'], event['endTime']]) + '.mseed'
    print('downloading: %s' % file_name)
    path = os.path.join(data_path, file_name)
    with open(path, 'wb') as c:
        c.write(req.content)
    return path


if __name__ == '__main__':
    path = os.path.expanduser('~/.seisml/mars/all_BH/')
    download_data_availability(path)
    ava = split_availability(path)
    for event in ava:
        download_mseed(event, data_path=os.path.join(path, 'raw/'))