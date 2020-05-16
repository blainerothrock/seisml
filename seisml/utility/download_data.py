import os, shutil
import tarfile
import requests
import hashlib
from enum import Enum
from tqdm import tqdm


class DownloadableData(str, Enum):
    SAMPLE_DATA = 'triggered_earthquake_sample_data'
    TRIGGERED_EARTHQUAKE = 'triggered_earthquakes'


def downloadable_data_path(downloadable_data):
    if downloadable_data == DownloadableData.SAMPLE_DATA:
        return 'https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/triggered_earthquake_sample_data.tar.gz'
    if downloadable_data == DownloadableData.TRIGGERED_EARTHQUAKE:
        return 'https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/triggered_earthquakes.tar.gz'


DATA_PATH = os.path.expanduser('~/.seisml/data/')


def download_and_verify(name, download_path, force=False):
    print('downloading {}: '.format(name))
    if os.path.isdir(DATA_PATH + name) and not force:
        return
    if os.path.isdir(DATA_PATH + name) and force:
        print('  - removing existing sample_data')
        shutil.rmtree(DATA_PATH + 'sample_data')

    url = download_path
    target_path = os.path.join(DATA_PATH, '{}.tar.gz'.format(name))

    print('  - downloading')
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(res.raw.read())

        print('  - validating checksum')
        with open(target_path, 'rb') as f:
            data = f.read()
            md5_to_check = hashlib.md5(data).hexdigest()

        with open(os.path.join('seisml/utility/checksums', '{}.md5'.format(name)), 'r') as f:
            data = f.readlines()[0]
            data = data.split('  ')
            md5 = data[0]

        if md5 == md5_to_check:
            print('  - checksum verified')
        else:
            print('  - checksum invalid, aborting')
            os.remove(target_path)
            return

        print('  - extracting sample data')
        tf = tarfile.open(target_path, 'r:gz')
        tf.extractall(DATA_PATH)
        tf.close()
        os.remove(target_path)


# def download_triggered_earthquake_data(force=False):
#     print('downloading triggered earthquake data: ')
#     if os.path.isdir(DATA_PATH + 'triggered_earthquake') and not force:
#         return
#     if os.path.isdir(DATA_PATH + 'triggered_earthquake') and force:
#         print('  - removing existing triggered_earthquake')
#         shutil.rmtree(DATA_PATH + 'triggered_earthquake')
#
#     url = 'https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/triggered_earthquakes.tar.gz'
#
#     file_path = os.path.join(DATA_PATH, 'triggered_earthquake.tar.gz')
#     with open(file_path, 'wb') as f:
#         print('  - downloading %s' % file_path)
#         res = requests.get(url, stream=True)
#         total_size = int(res.headers.get('Content-Length'))
#
#         pbar = tqdm(
#             total=total_size,
#             initial=0,
#             unit='B',
#             unit_scale=True,
#             desc=file_path.split('/')[-1]
#         )
#
#         if total_size is None:
#             f.write(res.content)
#         else:
#             for data in res.iter_content(chunk_size=4096):
#                 f.write(data)
#                 pbar.update(4096)
#     pbar.close()
#
#     print('  - validating checksum')
#     with open(file_path, 'rb') as f:
#         data = f.read()
#         md5_to_check = hashlib.md5(data).hexdigest()
#
#     with open(os.path.join('seisml/utility/checksums', 'triggered_earthquakes.md5'), 'r') as f:
#         data = f.readlines()[0]
#         data = data.split('  ')
#         md5 = data[0]
#
#     if md5 == md5_to_check:
#         print('  - checksum verified')
#     else:
#         print('  - checksum invalid, aborting')
#         os.remove(file_path)
#         return
#
#     print('  - extracting sample data')
#     tf = tarfile.open(file_path, 'r:gz')
#     tf.extractall(DATA_PATH)
#     tf.close()
#     os.remove(file_path)
