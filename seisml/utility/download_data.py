import os, shutil
import tarfile
import requests
import hashlib
from pathlib import Path

DATA_PATH = os.path.expanduser('~/.seisml/')
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)


def download_sample_data(force=False):
    print('downloading sample_data: ')
    if os.path.isdir(DATA_PATH + 'sample_data') and not force:
        return
    if os.path.isdir(DATA_PATH + 'sample_data') and force:
        print('  - removing existing sample_data')
        shutil.rmtree(DATA_PATH + 'sample_data')

    url = 'https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/sample_data.tar.gz'
    target_path = os.path.join(DATA_PATH, 'sample_data.tar.gz')

    print('  - downloading')
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(res.raw.read())

        print('  - validating checksum')
        with open(target_path, 'rb') as f:
            data = f.read()
            md5_to_check = hashlib.md5(data).hexdigest()

        with open(os.path.join('seisml/utility/checksums', 'sample_data.md5'), 'r') as f:
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
