import os
import io
import requests
import shutil
import zipfile

from ASAPPpy import ROOT_PATH

def get_confirm_token(response):
    '''
    Parameters
    ----------


    '''

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def download(gid='1kR0hfuuaduTFixY0ZLgqkLRCw_ET5xRW', save_path=ROOT_PATH):
    '''
    Parameters
    ----------


    '''

    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id':gid}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id':gid, 'confirm':token}
        response = session.get(url, params=params, stream=True, headers={'Accept-Encoding': None})

    if 'Content-Length' in response.headers:
        length = response.headers.get('Content-Length')
    else:
        # read directly from the raw urllib3 connection
        print("Retrieving file size...")
        raw_content = response.raw.read()
        length = len(raw_content)
        # replace the internal file-object to serve the data again
        response.raw._fp = io.BytesIO(raw_content)

    if length:
        length = int(length)
        blocksize = max(4096, length//100)
        buffer = io.BytesIO()
        current_progress = 0

        print("Downloading data...")
        for data in response.iter_content(blocksize):
            current_progress += len(data)
            buffer.write(data)
            done = int(50 * current_progress / length)
            print("[%s%s]" % ('=' * done, ' ' * (50-done)), end='\r')

        print("\nUnzipping file...")
        zip_file = zipfile.ZipFile(buffer)

        if save_path is None:
            zip_file.extractall()
        else:
            zip_file.extractall(save_path)
    else:
        print("Could not find the length of the file. Downloading without showing progress bar and unzipping.")

        zip_file = zipfile.ZipFile(io.BytesIO(response.content))

        if save_path is None:
            zip_file.extractall()
        else:
            zip_file.extractall(save_path)

    # check if __MACOSX was created after unzipping and delete it
    if save_path is None:
        macosx_folder_path = '__MACOSX'
    else:
        macosx_folder_path = os.path.join(save_path, '__MACOSX')
    if os.path.exists(macosx_folder_path):
        shutil.rmtree(macosx_folder_path)

if __name__ == "__main__":
    # TAKE ID FROM SHAREABLE LINK
    file_id = '1kR0hfuuaduTFixY0ZLgqkLRCw_ET5xRW'
    # DESTINATION FILE ON YOUR DISK
    destination = None
    download(file_id, destination)