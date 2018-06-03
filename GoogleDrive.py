# snippet for getting data to and from google drive while using google colab
import tqdm
from google.colab import auth
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.http import MediaFileUpload


def download_file_from_google_drive(file_id=None, out_file_name=None):
    assert file_id is not None and out_file_name is not None
    auth.authenticate_user()
    drive_service = build('drive', 'v3')
    request = drive_service.files().get_media(fileId=file_id)
    downloaded = io.BytesIO()
    downloader = MediaIoBaseDownload(downloaded, request)
    done = False
    pbar = tqdm.tqdm(total=100, desc=out_file_name)

    while done is False:
        # _ is a placeholder for a progress object that we ignore.
        # (Our file is small, so we skip reporting progress.)
        status, done = downloader.next_chunk()
        pbar.update(status.progress() * 100)
        # print("Downloaded: ", int(status.progress() * 100))

    downloaded.seek(0)
    # print('Downloaded file contents are: {}'.format(downloaded.read()[:10]))
    with open(out_file_name, 'wb') as out:
        out.write(downloaded.read())
    print("Data downloaded to: ", out_file_name)
    return out_file_name


def save_file_to_google_drive(local_filename, dest_filename, mimetype='application/octet-stream'):
    auth.authenticate_user()
    drive_service = build('drive', 'v3')

    file_metadata = {
        'name': dest_filename,
        'mimeType': mimetype
    }
    media = MediaFileUpload(local_filename,
                            mimetype=mimetype,
                            resumable=True)
    created = drive_service.files().create(body=file_metadata,
                                           media_body=media,
                                           fields='id').execute()
    print('File ID: {}'.format(created.get('id')))
    return created.get('id')