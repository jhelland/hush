"""
Script to download trained Keras models from Google Drive.

Credit: https://stackoverflow.com/a/39225272
"""

import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { "id" : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { "id" : id, "confirm" : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":
    print("...Downloading trained models")
    file_ids = ["1KSR7nVi_BQZq8DMF83CZokm1zv70i0Td", # pickled tokenizer
                "1BI_pQ80D0WOiMOzBuDZ2uv4TVymRJLJa", # hdf5 glove model
                "177T41ZOGIm89mH3liu1dTQp9BRAW5Mcj"] # hdf5 fasttext model
    destinations = ["./data/models/tokenizer_raw_binary.pkl",
                    "./data/models/bi_gru_raw_binary_glove.h5",
                    "./data/models/bi_gru_raw_binary_fasttext.h5"]
    for file_id, destination in zip(file_ids, destinations):
        download_file_from_google_drive(file_id, destination)
    print("...Models downloaded to \"./data/models/\"")
    print("DONE\n")
