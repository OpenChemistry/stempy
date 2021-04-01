import io

import pytest
import requests


DATA_URLS = {
    'electron_small': 'https://data.kitware.com/api/v1/file/6065f00d2fa25629b93bdabe/download',  # noqa
    'electron_large': 'https://data.kitware.com/api/v1/file/6065f2792fa25629b93c0303/download',  # noqa
}

DATA_RESPONSES = {}


def response(key):
    if key not in DATA_RESPONSES:
        r = requests.get(DATA_URLS[key])
        r.raise_for_status()
        DATA_RESPONSES[key] = r

    return DATA_RESPONSES[key]


def io_object(key):
    r = response(key)
    return io.BytesIO(r.content)


@pytest.fixture
def electron_data_small():
    return io_object('electron_small')


@pytest.fixture
def electron_data_large():
    return io_object('electron_large')
