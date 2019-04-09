from collections import namedtuple

import numpy as np

from stempy._io import _reader

class FileVersion(object):
    VERSION1 = 1
    VERSION2 = 2

class Reader(_reader):
    def __iter__(self):
        return self

    def __next__(self):
        b = self.read()
        if b is None:
            raise StopIteration
        else:
            return b

    def read(self):
        b = super(Reader, self).read()

        # We are at the end of the stream
        if b.header.version == 0:
            return None

        block = namedtuple('Block', ['header', 'data'])
        block._block = b
        block.header = b.header
        block.data = np.array(b, copy = False)

        return block

def reader(path, version=FileVersion.VERSION1):
    return Reader(path, version)
