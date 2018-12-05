from stempy._io import _reader
import numpy as np
from collections import namedtuple

class Reader(_reader):

    def read(self):
        b = super(Reader, self).read()

        # We are at the end of the stream
        if b.header.version == 0:
            return None

        block = namedtuple('Block', ['header', 'data'])
        block.header = b.header
        block.data = np.array(b, copy = False)

        return block

def reader(path):
    return Reader(path)
