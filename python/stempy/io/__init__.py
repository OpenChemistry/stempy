from stempy._io import _reader
import numpy as np
from collections import namedtuple

class Reader(_reader):

    def read(self):
        s = super(Reader, self).read()
        stream = namedtuple('Stream', ['header', 'data'])
        stream.header = s.header
        stream.data = np.array(s, copy = False)

        return stream

def reader(path):
    return Reader(path)