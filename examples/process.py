from stempy import io
import glob

for i, f in enumerate(glob.glob('/data/4dstem/smallScanningDiffraction/data*.dat')):
  print(f)
  reader = io.reader(f)
  reader.process(stream_id=i)

