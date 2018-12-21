from stempy import io
import glob
import os

node_id = os.environ['SLURM_NODEID']

path = '/global/project/projectdirs/ncemhub/simData/smallScanningDiffraction/data00%s.dat' % node_id.zfill(2)
reader = io.reader(path)
reader.process(url='http://128.55.206.19:60048/', stream_id=int(node_id))
