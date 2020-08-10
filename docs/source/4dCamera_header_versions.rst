
NCEM 4D Camera Raw Data Versions
================================

This is a record of the layout of raw 4D Camera data. It inlucdes
information about the headers and the data shape.

The raw files contain a header block and then a data block
for each frame or sector. This is the general layout:

.. code-block::

   raw file
   ├── frame
       ├── header
       └── data
   ...
   └── frame
       ├── header
       └── data

header version 1
----------------

To be added

header version 2
----------------

To be added

header version 3
----------------

The raw data is written as full frames.

*

 Header

 * scan number, 1 x uint32
 * frame number, 1 x uint16
 * STEM scan size, 2 x uint16
 * STEM position of this frame, 2x uint16

*

 Data with shape (576,576)

 * frame data, 331776 x uint16

*
 Numpy custom dtype

.. code-block:: python

   cameraDtype3 = np.dtype([('scan_num','<u4'),('frame_num','<u4'),('scan_size','2<u2'),('scan_pos','2<u2'),('data','331776<u2')])
   frame = cameraDtype3.data.reshape((576,576))

header version 4
----------------

The raw data is written in sectors as 1/4 of the full frame.

The positions of the sectors correspond to the module number in the
file name.

*

 Header

 * scan number, 1 x uint32
 * frame number, 1 x uint16
 * STEM scan size, 2 x uint16
 * STEM position of this frame, 2x uint16

*

 Data with shape (576, 144)

 * sector data, 82944 x uint16

*
 Numpy custom dtype and sector reshape order

.. code-block:: python

   cameraDtype4 = np.dtype([('scan_num','<u4'),('frame_num','<u4'),('scan_size','2<u2'),('scan_pos','2<u2'),('data','82944<u2')])
   frame = cameraDtype4.data.reshape((576, 144))

header version 5
----------------

The raw data is written in sectors as 1/4 of the full frame.
In this version the sectors are written transposed compared to version4
making the data contiguous.

The positions of the sectors correspond to the module number in the
file name. See below for manual stitching code.

*

 Header

 * scan number, 1 x uint32
 * frame number, 1 x uint16
 * STEM scan size, 2 x uint16
 * STEM position of this frame, 2x uint16

*

 Data with shape (144, 576)

 * sector data, 82944 x uint16

*
 Numpy custom dtype and sector reshape order.

.. code-block:: python

   cameraDtype5 = np.dtype([('scan_num','<u4'),('frame_num','<u4'),('scan_size','2<u2'),('scan_pos','2<u2'),('data','82944<u2')])
   frame = cameraDtype5.data.reshape((144, 576))

* Reconstruct full frames manually

.. code-block:: python

  import numpy as np
  modulePositions = ((0,144),(144,144*2),(144*2,144*3),(144*3,144*4))

  scan = np.zeros((scanXY[0],scanXY[1], 576, 576),dtype='<u2')

  for file in files:
      #print(file.stem)
      with open(file,'rb') as f1:
          start = str(file).find('module')
          m = int(str(file)[start+6])
          module = modulePositions[int(str(file)[start+6])]
          dataSet = np.fromfile(f1, dtype=cameraDtype, count = -1)
          for pos, data in zip(dataSet['pos'], dataSet['data']):
              scan[pos[0], pos[1], module[0]:module[1], :] = data.reshape((detectorI, detectorJ))