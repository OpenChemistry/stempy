#include "reader_h5.h"
#include <stempy/reader.h>

namespace stempy {

PyBlock::PyBlock(py::object h5dataSet, uint32_t lowerBound, uint32_t upperBound)
{

  auto getItems = h5dataSet.attr("__getitem__");
  auto sliceIndex = py::slice(lowerBound, upperBound, 1);
  auto partArray = getItems(sliceIndex);
  py::array dataarray = py::array(partArray);
  
  m_array=dataarray;

  //py::buffer_info* persistDataBuf = new py::buffer_info();

  // if it is ok to only move the pointer in it???
  //*persistDataBuf = std::move(dataarray.request());

  this->data=(DataHolder());

  this->data.set((uint16_t*)(m_array.request().ptr));

  //this->m_buffer.reset(persistDataBuf);

  std::cout << "construct pyblock with index " << lowerBound << "and "
            << upperBound << std::endl;

}

//PyBlock::~PyBlock() {
//  std::cout << "~PyBlock" << std::endl;/
//}



H5Reader::H5Reader(py::object h5DataSet, std::vector<uint32_t>& imageNumbers,
                   uint32_t imageWidth, uint32_t imageHeight,
                   uint32_t scanWidth, uint32_t scanHeight, uint32_t blockSize,
                   uint32_t blockNumInFile, uint32_t totalImageNum)
  : m_h5dataset(h5DataSet), m_imageWidth(imageWidth),
    m_imageHeight(imageHeight), m_scanWidth(scanWidth),
    m_scanHeight(scanHeight), m_imageNumInBlock(blockSize),
    m_blockNumInFile(blockNumInFile), m_totalImageNum(totalImageNum),
    m_imageNumbers(imageNumbers)
{}


PyBlock H5Reader::read()
{

  // get to the end of the file, return empty Block
  if (m_currIndex >= m_totalImageNum) {
    std::cout << "data loading finish with index " << m_currIndex << std::endl;
    return PyBlock();
  }

  uint32_t upperBound = m_currIndex + m_imageNumInBlock;

  if (upperBound > m_totalImageNum) {
    upperBound = m_totalImageNum;
  }

  // auto sliceIndex = py::slice(m_currIndex, upperBound, 1);
  // auto partArray = getItems(sliceIndex);

  // py::array pyarray = py::array(partArray);

  // get the data pointer from the py object
  // auto getItems = m_h5dataset.attr("__getitem__");
  PyBlock b(m_h5dataset, m_currIndex, upperBound);

  // py::buffer_info buf = pyarray.request();
  // uint32_t arraySize = pyarray.size();

  // this->ptr = (uint16_t*)buf.ptr;

  // getData
  // return this->ptr;

  b.header = std::move(Header(m_imageWidth, m_imageHeight, m_imageNumInBlock, m_blockNumInFile,
           m_scanWidth, m_scanHeight, m_currIndex, m_imageNumbers));

  // std::copy(ptr, ptr + arraySize, b.data.get());

  std::cout << "check getData" << std::endl;

  // there are problems if this is called multiple times???
  //std::shared_ptr<uint16_t> data = b.getData();

  for (int i = 0; i < 10; i++) {
    std::cout << *(b.data.get() + i) << std::endl;
  }

  m_currIndex = upperBound;

  std::cout << "ok for get getData, return block" << std::endl;

  return b;
}

H5Reader::iterator H5Reader::begin()
{
  // read data at first time
  std::cout<<"debug reader begin in cpp" << std::endl;
  return iterator(this);
}

H5Reader::iterator H5Reader::end()
{
  return iterator(nullptr);
}

} // namespace stempy
