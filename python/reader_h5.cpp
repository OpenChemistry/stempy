#include "reader_h5.h"
#include <stempy/reader.h>

namespace stempy {

PyBlock::PyBlock(py::array pyarray)
{
  //std::cout << "debug construct1---" <<lowerBound<<std::endl;
  //auto getItems = h5dataSet.attr("__getitem__");
  //  std::cout << "debug construct2---" <<lowerBound<<std::endl;

  //auto sliceIndex = py::slice(lowerBound, upperBound, 1);
  //    std::cout << "debug construct3---" <<lowerBound<<std::endl;

  //m_array = getItems(sliceIndex);
  //      std::cout << "debug construct4---" <<lowerBound<<std::endl;

  m_array = pyarray;


  //py::array* persistArray = new py::array();
  //m_array = std::move(py::array(tempArray));
  //this->m_array.reset(persistArray);
  
  //m_array=dataarray;

  //py::buffer_info* persistDataBuf = new py::buffer_info();

  // if it is ok to only move the pointer in it???
  //*persistDataBuf = std::move(dataarray.request());
  // this->data.reset((uint16_t*)(m_array.request().ptr));
   
  this->data=(DataHolder());
  this->data.innerdata=(uint16_t*)(m_array.request().ptr);
  //this->data.set((uint16_t*)(m_array.request().ptr));

  //this->m_buffer.reset(persistDataBuf);

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
  py::gil_scoped_acquire acquire;

  // get to the end of the file, return empty Block
  if (m_currIndex >= m_totalImageNum) {
    std::cout << "data loading finish with index " << m_currIndex << std::endl;
    return PyBlock();
  }

  // get the data pointer from the py object
  auto getItems = m_h5dataset.attr("__getitem__");
  uint32_t upperBound = m_currIndex + m_imageNumInBlock;

  if (upperBound > m_totalImageNum) {
    upperBound = m_totalImageNum;
  }

  auto sliceIndex = py::slice(m_currIndex, upperBound, 1);
  auto part = getItems(sliceIndex);

  py::array pyarray = py::array(part);
  
  PyBlock* b = new PyBlock(pyarray);

  // py::buffer_info buf = pyarray.request();
  // uint32_t arraySize = pyarray.size();

  // this->ptr = (uint16_t*)buf.ptr;

  // getData
  // return this->ptr;

  b->header = std::move(Header(m_imageWidth, m_imageHeight, m_imageNumInBlock, m_blockNumInFile,
           m_scanWidth, m_scanHeight, m_currIndex, m_imageNumbers));

  // std::copy(ptr, ptr + arraySize, b.data.get());

  std::cout << "check getData" << std::endl;

  // there are problems if this is called multiple times???
  //std::shared_ptr<uint16_t> data = b.getData();

  for (int i = 0; i < 10; i++) {
    std::cout << *(b->data.get() + i) << std::endl;
  }

  m_currIndex = upperBound;

  std::cout << "ok for get getData, return block" << std::endl;

  return *b;
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
