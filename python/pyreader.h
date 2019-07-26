#ifndef stempyreaderpy_h
#define stempyreaderpy_h

#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stempy/reader.h>

namespace py = pybind11;

namespace stempy {

struct DataHolder
{
  DataHolder() = default;
  const uint16_t* innerdata = nullptr;
  const uint16_t* get() { return innerdata; }
  void reset() { return; }
};

struct PYBIND11_EXPORT PyBlock
{
  Header header;
  py::array_t<uint16_t> m_array;
  DataHolder data;
  PyBlock() = default;
  PyBlock(py::array_t<uint16_t> pyarray);
};

class PYBIND11_EXPORT PyReader
{

public:
  PyReader(py::object pyDataSet, std::vector<uint32_t>& imageNumbers,
           uint32_t scanWidth, uint32_t scanHeight, uint32_t blockSize,uint32_t totalImageNum);

  PyBlock read();

  class iterator;
  iterator begin();
  iterator end();

  class iterator
  {

  public:
    using self_type = iterator;
    using value_type = PyBlock;
    using reference = PyBlock&;
    using pointer = PyBlock*;
    using iterator_category = std::input_iterator_tag;
    using difference_type = void; // Differences not allowed here

    iterator(PyReader* pyreader) : m_PyReader(pyreader)
    {
      if (pyreader == nullptr) {
        return;
      }
      // read data at first time
      m_block = m_PyReader->read();
      if (m_block.data.get() == nullptr) {
        m_PyReader = nullptr;
      }
    }

    self_type operator++()
    {
      m_block = m_PyReader->read();
      if (!m_block.data.get()) {
        this->m_PyReader = nullptr;
      }

      return *this;
    }

    reference operator*() { return m_block; }

    pointer operator->() { return &m_block; }

    bool operator==(const self_type& rhs)
    {
      return m_PyReader == rhs.m_PyReader;
    }

    bool operator!=(const self_type& rhs)
    {
      return (this->m_PyReader != rhs.m_PyReader);
    }

  private:
    PyReader* m_PyReader = nullptr;
    value_type m_block;
  };

private:
  py::object m_pydataset;
  std::vector<uint32_t> m_imageNumbers;
  uint32_t m_currIndex = 0;
  uint32_t m_scanWidth;
  uint32_t m_scanHeight;
  uint32_t m_imageNumInBlock;
  uint32_t m_blockNumInFile;
  uint32_t m_totalImageNum;
};

} // namespace stempy

#endif
