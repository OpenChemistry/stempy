#ifndef stempyreaderpy_h
#define stempyreaderpy_h

#include <iostream>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stempy/reader.h>

namespace py = pybind11;

namespace stempy {

struct PYBIND11_EXPORT DataHolder
{
  DataHolder() = default;
  DataHolder(const DataHolder&) = default;
  DataHolder(DataHolder&&) = default;

  DataHolder& operator=(const DataHolder& other)
  {
    if (this != &other) {
      // Need to acquire the gil before deleting the python array
      // or there may be a crash.
      py::gil_scoped_acquire gil;
      this->array = other.array;
    }

    return *this;
  }

  DataHolder& operator=(DataHolder&& other)
  {
    if (this != &other) {
      // Need to acquire the gil before deleting the python array
      // or there may be a crash.
      py::gil_scoped_acquire gil;
      this->array = std::move(other.array);
    }

    return *this;
  }

  ~DataHolder()
  {
    // Need to acquire the gil before deleting the python array
    // or there may be a crash.
    reset();
  }

  const uint16_t* get()
  {
    if (!this->array) {
      return nullptr;
    }

    return this->array->data();
  }

  void reset()
  {
    py::gil_scoped_acquire gil;
    this->array.reset();
  }

  std::shared_ptr<py::array_t<uint16_t>> array;
};

struct PYBIND11_EXPORT PyBlock
{
  Header header;
  DataHolder data;
  PyBlock() = default;
  PyBlock(py::array_t<uint16_t> pyarray);
};

class PYBIND11_EXPORT PyReader
{

public:
  PyReader(py::object pyDataSet, std::vector<uint32_t>& imageNumbers,
           Dimensions2D scanDimensions, uint32_t blockSize,
           uint32_t totalImageNum);

  PyBlock read();
  void reset();

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
  Dimensions2D m_scanDimensions;
  std::vector<uint32_t> m_imageNumbers;
  uint32_t m_currIndex = 0;
  uint32_t m_imageNumInBlock;
  uint32_t m_blockNumInFile;
  uint32_t m_totalImageNum;
};

} // namespace stempy

#endif
