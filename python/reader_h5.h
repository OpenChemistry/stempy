#ifndef stempyreaderh5_h
#define stempyreaderh5_h

#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stempy/reader.h>

namespace py = pybind11;

namespace stempy {

struct DataHolder{
  DataHolder()=default;
  uint16_t* innerdata = nullptr;
  uint16_t* get() {return innerdata;}
  void set(uint16_t*inputdata) {innerdata=inputdata;}
  void reset(){return;}
};


struct PyBlock{
  Header header;
  //std::shared_ptr<py::buffer_info> m_buffer=nullptr;
  //std::shared_ptr<py::array> m_array;
  py::array m_array;
  DataHolder data ;
  //std::shared_ptr<uint16_t> data=nullptr;
  //std::shared_ptr<py::array> m_array=nullptr;
  PyBlock() = default;
  PyBlock(py::array_t<uint16_t> pyarray);
  //PyBlock& operator=(PyBlock&& i) = default;
  //~PyBlock() {std::cout <<"Pyblock is destructed" << std::endl;}
};

class H5Reader
{

public:
  H5Reader(py::object h5DataSet,std::vector<uint32_t>& imageNumbers, uint32_t imageWidth,
          uint32_t imageHeight, uint32_t scanWidth,
          uint32_t scanHeight, uint32_t blockSize,
          uint32_t blockNumInFile, uint32_t totalImageNum);

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

    iterator(H5Reader* h5reader) : m_H5Reader(h5reader)
    {
      if(h5reader==nullptr){
        return;
      }
      // read data at first time
      m_block = m_H5Reader->read();
      std::cout<<"debug ok to get block from read in iterator" << std::endl;
      if (m_block.data.get()==nullptr) {
        std::cout<<"m_H5Reader is null" << std::endl;
        m_H5Reader = nullptr;
      }
      std::cout<<"ok to init iterator"<<std::endl;
    }

    self_type operator++()
    {
      m_block = m_H5Reader->read();
      if (!m_block.data.get()) {
        std::cout<<"data in Block is empty, data loading is finished"<<std::endl;
        this->m_H5Reader = nullptr;
      }

      return *this;
    }

    reference operator*() { 
      return m_block;  
    }

    pointer operator->() { return &m_block; }

    bool operator==(const self_type& rhs)
    {
      return m_H5Reader == rhs.m_H5Reader;
    }

    bool operator!=(const self_type& rhs) { return (this->m_H5Reader !=rhs.m_H5Reader ); }

  private:
    H5Reader* m_H5Reader=nullptr;
    value_type m_block;
  };

private:
  py::object m_h5dataset;
  std::vector<uint32_t> m_imageNumbers;
  uint32_t m_currIndex = 0;
  uint32_t m_imageWidth;
  uint32_t m_imageHeight;
  uint32_t m_scanWidth;
  uint32_t m_scanHeight;
  uint32_t m_imageNumInBlock;
  uint32_t m_blockNumInFile;
  uint32_t m_totalImageNum;

};

} // namespace stempy

#endif
