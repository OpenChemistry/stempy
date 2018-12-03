#ifndef stempyreader_h
#define stempyreader_h

#include <iostream>
#include <vector>
#include <memory>
#include <fstream>

namespace stempy {

struct Header {
  uint32_t imagesInBlock, rows, columns, version, timestamp;
  std::vector<uint32_t> imageNumbers;
};

struct Stream {
  Header header;
  std::unique_ptr<uint16_t[]> data;
};

class StreamReader {

public:
  StreamReader(const std::string &path);

  Stream read();

private:
  std::ifstream m_stream;

  Header readHeader();

  template<typename T>
  std::istream & read(T& value);
  template<typename T>
  std::istream & read(T* value, std::streamsize size);
};

}

#endif
