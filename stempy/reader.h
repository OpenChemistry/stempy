#ifndef stempyreader_h
#define stempyreader_h

#include <iostream>
#include <vector>
#include <memory>
#include <fstream>

namespace stempy {

struct EofException : public std::exception
{
  const char* what () const throw () { return "EOF Exception"; }
};

struct Header {
  uint32_t imagesInBlock = 0, rows = 0, columns = 0, version = 0, timestamp = 0;
  std::vector<uint32_t> imageNumbers;
};

struct Block {
  Header header;
  std::unique_ptr<uint16_t[]> data;
};

class StreamReader {

public:
  StreamReader(const std::string &path);

  Block read();

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
