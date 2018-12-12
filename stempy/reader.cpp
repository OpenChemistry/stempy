#include "reader.h"

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;


namespace stempy {

Block::Block(const Header& header) :
  header(header),
  data(new uint16_t[header.rows*header.columns*header.imagesInBlock],
      std::default_delete<uint16_t[]>())
{}

StreamReader::StreamReader(const std::string& path)
{
  m_stream.open (path, ios::in | ios::binary);
  if (!m_stream.is_open()) {
    ostringstream msg;
    msg << "Unable to open file: " << path;
    throw invalid_argument(msg.str());
  }
}

template<typename T>
istream & StreamReader::read(T& value){
    return read(&value, sizeof(value));
}

template<typename T>
istream & StreamReader::read(T* value, streamsize size){
    m_stream.read(reinterpret_cast<char*>(value), size);

    if (m_stream.eof()) {
      throw EofException();
    }

    return m_stream;
}

Header StreamReader::readHeader() {

  Header header;

  uint32_t headerData[1024];
  read(headerData, 1024*sizeof(uint32_t));

  int index = 0;
  header.imagesInBlock = headerData[index++];
  header.rows = headerData[index++];
  header.columns = headerData[index++];
  header.version = headerData[index++];
  header.timestamp =  headerData[index++];
  // Skip over 6 - 10 - reserved
  index += 5;

  // Now get the image numbers
  header.imageNumbers.resize(header.imagesInBlock);
  auto imageNumbersSize = sizeof(uint32_t)*header.imagesInBlock;
  copy(headerData + index,
       headerData + index + header.imagesInBlock,
       header.imageNumbers.data());

  return header;
}

Block StreamReader::read() {


  // Check that we have a block to read
  auto c = m_stream.peek();
  if (c != EOF) {
    try {
      auto header = readHeader();
      Block b(header);

      auto dataSize = b.header.rows*b.header.columns*b.header.imagesInBlock;
      read(b.data.get(), dataSize*sizeof(uint16_t));

      return b;
    }
    catch (EofException& e) {
      throw invalid_argument("Unexpected EOF while processing stream.");
    }
  }
  return Block();
}

}
