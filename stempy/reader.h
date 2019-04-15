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

  Header() = default;
  Header(const Header& header) = default;
  Header(Header&& header) noexcept = default;
  Header& operator=(Header&& header) noexcept = default;

};

struct Block {
  Header header;
  std::shared_ptr<uint16_t> data;

  Block() = default;
  Block(const Block&) = default;
  Block(const Header& header);
  Block(Block&& i) noexcept = default;
  Block& operator=(Block&& i) noexcept = default;
};

class StreamReader {

public:
  StreamReader(const std::string &path, uint8_t version=1);
  StreamReader(const std::vector<std::string>& files, uint8_t version = 1);

  Block read();
  void process(int streamId, int concurrency=-1, int width=160, int height=160,
      const std::string& url="http://127.0.0.1:5000");

  // Whether or not we are at the end of all of the files
  bool atEnd() const { return m_curFileIndex >= m_files.size(); }

private:
  std::ifstream m_stream;
  std::vector<std::string> m_files;
  size_t m_curFileIndex = 0;
  int m_version;

  void openNextFile();

  Header readHeaderVersion1();
  Header readHeaderVersion2();

  template<typename T>
  std::istream & read(T& value);
  template<typename T>
  std::istream & read(T* value, std::streamsize size);
};

inline StreamReader::StreamReader(const std::string& path, uint8_t version)
  : StreamReader(std::vector<std::string>{ path }, version)
{}
}

#endif
