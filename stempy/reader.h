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

  Block& read();
  void process(int streamId, int concurrency=-1, int width=160, int height=160,
      const std::string& url="http://127.0.0.1:5000");

  Block& currentBlock() { return m_curBlock; }

  class iterator;
  iterator begin() { return iterator(this); }
  iterator end() { return iterator(nullptr); }

  class iterator
  {
  public:
    using self_type = iterator;
    using value_type = Block;
    using reference = Block&;
    using pointer = Block*;
    using iterator_category = std::input_iterator_tag;
    using difference_type = void; // Differences not allowed here

    // This class is required for "*it++" to work properly
    // But we really shouldn't use "*it++", because it is expensive
    class postinc_return
    {
    public:
      postinc_return(reference value_) : value(value_) {}
      value_type operator*() { return value; }

    private:
      value_type value;
    };

    iterator(StreamReader* reader) : m_streamReader(reader) {}
    self_type operator++()
    {
      m_streamReader->read();
      if (!m_streamReader->currentBlock().data)
        m_streamReader = nullptr;
      return *this;
    }
    postinc_return operator++(int)
    {
      postinc_return temp(m_streamReader->currentBlock());
      ++(*this);
      return temp;
    }
    reference operator*() { return m_streamReader->currentBlock(); }
    pointer operator->() { return &m_streamReader->currentBlock(); }
    bool operator==(const self_type& rhs)
    {
      return m_streamReader == rhs.m_streamReader;
    }
    bool operator!=(const self_type& rhs) { return !(*this == rhs); }

  private:
    StreamReader* m_streamReader;
  };

private:
  std::ifstream m_stream;
  Block m_curBlock;
  std::vector<std::string> m_files;
  size_t m_curFileIndex = 0;
  int m_version;

  void openNextFile();

  // Whether or not we are at the end of all of the files
  bool atEnd() const { return m_curFileIndex >= m_files.size(); }

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
