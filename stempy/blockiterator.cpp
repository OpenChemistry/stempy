
#include "blockiterator.h"

#include <iostream>

using std::cout;
using std::cerr;
using std::endl;

using std::string;
using std::vector;

namespace stempy {

BlockIterator::BlockIterator(const vector<string>& fileList)
{
  m_streams.reserve(fileList.size());
  for (const auto& file: fileList)
    m_streams.push_back(StreamReader(file));

  if (m_streams.empty())
    cerr << "Warning: no files provided for BlockIterator" << endl;

  // Get the first block
  ++(*this);
}

const Block& BlockIterator::operator*() const
{
  return m_block;
}

const Block* BlockIterator::operator->() const
{
  return &m_block;
}

BlockIterator& BlockIterator::operator++ ()
{
  m_block = Block();
  if (m_streams.size() <= m_curStreamIndex) {
    cerr << "BlockIterator: current stream index is out of bounds!" << endl;
    return *this;
  }

  m_block = m_streams[m_curStreamIndex].read();

  if (!m_block.data) {
    // Increment the current stream index and try again
    ++m_curStreamIndex;
    if (m_streams.size() == m_curStreamIndex)
      return *this;
    return ++(*this);
  }

  ++m_numBlocksRead;

  return *this;
}

bool BlockIterator::atEnd() const
{
  return m_curStreamIndex >= m_streams.size();
}

} // end namespace stempy
