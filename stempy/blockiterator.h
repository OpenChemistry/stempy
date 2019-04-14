#ifndef stempy_blockiterator_h
#define stempy_blockiterator_h

#include <string>
#include <vector>

#include "reader.h"

namespace stempy {

class BlockIterator {
public:
  BlockIterator(const std::vector<std::string>& fileList);
  BlockIterator& operator++(); //prefix increment
  // Note: no postfix increment, because copying BlockIterator is not allowed
  const Block& operator*() const;
  const Block* operator->() const;
  bool atEnd() const;
  size_t numBlocksRead() const { return m_numBlocksRead; }
private:
  std::vector<StreamReader> m_streams;
  size_t m_curStreamIndex = 0;
  size_t m_numBlocksRead = 0;
  Block m_block;
};

} // end namespace stempy

#endif // stempy_blockiterator_h
