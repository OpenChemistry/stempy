#ifndef stempystreamview_h
#define stempystreamview_h

#include <streambuf>

namespace stempy {
class StreamView : public std::streambuf
{
public:
  StreamView(char* data, size_t size) { this->setg(data, data, data + size); }
};
} // namespace stempy

#endif