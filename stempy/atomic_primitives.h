#ifndef stempy_atomicprimitives_h_
#define stempy_atomicprimitives_h_

#ifdef _MSC_VER
#include <Windows.h>
#endif

inline uint64_t sync_fetch_and_add(volatile uint64_t* ptr, uint64_t add)
{
#ifdef _MSC_VER
  // Windows is requiring int64_t. Let's hope if there's an overflow, that
  // it will still add and convert back to uint64_t correctly.
  return InterlockedExchangeAdd64(reinterpret_cast<volatile int64_t*>(ptr),
                                  static_cast<int64_t>(add));
#elif defined __GNUC__
  return __sync_fetch_and_add(ptr, add);
#else
#error "Unhandled system for sync_fetch_and_add_32"
#endif
}

#endif // stempy_atomicprimitives_h_
