#ifndef stempysioclient_h
#define stempysioclient_h

#include <sio_client.h>
#include <mutex>
#include <condition_variable>
#include <functional>


namespace stempy {

class SocketIOClient {

public:
  SocketIOClient(const std::string &url, const std::string &ns="");

  void connect();

  void onConnect();
  void onClose(const sio::client::close_reason& reason);
  void onFail();
  void emit(const std::string& eventName, const sio::message::ptr msg);

private:
  sio::client client;
  sio::socket::ptr socket;
  std::mutex connectLock;
  std::condition_variable_any connectCondition;
  bool connected = false;
  std::string url = "http://127.0.0.1:3000";
  std::string ns = "";
};

}

#endif
