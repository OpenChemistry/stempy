#include "sioclient.h"

#include <iostream>

using std::string;

namespace stempy {

SocketIOClient::SocketIOClient(const string& u, const string& n) : url(u), ns(n)
{
  this->client.set_open_listener(bind(&SocketIOClient::onConnect, this));
  this->client.set_close_listener(bind(&SocketIOClient::onClose, this, placeholders::_1));
  this->client.set_fail_listener(bind(&SocketIOClient::onFail, this));
}

void SocketIOClient::connect()
{
  this->client.connect(this->url);
  this->connectLock.lock();
  if (!this->connected) {
    this->connectCondition.wait(this->connectLock);
  }
  this->connectLock.unlock();
  this->socket = this->client.socket(this->ns);
}


void SocketIOClient::onConnect()
{
    this->connectLock.lock();
    this->connectCondition.notify_all();
    connected = true;
    this->connectLock.unlock();
}

void SocketIOClient::onClose(const sio::client::close_reason& reason)
{
    cout << "Connection closed: " << reason << endl;
    connected = false;
}

void SocketIOClient::onFail()
{
    cout << "Connect failed." << endl;
}

void SocketIOClient::emit(const string& eventName,  const sio::message::ptr msg)
{
  if (!this->connected) {
    throw runtime_error("Not connected");
  }

  this->socket->emit(eventName, msg);
}

}
