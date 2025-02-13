#include "WebSocketController.h"

WebSocketController* pInstance = nullptr;

WebSocketController::WebSocketController()
{
    pInstance = this;
}

WebSocketController* WebSocketController::getInstance()
{
    return pInstance;
}



std::vector<int> WebSocketController::getConnectedClientIds()
{
    this->_mtx.lock();
    std::vector<int> cIds;

    for (auto& Con : this->_connected)
        if (Con->hasContext())
            cIds.push_back(*Con->getContext<int>().get());
    this->_mtx.unlock();

    return cIds;
}



std::vector<drogon::WebSocketConnectionPtr>& WebSocketController::getConnectedClients()
{
    return this->_connected;
}



void WebSocketController::addClientConnection(const drogon::WebSocketConnectionPtr& Connection)
{
    if (!Connection->hasContext())
    {
        static int id = 0;
        auto ctx = std::make_shared<int>(id++);
        auto s0 = ctx.get();
        Connection->setContext(ctx);
    }

    this->_mtx.lock();
    this->_connected.push_back(Connection);
    this->_mtx.unlock();
}



void WebSocketController::removeClientConnection(const drogon::WebSocketConnectionPtr& Connection)
{
    int id = -1;
    if (Connection->hasContext())
    {
        id = *Connection->getContext<int>().get();

        this->_mtx.lock();

        auto found = this->_connected.end();
        auto it = this->_connected.begin();
        while (it != this->_connected.end())
        {
            if (*(*it)->getContext<int>().get() == id)
            {
                found = it;
                break;
            }
            it++;
        }

        if (found != this->_connected.end())
            this->_connected.erase(found);

        this->_mtx.unlock();
    }
}



drogon::WebSocketConnectionPtr WebSocketController::getConnectedClient(int Id)
{
    drogon::WebSocketConnectionPtr retval = nullptr;

    this->_mtx.lock();

    for (auto& pt : this->_connected)
    {
        if (pt->connected() && *pt->getContext<int>().get() == Id)
        {
            retval = pt;
            break;
        }
    }

    this->_mtx.unlock();
    return retval;
}



void WebSocketController::handleNewMessage(const drogon::WebSocketConnectionPtr&, std::string&&, const drogon::WebSocketMessageType&)
{
}


void WebSocketController::handleNewConnection(const drogon::HttpRequestPtr& Req, const drogon::WebSocketConnectionPtr& WS)
{
    this->addClientConnection(WS);
    std::cout << "Connected " << *WS->getContext<int>().get() << std::endl;
}


void WebSocketController::handleConnectionClosed(const drogon::WebSocketConnectionPtr& WS)
{
    std::cout << "Closed " << *WS->getContext<int>().get() << std::endl;
    this->removeClientConnection(WS);
}
