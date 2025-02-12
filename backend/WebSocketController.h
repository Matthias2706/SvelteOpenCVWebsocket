#pragma once

#include <drogon/WebSocketController.h>

#include <opencv2/opencv.hpp>
#include "base64.h"

class WebSocketController : public drogon::WebSocketController<WebSocketController>
{
    private:

        std::vector<drogon::WebSocketConnectionPtr> _connected;
        std::mutex _mtx;

    public:

        WebSocketController();

        static WebSocketController* getInstance();

        std::vector<int> getConnectedClientIds();
        std::vector<drogon::WebSocketConnectionPtr>& getConnectedClients();

        void addClientConnection(const drogon::WebSocketConnectionPtr& Connection);
        void removeClientConnection(const drogon::WebSocketConnectionPtr& Connection);
        drogon::WebSocketConnectionPtr getConnectedClient(int Id);

        virtual void handleNewMessage(const drogon::WebSocketConnectionPtr&, std::string&&, const drogon::WebSocketMessageType&) override;
        virtual void handleNewConnection(const drogon::HttpRequestPtr&, const drogon::WebSocketConnectionPtr&) override;
        virtual void handleConnectionClosed(const drogon::WebSocketConnectionPtr&) override;

        WS_PATH_LIST_BEGIN
        WS_PATH_ADD("/camera/stream", {drogon::Get});
        WS_PATH_LIST_END
};

