
/*
 * Autor: Matthias Micheler
 * Datum: 12. Februar 2025
 * Lizenz: MIT
 *
 * MIT License
 *
 * Copyright (c) 2025 Matthias Micheler
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <drogon/drogon.h>
#include <opencv2/opencv.hpp>
#include "WebSocketController.h"

int main()
{
    // Start webserver in thread
    std::thread t([] {
        drogon::app()
            .addListener("0.0.0.0", 80)
            .loadConfigFile("./drogon.json")
            .run();
        });

    // Start cam
    cv::VideoCapture vc(0, cv::CAP_DSHOW);
    
    if (vc.isOpened())
    {
        while (true)
        {
            cv::Mat imgCap;
            vc >> imgCap;

            if (!imgCap.empty())
            {
                std::vector<uchar> buf;
                cv::imencode(".jpg", imgCap, buf);
                auto base64 = base64_encode(std::string(buf.begin(), buf.end()));

                auto ccIds = WebSocketController::getInstance()->getConnectedClientIds();
                // Send images to all connected clients
                for (auto ccId : ccIds)
                {
                    auto cc = WebSocketController::getInstance()->getConnectedClient(ccId);
                    if (cc != nullptr)
                        cc->send(base64);
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Stop cam
    vc.release();

    // Stop webserver
    std::this_thread::sleep_for(std::chrono::seconds(1));
    drogon::app().quit();
    t.join();

	return 0;
}