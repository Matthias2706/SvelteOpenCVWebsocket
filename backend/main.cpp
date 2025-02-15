
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

// Yolo model https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt

#include <drogon/drogon.h>
#include <opencv2/opencv.hpp>
#include "WebSocketController.h"

#include <opencv2/dnn.hpp>

#include "FilterBlur.h"

#define NET_WIDTH 640
#define NET_HEIGHT 640

struct OutputParams {
    int id;
    float confidence;
    cv::Rect box;
    cv::RotatedRect rotatedBox;
    cv::Mat boxMask;

};

struct MaskParams {
    int netWidth = NET_WIDTH;
    int netHeight = NET_HEIGHT;
    float maskThreshold = 0.5;
    cv::Size srcImgShape;
    cv::Vec4d params;
};

void LetterBox(
    const cv::Mat& image,
    cv::Mat& outImage,
    cv::Vec4d& params,
    const cv::Size& newShape = cv::Size(NET_WIDTH, NET_HEIGHT),
    bool autoShape = false,
    bool scaleFill = false,
    bool scaleUp = true,
    int stride = 32,
    const cv::Scalar& color = cv::Scalar(114, 114, 114))
{
    if (false) {
        int maxLen = MAX(image.rows, image.cols);
        outImage = cv::Mat::zeros(cv::Size(maxLen, maxLen), CV_8UC3);
        image.copyTo(outImage(cv::Rect(0, 0, image.cols, image.rows)));
        params[0] = 1;
        params[1] = 1;
        params[3] = 0;
        params[2] = 0;
    }

    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
        (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{ r, r };
    int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

    auto dw = (float)(newShape.width - new_un_pad[0]);
    auto dh = (float)(newShape.height - new_un_pad[1]);

    if (autoShape)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        new_un_pad[0] = newShape.width;
        new_un_pad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
    {
        cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
    }
    else {
        outImage = image.clone();
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    params[0] = ratio[0];
    params[1] = ratio[1];
    params[2] = left;
    params[3] = top;
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}


void GetMask2(const cv::Mat& maskProposals, const cv::Mat& maskProtos, OutputParams& output, const MaskParams& maskParams)
{
    int net_width = maskParams.netWidth;
    int net_height = maskParams.netHeight;
    int seg_channels = maskProtos.size[1];
    int seg_height = maskProtos.size[2];
    int seg_width = maskProtos.size[3];
    float mask_threshold = maskParams.maskThreshold;
    cv::Vec4f params = maskParams.params;
    cv::Size src_img_shape = maskParams.srcImgShape;

    cv::Rect temp_rect = output.box;
    //crop from mask_protos
    int rang_x = floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width);
    int rang_y = floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height);
    int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / net_width * seg_width) - rang_x;
    int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / net_height * seg_height) - rang_y;

    rang_w = MAX(rang_w, 1);
    rang_h = MAX(rang_h, 1);
    if (rang_x + rang_w > seg_width) {
        if (seg_width - rang_x > 0)
            rang_w = seg_width - rang_x;
        else
            rang_x -= 1;
    }
    if (rang_y + rang_h > seg_height) {
        if (seg_height - rang_y > 0)
            rang_h = seg_height - rang_y;
        else
            rang_y -= 1;
    }

    std::vector<cv::Range> roi_rangs;
    roi_rangs.push_back(cv::Range(0, 1));
    roi_rangs.push_back(cv::Range::all());
    roi_rangs.push_back(cv::Range(rang_y, rang_h + rang_y));
    roi_rangs.push_back(cv::Range(rang_x, rang_w + rang_x));

    //crop
    cv::Mat temp_mask_protos = maskProtos(roi_rangs).clone();
    cv::Mat protos = temp_mask_protos.reshape(0, { seg_channels,rang_w * rang_h });
    cv::Mat matmul_res = (maskProposals * protos).t();
    cv::Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
    cv::Mat dest, mask;

    //sigmoid
    cv::exp(-masks_feature, dest);
    dest = 1.0 / (1.0 + dest);

    int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
    int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
    int width = ceil(net_width / seg_width * rang_w / params[0]);
    int height = ceil(net_height / seg_height * rang_h / params[1]);

    resize(dest, mask, cv::Size(width, height), cv::INTER_NEAREST);
    cv::Rect mask_rect = temp_rect - cv::Point(left, top);
    mask_rect &= cv::Rect(0, 0, width, height);
    mask = mask(mask_rect) > mask_threshold;
    if (mask.rows != temp_rect.height || mask.cols != temp_rect.width) { //https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp/pull/30
        resize(mask, mask, temp_rect.size(), cv::INTER_NEAREST);
    }
    output.boxMask = mask;

}

int main()
{
    
    /*
    auto net = cv::dnn::readNetFromONNX("human.onnx");
    
    if (net.empty()) {
        std::cerr << "Fehler beim Laden des Modells!" << std::endl;
        return -1;
    }
    std::cout << "YOLOv8 ONNX-Modell erfolgreich geladen!" << std::endl;

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    */

    // Start webserver in thread
    std::thread t([] {
        drogon::app()
            .addListener("0.0.0.0", 80)
            .loadConfigFile("./drogon.json")
            .run();
        });

    // Start cam
    cv::VideoCapture vc(0, cv::CAP_DSHOW);
    
    FilterBlur* fb = new FilterBlur("human.onnx");

    if (vc.isOpened())
    {
        while (true)
        {
            cv::Mat imgCap;
            vc >> imgCap;

            if (!imgCap.empty())
            {
                /*
                cv::resize(imgCap, imgCap, { NET_WIDTH,NET_HEIGHT });

                cv::Mat netInputImg;
                cv::Vec4d params;
                LetterBox(imgCap, netInputImg, params, cv::Size(NET_WIDTH, NET_HEIGHT));
                auto matBlob = cv::dnn::blobFromImage(netInputImg, 1.0 / 255.0, { NET_WIDTH, NET_HEIGHT }, cv::Scalar(), true, false);
                net.setInput(matBlob);

                std::vector<cv::Mat> outputs;
                net.forward(outputs, net.getUnconnectedOutLayersNames());
                cv::Mat output0 = cv::Mat(cv::Size(outputs[0].size[2], outputs[0].size[1]), CV_32F, (float*)outputs[0].data).t();

                float _classThreshold = 0.5;
                float _nmsThreshold = 0.45;
                float _maskThreshold = 0.35;

                int rows = output0.rows;
                int net_width = output0.cols;
                int socre_array_length = net_width - 4 - outputs[1].size[1];
                float* pdata = (float*)output0.data;

                std::vector<std::vector<float>> picked_proposals;
                std::vector<int> class_ids;
                std::vector<float> confidences;
                std::vector<cv::Rect> boxes;

                for (int r = 0; r < rows; ++r)
                {
                    cv::Mat scores(1, socre_array_length, CV_32FC1, pdata + 4);
                    cv::Point classIdPoint;
                    double max_class_socre;
                    minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
                    max_class_socre = (float)max_class_socre;

                    //std::cout << max_class_socre << std::endl;

                    if (max_class_socre >= _classThreshold)
                    {
                        std::vector<float> temp_proto(pdata + 4 + socre_array_length, pdata + net_width);
                        picked_proposals.push_back(temp_proto);
                        //rect [x,y,w,h]
                        float x = (pdata[0] - params[2]) / params[0];
                        float y = (pdata[1] - params[3]) / params[1];
                        float w = pdata[2] / params[0];
                        float h = pdata[3] / params[1];
                        int left = MAX(int(x - 0.5 * w + 0.5), 0);
                        int top = MAX(int(y - 0.5 * h + 0.5), 0);
                        class_ids.push_back(classIdPoint.x);
                        confidences.push_back(max_class_socre);
                        boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
                    }
                    pdata += net_width;//next line
                }

                std::vector<OutputParams> output;
                std::vector<int> nms_result;
                cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
                std::vector<std::vector<float>> temp_mask_proposals;
                cv::Rect holeImgRect(0, 0, imgCap.cols, imgCap.rows);
                for (int i = 0; i < nms_result.size(); ++i) {

                    int idx = nms_result[i];
                    OutputParams result;
                    result.id = class_ids[idx];
                    result.confidence = confidences[idx];
                    result.box = boxes[idx] & holeImgRect;
                    if (result.box.area() < 1)
                        continue;
                    temp_mask_proposals.push_back(picked_proposals[idx]);
                    output.push_back(result);
                }

                MaskParams mask_params;
                mask_params.params = params;
                mask_params.srcImgShape = imgCap.size();
                mask_params.netHeight = NET_WIDTH;
                mask_params.netWidth = NET_HEIGHT;
                mask_params.maskThreshold = _maskThreshold;
                for (int i = 0; i < temp_mask_proposals.size(); ++i) {
                    GetMask2(cv::Mat(temp_mask_proposals[i]).t(), outputs[1], output[i], mask_params);
                }

                // Draw masks
                cv::Mat imgMask(imgCap.size(),CV_8UC1, cv::Scalar::all(255));

                for (auto& o : output)
                {
                    cv::Mat imgROI = imgMask(o.box);
                    imgROI.setTo(cv::Scalar(0), o.boxMask);
                }

                cv::Mat blurred, softMask;
                cv::GaussianBlur(imgCap, blurred, cv::Size(51, 51), 0);
                cv::GaussianBlur(imgMask, softMask, cv::Size(31, 31), 0);
                softMask.convertTo(softMask, CV_32F, 1.0 / 255.0);
                imgCap.convertTo(imgCap, CV_32F);
                blurred.convertTo(blurred, CV_32F);

                cv::Mat resultF = cv::Mat::zeros(imgCap.size(), imgCap.type());

                std::vector<cv::Mat> channelsImage, channelsBlurred, channelsResult;
                cv::split(imgCap, channelsImage);
                cv::split(blurred, channelsBlurred);

                for (size_t i = 0; i < channelsImage.size(); i++) {
                    cv::Mat blended = channelsBlurred[i].mul(softMask) + channelsImage[i].mul(1.0 - softMask);
                    channelsResult.push_back(blended);
                }

                cv::merge(channelsResult, resultF);
                resultF.convertTo(imgCap, CV_8U);
                */

                cv::Mat out = fb->run(imgCap);

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