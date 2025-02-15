
#include "FilterBlur.h"

FilterBlur::FilterBlur(const std::string& modelPath) {
    net = cv::dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
        std::cerr << "Fehler beim Laden des Modells!" << std::endl;
        throw std::runtime_error("Fehler beim Laden des Modells!");
    }
    std::cout << "YOLOv8 ONNX-Modell erfolgreich geladen!" << std::endl;
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

void FilterBlur::LetterBox(const cv::Mat& image,
    cv::Mat& outImage,
    cv::Vec4d& params,
    const cv::Size& newShape,
    bool autoShape,
    bool scaleFill,
    bool scaleUp,
    int stride,
    const cv::Scalar& color)
{
    // Dieser Block wird aktuell nicht ausgeführt – evtl. für Debug-Zwecke:
    if (false) {
        int maxLen = std::max(image.rows, image.cols);
        outImage = cv::Mat::zeros(cv::Size(maxLen, maxLen), CV_8UC3);
        image.copyTo(outImage(cv::Rect(0, 0, image.cols, image.rows)));
        params[0] = 1;
        params[1] = 1;
        params[2] = 0;
        params[3] = 0;
        return;
    }

    cv::Size shape = image.size();
    float r = std::min(static_cast<float>(newShape.height) / shape.height,
        static_cast<float>(newShape.width) / shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2] = { r, r };
    int new_un_pad[2] = { static_cast<int>(std::round(shape.width * r)),
                          static_cast<int>(std::round(shape.height * r)) };

    float dw = static_cast<float>(newShape.width - new_un_pad[0]);
    float dh = static_cast<float>(newShape.height - new_un_pad[1]);

    if (autoShape) {
        dw = static_cast<float>(static_cast<int>(dw) % stride);
        dh = static_cast<float>(static_cast<int>(dh) % stride);
    }
    else if (scaleFill) {
        dw = 0.0f;
        dh = 0.0f;
        new_un_pad[0] = newShape.width;
        new_un_pad[1] = newShape.height;
        ratio[0] = static_cast<float>(newShape.width) / shape.width;
        ratio[1] = static_cast<float>(newShape.height) / shape.height;
    }
    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != new_un_pad[0] || shape.height != new_un_pad[1])
        cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
    else
        outImage = image.clone();

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));
    params[0] = ratio[0];
    params[1] = ratio[1];
    params[2] = left;
    params[3] = top;
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}



cv::Mat FilterBlur::run(const cv::Mat& inputImage) {
    if (inputImage.empty()) {
        std::cerr << "Eingabebild ist leer!" << std::endl;
        return cv::Mat();
    }
    cv::Mat imgCap = inputImage.clone();

    // Resize auf NET_WIDTH x NET_HEIGHT
    cv::resize(imgCap, imgCap, cv::Size(NET_WIDTH, NET_HEIGHT));

    cv::Mat netInputImg;
    cv::Vec4d params;
    LetterBox(imgCap, netInputImg, params, cv::Size(NET_WIDTH, NET_HEIGHT));

    cv::Mat matBlob = cv::dnn::blobFromImage(netInputImg, 1.0 / 255.0,
        cv::Size(NET_WIDTH, NET_HEIGHT),
        cv::Scalar(), true, false);
    net.setInput(matBlob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Umformen des ersten Outputs
    cv::Mat output0 = cv::Mat(cv::Size(outputs[0].size[2],
        outputs[0].size[1]),
        CV_32F, (float*)outputs[0].data).t();

    // Schwellenwerte
    float _classThreshold = 0.5f;
    float _nmsThreshold = 0.45f;
    float _maskThreshold = 0.35f;

    int rows = output0.rows;
    int net_width = output0.cols;
    int score_array_length = net_width - 4 - outputs[1].size[1];

    float* pdata = (float*)output0.data;

    std::vector<std::vector<float>> picked_proposals;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int r = 0; r < rows; ++r) {
        cv::Mat scores(1, score_array_length, CV_32FC1, pdata + 4);
        cv::Point classIdPoint;
        double max_class_score;
        cv::minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);
        max_class_score = static_cast<float>(max_class_score);

        if (max_class_score >= _classThreshold) {
            std::vector<float> temp_proto(pdata + 4 + score_array_length, pdata + net_width);
            picked_proposals.push_back(temp_proto);
            // Berechnung der Box-Koordinaten (rect: [x,y,w,h])
            float x = (pdata[0] - params[2]) / params[0];
            float y = (pdata[1] - params[3]) / params[1];
            float w = pdata[2] / params[0];
            float h = pdata[3] / params[1];
            int left = std::max(int(x - 0.5 * w + 0.5), 0);
            int top = std::max(int(y - 0.5 * h + 0.5), 0);
            class_ids.push_back(classIdPoint.x);
            confidences.push_back(max_class_score);
            boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
        }
        pdata += net_width;
    }

    std::vector<OutputParams> outputResults;
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
        outputResults.push_back(result);
    }

    MaskParams mask_params;
    mask_params.params = params;
    mask_params.srcImgShape = imgCap.size();
    mask_params.netHeight = NET_WIDTH;
    mask_params.netWidth = NET_HEIGHT;
    mask_params.maskThreshold = _maskThreshold;
    for (int i = 0; i < temp_mask_proposals.size(); ++i) {
        cv::Mat maskProposalMat(temp_mask_proposals[i], true);
        maskProposalMat = maskProposalMat.t();
        GetMask2(maskProposalMat, outputs[1], outputResults[i], mask_params);
    }

    // Zeichne Masken
    cv::Mat imgMask(imgCap.size(), CV_8UC1, cv::Scalar::all(255));
    for (auto& o : outputResults) {
        cv::Mat imgROI = imgMask(o.box);
        imgROI.setTo(cv::Scalar(0), o.boxMask);
    }

    // Weichzeichnen und Mischungslogik
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
    resultF.convertTo(resultF, CV_8U);

    return resultF;
}

void FilterBlur::GetMask2(const cv::Mat& maskProposals,
    const cv::Mat& maskProtos,
    OutputParams& output,
    const MaskParams& maskParams)
{
    int net_width = maskParams.netWidth;
    int net_height = maskParams.netHeight;
    int seg_channels = maskProtos.size[1];
    int seg_height = maskProtos.size[2];
    int seg_width = maskProtos.size[3];
    float mask_threshold = maskParams.maskThreshold;
    cv::Vec4f params(
        static_cast<float>(maskParams.params[0]),
        static_cast<float>(maskParams.params[1]),
        static_cast<float>(maskParams.params[2]),
        static_cast<float>(maskParams.params[3])
    );
    cv::Size src_img_shape = maskParams.srcImgShape;

    cv::Rect temp_rect = output.box;
    // Bestimme den ROI in den Maskenprototypen:
    int rang_x = static_cast<int>(std::floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width));
    int rang_y = static_cast<int>(std::floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height));
    int rang_w = static_cast<int>(std::ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / static_cast<float>(net_width) * seg_width)) - rang_x;
    int rang_h = static_cast<int>(std::ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / static_cast<float>(net_height) * seg_height)) - rang_y;

    rang_w = std::max(rang_w, 1);
    rang_h = std::max(rang_h, 1);
    if (rang_x + rang_w > seg_width) {
        if (seg_width - rang_x > 0)
            rang_w = seg_width - rang_x;
        else
            rang_x = std::max(rang_x - 1, 0);
    }
    if (rang_y + rang_h > seg_height) {
        if (seg_height - rang_y > 0)
            rang_h = seg_height - rang_y;
        else
            rang_y = std::max(rang_y - 1, 0);
    }

    std::vector<cv::Range> roi_ranges;
    roi_ranges.push_back(cv::Range(0, 1));
    roi_ranges.push_back(cv::Range::all());
    roi_ranges.push_back(cv::Range(rang_y, rang_y + rang_h));
    roi_ranges.push_back(cv::Range(rang_x, rang_x + rang_w));

    // Zuschneiden der Prototypen
    cv::Mat temp_mask_protos = maskProtos(roi_ranges).clone();
    cv::Mat protos = temp_mask_protos.reshape(0, { seg_channels, rang_w * rang_h });
    cv::Mat matmul_res = (maskProposals * protos).t();
    cv::Mat masks_feature = matmul_res.reshape(1, { rang_h, rang_w });
    cv::Mat dest, mask;

    // Sigmoid
    cv::exp(-masks_feature, dest);
    dest = 1.0 / (1.0 + dest);

    int left = static_cast<int>(std::floor((net_width / static_cast<float>(seg_width) * rang_x - params[2]) / params[0]));
    int top = static_cast<int>(std::floor((net_height / static_cast<float>(seg_height) * rang_y - params[3]) / params[1]));
    int width = static_cast<int>(std::ceil((net_width / static_cast<float>(seg_width) * rang_w) / params[0]));
    int height = static_cast<int>(std::ceil((net_height / static_cast<float>(seg_height) * rang_h) / params[1]));

    cv::resize(dest, mask, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    cv::Rect mask_rect = temp_rect - cv::Point(left, top);
    mask_rect &= cv::Rect(0, 0, width, height);
    mask = mask(mask_rect) > mask_threshold;
    if (mask.rows != temp_rect.height || mask.cols != temp_rect.width) {
        cv::resize(mask, mask, temp_rect.size(), 0, 0, cv::INTER_NEAREST);
    }
    output.boxMask = mask;
}
