#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

class FilterBlur
{
    private:

        cv::dnn::Net net;

    public:

        static constexpr int NET_WIDTH = 640;
        static constexpr int NET_HEIGHT = 640;

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
            float maskThreshold = 0.5f;
            cv::Size srcImgShape;
            cv::Vec4d params;
        };

        /**
         * @brief Konstruktor: Lädt das ONNX-Modell.
         *
         * @param modelPath Pfad zum ONNX-Modell (Standard: "human.onnx").
         * @throws std::runtime_error wenn das Modell nicht geladen werden konnte.
         */
        FilterBlur(const std::string& modelPath = "human.onnx");

        /**
         * @brief Wendet das Netzwerk sowie den Masken- und Blur-Filter auf ein Bild an.
         *
         * @param inputImage Eingabebild.
         * @return cv::Mat Ergebnisbild.
         */
        cv::Mat run(const cv::Mat& inputImage);

        /**
         * @brief Erzeugt ein letterboxed Bild.
         *
         * @param image Eingangsbild.
         * @param outImage Ausgabebild (letterboxed).
         * @param params Parameter: [ratio_x, ratio_y, pad_left, pad_top].
         * @param newShape Zielgröße (Standard: 640x640).
         * @param autoShape Falls true, wird das Padding mod stride angepasst.
         * @param scaleFill Falls true, wird das Bild gestreckt, sodass kein Padding entsteht.
         * @param scaleUp Falls false, wird das Bild nur verkleinert.
         * @param stride Schrittweite (Standard: 32).
         * @param color Farbe für das Padding (Standard: cv::Scalar(114,114,114)).
         */
        static void LetterBox(const cv::Mat& image,
            cv::Mat& outImage,
            cv::Vec4d& params,
            const cv::Size& newShape = cv::Size(NET_WIDTH, NET_HEIGHT),
            bool autoShape = false,
            bool scaleFill = false,
            bool scaleUp = true,
            int stride = 32,
            const cv::Scalar& color = cv::Scalar(114, 114, 114));

    private:

        /**
         * @brief Extrahiert die Maske für eine erkannte Box.
         *
         * @param maskProposals Vektor (als cv::Mat) der Maskenvorschläge für die Box.
         * @param maskProtos Maskenprototypen aus dem Netz.
         * @param output Enthält die Box, in die die Maske geschrieben wird.
         * @param maskParams Parameter zur Skalierung/Transformation.
         */
        static void GetMask2(const cv::Mat& maskProposals,
            const cv::Mat& maskProtos,
            OutputParams& output,
            const MaskParams& maskParams);
};