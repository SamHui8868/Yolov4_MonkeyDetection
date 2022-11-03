
#include <ws2tcpip.h>
#include <WINSOCK2.H>
#include <cstdio>
#include <windows.h>
#include <iostream>
#include <fstream>
#include <process.h>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <time.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2\imgproc\types_c.h>
#include <opencv2/core.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace cuda;
clock_t tStart = clock();
#define BUFFERSIZE 100
#define SPORT 4000
//#define SPORT 4100
#pragma comment(lib, "Ws2_32.lib")


std::vector<std::string> classes;
SOCKET connect_DATABASE(const char* ip, int port);

bool postprocess(Mat& frame, const std::vector<Mat>& out, Net& net);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

void callback(int pos, void* userdata);
bool Yolo(Mat target, Net net, vector<String> outNames);

float confThreshold = 0.5;
float nmsThreshold = 0.4;
float scale = 0.003921;
bool swapRB = 1;
int inpWidth = 416;
int inpHeight = 416;
string kWinName;
int win = 1;
int main()
{
    SOCKET rsck;
    char buf[BUFFERSIZE], IP[20];
    int i = 0;
    Mat  targetA, targetB, targetC, WarmUp;
    int A, B, C;
    A = 0;
    B = 0;
    C = 0;

    WSADATA			WsaData;
    if (0 != WSAStartup(MAKEWORD(1, 1), &WsaData)) {
        WSACleanup();																			  //
        exit(0);
    }
    //rsck = connect_DATABASE("127.20.0.1", SPORT);
    rsck = connect_DATABASE("192.168.1.7", SPORT);
    //rsck = connect_DATABASE("172.24.31.249", SPORT);
    //rsck = connect_DATABASE("172.20.10.2", SPORT);
     //rsck = connect_DATABASE("192.168.43.1", SPORT);


     // Open file with classes names.
    std::string file = "obj.names";
    std::ifstream ifs(file.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }
    // Load a model.
    Net net = readNet("yolov4-obj_4000.weights", "yolov4-obj.cfg");
    net.setPreferableBackend(0);
    net.setPreferableTarget(0);
    std::vector<String> outNames = net.getUnconnectedOutLayersNames();
    VideoCapture captureTest("Monkey.jpg");
    captureTest >> WarmUp;
    Yolo(WarmUp, net, outNames);
    while (true)
    {
        //VideoCapture captureA("C:/Users/user/Desktop/OPENCV_(no_thread)/PositionA/OBJ" + to_string(A)+".jpg");;
          //VideoCapture captureA("C:/Users/mings/Desktop/OPENCV_(no_thread)/PositionA/OBJ" + to_string(A) + ".jpg");
        VideoCapture captureA("C:/Users/sam88/Desktop/OPENCV_GPU/PositionA/OBJ" + to_string(A) + ".jpg");
        captureA >> targetA;
        if (!targetA.empty()) {
            if (Yolo(targetA, net, outNames) == true)
            {
                printf("Position A[%d] HAVE MONKEY!!!!\n", A);
                sprintf_s(buf, BUFFERSIZE, "1");
                send(rsck, buf, BUFFERSIZE, 0);
            }

            else
            {
              //  printf("Position A  Safe!!!\n");
            }
            A++;
            printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
        }
        //VideoCapture captureB("C:/Users/user/Desktop/OPENCV_(no_thread)/PositionB/OBJ" + to_string(B) + ".jpg");
        //VideoCapture captureB("C:/Users/mings/Desktop/OPENCV_(no_thread)/PositionB/OBJ" + to_string(B) + ".jpg");
        VideoCapture captureB("C:/Users/sam88/Desktop/OPENCV_GPU/PositionB/OBJ" + to_string(B) + ".jpg");
        captureB >> targetB;
        if (!targetB.empty()) {
            if (Yolo(targetB, net, outNames) == true)
            {
                printf("Position B[%d] HAVE MONKEY!!!!\n", B);
                sprintf_s(buf, BUFFERSIZE, "2");
                send(rsck, buf, BUFFERSIZE, 0);
            }

            else
            {
               // printf("Position B  Safe!!!\n");
            }


            B++;
            printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
        }
        //VideoCapture captureC("C:/Users/user/Desktop/OPENCV_(no_thread)/PositionC/OBJ" + to_string(C) + ".jpg");;
        //VideoCapture captureC("C:/Users/mings/Desktop/OPENCV_(no_thread)/PositionC/OBJ" + to_string(C) + ".jpg");
        VideoCapture captureC("C:/Users/sam88/Desktop/OPENCV_GPU/PositionC/OBJ" + to_string(C) + ".jpg");
        captureC >> targetC;
        if (!targetC.empty()) {
            if (Yolo(targetC, net, outNames) == true)
            {
                printf("Position C[%d] HAVE MONKEY!!!!\n", C);
                sprintf_s(buf, BUFFERSIZE, "3");
                send(rsck, buf, BUFFERSIZE, 0);
            }

            else
            {
                //printf("Position C  Safe!!!\n");
            }
            C++;
            printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
        }
    }
    return 0;
}

bool postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net)
{
    int flag = 0;
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    CV_Assert(outs.size() > 0);
    if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                    flag = 1;
                }
            }
        }
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
            box.x + box.width, box.y + box.height, frame);
    }
    if (flag == 1)
    {
        return TRUE;
    }
    else
        return FALSE;
}
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
        Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

void callback(int pos, void*)
{
    confThreshold = pos * 0.01f;
}

bool Yolo(Mat target, Net net, vector<String> outNames) {
    Mat blob;
    int flag = 0;
    // Create a window
    static const std::string kWinName = "Deep learning object detection in OpenCV";
    Size inpSize(inpWidth > 0 ? inpWidth : target.cols,
        inpHeight > 0 ? inpHeight : target.rows);
    blobFromImage(target, blob, scale, Size(127, 127), 0, swapRB, false);

    net.setInput(blob);
    if (net.getLayer(0)->outputNameToIndex("im_info") != -1)
    {
        resize(target, target, inpSize);
        Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
        net.setInput(imInfo, "im_info");
    }
    std::vector<Mat> outs;
    net.forward(outs, outNames);
    if (postprocess(target, outs, net) == TRUE)
    {
        flag = 1;
    }
    imshow(kWinName, target);
    waitKey(1);
    win++;
    if (flag == 1)
    {
        return TRUE;
    }
    else
        return FALSE;

}
SOCKET connect_DATABASE(const char* ip, int port)
{

    SOCKET			sc;							//自己的socket
    SOCKADDR_IN		sa;							//自己的SOCKADDR_IN結構
                                                                                              //
//---------------------------------------------建立SCOKET---------------------------------------------//
    sc = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);											  //
    if (INVALID_SOCKET == sc) {
        WSACleanup();																			  //
        exit(0);
    }																								  //

    sa.sin_family = AF_INET;																	  //
    sa.sin_port = htons(port);															  //
    inet_pton(AF_INET, ip, &(sa.sin_addr.s_addr));
    if (SOCKET_ERROR == connect(sc, (LPSOCKADDR)&sa, sizeof(sa))) {
        printf("connect server error!");//
        getchar();
        WSACleanup();
        exit(0);
    }

    return sc;
}