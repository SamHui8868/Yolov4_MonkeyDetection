
#include <windows.h>
#include <iostream>
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
#include <opencv2/imgproc\types_c.h>
#include <opencv2/core.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
using namespace cv;
using namespace std;
clock_t tStart = clock();
Mat preporcess(Mat input);
//void OriginPlay(void* p);

int main()
{
    stringstream ss;
    string folderName1 = "PositionA";
    string folderName2 = "PositionB";
    string folderName3 = "PositionC";
    string folderCreateCommand1 = "mkdir " + folderName1;
    string folderCreateCommand2 = "mkdir " + folderName2;
    string folderCreateCommand3 = "mkdir " + folderName3;
    system(folderCreateCommand1.c_str());
    system(folderCreateCommand2.c_str());
    system(folderCreateCommand3.c_str());
    int x, y, Area, oldArea, A, B, C, xc;
    oldArea = 0;
    char Position;
    Position = 'A';
    char file_name[100];
    int no_of_target = 0;
    int Change_bkg = 0;
    int i = 0;

    A = 0;
    B = 0;
    C = 0;
    Mat frame, fgMask, gray, diff, image, blurbkg, blur, dilatef, thr, target, output, er, origin;
    //VideoCapture capture("monkey(origin).mp4");
    VideoCapture capture("rtsp://192.168.1.4/v1");
    //VideoCapture capture(0);
    //VideoCapture Origincapture("DEMO.mp4");
    //VideoCapture bkgcapture("monkey(origin).mp4");
    //bkgcapture >> blurbkg;
    capture.read(frame);
    blurbkg = frame.clone();
    resize(blurbkg, blurbkg, Size(480, 270));
    blurbkg = preporcess(blurbkg);
    while (true) {
        // Origincapture.read(origin);
        capture.read(frame);
        origin = frame.clone();
        if (i == 30)
        {
            blurbkg = frame.clone();
            resize(blurbkg, blurbkg, Size(480, 270));
            blurbkg = preporcess(blurbkg);
            i = 0;
        }
        if (frame.empty())
            break;
        resize(frame, frame, Size(480, 270));
        output = frame.clone();
        // grey scale
        blur = preporcess(frame);
        //absdiff
        absdiff(blur, blurbkg, diff);
        threshold(diff, thr, 40, 255, THRESH_BINARY);
        erode(thr, er, getStructuringElement(MORPH_RECT, Size(5, 5)), Point(-1, -1), 2);
        dilate(er, dilatef, getStructuringElement(MORPH_RECT, Size(25, 25)), Point(-1, -1), 2);

        Mat canny_output;
        Canny(dilatef, canny_output, 30, 60);
        vector<vector<Point> > contours;
        findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
        vector<vector<Point> > contours_poly(contours.size());
        vector<Rect> boundRect(contours.size());
        vector<Point2f>centers(contours.size());
        vector<float>radius(contours.size());
        for (size_t i = 0; i < contours.size(); i++)
        {
            approxPolyDP(contours[i], contours_poly[i], 3, true);
            boundRect[i] = boundingRect(contours_poly[i]);
        }
        Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
        for (size_t i = 0; i < contours.size(); i++)
        {
            Scalar color = Scalar(256, 256, 256);
            x = boundRect[i].br().x - boundRect[i].tl().x;//width
            y = boundRect[i].tl().y - boundRect[i].br().y;//length
            xc = boundRect[i].tl().x + x / 2;
            Area = abs(x * y);
            printf("x=%d  Position %c [%d]\n", xc, Position, Area);
            if ((Area >= 150 * 150) && (abs(oldArea - Area) >= 600))
           // if ((Area >= 100 * 100) && (abs(oldArea - Area) >= 100))
            {
                if (xc <= 160)
                {
                    ss << folderName1 << "/" << "OBJ" << A << ".jpg";
                    Position = 'A';
                    A++;
                }
                if ((xc > 160) && (xc <= 320))
                {
                    ss << folderName2 << "/" << "OBJ" << B << ".jpg";
                    Position = 'B';
                    B++;
                }
                if ((xc > 320) && (xc <= 480))
                {
                    ss << folderName3 << "/" << "OBJ" << C << ".jpg";
                    Position = 'C';
                    C++;
                }
                rectangle(output, boundRect[i].tl(), boundRect[i].br(), color, 2);
                target = frame(boundRect[i]);
                string fullPath = ss.str();
                ss.str("");
                imwrite(fullPath, target);
                no_of_target++;
                oldArea = Area;
            }
        }
        imshow("Frame", frame);
        imshow("Origin", origin);
        imshow("Diff", thr);
        //imshow("Blur", blur);
        imshow("Output", output);
        imshow("erode", er);
        imshow("Dilate", dilatef);
        waitKey(1);
        i++;
    }
    printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}
Mat preporcess(Mat input)
{
    Mat output;
    cvtColor(input, input, CV_BGR2GRAY);
    GaussianBlur(input, output, Size(7, 7), 0);
    return output;
}

//void OriginPlay(void*p)
//{
//    Mat origin;
//    VideoCapture Origincapture("monkey.mp4");
//    while (true)
//    {
//        Origincapture >> origin;
//        imshow("Origin", origin);
//        Sleep(1);
//    }
//
//}

