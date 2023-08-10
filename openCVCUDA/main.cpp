// Object detection based on YOLO 
// Using VS2022 openCV-4.7.0 cuda11.7 (cudnn8.9.3 July 11 2023)
// Included hi-res timer for FPS
// Added various pre trained models
// Target CUDA

#include <fstream>
#include <iostream>
#include <conio.h>
#include <windows.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cudnn.h> // Include cudnn11.7 header

using namespace std;
using namespace cv;
using namespace dnn;

// Initialize the parameters
double confThreshold = 0.60; // Confidence threshold
double nmsThreshold = 0.75;  // Non-maximum suppression threshold

vector<string> classes;
void postprocess(Mat& frame, const vector<Mat>& out);                                           // Remove the bounding boxes with low confidence using non-maxima suppression
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);   // Draw the predicted bounding box
vector<String> getOutputsNames(const Net& net);                                                 // Get the names of the output layers

int main(void)
{
    Mat blob, blobb, frame, frame2, frame3, frame4, frame5; // Create surfaces
    int width, height, reswidth, resheight;
    float sx = 0.75, sy = 0.75;
    int dCount = 0, cores = 0, mp = 0, dev, driverVersion = 0, runtimeVersion = 0;
    LARGE_INTEGER fr, t1, t2;

    // Detect OpenCV and CUDA versions
    cout << "OpenCV Version " << CV_VERSION << "\n";
    cudaError_t error_id = cudaGetDeviceCount(&dCount);
    if (dCount == 0) { printf("There are no available device(s) that support CUDA\n"); }
    else { printf("Detected %d CUDA device(s)\n", dCount); }

    // Loop through avalible devices
    for (dev = 0; dev < dCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        mp = deviceProp.multiProcessorCount;
        printf("Device %2d        : %s\n", dev, deviceProp.name);
        printf("CUDA Version     : %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
        printf("Revision         : %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Total memory     : %.0f MBytes\n", deviceProp.totalGlobalMem / 1048576.0f);
        printf("Multiprocessors  : %d\n", mp);
        printf("GPU Clock rate   : %.2f GHz\n", deviceProp.clockRate * 1e-6f);
        printf("Memory Clock rate: %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
        printf("Memory Bus Width : %d-bit\n", deviceProp.memoryBusWidth);

        switch (deviceProp.major)
        {
        case 2: // Fermi
            printf("Architecture     : Fermi\n");
            if (deviceProp.minor == 1) cores = mp * 48;
            else cores = mp * 32; break;
        case 3: // Kepler
            printf("Architecture     : Kepler\n");
            cores = mp * 192; break;
        case 5: // Maxwell
            printf("Architecture     : Maxwell\n");
            cores = mp * 128; break;
        case 6: // Pascal
            printf("Architecture     : Pascal\n");
            if ((deviceProp.minor == 1) || (deviceProp.minor == 2)) cores = mp * 128;
            else if (deviceProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n"); break;
        case 7: // Volta and Turing
            if ((deviceProp.minor == 0) || (deviceProp.minor == 5)) cores = mp * 64;
            else printf("Unknown device type\n"); break;
        case 8: // Ampere
            if (deviceProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n"); break;
        default:
            printf("Unknown device type\n"); break;
        } // End switch
        printf("Cores            : %d (%dx%d)\n\n", cores, mp, cores / mp);
    } // End of devicecount loop

    // Get the configuration and weight files for the model
    string modelConfiguration = "yolov4.cfg";
    string modelWeights = "yolov4.weights";
    //string modelConfiguration = "yolov3.cfg";
    //string modelWeights       = "yolov3.weights";
    //string modelConfiguration = "yolov3-tiny.cfg";
    //string modelWeights       = "yolov3-tiny.weights";
    string classesFile = "coco.names"; // List of 80 objects model is trained on
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Load the network
    Net net = readNet(modelConfiguration, modelWeights);

    // Set backend and target to use cuda
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);

    // Load the webcam or MP4 file
    //VideoCapture capture(0,CAP_DSHOW);
    VideoCapture capture("Traffic1.mp4", CAP_FFMPEG);
    //VideoCapture capture("newyork1970.mp4", CAP_FFMPEG);

    if (!capture.isOpened()) { cerr << "Unable to open video" << endl; _getch(); return 0; }
    capture.read(frame);

    reswidth = capture.get(CAP_PROP_FRAME_WIDTH);   // Comment if resizing video
    resheight = capture.get(CAP_PROP_FRAME_HEIGHT);
    //capture.set(CAP_PROP_FRAME_WIDTH, 1280.0);    // Uncomment to resize video input, causes reduced performance but allows 16:9 video input
    //capture.set(CAP_PROP_FRAME_HEIGHT, 1024.0);

// Video frame loop
    while (true)
    {
        QueryPerformanceCounter(&t1);

        capture >> frame;
        capture >> frame2;
        // width = 1280.0; // Uncomment to resize video input, causes reduced performance but allows 16:9 video input
        // height = 1024.0;
        width = capture.get(CAP_PROP_FRAME_WIDTH);      // Comment this
        height = capture.get(CAP_PROP_FRAME_HEIGHT);    // if resizing video input
        if (frame.empty()) break;

        // Reduce input frame size
        resize(frame, frame, Size(reswidth, resheight));

        // Filters to reduce computational power
        //bitwise_not(frame,frame);                         // Invert colours of original frame
        //GaussianBlur(frame, frame, cv::Size(5, 5), 0.8);  // (input image, output image, smoothing window width and height in pixels, sigma value)

        // Create 4D blobs
        blobFromImage(frame, blob, 1 / 255.0, Size(width, height), Scalar(0, 0, 0), true, false, CV_32F);
        net.setInput(blob);

        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));

        // Remove the bounding boxes with low confidence
        postprocess(frame, outs);

        QueryPerformanceCounter(&t2);
        QueryPerformanceFrequency(&fr);
        double t = (t2.QuadPart - t1.QuadPart) / (double)fr.QuadPart;

        // Print info in frame
        rectangle(frame, Point(0, 0), Point(200, 20), Scalar(255, 255, 255), FILLED);
        string label = format("FPS %.2fs %dx%d", 1.0 / t, reswidth, resheight);
        putText(frame, label, Point(5, 15), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 0));
        putText(frame, label, Point(5, 15), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 0));

        imshow("Object Detection", frame);

        // Get input from keyboard to quit
        int keyboard = waitKey(30); if (keyboard == 'q' || keyboard == 27) break; // Get input from keyboard to quit
    } // End of video frame loop
    return 0;
} // End of main


 // Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        /* Scan through all the bounding box outputs from the network and keep only the
         ones with high confidence scores. Assign the box's class label as the class
         with the highest score for the box.*/
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
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
            }
        }
    }


    // Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
    }


}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    // Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255 - classId * 3, 0, 0), 1);//
    //                                                                               B   G    R
    if (classId == 0)rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 1);    // Person
    if (classId == 1)rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 0, 0), 1);    // Bicycle
    if (classId == 2)rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 1);    // Car
    if (classId == 5)rectangle(frame, Point(left, top), Point(right, bottom), Scalar(127, 255, 0), 1);  // Bus
    if (classId == 7)rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 0, 0), 1);    // Truck

    // Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    // Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_PLAIN, 1.0, 1, &baseLine);
    top = max(top, labelSize.height);
    putText(frame, label, Point(left + 1, top), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 0), 1);
    putText(frame, label, Point(left, top), FONT_HERSHEY_PLAIN, 1.0, Scalar(255, 255, 255), 1);

}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        // Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        // Get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}
