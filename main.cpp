// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "helpers/common.h"
#include "helpers/Functions.h"

void testOpenImage() {
    char fname[MAX_PATH];
    while (openFileDlg(fname)) {
        Mat src;
        src = imread(fname);
        imshow("image", src);
        waitKey();
    }
}

void testOpenImagesFld() {
    char folderName[MAX_PATH];
    if (openFolderDlg(folderName) == 0)
        return;
    char fname[MAX_PATH];
    FileGetter fg(folderName, "bmp");
    while (fg.getNextAbsFile(fname)) {
        Mat src;
        src = imread(fname);
        imshow(fg.getFoundFileName(), src);
        if (waitKey() == 27) //ESC pressed
            break;
    }
}

void testImageOpenAndSave() {
    Mat src, dst;

    src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);    // Read the image

    if (!src.data)    // Check for invalid input
    {
        printf("Could not open or find the image\n");
        return;
    }

    // Get the image resolution
    Size src_size = Size(src.cols, src.rows);

    // Display window
    const char *WIN_SRC = "Src"; //window for the source image
    namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
    cvMoveWindow(WIN_SRC, 0, 0);

    const char *WIN_DST = "Dst"; //window for the destination (processed) image
    namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
    cvMoveWindow(WIN_DST, src_size.width + 10, 0);

    cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

    imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

    imshow(WIN_SRC, src);
    imshow(WIN_DST, dst);

    printf("Press any key to continue ...\n");
    waitKey(0);
}

void testNegativeImage() {
    char fname[MAX_PATH];
    while (openFileDlg(fname)) {
        double t = (double) getTickCount(); // Get the current time [s]

        Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
        int height = src.rows;
        int width = src.cols;
        Mat dst = Mat(height, width, CV_8UC1);
        // Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
        // Varianta ineficienta (lenta)
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                uchar val = src.at<uchar>(i, j);
                uchar neg = MAX_PATH - val;
                dst.at<uchar>(i, j) = neg;
            }
        }

        // Get the current time again and compute the time difference [s]
        t = ((double) getTickCount() - t) / getTickFrequency();
        // Print (in the console window) the processing time in [ms]
        printf("Time = %.3f [ms]\n", t * 1000);

        imshow("input image", src);
        imshow("negative image", dst);
        waitKey();
    }
}

void testParcurgereSimplaDiblookStyle() {
    char fname[MAX_PATH];
    while (openFileDlg(fname)) {
        Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
        int height = src.rows;
        int width = src.cols;
        int w = src.step; // no dword alignment is done !!!
        Mat dst = src.clone();

        double t = (double) getTickCount(); // Get the current time [s]

        // the fastest approach using the “diblook style”
        uchar *lpSrc = src.data;
        uchar *lpDst = dst.data;
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
                uchar val = lpSrc[i * w + j];
                lpDst[i * w + j] = 255 - val;
                /* sau puteti scrie:
                uchar val = lpSrc[i*width + j];
                lpDst[i*width + j] = 255 - val;
                //	w = width pt. imagini cu 8 biti / pixel
                //	w = 3*width pt. imagini cu 24 biti / pixel
                */
            }

        // Get the current time again and compute the time difference [s]
        t = ((double) getTickCount() - t) / getTickFrequency();
        // Print (in the console window) the processing time in [ms]
        printf("Time = %.3f [ms]\n", t * 1000);

        imshow("input image", src);
        imshow("negative image", dst);
        waitKey();
    }
}

void testColor2Gray() {
    char fname[MAX_PATH];
    while (openFileDlg(fname)) {
        Mat src = imread(fname);

        int height = src.rows;
        int width = src.cols;

        Mat dst = Mat(height, width, CV_8UC1);

        // Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
        // Varianta ineficienta (lenta)
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                Vec3b v3 = src.at<Vec3b>(i, j);
                uchar b = v3[0];
                uchar g = v3[1];
                uchar r = v3[2];
                dst.at<uchar>(i, j) = (r + g + b) / 3;
            }
        }

        imshow("input image", src);
        imshow("gray image", dst);
        waitKey();
    }
}

void testBGR2HSV() {
    char fname[MAX_PATH];
    while (openFileDlg(fname)) {
        Mat src = imread(fname);
        int height = src.rows;
        int width = src.cols;
        int w = src.step; // latimea in octeti a unei linii de imagine

        Mat dstH = Mat(height, width, CV_8UC1);
        Mat dstS = Mat(height, width, CV_8UC1);
        Mat dstV = Mat(height, width, CV_8UC1);

        // definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
        uchar *dstDataPtrH = dstH.data;
        uchar *dstDataPtrS = dstS.data;
        uchar *dstDataPtrV = dstV.data;

        Mat hsvImg;
        cvtColor(src, hsvImg, CV_BGR2HSV);
        // definire pointer la matricea (24 biti/pixeli) a imaginii HSV
        uchar *hsvDataPtr = hsvImg.data;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int hi = i * width * 3 + j * 3;
                // sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
                int gi = i * width + j;

                dstDataPtrH[gi] = hsvDataPtr[hi] * 510 / 360;        // H = 0 .. 255
                dstDataPtrS[gi] = hsvDataPtr[hi + 1];            // S = 0 .. 255
                dstDataPtrV[gi] = hsvDataPtr[hi + 2];            // V = 0 .. 255
            }
        }

        imshow("input image", src);
        imshow("H", dstH);
        imshow("S", dstS);
        imshow("V", dstV);
        waitKey();
    }
}

void testResize() {
    char fname[MAX_PATH];
    while (openFileDlg(fname)) {
        Mat src;
        src = imread(fname);
        Mat dst1, dst2;
        //without interpolation
        resizeImg(src, dst1, 320, false);
        //with interpolation
        resizeImg(src, dst2, 320, true);
        imshow("input image", src);
        imshow("resized image (without interpolation)", dst1);
        imshow("resized image (with interpolation)", dst2);
        waitKey();
    }
}

void testCanny() {
    char fname[MAX_PATH];
    while (openFileDlg(fname)) {
        Mat src, dst, gauss;
        src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
        int k = 0.4;
        int pH = 50;
        int pL = k * pH;
        GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
        Canny(gauss, dst, pL, pH, 3);
        imshow("input image", src);
        imshow("canny", dst);
        waitKey();
    }
}

void testVideoSequence() {
    VideoCapture cap("Videos/rubic.avi"); // off-line video from file
    //VideoCapture cap(0);	// live video from web cam
    if (!cap.isOpened()) {
        printf("Cannot open video capture device.\n");
        waitKey();
        return;
    }

    Mat edges;
    Mat frame;
    char c;

    while (cap.read(frame)) {
        Mat grayFrame;
        cvtColor(frame, grayFrame, CV_BGR2GRAY);
        Canny(grayFrame, edges, 40, 100, 3);
        imshow("source", frame);
        imshow("gray", grayFrame);
        imshow("edges", edges);
        c = cvWaitKey();  // waits a key press to advance to the next frame
        if (c == 27) {
            // press ESC to exit
            printf("ESC pressed - capture finished\n");
            break;  //ESC pressed
        };
    }
}


void testSnap() {
    VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
    if (!cap.isOpened()) // openenig the video device failed
    {
        printf("Cannot open video capture device.\n");
        return;
    }

    Mat frame;
    char numberStr[256];
    char fileName[256];

    // video resolution
    Size capS = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),
                     (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));

    // Display window
    const char *WIN_SRC = "Src"; //window for the source frame
    namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
    cvMoveWindow(WIN_SRC, 0, 0);

    const char *WIN_DST = "Snapped"; //window for showing the snapped frame
    namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
    cvMoveWindow(WIN_DST, capS.width + 10, 0);

    char c;
    int frameNum = -1;
    int frameCount = 0;

    for (;;) {
        cap >> frame; // get a new frame from camera
        if (frame.empty()) {
            printf("End of the video file\n");
            break;
        }

        ++frameNum;

        imshow(WIN_SRC, frame);

        c = cvWaitKey(10);  // waits a key press to advance to the next frame
        if (c == 27) {
            // press ESC to exit
            printf("ESC pressed - capture finished");
            break;  //ESC pressed
        }
        if (c == 115) { //'s' pressed - snapp the image to a file
            frameCount++;
            fileName[0] = NULL;
            sprintf(numberStr, "%d", frameCount);
            strcat(fileName, "Images/A");
            strcat(fileName, numberStr);
            strcat(fileName, ".bmp");
            bool bSuccess = imwrite(fileName, frame);
            if (!bSuccess) {
                printf("Error writing the snapped image\n");
            } else
                imshow(WIN_DST, frame);
        }
    }

}

void MyCallBackFunc(int event, int x, int y, int flags, void *param) {
    //More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
    Mat *src = (Mat *) param;
    Mat hsv;
    cvtColor(*src, hsv, CV_BGR2HSV);
    if (event == CV_EVENT_LBUTTONDOWN) {
        printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
               x, y,
               (int) (*src).at<Vec3b>(y, x)[2],
               (int) (*src).at<Vec3b>(y, x)[1],
               (int) (*src).at<Vec3b>(y, x)[0]);
        printf("Pos(x,y): %d,%d  Color(HSV): H:%d, S:%d, V:%d\n",
               x, y,
               (int) (hsv).at<Vec3b>(y, x)[2],
               (int) (hsv).at<Vec3b>(y, x)[1],
               (int) (hsv).at<Vec3b>(y, x)[0]);

    }
}

void testMouseClick() {
    Mat src;
    // Read image from file
    char fname[MAX_PATH];
    while (openFileDlg(fname)) {
        src = imread(fname);
        //Create a window
        namedWindow("My Window", 1);

        //set the callback function for any mouse event
        setMouseCallback("My Window", MyCallBackFunc, &src);

        //show the image
        imshow("My Window", src);

        // Wait until user press some key
        waitKey(0);
    }
}

////////////////////////////////////////////////// LAB 2  /////////////////////////////////////////////////////////////
//compute histogram
void compute_histogram(int *a, Mat src, int n) {

    int height = src.rows;
    int width = src.cols;
    //intializare vector
    for (int i = 0; i < n; i++) {
        a[i] = 0;
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            a[src.at<uchar>(i, j)] += 1;
        }
    }

}


void showHistogramHSV() {

    char fname[MAX_PATH];
    while (openFileDlg(fname)) {

        Mat src = imread(fname);
        int height = src.rows;
        int width = src.cols;

        Mat dstH = Mat(height, width, CV_8UC1);
        Mat dstS = Mat(height, width, CV_8UC1);
        Mat dstV = Mat(height, width, CV_8UC1);

        // definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
        uchar *dstDataPtrH = dstH.data;
        uchar *dstDataPtrS = dstS.data;
        uchar *dstDataPtrV = dstV.data;

        Mat hsvImg;
        cvtColor(src, hsvImg, CV_BGR2HSV);
        // definire pointer la matricea (24 biti/pixeli) a imaginii HSV
        uchar *hsvDataPtr = hsvImg.data;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int hi = i * width * 3 + j * 3;
                // sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
                int gi = i * width + j;

                dstDataPtrH[gi] = hsvDataPtr[hi] * 510 / 360;        // H = 0 .. 255
                dstDataPtrS[gi] = hsvDataPtr[hi + 1];            // S = 0 .. 255
                dstDataPtrV[gi] = hsvDataPtr[hi + 2];            // V = 0 .. 255
            }
        }

        int h_histo[256];
        int s_histo[256];
        int v_histo[256];
        compute_histogram(h_histo, dstH, 256);
        compute_histogram(s_histo, dstS, 256);
        compute_histogram(v_histo, dstV, 256);

        imshow("input image", src);
        imshow("H", dstH);
        imshow("S", dstS);
        imshow("V", dstV);
        showHistogram("H Histogram", h_histo, 256, 100, true);
        showHistogram("S Histogram", s_histo, 256, 100, true);
        showHistogram("V Histogram", v_histo, 256, 100, true);
        waitKey();
    }
}

void binarize_HSV() {

    char fname[MAX_PATH];

    while (openFileDlg(fname)) {

        Mat src = imread(fname);
        int height = src.rows;
        int width = src.cols;

        Mat dstH = Mat(height, width, CV_8UC1);
        Mat dstS = Mat(height, width, CV_8UC1);
        Mat dstV = Mat(height, width, CV_8UC1);

        // definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
        uchar *dstDataPtrH = dstH.data;
        uchar *dstDataPtrS = dstS.data;
        uchar *dstDataPtrV = dstV.data;

        Mat hsvImg;
        cvtColor(src, hsvImg, CV_BGR2HSV);
        // definire pointer la matricea (24 biti/pixeli) a imaginii HSV
        uchar *hsvDataPtr = hsvImg.data;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int hi = i * width * 3 + j * 3;
                // sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
                int gi = i * width + j;

                dstDataPtrH[gi] = hsvDataPtr[hi] * 510 / 360;        // H = 0 .. 255
                dstDataPtrS[gi] = hsvDataPtr[hi + 1];            // S = 0 .. 255
                dstDataPtrV[gi] = hsvDataPtr[hi + 2];            // V = 0 .. 255
            }
        }

        int h_histo[256];
        int s_histo[256];
        int v_histo[256];
        compute_histogram(h_histo, dstH, 256);
        compute_histogram(s_histo, dstS, 256);
        compute_histogram(v_histo, dstV, 256);

        imshow("input image", src);
        imshow("H", dstH);
        imshow("S", dstS);
        imshow("V", dstV);
        showHistogram("H Histogram", h_histo, 256, 100, true);
        showHistogram("S Histogram", s_histo, 256, 100, true);
        showHistogram("V Histogram", v_histo, 256, 100, true);

        int threshold = -1;
        while (threshold < 0 || threshold > 255) {
            printf("Write threshold: ");
            scanf("%d", &threshold);
        }

        Mat dst = Mat(height, width, CV_8UC1);
        Vec3b aux;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                dst.at<uchar>(i, j) = dstH.at<uchar>(i, j) < threshold ? 0 : 255;
            }
        }

        imshow("white-black image", dst);

        waitKey();
    }

}

void compute_FDP(float *p, int n, Mat src) {
    int a[256];
    compute_histogram(a, src, n);
    int M = src.rows * src.cols;
    for (int i = 0; i < n; i++) {
        p[i] = (float) ((float) a[i]) / M;
    }
}

void compute_histograme(int *h, int *hc, float *p, int n, Mat src) {

    compute_histogram(h, src, n);
    compute_FDP(p, n, src);
    //compute comultative histogram
    hc[0] = h[0];
    for (int g = 1; g < 256; g++) {
        hc[g] = hc[g - 1] + h[g];
    }

}

float determinare_prag_binarizare_gobala(Mat src) {
    int n = 256;
    int h[256];
    int hc[256];
    float p[256];
    compute_histograme(h, hc, p, n, src);
    int Imin, Imax, gmin, gmax;
    //se parcurge h
    for (int g = 0; g < n; g++) {
        if (h[g] > 0) {
            Imin = g;
            break;
        }
    }
    //determinare Imax
    for (int g = n - 1; g >= 0; g--) {
        if (h[g] > 0) {
            Imax = g;
            break;
        }
    }

    float e = 0.5f;
    float Told = 0.0f;
    float T = (Imin + Imax) / 2.0f;
    do {
        Told = T;
        float m1 = 0.0f, m2 = 0.0f;
        for (int g = 0; g < Told; g++) {
            m1 += g * p[g];
        }
        for (int g = Told; g < n; g++) {
            m2 += g * p[g];
        }
        T = (m1 + m2) / 2.0f;
    } while (abs(T - Told) > e);
    return T;
}

void binarize_automatic_threshold() {

    char fname[MAX_PATH];

    while (openFileDlg(fname)) {

        Mat src = imread(fname);
        int height = src.rows;
        int width = src.cols;

        Mat dstH = Mat(height, width, CV_8UC1);
        Mat dstS = Mat(height, width, CV_8UC1);
        Mat dstV = Mat(height, width, CV_8UC1);

        // definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
        uchar *dstDataPtrH = dstH.data;
        uchar *dstDataPtrS = dstS.data;
        uchar *dstDataPtrV = dstV.data;

        Mat hsvImg;
        cvtColor(src, hsvImg, CV_BGR2HSV);
        // definire pointer la matricea (24 biti/pixeli) a imaginii HSV
        uchar *hsvDataPtr = hsvImg.data;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int hi = i * width * 3 + j * 3;
                // sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
                int gi = i * width + j;

                dstDataPtrH[gi] = hsvDataPtr[hi] * 510 / 360;        // H = 0 .. 255
                dstDataPtrS[gi] = hsvDataPtr[hi + 1];            // S = 0 .. 255
                dstDataPtrV[gi] = hsvDataPtr[hi + 2];            // V = 0 .. 255
            }
        }

        int h_histo[256];
        int s_histo[256];
        int v_histo[256];
        compute_histogram(h_histo, dstH, 256);
        compute_histogram(s_histo, dstS, 256);
        compute_histogram(v_histo, dstV, 256);

        imshow("input image", src);
        imshow("H", dstH);
        imshow("S", dstS);
        imshow("V", dstV);
        showHistogram("H Histogram", h_histo, 256, 100, true);
        showHistogram("S Histogram", s_histo, 256, 100, true);
        showHistogram("V Histogram", v_histo, 256, 100, true);


        float threshold = determinare_prag_binarizare_gobala(dstV);
        printf("Threshold:  %f", threshold);

        Mat dst = Mat(height, width, CV_8UC1);
        Vec3b aux;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                dst.at<uchar>(i, j) = dstH.at<uchar>(i, j) < threshold ? 255 : 0;
            }
        }

        imshow("white-black image", dst);

        waitKey();
    }
}

///////////////////////////////////////////////////LAB 3////////////////////////////////////////////////
#define MAX_HUE 256 //variabile globale
int histG_hue[MAX_HUE]; // histograma globala / cumulativa

void L3_ColorModel_Init() {
    memset(histG_hue, 0, sizeof(unsigned int) * MAX_HUE);
}

Point Pstart, Pend; // Punctele/colturile aferente selectiei ROI curente (declarate global)

void write_global_hue_in_file() {
    FILE *fp;
    fp = fopen("./global_hue.txt", "w");
    if (fp == NULL) {
        printf("Could not open file!");
        exit(1);
    }
    for (int hue = 0; hue < MAX_HUE; hue++) {
        fprintf(fp, "%d\n", histG_hue[hue]);
    }
    fclose(fp);
}

void read_from_file_global_histogram() {
    FILE *fp;
    fp = fopen("./global_hue.txt", "r");
    if (fp == NULL) {
        printf("Could not open file!");
        exit(1);
    }

    for (int hue = 0; hue < MAX_HUE; hue++) {
        fscanf(fp, "%d", &histG_hue[hue]);
    }
    fclose(fp);
}

void CallBackFuncL3(int event, int x, int y, int flags, void *userdata) {
    Mat *H = (Mat *) userdata;
    Rect roi; // regiunea de interes curenta (ROI)
    if (event == EVENT_LBUTTONDOWN) {
        // punctul de start al ROI
        Pstart.x = x;
        Pstart.y = y;
        printf("Pstart: (%d, %d)  ", Pstart.x, Pstart.y);
    } else if (event == EVENT_RBUTTONDOWN) {
        // punctul de final (diametral opus) al ROI
        Pend.x = x;
        Pend.y = y;
        printf("Pend: (%d, %d)  ", Pend.x, Pend.y);
        // sortare puncte dupa x si y  //(parametrii width si height ai structurii Rect > 0)
        roi.x = min(Pstart.x, Pend.x);
        roi.y = min(Pstart.y, Pend.y);
        roi.width = abs(Pstart.x - Pend.x);
        roi.height = abs(Pstart.y - Pend.y);
        printf("Local ROI: (%d, %d), (%d, %d)\n", roi.x, roi.y, roi.x + roi.width, roi.y + roi.height);

        int hist_hue[MAX_HUE] = {0}; // histograma locala a lui Hue
        //memset(hist_hue, 0, MAX_HUE*sizeof(int));
        //Din toata imaginea H se selecteaza o subimagine (Hroi) aferenta ROI
        Mat Hroi = (*H)(roi);
        uchar hue;   //construieste histograma locala aferente ROI
        for (int y = 0; y < roi.height; y++)
            for (int x = 0; x < roi.width; x++) {
                hue = Hroi.at<uchar>(y, x);
                hist_hue[hue]++;
            }
        //acumuleaza histograma locala in cea globala
        for (int i = 0; i < MAX_HUE; i++)
            histG_hue[i] += hist_hue[i];
        // afiseaza histohrama locala
        showHistogram("H local histogram", hist_hue, MAX_HUE, 200, true);
        // afiseaza histohrama globala / cumulativa
        showHistogram("H global histogram", histG_hue, MAX_HUE, 200, true);
        write_global_hue_in_file();
    }
}

void L3_ColorModel_Build() {
    Mat src;
    Mat hsv;  // Read image from file
    char fname[MAX_PATH];
    L3_ColorModel_Init();
    while (openFileDlg(fname)) {
        src = imread(fname);
        int height = src.rows;
        int width = src.cols;
        // Aplicare FTJ gaussian pt. eliminare zgomote: essential sa il aplicati
        GaussianBlur(src, src, Size(5, 5), 0, 0);

        //Creare fereastra pt. afisare
        namedWindow("src", 1);
        // Componenta de culoare Hue a modelului HSV
        Mat H = Mat(height, width, CV_8UC1);
        // definire pointeri la matricea (8 biti/pixeli) folosita la stocarea // componentei individuale H
        uchar *lpH = H.data;
        cvtColor(src, hsv,
                 CV_BGR2HSV); // conversie RGB -> HSV    // definire pointer la matricea (24 biti/pixeli) a imaginii HSV
        uchar *hsvDataPtr = hsv.data;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                // index in matricea hsv (24 biti/pixel)
                int hi = i * width * 3 + j * 3;
                int gi = i * width + j; // index in matricea H (8 biti/pixel)
                lpH[gi] = hsvDataPtr[hi] * 510 / 360; // lpH = 0 .. 255
            }
        }
        //Asociere functie de tratare a avenimentelor MOUSE cu ferestra curenta
        // Ultimul parametru este matricea H (valorile compunentei Hue)
        setMouseCallback("src", CallBackFuncL3, &H);
        imshow("src", src);     // Wait until user press some key
        waitKey(0);
    }
}

#define FILTER_HISTOGRAM 1  //pentru filtraea histrogramei

void save_in_file_parameters(int *histF_hue) {
    //compute h_mean and h_std
    float hue_mean;
    float hue_std;
    int M = 0;
    int sum = 0;
    for (int g = 0; g < MAX_HUE; g++) {
        M += histF_hue[g];
        sum += (g * histF_hue[g]);
    }
    hue_mean = ((float) sum) / M;

    float sum2 = 0.0f;
    for (int g = 0; g < MAX_HUE; g++) {
        float p = ((float) histF_hue[g]) / M;
        sum2 += ((g - hue_mean) * (g - hue_mean) * p);
    }
    hue_std = sqrt(sum2);

    //wirte in file
    FILE *fp;
    // Hue
    fp = fopen("./model.txt", "wt");
    fprintf(fp, "H=[\n");
    for (int hue = 0; hue < MAX_HUE; hue++) {
        fprintf(fp, "%d\n", histF_hue[hue]);
    }
    fprintf(fp, "];\n");
    fprintf(fp, "Hmean = %.0f ;\n", hue_mean);
    fprintf(fp, "Hstd = %.0f ;\n", hue_std);
    fclose(fp);

}

void L3_ColorModel_Save() {
    int hue, sat, i, j;
    int histF_hue[MAX_HUE]; // histograma filtrata cu FTJ
    memset(histF_hue, 0, MAX_HUE * sizeof(unsigned int));
    read_from_file_global_histogram();
    if (FILTER_HISTOGRAM == 1) {
        float gauss[7];
        float sqrt2pi = sqrtf(2 * PI);
        float sigma = 1.5;
        float e = 2.718;
        float sum = 0;
        // Construire gaussian
        for (i = 0; i < 7; i++) {
            gauss[i] = 1.0f / (sqrt2pi * sigma) *
                       powf(e, -(float) (i - 3) * (i - 3) / (2 * sigma * sigma));
            sum += gauss[i];
        }
        for (j = 3; j < MAX_HUE - 3; j++) {
            for (i = 0; i < 7; i++)
                histF_hue[j] += (float) histG_hue[j + i - 3] * gauss[i];
        }
    } else {
        for (j = 0; j < MAX_HUE; j++)
            histF_hue[j] = histG_hue[j];
    }
    //find MAX Hue
    int max_value = 0;
    for (j = 0; j < MAX_HUE; j++)
        if (max_value < histF_hue[j])
            max_value = histF_hue[j];
    int threshold = (int) (max_value * 0.1f);
    for (j = 0; j < MAX_HUE; j++)
        if (threshold > histF_hue[j])
            histF_hue[j] = 0;
    showHistogram("H global histogram", histG_hue, MAX_HUE, 200, true);
    showHistogram("H global filtered histogram", histF_hue, MAX_HUE, 200, true);
    // Wait until user press some key
    save_in_file_parameters(histF_hue);
    waitKey(0);

}

#define OBJECT 255
#define BACKGROUND 0

void segmentation_process() {
    Mat src;
    Mat hsv;  // Read image from file
    char fname[MAX_PATH];
    while (openFileDlg(fname)) {
        src = imread(fname);
        int height = src.rows;
        int width = src.cols;
        // Aplicare FTJ gaussian pt. eliminare zgomote: essential sa il aplicati
        GaussianBlur(src, src, Size(5, 5), 0, 0);

        //Creare fereastra pt. afisare
        namedWindow("src", 1);
        // Componenta de culoare Hue a modelului HSV
        Mat H = Mat(height, width, CV_8UC1);
        // definire pointeri la matricea (8 biti/pixeli) folosita la stocarea // componentei individuale H
        uchar *lpH = H.data;
        cvtColor(src, hsv,
                 CV_BGR2HSV); // conversie RGB -> HSV    // definire pointer la matricea (24 biti/pixeli) a imaginii HSV
        uchar *hsvDataPtr = hsv.data;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                // index in matricea hsv (24 biti/pixel)
                int hi = i * width * 3 + j * 3;
                int gi = i * width + j; // index in matricea H (8 biti/pixel)
                lpH[gi] = hsvDataPtr[hi] * 510 / 360; // lpH = 0 .. 255
            }
        }
        ///// procesare H
        int hue_std = 5;
        int hue_mean = 16;
        float k = 2.5f;
        float minVal = hue_mean - k * hue_std;
        float maxVal = hue_mean + k * hue_std;
        Mat dst = Mat(height, width, CV_8UC1);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (H.at<uchar>(i, j) > minVal && H.at<uchar>(i, j) < maxVal)
                    dst.at<uchar>(i, j) = OBJECT;
                else
                    dst.at<uchar>(i, j) = BACKGROUND;
            }
        }
        //////postprocesare
        // creare element structural de dimensiune 5x5 de tip cruce
        Mat element1 = getStructuringElement(MORPH_CROSS, Size(3, 3));
        //eroziune cu acest element structural (aplicata 1x)
        erode(dst, dst, element1, Point(-1, -1), 2);
        // creare element structural de dimensiune 3x3 de tip patrat (V8)
        Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3));
        // dilatare cu acest element structural (aplicata 2x)
        dilate(dst, dst, element2, Point(-1, -1), 4);
        Mat element3 = getStructuringElement(MORPH_RECT, Size(3, 3));
        // dilatare cu acest element structural (aplicata 2x)
        erode(dst, dst, element3, Point(-1, -1), 2);

        ///vie labeling
        Labeling("After all process", dst, false);
        imshow("src", src);     // Wait until user press some key
        imshow("dst", dst);     // Wait until user press some key
        waitKey(0);
    }
}


int main() {
    int op;
    do {
        system("cls");
        destroyAllWindows();
        printf("Menu:\n");
        printf(" 1 - Open image\n");
        printf(" 2 - Open BMP images from folder\n");
        printf(" 3 - Image negative - diblook style\n");
        printf(" 4 - BGR->HSV\n");
        printf(" 5 - Resize image\n");
        printf(" 6 - Canny edge detection\n");
        printf(" 7 - Edges in a video sequence\n");
        printf(" 8 - Snap frame from live video\n");
        printf(" 9 - Mouse callback demo\n");
        printf(" 11 - HSV histograms\n");
        printf(" 12 - Bynarize with a threshold\n");
        printf(" 13 - Bynarize with a automatic threshold\n");
        printf("-------------LAB 3 -----------------------\n");
        printf(" 14- Histograma globala\n");
        printf(" 15- Read from file Histograma globala\n");
        printf(" 16- Save model\n");
        printf(" 17- Segmentation\n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        scanf("%d", &op);
        switch (op) {
            case 1:
                testOpenImage();
                break;
            case 2:
                testOpenImagesFld();
                break;
            case 3:
                testParcurgereSimplaDiblookStyle(); //diblook style
                break;
            case 4:
                //testColor2Gray();
                testBGR2HSV();
                break;
            case 5:
                testResize();
                break;
            case 6:
                testCanny();
                break;
            case 7:
                testVideoSequence();
                break;
            case 8:
                testSnap();
                break;
            case 9:
                testMouseClick();
                break;
            case 11:
                showHistogramHSV();
                break;
            case 12:
                binarize_HSV();
                break;
            case 13:
                binarize_automatic_threshold();
                break;
            case 14:
                L3_ColorModel_Build();
                break;
            case 15:
                read_from_file_global_histogram();
                break;
            case 16:
                L3_ColorModel_Save();
                break;
            case 17:
                segmentation_process();
                break;


        }
    } while (op != 0);
    return 0;
}