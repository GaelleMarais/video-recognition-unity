using UnityEngine;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.IO;

using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

public class VideoCaptureScript : MonoBehaviour
{
    VideoCapture video;
    [Range(0, 255)]
    public double hmin;
    [Range(0, 255)]
    public double hmax;
    [Range(0, 255)]
    public double smin;
    [Range(0, 255)]
    public double smax;
    [Range(0, 255)]
    public double vmin;
    [Range(0, 255)]
    public double vmax;


    // Start is called before the first frame update
    void Start()
    {
        // video = new VideoCapture("Assets/Videos/lego.avi");
        video = new VideoCapture(0);
    }

    // Update is called once per frame
    void Update()
    {
        Mat orig;
        orig = video.QueryFrame();
        if(orig != null)
        {
            CvInvoke.Imshow("1", orig);
            CvInvoke.WaitKey(24);

            Mat image2 = orig.Clone();
            Mat final = ImageTreatment(image2);
            CvInvoke.Imshow("2", final);

            Mat image3 = final.Clone();
            Mat borders = Borders(final, orig);
            CvInvoke.Imshow("3", borders);

            /*
            Mat grayImage = image.Clone();
            CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);
            CvInvoke.Flip(grayImage, grayImage, FlipType.Vertical);
            CvInvoke.Imshow("3", grayImage);

            Mat hsvImage = image.Clone();
            CvInvoke.CvtColor(image, hsvImage, ColorConversion.Bgr2Hsv);
            CvInvoke.Flip(hsvImage, hsvImage, FlipType.Vertical);
            CvInvoke.Flip(hsvImage, hsvImage, FlipType.Horizontal);
            CvInvoke.Imshow("4", hsvImage);
            */
        }
        else
        {
            CvInvoke.DestroyAllWindows();
        }

    }

    void OnDestroy()
    {
        CvInvoke.DestroyAllWindows();
    }

     Mat ImageTreatment(Mat image)
    {
        CvInvoke.CvtColor(image, image, ColorConversion.Bgr2Hsv);
        CvInvoke.MedianBlur(image, image, 21);


        Hsv lower = new Hsv(hmin, smin, vmin);
        Hsv higher = new Hsv(hmax, smax, vmax);

        Image<Hsv, Byte> i = image.ToImage<Hsv, Byte>();
        Mat result = i.InRange(lower, higher).Mat;

        int operationSize = 1;

        Mat structuringElement = CvInvoke.GetStructuringElement(ElementShape.Ellipse,
                                                                new Size(2 * operationSize +1, 2 * operationSize + 1),
                                                                new Point(operationSize, operationSize));

        CvInvoke.Erode(result, result, structuringElement, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));
        CvInvoke.Dilate(result, result, structuringElement, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));

        return result;
    }

    Mat Borders(Mat image, Mat orig)
    {
        VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint(100);
        VectorOfPoint biggestContour = new VectorOfPoint(100);
        int biggestContourIndex = -1;
        double biggestContourArea = 0;

        Mat hierarchy = new Mat();
        CvInvoke.FindContours(image, contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxNone);

        for(int i = 0; i < contours.Size; i++)
        {
            if( CvInvoke.ContourArea(contours[i]) > biggestContourArea)
            {
                biggestContour = contours[i];
                biggestContourIndex = i;
                biggestContourArea = CvInvoke.ContourArea(contours[i]);
            }
        }

        if(biggestContourIndex > -1)
        {
            CvInvoke.DrawContours(orig, contours, biggestContourIndex, new MCvScalar(244, 0, 0));
        }

        return orig;
    }
}
