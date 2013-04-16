package pdetection;

import com.googlecode.javacv.FrameGrabber;
import com.googlecode.javacv.cpp.opencv_objdetect;

import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_highgui.*;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_BGR2GRAY;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvCvtColor;
import static com.googlecode.javacv.cpp.opencv_objdetect.cvHaarDetectObjects;

public class BodyDetection {

    private static final String CASCADE_FILE = "src/main/resources/algorithm/HS.xml";

    public static void main(String[] args) throws Exception {
        FrameGrabber grabber = FrameGrabber.createDefault(0);
        grabber.start();

        IplImage originalImage;
        while (true) {
            originalImage = grabber.grab();

            IplImage grayImage = IplImage.create(originalImage.width(), originalImage.height(), IPL_DEPTH_8U, 1);
            cvCvtColor(originalImage, grayImage, CV_BGR2GRAY);

            CvMemStorage storage = CvMemStorage.create();
            opencv_objdetect.CvHaarClassifierCascade cascade = new opencv_objdetect.CvHaarClassifierCascade(cvLoad(CASCADE_FILE));

            CvSeq faces = cvHaarDetectObjects(grayImage, cascade, storage, 1.1, 1, 0);
            for (int i = 0; i < faces.total(); i++) {
                CvRect r = new CvRect(cvGetSeqElem(faces, i));
                cvRectangle(originalImage, cvPoint(r.x(), r.y()), cvPoint(r.x() + r.width(), r.y() + r.height()), CvScalar.YELLOW, 1, CV_AA, 0);

            }
            cvShowImage("S1", originalImage);
            cvWaitKey(33);
        }
    }
}