
//
//  main.cpp
//  Comp558 Project
//  Nicolas Langley & Peter Davoust
//

// System Includes
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>

// OpenCV Includes
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/stitcher.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <mach-o/dyld.h>

using namespace std;
// Namespaces for OpenCV
using namespace cv;
using namespace detail;

bool try_use_gpu = false;
vector<Mat> imgs;
// Replace this with the local location of your Xcode Project
string result_name = "/Users/nicolaslangley/Desktop/result.jpg";

void printUsage();
int parseCmdArgs(int argc, char** argv);
bool vectorComp(const vector<DMatch>& a,const vector<DMatch>& b);
void featureDetectionTest();
Mat composeImages(Mat image1, Mat image2, bool displayMatches = false, bool displayFeatures = false);

int main(int argc, char* argv[]) {
    // Handle command line arguments
    int retval = parseCmdArgs(argc, argv);
    if (retval) return -1;
    
    // First step is to obtain Image Features
    // Create feature finder for SURF using OpenCV
    SurfFeaturesFinder surfFinder = SurfFeaturesFinder(650.);
    vector<ImageFeatures> imgFeatures;
    // Obtain features for each image
    for (int i = 0; i < imgs.size(); i++) {
        ImageFeatures features;
        surfFinder(imgs.at(i), features);
        imgFeatures.push_back(features);
    }
//    cout << imgFeatures.at(0).descriptors.rows << " " << imgFeatures.at(1).descriptors.rows << endl;
    
    /* 
     * Use FLANN to find approximate nearest neighbours [BL97]
     * In order to find matches, do we need to perform this for each image with every other image?
     * This could be very inefficient.
     * Instead only choose look at images with a high number of matching features
     */
    int numImages = (int) imgFeatures.size();
    // Initialize FLANN matcher
    FlannBasedMatcher flann = FlannBasedMatcher();
    
    // Typedef for sets of matches
    map<vector<DMatch>, int> matchToImage;
    typedef vector<vector<DMatch>> MatchSet;
    vector<MatchSet> allMatches;
    // Iterate through all image pairs
    for (int i = 0; i < numImages; i++) {
        MatchSet curMatchSet;
        for (int j = 0; j < numImages; j++) {
            //if (j == 1) continue;
            // Add matches for each image to current match set
            vector<DMatch> curMatches;
            flann.match(imgFeatures.at(i).descriptors, imgFeatures.at(j).descriptors, curMatches);
            // Only choose matches that are considered valid
            vector<DMatch> curGoodMatches;
            double max_dist = 0;
            double min_dist = 100;
            
            // Quick calculation of max and min distances between keypoints
            for (int k = 0; k < imgFeatures.at(i).descriptors.rows; k++) {
                double dist = curMatches[k].distance;
                if (dist < min_dist) min_dist = dist;
                if (dist > max_dist) max_dist = dist;
            }
            
            // Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
            vector< DMatch > good_matches;
            
            for (int k = 0; k < imgFeatures.at(i).descriptors.rows; k++) {
                if (curMatches[k].distance <= 2 * min_dist ) {
                    curGoodMatches.push_back(curMatches[k]);
                }
            }
            matchToImage[curGoodMatches] = j;
            curMatchSet.push_back(curGoodMatches);
        }
        // Sort image matches based on number of features and only keep top m = 6
        sort(curMatchSet.begin(), curMatchSet.end(), vectorComp);
        while (curMatchSet.size() > 6) {
            vector<DMatch> last = curMatchSet.back();
            curMatchSet.pop_back();
            matchToImage.erase(last);
        }
        allMatches.push_back(curMatchSet);
    }
    
    
    // Typedefs
    typedef vector<Mat> HomographySet;
    typedef vector<Mat> MaskSet;
    
    // Create lists of all homographies and inlier/outlier masks
    // Mask values - 0 = outlier value = inlier
    vector<HomographySet> allHomographies;
    vector<MaskSet> allMasks;
    
    // Iterate through all pairs of images
    for (int i = 0; i < allMatches.size(); i++) {
        HomographySet curHomographySet;
        MaskSet curMaskSet;
        for (int j = 0; j < allMatches.at(i).size(); j++) {
            // Get keypoints for each set of images
            vector<Point2f> obj;
            vector<Point2f> scene;
            vector<DMatch> curMatches = allMatches.at(i).at(j);
            for (int k = 0; k < curMatches.size(); k++) {
                int qIndex = curMatches.at(k).queryIdx;
                int tIndex = curMatches.at(k).trainIdx;
                obj.push_back(imgFeatures.at(i).keypoints[qIndex].pt);
                scene.push_back(imgFeatures.at(j).keypoints[tIndex].pt);
            }
            
            // Compute Homography and inlier/outlier mask
            Mat curMask;
            cout << "Object " << obj.size() << " Scene " << scene.size() << endl;
            if (obj.size() >= 4 && scene.size() >= 4) {
                Mat H = findHomography(obj, scene, CV_RANSAC, 10, curMask);
                curHomographySet.push_back(H);
                curMaskSet.push_back(curMask);
                //cout << "Homography: " << endl << H << endl;
                //cout << "Cur Mask: " << endl << curMask << endl;
            }
        }
        allHomographies.push_back(curHomographySet);
        allMasks.push_back(curMaskSet);
    }
    
    // Check that homography for each pair of images is valid
    // n_i > alpha + beta* n_f where alpha = 8.0, beta = 0.3, n_f = # features in overlap, n_i = # inliers
    // Iterate through images and remove all matches that do not pass above condition
    double alpha = 8.0;
    double beta = 0.3;
    vector<int> imageOrder;
    for (int i = 0; i < allMasks.size(); i++) {
        for (int j = 0; j < allMasks.at(i).size(); j++) {
            // Compute validity of homography
            int numInliers = 0;
            int numOutliers = 0;
            Mat inliers = allMasks.at(i).at(j);
            for(int row = 0; row < inliers.rows; ++row) {
                uchar* p = inliers.ptr(row);
                for(int col = 0; col < inliers.cols; ++col) {
                    int pt = (int) *p;
                    if (pt == 0) {
                        numInliers++;
                    } else {
                        numOutliers++;
                    }
                    *p++;
                }
            }
            bool isGoodMatch = false;
            cout << "Checking... " << i << ", " << j << endl;
            cout << "Num inliers " << numInliers << " Num outliers " << numOutliers << endl;
            if ((double) numInliers > (alpha + beta * (double) numOutliers)) {
                isGoodMatch = true;
                cout << "Good match occured at " << i << ", " << j << endl;
                if (find(imageOrder.begin(), imageOrder.end(), i) == imageOrder.end()) imageOrder.push_back(i);
                if (find(imageOrder.begin(), imageOrder.end(), j) == imageOrder.end()) imageOrder.push_back(j);
            }
        }
    }
    vector<Mat> orderedImgs;
    vector<Mat> orderedHomographies;
    for (int i = 0; i < imageOrder.size(); i++) {
        orderedImgs.push_back(imgs.at(imageOrder.at(i)));
        if (i < imageOrder.size() - 1) orderedHomographies.push_back(allHomographies.at(i).at(i+1));
    }
    
    Mat pano;
    for (int i = 0; i < orderedImgs.size() - 1; i++) {
        pano = composeImages(imgs.at(i+1), imgs.at(i), false, false);
        imshow("Stitch", pano);
        imwrite(result_name, pano);
    }
    
    return 0;
}

// Function for comparing size of two vectors - used in sorting
bool vectorComp(const vector<DMatch>& a,const vector<DMatch>& b) {
    return a.size() < b.size();
}

void printUsage() {
    cout <<
    
    "Rotation model images stitcher.\n\n"
    "stitching img1 img2 [...imgN]\n\n"
    "Flags:\n"
    "  --try_use_gpu (yes|no)\n"
    "      Try to use GPU. The default value is 'no'. All default values\n"
    "      are for CPU mode.\n"
    "  --output <result_img>\n"
    "      The default is 'result.jpg'.\n";
}


int parseCmdArgs(int argc, char** argv) {
    // Check that there it at least one argument - if not print usage
    if (argc == 1) {
        printUsage();
        return -1;
    }
    // Iterate through arguments
    for (int i = 1; i < argc; ++i) {
        // Handle case that argument is --help or /?
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?") {
            printUsage();
            return -1;
        } else if (string(argv[i]) == "--try_gpu") {
            // Handle --try_gpu flag
            if (string(argv[i + 1]) == "no") try_use_gpu = false;
            else if (string(argv[i + 1]) == "yes") try_use_gpu = true;
            else {
                cout << "Bad --try_use_gpu flag value\n";
                return -1;
            }
            i++;
        }
        // Handle --output flag
        else if (string(argv[i]) == "--output") {
            result_name = argv[i + 1];
            i++;
        } else {
            // Read input images into imgs
            Mat img = imread(argv[i]);
            if (img.empty()) {
                cout << "Can't read image '" << argv[i] << "'\n";
                return -1;
            }
            imgs.push_back(img);
        }
    }
    return 0;
}

// Function that takes 2 image and composes them - used for after order has been found.
Mat composeImages(Mat image1, Mat image2, bool displayMatches, bool displayFeatures) {
    
    // Load the images
    Mat gray_image1;
    Mat gray_image2;
    // Convert to Grayscale
    cvtColor( image1, gray_image1, CV_RGB2GRAY );
    cvtColor( image2, gray_image2, CV_RGB2GRAY );
    
    if( !gray_image1.data || !gray_image2.data )
    { std::cout<< " --(!) Error reading images " << std::endl;}
    
    // First step is to obtain Image Features
    // Create feature finder for SURF using OpenCV
    SurfFeaturesFinder surfFinder = SurfFeaturesFinder(650.);
    ImageFeatures imScene, imObject;
    surfFinder(gray_image1, imObject);
    surfFinder(gray_image2, imScene);
    Mat descriptors_object, descriptors_scene;
    descriptors_object = imObject.descriptors;
    descriptors_scene = imScene.descriptors;
    vector<KeyPoint> keypoints_object, keypoints_scene;
    keypoints_object = imObject.keypoints;
    keypoints_scene = imScene.keypoints;
    
    if (displayFeatures == true) {
    
        Mat img_keypoints_1; Mat img_keypoints_2;
        drawKeypoints( image1, keypoints_object, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        drawKeypoints( image2, keypoints_scene, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    
        // Draw keypoints
        imwrite("/Users/nicolaslangley/Desktop/keypts1.jpg", img_keypoints_1 );
        imwrite("/Users/nicolaslangley/Desktop/keypts2.jpg", img_keypoints_2 );
    }
    
    FlannBasedMatcher flann;
    vector<DMatch> matches;
    flann.match(descriptors_object, descriptors_scene, matches);
    // Only choose matches that are considered valid
    vector<DMatch> goodMatches;
    double max_dist = 0;
    double min_dist = 100;
    
    // Quick calculation of max and min distances between keypoints
    for (int k = 0; k < descriptors_object.rows; k++) {
        double dist = matches[k].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    
    // Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
    vector< DMatch > good_matches;
    
    for (int k = 0; k < descriptors_object.rows; k++) {
        if (matches[k].distance <= 2 * min_dist ) {
            goodMatches.push_back(matches[k]);
        }
    }
    
    if (displayMatches == true) {
        // Display matches
        Mat img_matches;
        drawMatches(image1, keypoints_object, image2, keypoints_scene,
                    goodMatches, img_matches, Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
        // Write matches to file
        imwrite("/Users/nicolaslangley/Desktop/matches1.jpg", img_matches);
    }
    
    vector<Point2f> obj;
    vector<Point2f> scene;
    
    for( int i = 0; i < goodMatches.size(); i++ ) {
        obj.push_back( keypoints_object[ goodMatches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ goodMatches[i].trainIdx ].pt );
    }
    
    // Find the Homography Matrix
    Mat H = findHomography( obj, scene, CV_RANSAC );
    // Use the Homography Matrix to warp the images
    Mat result;
    warpPerspective(image1,result,H,cv::Size(image1.cols+image2.cols,image1.rows));
    cv::Mat half(result,cv::Rect(0,0,image2.cols,image2.rows));
    image2.copyTo(half);
    
    return result;
}






