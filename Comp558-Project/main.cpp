
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
string result_name = "result.jpg";

void printUsage();
int parseCmdArgs(int argc, char** argv);
bool vectorComp(const vector<DMatch>& a,const vector<DMatch>& b);
void featureDetectionTest();

int main(int argc, char* argv[]) {
    // Handle command line arguments
    int retval = parseCmdArgs(argc, argv);
    if (retval) return -1;
    
    //featureDetectionTest();
    
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
    cout << imgFeatures.at(0).descriptors.rows << " " << imgFeatures.at(1).descriptors.rows << endl;
    
    /* 
     * Use FLANN to find approximate nearest neighbours [BL97]
     * In order to find matches, do we need to perform this for each image with every other image?
     * This could be very inefficient.
     * Instead only choose look at images with a high number of matching features
     */
    int numFeatures = (int) imgFeatures.size();
    // Initialize FLANN matcher
    FlannBasedMatcher flann = FlannBasedMatcher();
    
    // Typedef for sets of matches
    map<vector<DMatch>, int> matchToImage;
    typedef vector<vector<DMatch>> MatchSet;
    vector<MatchSet> allMatches;
    // Iterate through all image pairs
    for (int i = 0; i < numFeatures-1; i++) {
        MatchSet curMatchSet;
        for (int j = 1; j < numFeatures; j++) {
            // Add matches for each image to current match set
            vector<DMatch> curMatches;
            flann.match(imgFeatures.at(i).descriptors, imgFeatures.at(j).descriptors, curMatches);
            // Only choose matches that are considered valid
            vector<DMatch> curGoodMatches;
            double max_dist = 0;
            double min_dist = 100;
            
            //-- Quick calculation of max and min distances between keypoints
            for (int k = 0; k < imgFeatures.at(i).descriptors.rows; k++) {
                double dist = curMatches[k].distance;
                if (dist < min_dist) min_dist = dist;
                if (dist > max_dist) max_dist = dist;
            }
            
            //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
            std::vector< DMatch > good_matches;
            
            for (int k = 0; k < imgFeatures.at(i).descriptors.rows; k++) {
                if (curMatches[k].distance <= 2 * min_dist ) {
                    curGoodMatches.push_back(curMatches[i]);
                }
            }
            cout << "Number of matches method 2 " << curGoodMatches.size() << endl;
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
    
    // Create lists of all homographis and inlier/outlier masks
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
            for (int k = 0; k < allMatches.at(i).at(j).size(); k++) {
                int qIndex = allMatches.at(i).at(j).at(k).queryIdx;
                int tIndex = allMatches.at(i).at(j).at(k).trainIdx;
                obj.push_back(imgFeatures.at(i).keypoints[qIndex].pt);
                scene.push_back(imgFeatures.at(j).keypoints[tIndex].pt);
            }
            // Compute Homography and inlier/outlier mask
            Mat curMask;
            cout << "Object " << obj.size() << " Scene " << scene.size() << endl;
            if (obj.size() >= 4 && scene.size() >= 4) {
                Mat H = findHomography(obj, scene, CV_RANSAC, 5, curMask);
                curHomographySet.push_back(H);
                curMaskSet.push_back(curMask);
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
    int count = 0;
    vector<int> imageOrder;
    for (int i = 0; i < allMasks.size(); i++) {
        for (int j = 0; j < allMasks.at(i).size(); j++) {
            // Compute validity of homography
            int numInliers = 0;
            Mat inliers = allMasks.at(i).at(j);
            for(int row = 0; row < inliers.rows; ++row) {
                //
                uchar* p = inliers.ptr(row);
                for(int col = 0; col < inliers.cols; ++col) {
                    if (*p != 0) {
                        numInliers++;
                    }
                    *p++;
                    count++;
                }
            }
            int numFeatures = (int) allMatches.at(i).at(j).size();
            bool isGoodMatch = false;
            cout << "Checking... " << i << ", " << j << endl;
            cout << "Num inliers " << numInliers << " Num features " << numFeatures << endl;
            if ((double) numInliers > (alpha + beta * (double) numFeatures)) {
                isGoodMatch = true;
                cout << "Good match occured at " << i << ", " << j << endl;
                // Put image into list
            }
        }
    }
    
    
    /*
    // Draw matches - Not part of actual pipeline
    Mat img1 = imgs.at(0);
    Mat img2 = imgs.at(1);
    vector<KeyPoint> keypoints1 = imgFeatures.at(0).keypoints;
    vector<KeyPoint> keypoints2 = imgFeatures.at(1).keypoints;
    vector<DMatch> matches = allMatches.at(0);
    Mat matchesToShow;
    drawMatches( img1, keypoints1, img2, keypoints2,
                matches, matchesToShow, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
    //-- Show detected matches
    imwrite(result_name, matchesToShow);
    */
    
    //cout << "Writing image to " << result_name << "\n";
    //imwrite(result_name, pano);
    return 0;
}

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

