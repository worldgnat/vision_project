//
//  main.cpp
//  Comp558 Project
//  Nicolas Langley & Peter Davoust
//

#include <iostream>
#include <fstream>
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/core/core.hpp>
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

int main(int argc, char* argv[]) {
    int retval = parseCmdArgs(argc, argv);
    if (retval) return -1;
    
    // First step is to obtain Image Features
    // Create feature finder for SURF using OpenCV
    SurfFeaturesFinder surfFinder = SurfFeaturesFinder();
    vector<ImageFeatures> imgFeatures;
    // Obtain features for each image
    for (int i = 0; i < imgs.size(); i++) {
        ImageFeatures features;
        surfFinder(imgs.at(i), features);
        imgFeatures.push_back(features);
    }
    
    /* 
     * Use FLANN to find approximate nearest neighbours [BL97]
     * In order to find matches, do we need to perform this for each image with every other image?
     * This could be very inefficient.
     * Instead only choose look at images with a high number of matching features
     */
    int numFeatures = (int) imgFeatures.size();
    FlannBasedMatcher flann = FlannBasedMatcher();
    typedef vector<vector<DMatch>> MatchSet;
    vector<MatchSet> allMatches;
    for (int i = 0; i < numFeatures-1; i++) {
        MatchSet curMatchSet;
        for (int j = 1; j < numFeatures; j++) {
            // Add matches for each image to current match set
            vector<DMatch> curMatches;
            flann.match(imgFeatures.at(i).descriptors, imgFeatures.at(j).descriptors, curMatches);
            curMatchSet.push_back(curMatches);
        }
        //
        sort(curMatchSet.begin(), curMatchSet.end(), vectorComp);
        while (curMatchSet.size() > 6) curMatchSet.pop_back();
    }
    
    // Use FindHomography
    // Convert from vector<dMatch> to vector<Point2f>
    // calib3d.FindHomography()
    // Mat findHomography(InputArray srcPoints, InputArray dstPoints, int method=0, double ransacReprojThreshold=3, OutputArray mask=noArray())
    vector<vector<Mat>> allHomographies;
    Mat homography;
    Mat ransac_mask;
    vector<Point2f> srcPoints, dstPoints;
    homography = findHomography(srcPoints, dstPoints, CV_RANSAC, 5, ransac_mask);
    
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

