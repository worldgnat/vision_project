//
//  main.cpp
//  Comp558 Project
//  Nicolas Langley & Peter Davoust
//

#include <iostream>
#include <fstream>
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/stitching/stitcher.hpp"
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

int main(int argc, char* argv[])
{
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
     */
    int numFeatures = (int) imgFeatures.size();
    FlannBasedMatcher flann = FlannBasedMatcher();
    vector<vector<DMatch>> allMatches;
    for (int i = 0; i < numFeatures; i++) {
        for (int j = 0; j < numFeatures; j++) {
            vector<DMatch> curMatches;
            flann.match(imgFeatures.at(i).descriptors, imgFeatures.at(j).descriptors, curMatches);
            allMatches.push_back(curMatches);
        }
    }
    
    // Use FindHomography
    
    
    //cout << "Writing image to " << result_name << "\n";
    //imwrite(result_name, pano);
    return 0;
}


void printUsage()
{
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


int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage();
        return -1;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage();
            return -1;
        }
        else if (string(argv[i]) == "--try_gpu")
        {
            if (string(argv[i + 1]) == "no")
                try_use_gpu = false;
            else if (string(argv[i + 1]) == "yes")
                try_use_gpu = true;
            else
            {
                cout << "Bad --try_use_gpu flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--output")
        {
            result_name = argv[i + 1];
            i++;
        }
        else
        {
            Mat img = imread(argv[i]);
            if (img.empty())
            {
                cout << "Can't read image '" << argv[i] << "'\n";
                return -1;
            }
            imgs.push_back(img);
        }
    }
    return 0;
}

