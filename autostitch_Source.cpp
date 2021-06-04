/* 	ELEC 474 Take Home Exam - Image Stitching
Completed by: Lukas Bauer, Alex Koch-Fitsialos
Student #: 20052327, 20055172
Due: November 29 2019
Statement of originality: We certify that this submission contains our own work, except as noted

INSTRUCTIONS:
- To use the program on your system, scroll down to the main() function and replace the fields at the very top indicated by "EDIT HERE"
- You will need to specify an output folder, a source images folder, and a base/anchor image

DESCRIPTION
- for the description please see the write-up.

 */


#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>


// OpenCV Imports
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp> // OpenCV Core Functionality
#include <opencv2/highgui/highgui.hpp> // High-Level Graphical User Interface
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;


//Function used to calculate the number of matches between imgAnchor and imgAttach - used much of the same process as Lab 4
vector<cv::DMatch> getNumMatches(string imgAnchor, string imgAttach, vector<KeyPoint> & kp_object, vector<KeyPoint> & kp_scene) {
	Mat imgA = imread(imgAttach, IMREAD_GRAYSCALE);
	Mat imgB = imread(imgAnchor, IMREAD_GRAYSCALE);
	vector<cv::DMatch> good_matches;
	vector<cv::DMatch> best_matches;

	if (!imgA.data || !imgB.data)
	{
		printf("No image data \n");
		return good_matches;
	}

	//parameters for descriptor extractor
	int nfeatures = 2500;
	float scaleFactor = 1.35f;
	int nlevels = 8;
	int edgeThreshold = 31;
	int firstLevel = 0;
	int WTA_K = 2;
	ORB::ScoreType scoreType = ORB::HARRIS_SCORE;
	int patchSize = 31;

	Ptr<FeatureDetector> detector = ORB::create(nfeatures,
		scaleFactor, nlevels, edgeThreshold, firstLevel,
		WTA_K, scoreType, patchSize);

	Ptr<DescriptorExtractor> descriptor = ORB::create(nfeatures,
		scaleFactor, nlevels, edgeThreshold, firstLevel,
		WTA_K, scoreType, patchSize);

	detector->detect(imgA, kp_object);
	detector->detect(imgB, kp_scene);

	Mat descriptors_obj, descriptors_scene;
	descriptor->compute(imgA, kp_object, descriptors_obj);
	descriptor->compute(imgB, kp_scene, descriptors_scene);

	std::vector<std::vector<cv::DMatch>> matches;
	cv::BFMatcher matcher;
	matcher.knnMatch(descriptors_obj, descriptors_scene, matches,2);  // Find two nearest matches

	// Further match filtering - see "Feature Matching Method" from section I. List of all Source Modules
	for (int i = 0; i < matches.size(); ++i)
	{
		const float ratio = 0.80; 
		if (matches[i][0].distance < ratio * matches[i][1].distance)
		{
			best_matches.push_back(matches[i][0]);
		}
	}
	return best_matches;
}

// Function used to create a copy of a vector (used for good_matches, kp_scene, kp_object
template<typename T>
std::vector<T> create_copy(std::vector<T> const& vec)
{
	std::vector<T> v(vec);
	return v;
}


int main(int argc, char** argv)
{
	//----------------------- (EDIT HERE) New system config section -------------------------------------

	const string folderBase = "C:/Users/Lukas/source/repos/laptopStitch/laptopStitch/pics/WLH/"; //folder for output of stitched image
	const string folder = folderBase + "toBeStitched/*.jpg"; //path for images to be stitched together
	string basePic = folderBase + "toBeStitched\\1.jpg"; //name of base/anchor image

	//----------------------- End of configuration needed --------------------------------

	string imgAnchor;
	string imgAttach;
	vector<cv::DMatch> temp_matches;
	vector<cv::DMatch> good_matches, matches;
	vector<KeyPoint> temp_object, temp_scene;
	vector<KeyPoint> kp_object, kp_scene;

	//take the specified anchor image and start a new stitched.jpg file in the folderBase directory
	Mat copyImg = imread(basePic);
	imwrite(folderBase + "stitched.jpg", copyImg);
	cout << "Moved anchor image to stitched.jpg in: " << folderBase << endl;

	vector<string> filenames;
	glob(folder, filenames, false);

	waitKey(1000);

	imgAnchor = folderBase + "stitched.jpg";
	int removeIdx;
	cout << "Base pic: " << basePic << "\n" << endl;

	//removing the anchor image from the list of possible attachment images
	for (size_t i = 0; i < filenames.size(); i++) {
		if (filenames[i] == basePic) {
			filenames.erase(filenames.begin() + i);
		}
	}

	//------------------END OF THE INITIALIZATIONS, BEGINNING OF RECURSIVE LOOP AND STEP 1---------------------

	cout << "Beginning joining loop!\n" << endl;

	//loop while there are still files that need to be checked if they can be attached
	while (filenames.size() > 0) {

		//clear all the vectors holding match information
		matches.clear();
		good_matches.clear();
		kp_object.clear();
		kp_scene.clear();
		int maxMatches = 0;
		for (int i = 0; i < filenames.size(); i++) {
			temp_object.clear();
			temp_scene.clear();
			temp_matches.clear();
			
			//left this in in case the anchor image was not properly removed
			if (filenames[i] == imgAnchor) {
				continue;
			}
			temp_matches = getNumMatches(imgAnchor, filenames[i], temp_object, temp_scene);
			cout << "Checking: " << filenames[i] << " Good matches: " << temp_matches.size() << endl;

			// to keep track of which image has the highest number of good_matches
			if (temp_matches.size() > maxMatches) {
				maxMatches = temp_matches.size();
				good_matches = create_copy(temp_matches);
				kp_object = create_copy(temp_object);
				kp_scene = create_copy(temp_scene);
				imgAttach = filenames[i];
				removeIdx = i;
			}
		}
		cout << "The chosen attach image is: " << imgAttach << endl;
		Mat imgA = imread(imgAttach, IMREAD_GRAYSCALE);
		Mat imgB = imread(imgAnchor, IMREAD_GRAYSCALE);

		//------------one of the metrics for determining if it is a good picture match - chosen through trial and error -------------
		if (good_matches.size() < 30) {
			cout << "Image did not pass the minimum good match threshold." << endl;
			cout << "All other files have less matches - ending program! " << filenames[removeIdx] << endl;
			cout << "Find the output in: " << folderBase << endl;
			return 0;
		}

		vector<Point2f> objectPoints;
		vector<Point2f> scenePoints;

		for (int i = 0; i < (int)good_matches.size(); i++) {
			objectPoints.push_back(kp_object[good_matches[i].queryIdx].pt);
			scenePoints.push_back(kp_scene[good_matches[i].trainIdx].pt);
		}

		Mat filter_output;
		drawMatches(imgA, kp_object, imgB, kp_scene,
			good_matches, filter_output, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		namedWindow("Matches", WINDOW_NORMAL);
		imshow("Matches", filter_output);

		//-----------------------------------End of Part 1, beginning of Part 2-------------------------------------------------

		Mat homogMask;
		Mat homography = findHomography(objectPoints, scenePoints, homogMask, RANSAC);
		cout << "\nPre-shift homography: \n" << homography << endl;


		Mat overRect = imread(imgAttach, IMREAD_COLOR);

		//-----------Start of extending dimensions------------

		float h = overRect.rows;
		float w = overRect.cols;

		vector<Point2f> borderPoints, transPoints;
		borderPoints.push_back(Point(0, 0));
		borderPoints.push_back(Point(0, h));
		borderPoints.push_back(Point(w, h));
		borderPoints.push_back(Point(w, 0));

		//figure out the new dimensions of the border if the points were transformed into the anchor image
		perspectiveTransform(borderPoints, transPoints, homography);

		//begin comparison of dimensions to figure out how we need to extend the bounds of the anchor image to fit the transformed image completely
		int xMin = 0;
		int yMin = 0;
		int xMax = imgB.cols;
		int yMax = imgB.rows;
		for (int i = 0; i < transPoints.size(); i++) {
			if (transPoints[i].x < xMin) {
				xMin = transPoints[i].x;
			}
			if (transPoints[i].x > xMax) {
				xMax = transPoints[i].x;
			}
			if (transPoints[i].y < yMin) {
				yMin = transPoints[i].y;
			}
			if (transPoints[i].y > yMax) {
				yMax = transPoints[i].y;
			}
		}

		int shiftxMin = 0;
		int shiftyMin = 0;
		int shiftxMax = 0;
		int shiftyMax = 0;

		cout << "\nCalculating necessary dimension shifts!" << endl;
		if (xMin < 0) {
			shiftxMin = xMin * -1;
			xMax += shiftxMin;
			xMin = 0;
		}

		if (yMin < 0) {
			shiftyMin = yMin * -1;
			yMax += shiftyMin;
			yMin = 0;
		}

		if (xMax > imgB.cols) {
			shiftxMax = xMax - shiftxMin - imgB.cols;
		}

		if (yMax > imgB.rows) {
			shiftyMax = yMax - shiftyMin - imgB.rows;
		}

		float transArr[3][3] = { {1.0,0,shiftxMin},{0,1.0,shiftyMin},{0,0,1.0} };
		Mat transMat;
		transMat = Mat(3, 3, CV_32F, &transArr);
		transMat.convertTo(transMat, 6);

		homography = transMat * homography;

		cout << "\nPost-shift homography: \n" << homography << endl;

		//------------end of Part 2 with homography * shift matrix, beginning of Part 3-------------

		//prepare the transformed image to be attached
		Mat warpedRect(yMax, xMax, CV_8UC3);
		warpPerspective(overRect, warpedRect, homography, warpedRect.size());
		imgB = imread(imgAnchor, IMREAD_COLOR);

		//make a border on anchor image with bounds as calculated above
		copyMakeBorder(imgB, imgB, shiftyMin, shiftyMax, shiftxMin, shiftxMax, BORDER_CONSTANT, 0);

		// Section to determine if the transformed image is about to be placed on top of an existing image (unnecessary stitch)
		perspectiveTransform(borderPoints, transPoints, homography);
		vector<cv::Point> borderCheckPoints;
		for (int i = 0; i < transPoints.size(); i++) { //have add a buffer in case the transform leads to pixel on the border
			if ((int)transPoints[i].x == xMax && (int)transPoints[i].y == yMax) {
				borderCheckPoints.push_back(Point((int)transPoints[i].x - 1, (int)transPoints[i].y - 1));
			}
			else if ((int)transPoints[i].x == xMax && (int)transPoints[i].y == 0) {
				borderCheckPoints.push_back(Point((int)transPoints[i].x - 1, (int)transPoints[i].y + 1));
			}
			else if ((int)transPoints[i].x == 0 && (int)transPoints[i].y == yMax) {
				borderCheckPoints.push_back(Point((int)transPoints[i].x + 1, (int)transPoints[i].y - 1));
			}
			else if ((int)transPoints[i].x == xMax) {
				borderCheckPoints.push_back(Point((int)transPoints[i].x - 1, (int)transPoints[i].y));
			}
			else if ((int)transPoints[i].x == 0) {
				borderCheckPoints.push_back(Point((int)transPoints[i].x + 1, (int)transPoints[i].y));
			}
			else if ((int)transPoints[i].y == yMax) {
				borderCheckPoints.push_back(Point((int)transPoints[i].x, (int)transPoints[i].y - 1));
			}
			else if ((int)transPoints[i].y == 0) {
				borderCheckPoints.push_back(Point((int)transPoints[i].x, (int)transPoints[i].y + 1));
			}
			else {
				borderCheckPoints.push_back(Point((int)transPoints[i].x, (int)transPoints[i].y));
			}
		}

		//cornCount stands for Corner count! I.E. how many corners are black pixels
		int cornCount = 0;
		for (int i = 0; i < transPoints.size(); i++) {
			Vec3b intensity = imgB.at<Vec3b>(borderCheckPoints[i]);
			if (intensity.val[0] != 0 && intensity.val[1] != 0 && intensity.val[2] != 0) {
				cornCount += 1;
			}
		}

		if (cornCount == 4) {
			cout << "THIS IMAGE iS INSIDE THE EXISTING IMAGE. NOT STITCHING." << endl;
			cout << "Removing file from list: " << filenames[removeIdx] << endl;
			filenames.erase(filenames.begin() + removeIdx);
			continue;
		}

		// ---------- Ending unnecessary stitch section -----------

		Mat joinedImg = imgB.clone();

		//handling the joining of the two images - draw a black polygon on the anchor image which is the size of the transformed image then bitwise_or them together!
		fillConvexPoly(joinedImg, borderCheckPoints, Scalar(0, 0, 0));
		bitwise_or(warpedRect, joinedImg, joinedImg);

		namedWindow("Anchor Image", cv::WINDOW_NORMAL);
		imshow("Anchor Image", imgB);

		namedWindow("Attached Image", cv::WINDOW_NORMAL);
		imshow("Attached Image", warpedRect);

		namedWindow("joinedImg", cv::WINDOW_NORMAL);
		imshow("joinedImg", joinedImg);

		resize(joinedImg, joinedImg, Size(joinedImg.cols * .9, joinedImg.rows * .9));

		imwrite(folderBase + "stitched.jpg", joinedImg);
		cout << "Removing file from list: " << filenames[removeIdx] << endl;
		filenames.erase(filenames.begin() + removeIdx);

		waitKey();

		//------------------- END of Part 3, loops back up to check another image --------------------------

	}
	cout << "Stitched all files! Find the output in: " << folderBase << endl;
	return 0;
}