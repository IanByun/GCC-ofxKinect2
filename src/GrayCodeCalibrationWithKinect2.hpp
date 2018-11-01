#pragma once
#include <opencv.hpp>
#include <ofxKinect2.h>
#include "myKinect2.hpp"
#include "structured_light/graycodepattern.hpp"

using namespace cv;

namespace custom_opencv_calibration_cpp {
	double calibrateCamera(
		InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints,
		Size imageSize, InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
		OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags = 0, 
		TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON));
}

class GrayCodeCalibrationWithKinect2 {
	const string TAG = "GrayCodeCalibrationWithKinect2";

protected:
	cv::structured_light::GrayCodePattern::Params params;
	cv::Ptr<cv::structured_light::GrayCodePattern> graycode;

	std::vector<cv::Mat> pattern;
	std::vector<cv::Mat>::iterator iteration;

	std::vector<std::vector<cv::Mat>> capturedFrames; // 0번은 프로젝션 패턴, 1번째부터 카메라 캡처

	struct CalibrationParameters {
		cv::Mat cameraMatrix;
		cv::Mat distCoeffs; // k1, k2, p1, p2, k3
		cv::Mat rotation;
		cv::Mat translation;
		float reprojError = FLT_MAX;
	};

	std::vector<shared_ptr<myKinect2>> cameras;
	std::vector<CalibrationParameters> calibrationResults;
	bool m_bSequenceEnded = false;

	int calibrationFlags =
		CV_CALIB_USE_INTRINSIC_GUESS
		+ CV_CALIB_FIX_ASPECT_RATIO
		+ CV_CALIB_ZERO_TANGENT_DIST //p1, p2?
		+ CV_CALIB_FIX_K3
		;
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1.0e-4);

	bool CheckPlanarity(const std::vector<cv::Point3f>& worldPoints) {
		Matx33f A;
		float cx = 0, cy = 0, cz = 0;
		int n = worldPoints.size();

		for (int i = 0; i < n; i++) {
			Point3f p = worldPoints[i];
			cx += p.x;
			cy += p.y;
			cz += p.z;
		}

		cx /= n;
		cy /= n;
		cz /= n;

		for (int i = 0; i < n; i++) {
			Point3f p = worldPoints[i];
			A(0, 0) += (p.x - cx)*(p.x - cx);
			A(1, 1) += (p.y - cy)*(p.y - cy);
			A(2, 2) += (p.z - cz)*(p.z - cz);

			A(0, 1) += (p.x - cx)*(p.y - cy);
			A(0, 2) += (p.x - cx)*(p.z - cz);
			A(1, 2) += (p.y - cy)*(p.z - cz);

			A(1, 0) += (p.x - cx)*(p.y - cy);
			A(2, 0) += (p.x - cx)*(p.z - cz);
			A(2, 1) += (p.y - cy)*(p.z - cz);
		}

		/*
		the eigenvalues are stored in the descending order.
		the eigenvectors are stored as subsequent matrix rows, in the same order as the corresponding eigenvalues.
		*/
		Mat eigenvalues, eigenvectors;
		eigen(A, eigenvalues, eigenvectors);

		/*
		"Fast plane detection for SLAM from noisy range images in both structured and unstructured environments." ICMA. 2011.

		Given its sorted eigenvalues λ1 ≤ λ2 ≤ λ3 from a grid g. The
		appearance of g can be decided by the following criteria:

		if size(g) < μ then
			g ⊂ sparse
		else if λ2 / λ3 ≤ α  then
			g ⊂ linear
		else if λ1 / λ2 ≤ β  then
			g ⊂ planar
		else
			g ⊂ spherical
		end if

		α = 0.006, β = 0.4. <- 이건 조건이 너무 가혹함. 진짜 큰 평면 찾을 때나?

		-----평면 측정 결과------
		eigenvalues
		[33599932;
		9509544;
		12823.601]

		eigenvectors
		[0.99627358, -0.023369854, 0.083025254;
		3.3988092e-06, 0.96260417, 0.27091196;
		-0.086251631, -0.26990211, 0.9590171]

		12823.601 / 9509544 = 0.001348497993174
		-----------------------
		*/

		//cout << "eigenvalues" << endl << eigenvalues << endl;
		//cout << "eigenvectors" << endl << eigenvectors << endl;

		float planarity = eigenvalues.at<float>(2) / eigenvalues.at<float>(1);
		bool planar = planarity < 0.05; // 0.2 ~ 0.01

		//cout << "planarity: " << planarity << " | " << "planar: " << (planar ? "true" : "false") << endl;

		return planar;
	}

	void CameraPointsFromProjPoints(IN int camera_index, IN vector<vector<Point> >& proj2color, OUT vector<Point3f>& objectPoints, OUT vector<Point2f>& imagePoints) {
		vector<Point2f> projPoints;
		vector<ofPoint> colorPoints;

		for (int y = 0; y < params.height; y++) {
			for (int x = 0; x < params.width; x++) {
				if (proj2color[y][x] != Point(-1, -1)) {
					Point2f projPoint = Point(x, y);
					ofPoint colorPoint(proj2color[y][x].x, proj2color[y][x].y);

					projPoints.push_back(projPoint);
					colorPoints.push_back(colorPoint);
				}
			}
		}

		objectPoints.clear();
		imagePoints.clear();
		vector<ofPoint> cameraPoints = cameras[camera_index]->colorToCamera(colorPoints);

		for (int i = 0; i < cameraPoints.size(); i++) {
			auto cam = cameraPoints[i];

			if (cam.z > 1000 && cam.z < 6500) {
				//cout << projPoints[i] << " " << colorPoints[i] << " " << cam << endl;

				imagePoints.push_back(projPoints[i]);
				objectPoints.push_back(Point3f(cam.x, cam.y, cam.z));
			}
		}
	}

	bool SampleNonPlanarPoints(IN const vector<Point3f>& objectPoints, IN const vector<Point2f>& imagePoints, OUT vector<Point3f>& objectPointsSampled, OUT vector<Point2f>& imagePointsSampled) {
		int sampleCount = 100;
		if (objectPoints.size() < sampleCount) return false;

		ofSeedRandom();

		bool planar = true;
		int nTries = 0;
		while (planar && (nTries++ < 1000)) {
			objectPointsSampled.clear();
			imagePointsSampled.clear();

			for (int j = 0; j < sampleCount; j++) {
				int k = ofRandom(objectPoints.size() - 1);
				objectPointsSampled.push_back(objectPoints[k]);
				imagePointsSampled.push_back(imagePoints[k]);
			}

			planar = CheckPlanarity(objectPointsSampled);
			//ofLogWarning(TAG) << "Non-Planarity Subset Trial #" << nTries << ": " << (planar ? "failed" : "success");
		}

		return planar;
	}

public:
	void setup(std::vector<shared_ptr<ofxKinect2::Device>> _cameras, int width, int height) {
		using namespace std;
		using namespace cv;
		using namespace structured_light;

		for (auto& camera : _cameras)
			cameras.push_back(shared_ptr<myKinect2>(reinterpret_cast<myKinect2*>(camera.get())));

		params.width = width;
		params.height = height;

		graycode = GrayCodePattern::create(params);
		graycode->generate(pattern);

		Mat white;
		Mat black;
		graycode->getImagesForShadowMasks(black, white);
		pattern.push_back(white);
		pattern.push_back(black);

		iteration = pattern.begin();
		capturedFrames.resize(cameras.size() + 1);
		calibrationResults.resize(cameras.size());
	}

	void process() {
		if (!m_bSequenceEnded) {
			ofSetFullscreen(false);
			pattern_project_capture();
			pattern_decode_calibrate();
			m_bSequenceEnded = true;
			ofSetFullscreen(true);
		}
	}

private:
	void pattern_project_capture() {
		using namespace std;
		using namespace cv;
		using namespace structured_light;

		if (iteration == pattern.begin()) {
			namedWindow("Pattern Window", WINDOW_NORMAL);
			resizeWindow("Pattern Window", params.width, params.height);
			setWindowProperty("Pattern Window", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

			namedWindow("Captured Window", WINDOW_NORMAL);
			moveWindow("Captured Window", params.width * 1.2, params.height / 3);

			waitKey(100);
		}

		while (iteration != pattern.end()) {
			imshow("Pattern Window", *iteration);
#ifdef _DEBUG
			waitKey(1000);
#else
			waitKey(500);
#endif
			capturedFrames[0].push_back((*iteration).clone());
			for (int camera_index = 0; camera_index < cameras.size(); camera_index++) {
				auto camera = cameras[camera_index];
				camera->update();

				Mat capture = camera->getColorMat().clone();
				int numChannels = capture.channels();
				if (numChannels == 3) cvtColor(capture, capture, CV_RGB2GRAY);
				else if (numChannels == 4) cvtColor(capture, capture, CV_RGBA2GRAY);

				capturedFrames[camera_index + 1].push_back(capture.clone());

				/*Mat m = (*iteration).clone();
				resize(m, m, Size(capture.cols, capture.rows));
				hconcat(m, capture.clone(), m);

				resizeWindow("Captured Window", m.cols, m.rows);
				imshow("Captured Window", m);*/
				//waitKey(0);
			}

			iteration++;
		}

		waitKey(500);
	}

	void pattern_decode_calibrate() {
		using namespace std;
		using namespace cv;
		using namespace structured_light;

		if (iteration == pattern.end()) {
			
			destroyWindow("Pattern Window");
			destroyWindow("Captured Window");

			//----------패턴 캡처 결과 저장----------
			{
				CreateDirectoryA("data/Patterns/", NULL);

				string current_working_directory = "data/Patterns/proj/";
				CreateDirectoryA(current_working_directory.c_str(), NULL);
				for (int j = 0; j < capturedFrames[0].size(); j++) {
					Mat m = capturedFrames[0][j];
					imwrite(current_working_directory + ofToString(j) + ".jpg", m);
				}

				for (int i = 1; i < capturedFrames.size(); i++) {
					current_working_directory = "data/Patterns/cam" + ofToString(i) + "/";
					CreateDirectoryA(current_working_directory.c_str(), NULL);

					for (int j = 0; j < capturedFrames[i].size(); j++) {
						Mat m = capturedFrames[i][j];
						imwrite(current_working_directory + ofToString(j) + ".jpg", m);
					}

				}

				waitKey(1);
			}

			//----------프로젝터-i번째 카메라 캘리브레이션----------
			for (int camera_index = 1; camera_index < capturedFrames.size(); camera_index++) { // 0번은 프로젝션 패턴, 1번째부터 카메라 캡처
				string TAG = "Proj-Cam" + to_string(camera_index);
				vector<vector<Mat>> patternImages = { capturedFrames[0], capturedFrames[camera_index] };
				vector<vector<Point> > proj2colorMapping;
				vector<Mat> blackImages = { capturedFrames[0].back(),
					camera_index == 1 ? capturedFrames[camera_index].back() // 키넥트 1 오토 화이트 밸런스가 너무 심해서 전역 검색하게 바닥값 이미지
					: Mat::zeros(capturedFrames[camera_index].back().size(), capturedFrames[camera_index].back().type())
				};
				vector<Mat> whiteImages = { *(capturedFrames[0].end() - 2), *(capturedFrames[camera_index].end() - 2) };

				ofLogNotice(TAG) << "Decoding pattern: this may take a while...";
				graycode->decode(patternImages, proj2colorMapping, blackImages, whiteImages, DECODE_PROCAM_ENSEMBLE);

				int numTries = 0;
				constexpr int maxTries = 10;

				vector<Point3f> objectPoints;
				vector<Point2f> imagePoints;
				CameraPointsFromProjPoints(camera_index - 1, proj2colorMapping, objectPoints, imagePoints);

				while ((calibrationResults[camera_index - 1].reprojError > 4) && (numTries++ < maxTries)) {
				IterateWithoutIncrement:
					vector<Point3f> objectPointsSubset;
					vector<Point2f> imagePointsSubset;
					bool planar = SampleNonPlanarPoints(objectPoints, imagePoints, objectPointsSubset, imagePointsSubset);
					if (planar) {
						ofLogWarning(TAG) << "Trial #" << numTries << " No Non-Planar Subset retrying...";
						goto IterateWithoutIncrement; // do another iteration without incrementing numTries flag
					}

					Mat cameraMatrix = (Mat1d(3, 3) <<
						params.width, 0, params.width / 2,
						0, params.width, params.height / 2,
						0, 0, 1);
					Mat distCoeffs = Mat::zeros(5, 1, CV_64F); // k1, k2, p1, p2, k3

					if (camera_index > 1) {
						cameraMatrix = calibrationResults[0].cameraMatrix.clone();
						distCoeffs = calibrationResults[0].distCoeffs.clone();
						calibrationFlags = 
							CV_CALIB_USE_INTRINSIC_GUESS
							+ CV_CALIB_FIX_ASPECT_RATIO
							+ CV_CALIB_ZERO_TANGENT_DIST //p1, p2?
							+ CV_CALIB_FIX_K3
							+ CV_CALIB_FIX_PRINCIPAL_POINT
							+ CV_CALIB_FIX_FOCAL_LENGTH;
					}

					vector<Mat>	rotations, translations;
					float reprojError =
						custom_opencv_calibration_cpp::calibrateCamera(
							vector<vector<Point3f>>(1, objectPointsSubset), 
							vector<vector<Point2f>>(1, imagePointsSubset),
							Size(params.width, params.height),
							cameraMatrix, distCoeffs, rotations, translations,
							calibrationFlags, criteria);

					vector<Point3f> objectPointsInliers;
					vector<Point2f> imagePointsInliers;
					for (int k = 0; k < objectPoints.size(); k++) {
						vector<Point2f> imagePointsProjected;
						cv::projectPoints(vector<Point3f>(1, objectPoints[k]), rotations[0], translations[0], cameraMatrix, distCoeffs, imagePointsProjected);
						Point2f gt = imagePoints[k];
						Point2f pj = imagePointsProjected[0];
						float dist = sqrt(pow((gt.x - pj.x), 2.f) + pow((gt.x - pj.x), 2.f));
						if (dist < 2.f) {
							objectPointsInliers.push_back(objectPoints[k]);
							imagePointsInliers.push_back(imagePoints[k]);
						}
					}

					bool enoughInliers = objectPointsInliers.size() > 500;
					if (enoughInliers) {
						/*
						카메라 매트릭스 위에서 구한 것을 초기값으로 쓰면 가끔 에러가 떠서 cpp에 calibrateCamera 소스 가져옴
						OpenCV Error: One of arguments' values is out of range (Principal point must be within the image
						*/
						cameraMatrix = (Mat1d(3, 3) <<
							params.width, 0, params.width / 2,
							0, params.width, params.height / 2,
							0, 0, 1);
						distCoeffs = Mat::zeros(5, 1, CV_64F); // k1, k2, p1, p2, k3

						if (camera_index > 1) {
							cameraMatrix = calibrationResults[0].cameraMatrix.clone();
							distCoeffs = calibrationResults[0].distCoeffs.clone();
							calibrationFlags =
								CV_CALIB_USE_INTRINSIC_GUESS
								+ CV_CALIB_FIX_ASPECT_RATIO
								+ CV_CALIB_ZERO_TANGENT_DIST //p1, p2?
								+ CV_CALIB_FIX_K3
								+ CV_CALIB_FIX_PRINCIPAL_POINT
								+ CV_CALIB_FIX_FOCAL_LENGTH;
						}

						rotations.clear();
						translations.clear();

						reprojError =
							custom_opencv_calibration_cpp::calibrateCamera(
								vector<vector<Point3f>>(1, objectPointsInliers),
								vector<vector<Point2f>>(1, imagePointsInliers),
								Size(params.width, params.height),
								cameraMatrix, distCoeffs, rotations, translations,
								calibrationFlags, criteria);

						if (reprojError < calibrationResults[camera_index - 1].reprojError) {
							calibrationResults[camera_index - 1].reprojError = reprojError;
							calibrationResults[camera_index - 1].cameraMatrix = cameraMatrix;
							calibrationResults[camera_index - 1].distCoeffs = distCoeffs;
							calibrationResults[camera_index - 1].rotation = rotations[0];
							calibrationResults[camera_index - 1].translation = translations[0];
						}
					}
					else {
						ofLogWarning(TAG) << "Trial #" << numTries << " Not Enough Inliers retrying...";
						goto IterateWithoutIncrement; // do one more iteration without incrementing numTries flag
					}

					ofLogNotice(TAG) << "Trial #" << numTries << " reprojError: " << reprojError;
				}
			}

			print();
			save();
		}

	}

public:
	void draw(int camera_index = 0) {
		using namespace std;
		using namespace cv;

		if (cameras.size() < camera_index || cameras[camera_index] == nullptr) return;

		size_t height = cameras[camera_index]->getDepthHeight();
		size_t width = cameras[camera_index]->getDepthWidth();

		Mat gray = Mat(height, width, CV_8UC1, cameras[camera_index]->getDepthPixels().getPixels()).clone();

		auto get_gradient_image = [](Mat gray)->Mat {
			int scale = 1;
			int delta = 0;
			int ddepth = CV_16S;

			Mat grad_x, grad_y;
			Mat abs_grad_x, abs_grad_y;

			Scharr(gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
			convertScaleAbs(grad_x, abs_grad_x);

			Scharr(gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
			convertScaleAbs(grad_y, abs_grad_y);

			Mat grad;
			addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
			
			return grad;
		};

		Mat grad = get_gradient_image(gray);

		Mat binary;
		threshold(grad, binary, 64, 255, THRESH_BINARY);
		//adaptiveThreshold(grad, binary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, 5);

		vector<vector<Point>> contours;
		findContours(
			binary,
			contours,
			CV_RETR_EXTERNAL,
			CHAIN_APPROX_TC89_KCOS
		);

		Mat result(binary.size(), CV_8U, Scalar(0));
		drawContours(result, contours, -1, Scalar(255), 2);
		
		//imshow("depth", gray);
		//imshow("gradient", grad);
		imshow("binary", binary);
		//imshow("contours", result);
		waitKey(1);

		if (m_bSequenceEnded) {
			for (vector<Point>& cv_contour_depth : contours) {
				vector<ofPoint> of_contour_depth;
				for (auto p : cv_contour_depth) of_contour_depth.push_back(ofPoint(p.x, p.y));

				vector<ofPoint> of_contour_camera = cameras[camera_index]->depthToCamera(of_contour_depth);

				vector<Point3f> cv_contour_camera(of_contour_camera.size());
				memcpy(cv_contour_camera.data(), of_contour_camera.data(), cv_contour_camera.size() * sizeof(Point3f));

				vector<Point2f> cv_contour_project;
				projectPoints(
					cv_contour_camera, 
					calibrationResults[camera_index].rotation, 
					calibrationResults[camera_index].translation, 
					calibrationResults[camera_index].cameraMatrix,
					calibrationResults[camera_index].distCoeffs, 
					cv_contour_project);

				ofPolyline line;
				for (auto pt : cv_contour_project) {
					line.addVertex(ofPoint(pt.x, pt.y));
				}
				line.draw();
			}
		}
	}

	void print() {
		using namespace std;
		using namespace cv;
		using namespace structured_light;

		for (int camera_index = 1; camera_index < capturedFrames.size(); camera_index++) {
			string deviceType = "ofxKinect2";

			string TAG = "Proj-Cam" + to_string(camera_index);
			cout << "----------------------------------------------" << endl;
			ofLogNotice(TAG) << "Final Result";
			cout << "Device type " << deviceType << endl;
			cout << "Camera matrix" << endl;
			cout << calibrationResults[camera_index - 1].cameraMatrix << endl;
			cout << "Dist coeffs" << endl;
			cout << calibrationResults[camera_index - 1].distCoeffs << endl;
			cout << "Rotation" << endl;
			cout << calibrationResults[camera_index - 1].rotation << endl;
			cout << "Translation" << endl;
			cout << calibrationResults[camera_index - 1].translation << endl;
			cout << "Reprojection Error " << calibrationResults[camera_index - 1].reprojError << endl;
			cout << "----------------------------------------------" << endl << endl;
		}
	}

	void save(bool absolute = true) {
		using namespace std;
		using namespace cv;
		using namespace structured_light;

		string timestamp = ofGetTimestampString();
		for (int camera_index = 1; camera_index < capturedFrames.size(); camera_index++) {
			string deviceType = "ofxKinect2";
			string filename = timestamp + "_" + deviceType + "Projector.yml";

			FileStorage fs(ofToDataPath(filename, absolute), FileStorage::WRITE);
			fs << "deviceType" << deviceType;
			fs << "intrisics" << calibrationResults[camera_index - 1].cameraMatrix;
			fs << "projResX" << params.width;
			fs << "projResY" << params.height;
			fs << "rotation" << calibrationResults[camera_index - 1].rotation;
			fs << "translation" << calibrationResults[camera_index - 1].translation;
			fs << "distCoeffs" << calibrationResults[camera_index - 1].distCoeffs;
			fs << "reprojectionError" << calibrationResults[camera_index - 1].reprojError;
		}
	}
};