#pragma once

#include <opencv.hpp>
#include <ofxKinect2.h>

class myKinect2 : public ofxKinect2::Device {
public:
	inline ofPoint depthToCamera(const ofPoint& o) {
		return depthToCamera(vector<ofPoint>{ o })[0];
	}
	vector<ofPoint> depthToCamera(const vector<ofPoint>& o) {
		int COLOR_WIDTH = getColorStream()->getWidth();
		int COLOR_HEIGHT = getColorStream()->getHeight();
		int DEPTH_WIDTH = getDepthStream()->getWidth();
		int DEPTH_HEIGHT = getDepthStream()->getHeight();

		DepthSpacePoint* depths = new DepthSpacePoint[o.size()];
		UINT16* depthRaws = new UINT16[o.size()];
		for (size_t i = 0; i < o.size(); i++) {
			depths[i].X = reflipDepthX(o[i].x);
			depths[i].Y = reflipDepthY(o[i].y);
			depthRaws[i] = getDepthShortPixelsUnflipped()[(int)(depths[i].Y + 0.5)*DEPTH_WIDTH + (int)(depths[i].X + 0.5)];
		}
		CameraSpacePoint* cameras = new CameraSpacePoint[o.size()];
		this->getMapper()->MapDepthPointsToCameraSpace(o.size(), depths, o.size(), depthRaws, o.size(), cameras);

		vector<ofPoint> out(o.size());
		for (size_t i = 0; i < o.size(); i++) {
			out[i].x = cameras[i].X * 1000;
			out[i].y = cameras[i].Y * 1000;
			out[i].z = cameras[i].Z * 1000;
		}

		delete[] depths;
		delete[] depthRaws;
		delete[] cameras;
		return out;
	}

	inline ofPoint colorToCamera(const ofPoint& o) {
		return colorToCamera(vector<ofPoint>{ o })[0];
	}
	vector<ofPoint> colorToCamera(const vector<ofPoint>& o) {
		int COLOR_WIDTH = getColorStream()->getWidth();
		int COLOR_HEIGHT = getColorStream()->getHeight();
		int DEPTH_WIDTH = getDepthStream()->getWidth();
		int DEPTH_HEIGHT = getDepthStream()->getHeight();

		CameraSpacePoint* cameras = new CameraSpacePoint[COLOR_WIDTH*COLOR_HEIGHT];
		this->getMapper()->MapColorFrameToCameraSpace(
			DEPTH_WIDTH * DEPTH_HEIGHT,
			getDepthShortPixelsUnflipped().getData(),
			COLOR_WIDTH * DEPTH_HEIGHT, cameras);
		
		vector<ofPoint> out(o.size());
		for (size_t i = 0; i < o.size(); i++) {
			ofPoint pt;
			pt.x = reflipColorX(o[i].x);
			pt.y = reflipColorY(o[i].y);

			CameraSpacePoint camera = cameras[(int)(pt.y + 0.5)*COLOR_WIDTH + (int)(pt.x + 0.5)];
			out[i].x = camera.X * 1000;
			out[i].y = camera.Y * 1000;
			out[i].z = camera.Z * 1000;
		}
		delete[] cameras;
		return out;
	}

	inline ofPixels& getColorPixels() {
		return getColorStream()->getPixelsRef();
	} // 8비트  RGBA
	inline ofPixels getDepthPixels() {
		return getDepthStream()->getPixels();
	} // 8비트 그레이
	inline ofShortPixels& getDepthShortPixels() {
		return getDepthStream()->getShortPixelsRef();
	} // 16비트 그레이
	inline ofShortPixels& getDepthShortPixelsUnflipped() {
		return getDepthStream()->getShortPixelsUnflipped();
	}

	inline cv::Mat getColorMat() {
		auto& pixels = getColorPixels();
		cv::Mat color = cv::Mat(pixels.getHeight(), pixels.getWidth(), CV_8UC4, pixels.getData()).clone();
		cvtColor(color, color, CV_RGBA2BGR);
		return color;
	}
	inline cv::Mat getDepthMat() {
		auto& pixels = getDepthPixels();
		cv::Mat depth8 = cv::Mat(pixels.getHeight(), pixels.getWidth(), CV_8UC1, pixels.getData()).clone();
		return depth8;
	}
	inline cv::Mat getDepthShortMat() {
		auto& pixels = getDepthShortPixels();
		cv::Mat depth16 = cv::Mat(pixels.getHeight(), pixels.getWidth(), CV_16UC1, pixels.getData()).clone();
		return depth16;
	}

	inline float reflipDepthX(float x) {
		bool FLIP_HORIZONTAL = getDepthStream()->isMirrorHorizontal();
		int DEPTH_WIDTH = getDepthStream()->getWidth();

		return FLIP_HORIZONTAL ? DEPTH_WIDTH - x - 1 : x;
	}
	inline float reflipDepthY(float y) {
		bool FLIP_VERTICAL = getDepthStream()->isMirrorVertical();
		int DEPTH_HEIGHT = getDepthStream()->getHeight();

		return FLIP_VERTICAL ? DEPTH_HEIGHT - y - 1 : y;
	}
	inline float reflipColorX(float x) {
		bool FLIP_HORIZONTAL = getColorStream()->isMirrorHorizontal();
		int COLOR_WIDTH = getColorStream()->getWidth();

		return FLIP_HORIZONTAL ? COLOR_WIDTH - x - 1 : x;
	}
	inline float reflipColorY(float y) {
		bool FLIP_VERTICAL = getColorStream()->isMirrorVertical();
		int COLOR_HEIGHT = getColorStream()->getHeight();

		return FLIP_VERTICAL ? COLOR_HEIGHT - y - 1 : y;
	}

	inline int getColorWidth() { return getColorStream()->getWidth(); }
	inline int getColorHeight() { return getColorStream()->getHeight(); }
	inline int getDepthWidth() { return getDepthStream()->getWidth(); }
	inline int getDepthHeight() { return getDepthStream()->getHeight(); }
};