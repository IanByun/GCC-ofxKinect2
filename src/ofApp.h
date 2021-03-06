#pragma once

#include "ofMain.h"
#include <opencv.hpp>
#include <ofxKinect2.h>
#include "GrayCodeCalibrationWithKinect2.hpp"

class ofApp : public ofBaseApp {

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		
		shared_ptr<ofxKinect2::Device> kinectv2;
		ofxKinect2::ColorStream color_;
		ofxKinect2::DepthStream depth_;
		
		GrayCodeCalibrationWithKinect2 GCCalibrator;
		bool do_capture_pattern = false;
};
