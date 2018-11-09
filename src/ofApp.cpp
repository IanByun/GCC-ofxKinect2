#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetLineWidth(5);

	kinectv2 = make_shared<ofxKinect2::Device>();
	if (kinectv2->setup()) {
		if (depth_.setup(*kinectv2)) depth_.open();
		else ofLogError() << "DepthStream not opened";
		if (color_.setup(*kinectv2)) color_.open();
		else ofLogError() << "ColorStream not opened";
	}

	depth_.setMirror(false, true);
	color_.setMirror(false, true);

	while (!kinectv2->isReady()) {
		kinectv2->update();

		depth_.update();
		color_.update();
	}

	GCCalibrator.setup({ kinectv2 }, 1920, 1080);
}

//--------------------------------------------------------------
void ofApp::update() {
	kinectv2->update();
#if (NUM_CAMERAS == 2)
	kinectv1->update();
#endif
	if (do_capture_pattern) {
		GCCalibrator.process();
	}
}

//--------------------------------------------------------------
void ofApp::draw() {
	ofBackground(ofColor::black);

	ofPoint mid = ((myKinect2*)kinectv2.get())->colorToCamera({ 
		kinectv2->getColorStream()->getWidth() / 4.f,
		kinectv2->getColorStream()->getHeight() / 4.f });
	cout << mid << endl;

	kinectv2->getColorStream()->draw(
		0, 0,
		kinectv2->getColorStream()->getWidth() / 4, 
		kinectv2->getColorStream()->getHeight() / 4
	);
	
	GCCalibrator.draw(0);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	switch (key) {
	case 'F': case 'f': ofToggleFullscreen(); break;
	case ' ': do_capture_pattern = true; break;
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ) {

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) { 

}
