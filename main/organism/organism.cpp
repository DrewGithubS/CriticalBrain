#include "organism.h"

#include <cstdint>
#include <cmath>

#define COEFICIENT_STATIC_FRICTION (0.75)
#define COEFICIENT_FRICTION 	   (0.5)

#define LEG_MASS  (1)
#define BODY_MASS (0.5)

#define PI (3.14159265359)

static const float legAngleMax[4] = {PI, PI / 2, 2 * PI, 3 * PI / 2};
static const float legAngleMin[4] = {PI / 2, 0, 3 * PI / 2, PI};

#define abs(x) (x < 0 ? -x : x)

float Organism::getXPos() {
	return xPos;
}

float Organism::getYPos() {
	return yPos;
}

float Organism::getLegX(int legNum) {
	return legX[legNum];
}

float Organism::getLegY(int legNum) {
	return legY[legNum];
}

float Organism::getLegAngle(int legNum) {
	return asin((legX[legNum] - xPos) / (legY[legNum] - yPos));
}

float Organism::getGrip(int legNum) {
	return grip[legNum];
}

uint8_t Organism::getGripUint(int legNum) {
	return uintGrip[legNum];
}

void Organism::adjustGrip(int legNum, float newGrip) {
	grip[legNum] = newGrip;
}

void Organism::pullLegX(int legNum, float pull) {
	legPullX[legNum] = pull;
}

void Organism::pullLegY(int legNum, float pull) {
	legPullY[legNum] = pull;
}

void Organism::simulateStep() {
	for(int i = 0; i < 4; i++) {
		if(abs(legVelX[i]) < 0.01) {
			legVelX[i] = 0;
		}

		if(abs(legVelY[i]) < 0.01) {
			legVelY[i] = 0;
		}

		// Note: Idk how components of friction work.
		if(legVelX[i] == 0) {
			// Static friction case.
			if(legPullX[i] * grip[i] > LEG_MASS * COEFICIENT_STATIC_FRICTION) {
				legVelX[i] += legPullX[i] * grip[i] - LEG_MASS * COEFICIENT_STATIC_FRICTION;
			}
			xVel -= (legPullX[i] * grip[i] - LEG_MASS * COEFICIENT_STATIC_FRICTION);
		} else {
			legVelX[i] += legPullX[i] * grip[i] - LEG_MASS * COEFICIENT_FRICTION;
			xVel -= (legPullX[i] * grip[i] - LEG_MASS * COEFICIENT_FRICTION);
		}

		if(legVelY[i] == 0) {
			// Static friction case.
			if(legPullY[i] * grip[i] > LEG_MASS * COEFICIENT_STATIC_FRICTION) {
				legVelY[i] += legPullY[i] * grip[i] - LEG_MASS * COEFICIENT_STATIC_FRICTION;
			}
			yVel -= (legPullY[i] * grip[i] - LEG_MASS * COEFICIENT_STATIC_FRICTION);
		} else {
			legVelY[i] += legPullY[i] * grip[i] - LEG_MASS * COEFICIENT_FRICTION;
			yVel -= (legPullY[i] * grip[i] - LEG_MASS * COEFICIENT_FRICTION);
		}

		legVelX[i] *= 0.95;
		legVelY[i] *= 0.95;
	}
}