#include "organism.h"

#define COEFICIENT_STATIC_FRICTION (0.75)
#define COEFICIENT_FRICTION 	   (0.5)

#define LEG_MASS  (1)
#define BODY_MASS (0.5)

#define PI (3.14159265359)

static const float legAngleMax[4] = {PI, PI / 2, 2 * PI, 3 * PI / 2};
static const float legAngleMin[4] = {PI / 2, 0, 3 * PI / 2, PI};

float Organism::getXPos() {
	return xPos;
}

float Organism::getYPos() {
	return yPos;
}

float Organism::getLegX(int legNum) {
	return legExtension[legNum] * cos(legAngle[legNum]) + xPos;
}

float Organism::getLegY(int legNum) {
	return legExtension[legNum] * sin(legAngle[legNum]) + yPos;
}

float Organism::getLegAngle(int legNum) {
	return arcsin((legX[legNum] - xPos) / (legY[legNum] - yPos));
}

float Organism::getGrip(int legNum) {
	return grip[legNum];
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
			if(legPullX[i] * grip > LEG_MASS * COEFICIENT_STATIC_FRICTION) {
				legVelX += legPullX[i] * grip[i] - LEG_MASS * COEFICIENT_STATIC_FRICTION;
			}
			xVel -= (legPullX[i] * grip[i] - LEG_MASS * COEFICIENT_STATIC_FRICTION);
		} else {
			legVelX += legPullX[i] * grip[i] - LEG_MASS * COEFICIENT_FRICTION;
			xVel -= (legPullX[i] * grip[i] - LEG_MASS * COEFICIENT_FRICTION);
		}

		if(legVelY[i] == 0) {
			// Static friction case.
			if(legPullY[i] * grip > LEG_MASS * COEFICIENT_STATIC_FRICTION) {
				legVelY += legPullY[i] * grip[i] - LEG_MASS * COEFICIENT_STATIC_FRICTION;
			}
			yVel -= (legPullY[i] * grip[i] - LEG_MASS * COEFICIENT_STATIC_FRICTION);
		} else {
			legVelY += legPullY[i] * grip[i] - LEG_MASS * COEFICIENT_FRICTION;
			yVel -= (legPullY[i] * grip[i] - LEG_MASS * COEFICIENT_FRICTION);
		}

		legVelX[i] *= 0.95;
		legVelY[i] *= 0.95;
	}
}