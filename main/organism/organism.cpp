#include "organism.h"

#define LEG_VEL_COEF    (0.01)
#define LEG_ANGVEL_COEF (0.001)

#define LEG_MASS  (10)
#define BODY_MASS (5)

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
	return legAngle[legNum];
}

float Organism::getGrip(int legNum) {
	return grip[legNum];
}

void Organism::adjustGrip(int legNum, float newGrip) {
	grip[legNum] = newGrip;
}

void Organism::pullLeg(int legNum, float pull) {
	legVel[legNum] += LEG_VEL_COEF * pull;
}

void Organism::rotLeg(int legNum, float pull) {
	legAngularVel[legNum] += LEG_ANGVEL_COEF * pull;
}

void Organism::simulateStep() {

}