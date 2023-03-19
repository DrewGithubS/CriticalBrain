#include <cstdint>

class Organism {
private:
	float xPos;
	float yPos;

	float xVel;
	float yVel;

	float legX[4];
	float legY[4];

	float legVelX[4];
	float legVelY[4];

	float legPullX[4];
	float legPullY[4];

	float grip[4];
public:
	Organism();

	Organism(float xIn,
			 float yIn);

	Organism(float xIn,
			 float yIn,
			 float xVelIn,
			 float yVelIn);

	Organism(float xIn,
			 float yIn,
			 float xVelIn,
			 float yVelIn,
			 float * legXIn,
			 float * legYIn);

	Organism(float xIn,
			 float yIn,
			 float xVelIn,
			 float yVelIn,
			 float * legXIn,
			 float * legYIn,
			 float * legVelXIn,
			 float * legVelYIn);

	Organism(float xIn,
			 float yIn,
			 float xVelIn,
			 float yVelIn,
			 float * legXIn,
			 float * legYIn,
			 float * legVelXIn,
			 float * legVelYIn,
			 float * grip);
	
	float getXPos();
	float getYPos();
	float getLegAngle(int legNum);
	float getLegX(int legNum);
	float getLegY(int legNum);
	float getGrip(int legNum);
	uint8_t getGripUint(int legNum);

	void adjustGrip(int legNum, float newGrip);
	void pullLegX(int legNum, float pull);
	void pullLegY(int legNum, float pull);
	void simulateStep();
};