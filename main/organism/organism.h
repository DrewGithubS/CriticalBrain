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
	float getXPos();
	float getYPos();
	float getLegAngle(int legNum);
	float getLegX(int legNum);
	float getLegY(int legNum);
	float getGrip(int legNum);

	void adjustGrip(int legNum, float newGrip);
	void pullLegX(int legNum, float pull);
	void pullLegY(int legNum, float pull);
	void simulationStep();
}