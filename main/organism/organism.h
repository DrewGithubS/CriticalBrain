class Organism {
private:
	float xPos;
	float yPos;

	float xVel;
	float yVel;

	float legExtension[4];
	float legAngle[4];
	float grip[4];

	float legVel[4];
	float legAngularVel[4];
	
public:
	float getXPos();
	float getYPos();
	float getLegX(int legNum);
	float getLegY(int legNum);
	float getLegAngle(int legNum);
	float getGrip(int legNum);

	void adjustGrip(int legNum, float newGrip);
	void pullLeg(int legNum, float pull);
	void rotLeg(int legNum, float pull);
	void simulationStep();
}