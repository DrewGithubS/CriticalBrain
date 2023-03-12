#ifndef RENDERER_H
#define RENDERER_H

#include <SDL2/SDL.h>

class Renderer {
private:
	uint32_t width;
	uint32_t height;
	uint32_t imageBytes;
	SDL_Window * pWindow;
	SDL_Renderer * pRenderer;
	SDL_Surface * pScreen;
	SDL_Texture * pTexture;
	bool isRendererActive;
public:
	bool isRunning;
	Renderer(uint32_t widthIn, uint32_t heightIn);
	bool init();
	void checkForEvent();
	void getFrame();
	void render();
	void exit();
	void * getScreen();
	void setFrame(void * input);
	bool getActiveStatus() {return isRendererActive;}
};

#endif