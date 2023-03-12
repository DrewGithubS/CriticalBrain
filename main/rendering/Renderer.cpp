#include <stdlib.h>

#include "Renderer.h"

Renderer::Renderer(uint32_t widthIn, uint32_t heightIn) {
	isRunning = true;
	width = widthIn;
	height = heightIn;
	imageBytes = width * height * sizeof(uint32_t);
	if(!init()) {
		printf("Failed to initialize SDL.");
		exit();
	}
}

bool Renderer::init() {
	isRendererActive = true;

	if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
		return false;
	}

	pWindow = SDL_CreateWindow(
					"main",
					SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
					width, height,
					SDL_WINDOW_SHOWN
					);

	if(pWindow != NULL) {
        pRenderer = SDL_CreateRenderer(pWindow, -1, 0);
    } else {
        return false;
    }

	pScreen = SDL_CreateRGBSurface(0, width, height, 32,
										0x00FF0000,
										0x0000FF00,
										0x000000FF,
										0xFF000000);

	pTexture = SDL_CreateTexture(pRenderer,
									SDL_PIXELFORMAT_ARGB8888,
									SDL_TEXTUREACCESS_STREAMING | SDL_TEXTUREACCESS_TARGET,
									width, height);

	if (pTexture== NULL) {
        exit();
	}

	if (pScreen == NULL) {
        return false;
	}

	return true;
}

void Renderer::checkForEvent() {
	SDL_Event event;
	while(SDL_PollEvent(&event) != 0) {
        // Do something
        if(event.type == SDL_QUIT) {
	        exit();
	    }
    }
}

void Renderer::getFrame() {
	SDL_LockSurface(pScreen);
	SDL_UnlockSurface(pScreen);
}

void Renderer::setFrame(void * input) {
	SDL_LockSurface(pScreen);
	memcpy(getScreen(), input, width * height * sizeof(int));
	SDL_UnlockSurface(pScreen);
}

void Renderer::render() {
	SDL_UpdateTexture(pTexture, NULL, pScreen->pixels, pScreen->pitch);
	SDL_RenderClear(pRenderer);
	SDL_RenderCopy(pRenderer, pTexture, NULL, NULL);
	SDL_RenderPresent(pRenderer);
}

void Renderer::exit() {
	SDL_DestroyWindow(pWindow);
    pWindow = NULL;
    SDL_Quit();
	isRendererActive = false;
}

void * Renderer::getScreen() {
	return pScreen->pixels;
}