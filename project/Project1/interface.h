#ifndef hInterfaceLib
#define hInterfaceLib
#include <stdio.h>
#include <cstdlib>		// need this?
#include <string.h>		// need this?
#include <time.h>
#include <unistd.h> 	// includes usleep
#include <SDL2/SDL.h>	// for sound processing..
#include <math.h>

#include "randlib.h"
#include "crack.h"
#include "audio.h"
#include "gpu_main.h"
#include "animate.h"

GPU_Palette openPalette(int, int);
int usage();

#endif
