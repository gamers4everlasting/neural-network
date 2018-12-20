# SDL stuff (for sound processing)
F1= -I/usr/local/include -L/usr/local/lib -lSDL2
F2= -std=c++11 -lfftw3 -lm

#CUDA includes and libraries
F3= -L/usr/local/cuda/lib64
F4= -I/usr/local/cuda-9.2/targets/x86_64-linux/include -lcuda -lcudart
F5= -lglut -lGL

all: recognition

recognition: recognition.o audio.o
	g++ -g -o recognition recognition.o audio.o $(F1) $(F2)

proj: interface.o audio.o gpu_main.o animate.o
	g++ -g -o proj interface.o audio.o gpu_main.o animate.o $(F1) $(F2) $(F3) $(F4) $(F5)

# do we really need all these flags to compile interface??!!
interface.o: interface.cpp interface.h animate.h animate.cu
	g++ -g -w -c interface.cpp $(F1) $(F2) $(F3) $(F4)

recognition.o: recognition.cpp
	g++ -g -w -c recognition.cpp $(F1) $(F2)

gpu_main.o: gpu_main.cu gpu_main.h audio.c audio.h
	nvcc -w -c gpu_main.cu

animate.o: animate.cu animate.h gpu_main.h
	nvcc -w -c animate.cu


audio.o: audio.c audio.h
	g++ -g -w -c audio.c $(F2) $(F3) $(F4)





clean:
	rm interface.o;
	rm gpu_main.o;
	rm animate.o;
	rm recognition.o;
	rm audio.o;
	rm proj;
