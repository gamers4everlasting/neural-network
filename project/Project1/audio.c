/*******************************************************************************
*
*   Stuff for sound processing
*
*******************************************************************************/
#include "audio.h"

#define BYTES_PER_SAMP 4        // assumes 16 bit 2 channel input

extern volatile int packet_pos;
extern volatile int print_spectrum;
const int BUCKETS = 9;// 5 sound spectres: bass, midbass, midrange, upper midrange, high-end; or 9 octaves
static int g_endprogram = 0;

AudioData* GetAudioData(Visual_Pkg* package)
{
    return package->AudioData_ptr;
}
SDL_AudioSpec* GetSDL_AudioSpec(Visual_Pkg* package)
{
    return package->wavSpec_ptr;
}
FFTW_Result* GetFFTW_Result(Visual_Pkg* package)
{
    return package->FFTW_Result_ptr;
}
FFTWop* GetFFTWop(Visual_Pkg* package)
{
    return package->fftw_ptr;
}
void closeNow(Uint8* wavPtr, SDL_AudioDeviceID device)
{

    SDL_CloseAudioDevice(device);
    SDL_FreeWAV(wavPtr);

    SDL_Quit();
    exit(0);
}
void MyAudioCallback(void* userdata, Uint8* stream, int streamLength)
{
  struct Visual_Pkg* package = (struct Visual_Pkg*)userdata;
  AudioData* audio = GetAudioData(package);

  if(audio->currentLength == 0)
  {
    if(g_endprogram)
      closeNow(audio->wavPtr, package->device);
    SDL_memset(stream, 0, streamLength);  // just silence.
    g_endprogram=1;
    return;
  }
  printOutput(package);
  Uint32 length = (Uint32)streamLength;
  length = (length > audio->currentLength ? audio->currentLength : length);

  SDL_memcpy(stream, audio->currentPos, length);


  audio->currentPos += length;
  audio->currentLength -= length;

  packet_pos++;
}
static void InitializePackage(SDL_AudioSpec* wavSpec,  Uint8* wavPtr, Uint32 wavLength, Visual_Pkg* pkg)
{
  AudioData* audio = (AudioData*)malloc(sizeof(AudioData));

  audio->currentPos = wavPtr;
  audio->wavPtr = wavPtr;
  audio->wavLength = wavLength;
  audio->currentLength = wavLength;

  wavSpec->callback = MyAudioCallback;
  wavSpec->userdata = &(*pkg);

  pkg->AudioData_ptr = audio;
  pkg->wavSpec_ptr = wavSpec;

}
static void InitializeVariables(Visual_Pkg* pkg, SDL_AudioSpec have, SDL_AudioDeviceID device)
{
  int sizeof_packet, totalpackets, frame_size, i, ch;
  AudioData* audio;

  pkg->device = device;
  pkg->setupDFT = getDFT;

  SDL_AudioSpec* wavSpec = GetSDL_AudioSpec(pkg);
  //initialize function ptr
  if(wavSpec->format != have.format)
      wavSpec->format = have.format;

  pkg->bitsize = (int)SDL_AUDIO_BITSIZE(wavSpec->format);
  pkg->GetAudioSample = Get16bitAudioSample;

  if(wavSpec->channels != have.channels)
      wavSpec->channels = have.channels;

  if(wavSpec->samples != have.samples ){
      printf("original sample size: %d\n"
      "new sample size: %d\n", wavSpec->samples, have.samples);
      wavSpec->samples = have.samples;
  }

  //size of samples in bytes?
  sizeof_packet =  pkg->bitsize / 8 ;
  sizeof_packet *=wavSpec->channels;
  sizeof_packet *= wavSpec->samples;

  //find the total number of packets
  //wavLength is the size of audio data in bytes
  audio = GetAudioData(pkg);

  totalpackets= (int)ceil((float)audio->wavLength/sizeof_packet);
  pkg->total_packets = totalpackets;

  //A frame can consist of N channels
  frame_size = wavSpec->samples;
  pkg->frame_size = frame_size;
  pkg->total_frames = audio->wavLength;
  pkg->total_frames /= (((pkg->bitsize)/8) * wavSpec->channels);
  //FFTW Results for each packet
  //

  pkg->FFTW_Result_ptr = (FFTW_Result*)malloc(totalpackets * sizeof(FFTW_Result));

  for (i = 0; i < totalpackets; ++i){
      //for peak results
      pkg->FFTW_Result_ptr[i].peakfreq = (double*)malloc(wavSpec->channels*sizeof(double));
      pkg->FFTW_Result_ptr[i].peakpower = (double*)malloc(wavSpec->channels*sizeof(double));

      //for power spectrum (i.e. a double matrix) of
          //N BUCKETS that represent a frequency range
      pkg->FFTW_Result_ptr[i].peakmagMatrix = (double**)malloc(wavSpec->channels*sizeof(double));
      for(ch = 0; ch < wavSpec->channels ; ++ch){
          pkg->FFTW_Result_ptr[i].peakmagMatrix[ch] = (double*)malloc(BUCKETS*sizeof(double));
      }

  }
  pkg->fftw_ptr = (FFTWop*)malloc(wavSpec->channels*sizeof(FFTWop));

  //allocating space for dft operations
  for(i=0; i<wavSpec->channels; ++i){
      pkg->fftw_ptr[i].in = (double(*)[2])fftw_malloc(sizeof(fftw_complex) * frame_size);
      pkg->fftw_ptr[i].out = (double(*)[2])fftw_malloc(sizeof(fftw_complex) * frame_size);
      pkg->fftw_ptr[i].index = i;
  }
}
int runAudio(char* fileName, GPU_Palette* P, CPUAnimBitmap* A)
{
  SDL_AudioSpec wavSpec, have;
  SDL_AudioDeviceID device;	// audio file header info (channels, samp rate, etc)
  Uint8* wavPtr;					// address of start of audio data
  Uint32 wavLength;
  Visual_Pkg* pkg;
  int length;

	if(SDL_Init(SDL_INIT_AUDIO) < 0){
		printf("problem initializing SDL audio\n");
		return 1;
		}

  if(SDL_LoadWAV(fileName, &wavSpec, &wavPtr, &wavLength) == NULL){
    printf("error: file couldn't be found or isn't a proper .wav file\n");
    return 1;
  	}

	if (wavSpec.freq != 44100){
		printf(".wav frequency not 44100!\n");
		return 1;
		}

	// https://www.rubydoc.info/gems/sdl2_ffi/0.0.6/SDL2/Audio
	if (wavSpec.format != 0x8010){
		printf(".wav format is not S16LSB (signed 16 bit little endian)\n");
		return 1;
		}

	if (wavSpec.channels != 2){
		printf(".wav not 2 channel (stereo) file\n");
		return 1;
		}

	if (wavSpec.samples != 4096){
		printf(" SDL not using 4096 buffer size??!\n");
		return 1;
		}

  pkg = (Visual_Pkg*)malloc(sizeof(struct Visual_Pkg));
  InitializePackage(&wavSpec, wavPtr, wavLength, pkg);
  pkg->Filename = fileName;

  device = SDL_OpenAudioDevice(NULL, 0, &wavSpec, &have, 0);

  if(device == 0){
    printf("error: audio device not found\n");
    return 1;
  }
  InitializeVariables(pkg, have, device);
  processWAVFile(wavLength, have.size ,pkg);

  SDL_PauseAudioDevice(device, 0); //play song
  length = GetAudioData(pkg)->currentLength;

  while(length > 0)
  {
    // here is the main deal:
    myDraw(P, A, pkg, packet_pos);
    SDL_Delay(100);	// in ms
  }

  SDL_CloseAudioDevice(device);
  SDL_FreeWAV(wavPtr);
  SDL_Quit();
  return 0;
}
Visual_Pkg* openAudio(char* fileName)
{
  SDL_AudioSpec wavSpec, have;
  SDL_AudioDeviceID device;
  Uint8* wavPtr;					// address of start of audio data
  Uint32 wavLength;
  Visual_Pkg* pkg;

	if(SDL_Init(SDL_INIT_AUDIO) < 0){
		printf("problem initializing SDL audio\n");
    SDL_FreeWAV(wavPtr);
  	exit(1);
		}

  if(SDL_LoadWAV(fileName, &wavSpec, &wavPtr, &wavLength) == NULL){
    printf("error: file couldn't be found or isn't a proper .wav file\n");
    SDL_FreeWAV(wavPtr);
  	exit(1);
  	}
/*
	if (wavSpec.freq != 44100){
		printf(".wav frequency not 44100!\n");
    SDL_FreeWAV(wavPtr);
    exit(1);
		}

	// https://www.rubydoc.info/gems/sdl2_ffi/0.0.6/SDL2/Audio
	if (wavSpec.format != 0x8010){
		printf(".wav format is not S16LSB (signed 16 bit little endian)\n");
    SDL_FreeWAV(wavPtr);
  	exit(1);
		}

	if (wavSpec.channels != 2){
		printf(".wav not 2 channel (stereo) file\n");
    SDL_FreeWAV(wavPtr);
  	exit(1);
		}

	if (wavSpec.samples != 4096){
		printf(" SDL not using 4096 buffer size??!\n");
    SDL_FreeWAV(wavPtr);
  	exit(1);
		}
  */

  pkg = (Visual_Pkg*)malloc(sizeof(struct Visual_Pkg));
  InitializePackage(&wavSpec, wavPtr, wavLength, pkg);
  pkg->Filename = fileName;

  device = SDL_OpenAudioDevice(NULL, 0, &wavSpec, &have, 0);

  if(device == 0){
    printf("error: audio device not found\n");
    SDL_FreeWAV(wavPtr);
  	exit(1);
  }
  InitializeVariables(pkg, have, device);
  processWAVFile(wavLength, have.size ,pkg);

  SDL_CloseAudioDevice(device);
  SDL_FreeWAV(wavPtr);
  SDL_Quit();
  return pkg;
}
void printOutput(Visual_Pkg* package)
{
    int channels = GetSDL_AudioSpec(package)->channels;
    FFTW_Result* res= GetFFTW_Result(package);
    int thickness = 2, minutes, seconds, b, c, t, p, energy;
    double n_samples, tot_seconds;

    if(system("clear") < 0){}

    for(c = 0; c < channels; ++c){
        for(b = 0; b < BUCKETS; ++b){
            if((c+1)%2 == 0)
                energy = res[packet_pos].peakmagMatrix[c][b];
            else
                energy = res[packet_pos].peakmagMatrix[c][BUCKETS-b-1];
            for(t = 0; t < thickness; ++t){
                for (p = 0; p < energy; ++p)
                    putchar('#');

                putchar('|');
                putchar('\n');
            }
        }
        putchar('\n');
        fflush(stdout);
    }

    n_samples = GetAudioData(package)->currentLength;
    n_samples /= (package->bitsize/8);
    n_samples /= channels;

    tot_seconds = n_samples / GetSDL_AudioSpec(package)->freq;
    minutes = tot_seconds / 60;
    seconds = (int)tot_seconds%60;

    printf("\nFilename: %s\n", package->Filename);
    printf("Time left: %02d:%02d\n", minutes, seconds);
    printf("Channels: %d\n", channels);
    printf("Packet: %d\n", packet_pos);
    fflush(stdout);
}

void myDraw(GPU_Palette* P, CPUAnimBitmap* A, Visual_Pkg* package, int packetN){
		//cudaMemcpy(P->dft, ####, 1024 * sizeof(float), cH2D);
		updatePalette(P, package, packetN);
    A->drawPalette();
		//usleep(23000); // sleep 23 milliseconds
}

void getDFT(Visual_Pkg* pkg, Uint8* buffer, int bytesRead)
{
    int bytewidth = pkg->bitsize / 8;
    SDL_AudioSpec* wavSpec = GetSDL_AudioSpec(pkg);
    int channels = wavSpec->channels;
    SDL_AudioFormat fmt = wavSpec->format;
    int frames = bytesRead / (bytewidth * channels);
    struct FFTWop* fftwop = GetFFTWop(pkg);
    int count = 0, c;

    for(c = 0; c < channels; ++c){
        fftwop[c].p = fftw_plan_dft_1d(frames, fftwop[c].in,
                fftwop[c].out, FFTW_FORWARD, FFTW_MEASURE);
    }

    while(count < frames){
        for(c = 0; c< channels; ++c){
            fftwop[c].in[count][re] = pkg->GetAudioSample(buffer, fmt);
            fftwop[c].in[count][im] = 0.0;

            buffer+=bytewidth;
        }
        count++;
    }
}
void analyze_FFTW_Result(Visual_Pkg* packet, struct FFTWop fftwop ,
                                        int packet_index, int ch,size_t bytesRead)
{

    double real, imag;
    double peakmax = 1.7E-308 ;
    int max_index = -1, i, j;
    double magnitude;
    double* peakmaxArray = (double*)malloc(BUCKETS*sizeof(double));
    double nyquist = packet->wavSpec_ptr->freq / 2;
    //double freq_bin[] = {19.0, 140.0, 400.0, 2600.0, 5200.0, nyquist };
    double freq_bin[] = {16.0, 33.0, 65.0, 131.0, 262.0, 523.0, 1047.0, 2093.0, 4196.0, 8372.0, 15805.0};//9 octaves
    SDL_AudioSpec* wavSpec = GetSDL_AudioSpec(packet);

    int frames = bytesRead / (wavSpec->channels * packet->bitsize / 8);
    FFTW_Result* results = GetFFTW_Result(packet);


    for(i = 0; i<BUCKETS; ++i) peakmaxArray[i] = 1.7E-308;

    for(j = 0; j < frames/2; ++j){

        real =  fftwop.out[j][0];
        imag =  fftwop.out[j][1];
        magnitude = sqrt(real*real+imag*imag);
        double freq = j * (double)wavSpec->freq / frames;

        for (i = 0; i < BUCKETS; ++i){
            if((freq>freq_bin[i]) && (freq <=freq_bin[i+1])){
                if (magnitude > peakmaxArray[i]){
                    peakmaxArray[i] = magnitude;
                }
            }
        }

        if(magnitude > peakmax){
                peakmax = magnitude;
                max_index = j;
        }
    }

    results[packet_index].peakpower[ch] =  10*(log10(peakmax));
    results[packet_index].peakfreq[ch] = max_index*(double)wavSpec->freq/frames;

    for(i = 0; i< BUCKETS; ++i){
        results[packet_index].peakmagMatrix[ch][i]=10*(log(peakmaxArray[i]));// to fit the result into console screen;
                                                                            // use pow(peakmagMatrix[ch][i],10)
    }
    free(peakmaxArray);
}
int getFileSize(FILE *inFile)
{
    int fileSize = 0;
    fseek(inFile,0,SEEK_END);

    fileSize=ftell(inFile);

    fseek(inFile,0,SEEK_SET);

    return fileSize;
}
void processWAVFile(Uint32 wavLength, int buffer_size, Visual_Pkg* pkg)
{
    FFTWop* dft = GetFFTWop(pkg);
    int channels = GetSDL_AudioSpec(pkg)->channels;

    FILE* wavFile = fopen(pkg->Filename, "r");
    int filesize = getFileSize(wavFile);

    Uint8* buffer = (Uint8*)malloc(buffer_size*sizeof(Uint8));

    size_t bytesRead;
    int packet_index = 0, i;
    //Skip header information in .WAV file
    bytesRead = fread(buffer, sizeof buffer[0], filesize-wavLength, wavFile);

        //Reading actual audio data
    while ((bytesRead = fread(buffer, sizeof buffer[0],
        buffer_size/sizeof(buffer[0]), wavFile)) > 0){

        pkg->setupDFT(pkg, buffer, bytesRead);
        for(i = 0; i < channels; ++i){

            fftw_execute(dft[i].p);
            analyze_FFTW_Result(pkg, dft[i], packet_index, i ,bytesRead);
            fftw_destroy_plan(dft[i].p);
        }
        packet_index++;
    }


    free(buffer);
    for(i = 0; i<channels; ++i){

        free(dft[i].in);
        free(dft[i].out);
    }
    free(dft);
    fclose(wavFile);
}
float getAmp(Uint8* wavPtr, int start, int end, int offset)
{
  // accumulate distance from zero across the frame
  float acc = 0;
  for (int samp = start; samp < end; samp++){ // e.g.  0 - 1023
  		acc += fabs(Get16bitAudioSample(wavPtr, (samp*BYTES_PER_SAMP)+offset));
  	}

  int divisor = (end-start);
  return (float) 1.0 * acc/divisor;
}
double Get16bitAudioSample(Uint8* bytebuffer, SDL_AudioFormat format)
{
  Uint16 val =  0x0;

  if(SDL_AUDIO_ISLITTLEENDIAN(format))
      val = (uint16_t)bytebuffer[0] | ((uint16_t)bytebuffer[1] << 8);

  else
      val = ((uint16_t)bytebuffer[0] << 8) | (uint16_t)bytebuffer[1];

  if(SDL_AUDIO_ISSIGNED(format))
      return ((int16_t)val)/32768.0;

  return val/65535.0;
}
void FreePtr(Visual_Pkg* pkg)
{
  SDL_AudioSpec* wavSpec = GetSDL_AudioSpec(pkg);
  for (int i = 0; i < pkg->total_packets; ++i){
      //for peak results
      free(pkg->FFTW_Result_ptr[i].peakfreq);
      free(pkg->FFTW_Result_ptr[i].peakpower);

      //for power spectrum (i.e. a double matrix) of
          //N BUCKETS that represent a frequency range

      for(int ch = 0; ch < wavSpec->channels ; ++ch){
          free(pkg->FFTW_Result_ptr[i].peakmagMatrix[ch]);
      }
      free(pkg->FFTW_Result_ptr[i].peakmagMatrix);
  }

  //deallocating space for dft operations
  for(int i=0; i<wavSpec->channels; ++i){
      free(pkg->fftw_ptr[i].in);
      free(pkg->fftw_ptr[i].out);
  }
  free(pkg->FFTW_Result_ptr);
  free(pkg->fftw_ptr);
}
