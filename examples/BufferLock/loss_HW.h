#include <math.h>
#include <hls_stream.h>
#define FEATURE_SIZE 32
#define BATCH_SIZE 128



void loss_HW(	float X[BATCH_SIZE*FEATURE_SIZE],
        			float LABEL[BATCH_SIZE],
					float Loss[BATCH_SIZE]);
