

#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include "weights.h"
#ifdef __SDSCC__
#include "sds_lib.h"
#endif


#include "loss_HW.h"

#ifdef __SDSCC__
class perf_counter
{
public:
  uint64_t tot, cnt, calls;
  perf_counter() : tot(0), cnt(0), calls(0) {};
  inline void reset() { tot = cnt = calls = 0; }
  inline void start() { cnt = sds_clock_counter(); calls++; };
  inline void stop() { tot += (sds_clock_counter() - cnt); };
  inline uint64_t avg_cpu_cycles() { return (tot / calls); };
};
#endif

static void init_arrays(float X[BATCH_SIZE*FEATURE_SIZE],
                        float LABEL[BATCH_SIZE])
{
	for (int i = 0; i < BATCH_SIZE;i++)
    {
    	for (int j = 0; j < FEATURE_SIZE; j++)
    	{
    		X[i*FEATURE_SIZE+j] = rand() % (FEATURE_SIZE);
    	}
    	LABEL[i] = rand() % (BATCH_SIZE);
     }
}

void loss_golden(	float X[BATCH_SIZE*FEATURE_SIZE],
        			float LABEL[BATCH_SIZE],
					float Loss[BATCH_SIZE])
{
	float X_buffer[BATCH_SIZE][FEATURE_SIZE];
	float LABEL_norm[BATCH_SIZE];
	float X_norm[BATCH_SIZE][FEATURE_SIZE];
	float sum = 0;
	float denominator = 0;

	for (int i=0; i<BATCH_SIZE; i++)
	{
		for (int j=0; j<FEATURE_SIZE; j++)
		{
			X_buffer[i][j] = X[i*FEATURE_SIZE+j];
		}
		if(LABEL[i] > 0)
			LABEL_norm[i] = 1;
		else
			LABEL_norm[i] = 0;
	}


	for (int i=0; i<BATCH_SIZE; i++)
	{
		sum = 0;
		for (int j=0; j<FEATURE_SIZE; j++)
		{
			sum += X_buffer[i][j]*X_buffer[i][j];
		}
		denominator = sqrt(sum);
		for (int k=0; k<FEATURE_SIZE; k++)
		{
			X_norm[i][k] = X_buffer[i][k]/denominator;
		}
	}

	for (int i=0; i<BATCH_SIZE; i++)
	{
		sum = 0;
		for(int j=0; j<FEATURE_SIZE; j++)
		{
			sum += theta[j]*X_norm[i][j];
		}
		Loss[i] = (sum-LABEL_norm[i])*(sum-LABEL_norm[i])/2;

	}
}

static int result_check(float Loss_SW[BATCH_SIZE], float Loss_HW[BATCH_SIZE])
{
     for (int i = 0; i < BATCH_SIZE; i++)
     {
		if ((Loss_SW[i]-Loss_HW[i])/Loss_HW[i]>0.01 || (Loss_SW[i]-Loss_HW[i])/Loss_HW[i]<-0.01)
		{
			std::cout << "Mismeatch: data index=" << i << " d=" << Loss_SW[i]
					  << ", dout=" << Loss_HW[i] << std::endl;
			return 1;
		}
     }
     return 0;
}


int main()
{

#ifdef __SDSCC__
	float *X, *LABEL,  *Loss_SW, *Loss_HW;
     X = (float *)sds_alloc(BATCH_SIZE * FEATURE_SIZE * sizeof(float));
     LABEL = (float *)sds_alloc(BATCH_SIZE * sizeof(float));
     Loss_SW = (float *)sds_alloc(BATCH_SIZE * sizeof(float));
     Loss_HW = (float *)sds_alloc(BATCH_SIZE * sizeof(float));
     if (!X || !LABEL || !Loss_SW || !Loss_HW) {
    	 if (X) sds_free(X);
    	 if (LABEL) sds_free(LABEL);
    	 if (Loss_SW) sds_free(Loss_SW);
    	 if (Loss_HW) sds_free(Loss_HW);
         return 2;
     }
#else
     float X[BATCH_SIZE * FEATURE_SIZE];
     float LABEL[BATCH_SIZE];
     float Loss_SW[BATCH_SIZE];
     float Loss_HW[BATCH_SIZE];
#endif



     init_arrays(X, LABEL);
     loss_golden(X, LABEL,  Loss_SW);
     printf("\n\n\n\n\n\n\n");
     loss_HW(X, LABEL, Loss_HW);
     int test_failed = result_check(Loss_SW, Loss_HW);

     std::cout << "TEST " << (test_failed ? "FAILED" : "PASSED") << std::endl;

#ifdef __SDSCC__
     sds_free(X);
     sds_free(LABEL);
     sds_free(Loss_SW);
     sds_free(Loss_HW);
#endif
     return (test_failed ? -1 : 0);
}

