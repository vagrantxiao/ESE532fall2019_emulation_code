#include "loss_HW.h"


void norm(		float X[BATCH_SIZE*FEATURE_SIZE],
         	 	float LABEL[BATCH_SIZE],
				hls::stream<float> & X_norm,
				hls::stream<float> & LABEL_norm)
{
	float X_buffer[BATCH_SIZE][FEATURE_SIZE];
	float sum = 0;
	float denominator = 0;



	for (int i=0; i<BATCH_SIZE; i++)
	{
		for (int j=0; j<FEATURE_SIZE; j++)
		{
			X_buffer[i][j] = X[i*FEATURE_SIZE+j];
		}
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
			X_norm.write(X_buffer[i][k]/denominator);
		}
	}

	for (int i=0; i<BATCH_SIZE; i++)
	{
		if(LABEL[i] > 0)
			LABEL_norm.write(1);
		else
			LABEL_norm.write(0);
	}


}

void square_loss(	hls::stream<float> & X_norm,
					hls::stream<float> & LABEL_norm_s,
					float Loss[BATCH_SIZE])
{
#include "weights.h"
	float sum = 0;
	float LABEL_norm[BATCH_SIZE];
	for (int i=0; i<BATCH_SIZE; i++)
	{
		LABEL_norm[i] = LABEL_norm_s.read();
	}

	for (int i=0; i<BATCH_SIZE; i++)
	{

		sum = 0;
		for(int j=0; j<FEATURE_SIZE; j++)
		{
			float X_in = X_norm.read();
			sum += theta[j]*X_in;
		}

		Loss[i] = (sum-LABEL_norm[i])*(sum-LABEL_norm[i])/2;
	}
}

void loss_HW(	float X[BATCH_SIZE*FEATURE_SIZE],
        			float LABEL[BATCH_SIZE],
					float Loss[BATCH_SIZE])
{
	//float X_norm[BATCH_SIZE][FEATURE_SIZE];
#pragma HLS dataflow
	hls::stream <float> LABEL_norm;
	hls::stream <float> X_norm;
//Real length of LABEL_norm is 128
#pragma HLS stream variable=LABEL_norm depth=32
//Real length of X_norm is 4096
#pragma HLS stream variable=X_norm depth=1024

	norm(X, LABEL, X_norm, LABEL_norm);
	square_loss(X_norm, LABEL_norm, Loss);

}
