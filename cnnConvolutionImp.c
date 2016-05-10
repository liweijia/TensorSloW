#include "cnnConvolutionImp.h"
#include <stdlib.h>
#include <math.h>

/****
NO TEST:
image channel > 1
conv stride != 1 
***/


void __forward_im2col(Tensor* const in, Tensor* const W, Tensor* col_data){

	int K1 = W->K1;
	int K2 = W->K2;
	int Ni = W->Ni; //=N
	int No = W->No;

	int R = in->R;
	int C = in->C;
	int N = in->N;
	int B = in->B;

	int convR = R - K2 + 1;
	int convC = C - K1 + 1;

	col_data->data = (real*)malloc(REALSIZE*K1*K2*Ni * convR*convC*B);
	col_data->size = K1*K2*Ni * convR*convC*B;

	int data_label = 0;

	for(int img=0; img < B; ++img){
		for(int ir=0; ir < convR; ++ir){
			for(int ic=0; ic < convC; ++ic){
				for(int cha=0; cha < Ni; ++cha){
					for(int nr=0; nr < K2; ++nr)
						for(int nc=0; nc < K1; ++nc){
							real* startpos = in->data + img*Ni*R*C + cha*R*C;
							col_data->data[data_label++] = *(startpos + (ir+nr)*C + ic+nc);
							//printf("%f\n", *(startpos + (ir+nr)*C + ic+nc));
						}
				}
			}
		}
	}
	printf("data size is %d\n", col_data->size);
}

void __convForward(Tensor* const col_data, Tensor* const Weight, Tensor* const b, Tensor* ConvData){

	int n = Weight->No;
	int m = ConvData->R*ConvData->C*ConvData->B;
	int k = Weight->K1*Weight->K2*Weight->Ni;

	printf("m: %d, n: %d, k: %d\n", m, n, k);

	char ta = 'T';
  	char tb = 'N';
  	real alpha = 1.0;
  	real beta = 0.0;
  	real* tmp = (real *)malloc(REALSIZE*(ConvData->size));
  	printf("before sgemm, W: %d, col_data: %d\n", Weight->size, col_data->size);
	sgemm_(&ta, &tb, &m, &n, &k, &alpha, col_data->data, &k, Weight->data, &k, &beta, tmp, &m);
	//data_col fR*fC*cha * oR*oR*numImg
	//W.data fR*fC*cha * numFilter
	printf("sgemm_ is ok\n");
	for(int img = 0; img<ConvData->B; ++img)
		for(int flt = 0; flt < ConvData->N; ++flt)
			for(int r = 0; r < ConvData->R; ++r )
				for(int c = 0; c < ConvData->C; ++c )
				{
					*(ConvData->data + c + r*ConvData->C + ConvData->C*ConvData->R*flt+ConvData->C*ConvData->R*ConvData->N*img) = 
						1/(1+exp(-*(tmp+c+r*ConvData->C+img*(ConvData->R*ConvData->C)+flt*ConvData->C*ConvData->R*ConvData->B)-b->data[flt]));
				}


	free(tmp);
	printf("cnnConvolution2\n");
}

void __convForward2(Tensor* const input, Tensor* const Weight, Tensor* const b, Tensor* ConvData)
{
	//[filterDimRow,filterDimCol,channel,numFilters] = size(W);
	int filterDimRow = Weight->K2;
	int filterDimCol = Weight->K1;
	int channel = Weight->Ni;
	int numFilters = Weight->No;

	//[imageDimRow, imageDimCol,~, numImages] = size(images);
	int imageDimRow = input->R;
	int imageDimCol = input->C;
	int numImages = input->B;

	int convDimRow = imageDimRow - filterDimRow + 1;
	int convDimCol = imageDimCol - filterDimCol + 1;

	real* convolvedImage = (real*)malloc(REALSIZE*convDimRow*convDimCol); 
	for (int imageNum = 0; imageNum < numImages; imageNum++){
	  	for (int filterNum = 0; filterNum < numFilters; filterNum++){
	      	
	      	//zeros(convDimRow, convDimCol);
	      	memset(convolvedImage, 0, REALSIZE*convDimRow*convDimCol);
	      	for (int channelNum = 0; channelNum < channel; ++channelNum){

	            float* filter = Weight->data + filterNum*(filterDimRow*filterDimCol*channel) +\
	             		channelNum*(filterDimCol*filterDimRow); 

	            float* im; 
	            im = input->data + channelNum*(imageDimRow*imageDimCol) + imageNum*(imageDimRow*imageDimRow*channel);

	            for(int i = 0; i < convDimRow; ++i){
	            	for(int j = 0; j < convDimCol; ++j){
	            		float  res = 0.0;
	            		int startpos = i*imageDimCol + j;
	            		for(int ii=0; ii<filterDimRow; ++ii){
	            			for(int jj=0; jj<filterDimCol; ++jj){
	            				res += *(im + startpos + jj + ii*imageDimCol) * (*(filter+ii*filterDimCol+jj));
	            				//printf("%f ", *(im + startpos + jj + ii*imageDimCol));
	            			}
	            			//printf("\n");
	            		}
	            		//exit(0);
	            		convolvedImage[i*convDimCol+j] +=  1/(1+exp(-res-b->data[filterNum]));
	            		//printf("%f ", 1/(1+exp(-res-b[filterNum])) );
	            		//printf("loop %d %d\n", i, j);
	            	}
	            	//printf("\n");
	            }
	  		}

	      	memcpy( ConvData->data + filterNum*(convDimRow*convDimCol) + imageNum*(numFilters*convDimRow*convDimCol)\
	      		, convolvedImage, sizeof(float)*convDimRow*convDimCol);


	  	}
	}
	free(convolvedImage);
}

