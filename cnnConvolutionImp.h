#ifndef _CNNCONVOLUTIONIMP_H_
#define _CNNCONVOLUTIONIMP_H_

#include "blob.h"

void __forward_im2col(Tensor* const in, Tensor* const W, Tensor* col_data);
void __convForward(Tensor* const col_data, Tensor* const Weight, Tensor* const b, Tensor* ConvData);
void __convForward2(Tensor* const col_data, Tensor* const Weight, Tensor* const b, Tensor* ConvData);

#endif
