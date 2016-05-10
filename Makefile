CXX = g++
GCC = gcc
CFLAGS = -O3

test: mnist.o cnnConvolutionImp.o cnnPoolingImp.o blob.o test.o /opt/OpenBlas/lib/libopenblas.a
		$(CXX) $(CFLAGS) $^ -o $@ -I.  -L/opt/OpenBLAS/lib -lopenblas -lpthread
%.o: %.cpp
		$(CXX) $(CFLAGS) -c $< -I.
%.o: %.c
		$(GCC) $(CFLAGS) -c $< -I.
#
#cnnPool.o: cnnPool.cpp
#		$(CXX) $(CFLAGS) -c $< -I.
#cnnConvolve.o: cnnConvolve.cpp
#		$(CXX) $(CFLAGS) -c $< -I /opt/OpenBlas/include/
#test.o: test.cpp
#		$(CXX) $(CFLAGS) -c $< -I. 
#mnist.o: mnist.c
#		$(CXX) $(CFLAGS) -c $< -I.
#blob.o: blob.c
#		$(CXX) $(CFLAGS) -c $< -I.
clean:
		rm *.o test
		rm log/*

