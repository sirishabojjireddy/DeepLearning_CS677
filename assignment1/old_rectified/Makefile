CXX2 = gcc
CXX = nvcc
CXXFLAGS = -O3
TARGET = dotproduct

all: $(TARGET)

#$(TARGET) : dprod.cu dproductkernel.cu dproductkernel.h
	#$(CXX) $(CXXFLAGS) -lm -o $(TARGET) dprod.cu dproductkernel.cu
$(TARGET) : dotproduct.cu dotproductkernel.cu dotproductkernel.h
	$(CXX) $(CXXFLAGS) -lm -o $(TARGET) dotproduct.cu dotproductkernel.cu

clean :
	rm -f $(TARGET)


