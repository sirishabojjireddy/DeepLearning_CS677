CC = gcc
CXX = nvcc
CXXFLAGS = -O3
TARGET = dproduct

all: $(TARGET)

$(TARGET) : dproduct.cu dproductkernel.cu dproductkernel.h
	$(CXX) $(CXXFLAGS) -lm -o $(TARGET) dproduct.cu dproductkernel.cu

clean :
	rm -f $(TARGET)


