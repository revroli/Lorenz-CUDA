INCL_DIR = -I$(CURDIR)/DeviceProperties
CMPL_OPT = -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_86 -lineinfo #-maxrregcount=32

SRC ?= main.cu
TARGET := $(basename $(SRC)).exe

all: $(TARGET)

$(TARGET): $(SOURCE)
	nvcc -o	$(TARGET) $(SRC) $(INCL_DIR) $(CMPL_OPT)

clean:
	rm -f $(TARGET)