INCL_DIR = -I$(HOME)/Lorenz/DeviceProperties
CMPL_OPT = -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_86 -lineinfo #-maxrregcount=32

SOURCE   = main_rk4.cu

all: main_rk4.exe

main_rk4.exe: $(SOURCE)
	nvcc -o	main_rk4.exe $(SOURCE) $(INCL_DIR) $(CMPL_OPT)

clean:
	rm -f main_rk4.exe