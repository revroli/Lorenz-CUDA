void ListCUDADevices()
{
	int NumberOfDevices;
	cudaGetDeviceCount(&NumberOfDevices);
	for (int i = 0; i < NumberOfDevices; i++)
	{
		cudaDeviceProp CurrentDeviceProperties;
		cudaGetDeviceProperties(&CurrentDeviceProperties, i);
		std::cout << std::endl;
		std::cout << "Device number: " << i << std::endl;
		std::cout << "Device name:   " << CurrentDeviceProperties.name << std::endl;
		std::cout << "--------------------------------------" << std::endl;
		std::cout << "Total global memory:        " << CurrentDeviceProperties.totalGlobalMem / 1024 / 1024 << " Mb" << std::endl;
		std::cout << "Total shared memory:        " << CurrentDeviceProperties.sharedMemPerBlock / 1024 << " Kb" << std::endl;
		std::cout << "Number of 32-bit registers: " << CurrentDeviceProperties.regsPerBlock << std::endl;
		std::cout << "Total constant memory:      " << CurrentDeviceProperties.totalConstMem / 1024 << " Kb" << std::endl;
		std::cout << std::endl;
		std::cout << "Number of multiprocessors:  " << CurrentDeviceProperties.multiProcessorCount << std::endl;
		std::cout << "Compute capability:         " << CurrentDeviceProperties.major << "." << CurrentDeviceProperties.minor << std::endl;
		std::cout << "Core clock rate:            " << CurrentDeviceProperties.clockRate / 1000 << " MHz" << std::endl;
		std::cout << "Memory clock rate:          " << CurrentDeviceProperties.memoryClockRate / 1000 << " MHz" << std::endl;
		std::cout << "Memory bus width:           " << CurrentDeviceProperties.memoryBusWidth  << " bits" << std::endl;
		std::cout << "Peak memory bandwidth:      " << 2.0*CurrentDeviceProperties.memoryClockRate*(CurrentDeviceProperties.memoryBusWidth/8)/1.0e6 << " GB/s" << std::endl;
		std::cout << std::endl;
		std::cout << "Warp size:                  " << CurrentDeviceProperties.warpSize << std::endl;
		std::cout << "Max. warps per multiproc:   " << CurrentDeviceProperties.maxThreadsPerMultiProcessor / CurrentDeviceProperties.warpSize << std::endl;
		std::cout << "Max. threads per multiproc: " << CurrentDeviceProperties.maxThreadsPerMultiProcessor << std::endl;
		std::cout << "Max. threads per block:     " << CurrentDeviceProperties.maxThreadsPerBlock << std::endl;
		std::cout << "Max. block dimensions:      " << CurrentDeviceProperties.maxThreadsDim[0] << " * " << CurrentDeviceProperties.maxThreadsDim[1] << " * " << CurrentDeviceProperties.maxThreadsDim[2] << std::endl;
		std::cout << "Max. grid dimensions:       " << CurrentDeviceProperties.maxGridSize[0] << " * " << CurrentDeviceProperties.maxGridSize[1] << " * " << CurrentDeviceProperties.maxGridSize[2] << std::endl;
		std::cout << std::endl;
		std::cout << "Concurrent memory copy:     " << CurrentDeviceProperties.deviceOverlap << std::endl;
		std::cout << "Execution multiple kernels: " << CurrentDeviceProperties.concurrentKernels << std::endl;
		std::cout << "ECC support turned on:      " << CurrentDeviceProperties.ECCEnabled << std::endl << std::endl;
	}
	std::cout << std::endl;
}

int SelectDeviceByClosestRevision(int MajorRevision, int MinorRevision)
{
	int SelectedDevice;
	
	cudaDeviceProp SelectedDeviceProperties;
	memset( &SelectedDeviceProperties, 0, sizeof(cudaDeviceProp) );
	
	SelectedDeviceProperties.major = MajorRevision;
	SelectedDeviceProperties.minor = MinorRevision;
		cudaChooseDevice( &SelectedDevice, &SelectedDeviceProperties );
	
	std::cout << "CUDA Device Number Closest to Revision " << SelectedDeviceProperties.major << "." << SelectedDeviceProperties.minor << ": " << SelectedDevice << std::endl << std::endl << std::endl;
	
	return SelectedDevice;
}

void PrintPropertiesOfSpecificDevice(int SelectedDevice)
{
	cudaDeviceProp SelectedDeviceProperties;
	cudaGetDeviceProperties(&SelectedDeviceProperties, SelectedDevice);
	
	std::cout << "Selected device number: " << SelectedDevice << std::endl;
	std::cout << "Selected device name:   " << SelectedDeviceProperties.name << std::endl;
	std::cout << "--------------------------------------" << std::endl;
	std::cout << "Total global memory:        " << SelectedDeviceProperties.totalGlobalMem / 1024 / 1024 << " Mb" << std::endl;
	std::cout << "Total shared memory:        " << SelectedDeviceProperties.sharedMemPerBlock / 1024 << " Kb" << std::endl;
	std::cout << "Number of 32-bit registers: " << SelectedDeviceProperties.regsPerBlock << std::endl;
	std::cout << "Total constant memory:      " << SelectedDeviceProperties.totalConstMem / 1024 << " Kb" << std::endl;
	std::cout << std::endl;
	std::cout << "Number of multiprocessors:  " << SelectedDeviceProperties.multiProcessorCount << std::endl;
	std::cout << "Compute capability:         " << SelectedDeviceProperties.major << "." << SelectedDeviceProperties.minor << std::endl;
	std::cout << "Core clock rate:            " << SelectedDeviceProperties.clockRate / 1000 << " MHz" << std::endl;
	std::cout << "Memory clock rate:          " << SelectedDeviceProperties.memoryClockRate / 1000 << " MHz" << std::endl;
	std::cout << "Memory bus width:           " << SelectedDeviceProperties.memoryBusWidth  << " bits" << std::endl;
	std::cout << "Peak memory bandwidth:      " << 2.0*SelectedDeviceProperties.memoryClockRate*(SelectedDeviceProperties.memoryBusWidth/8)/1.0e6 << " GB/s" << std::endl;
	std::cout << std::endl;
	std::cout << "Warp size:                  " << SelectedDeviceProperties.warpSize << std::endl;
	std::cout << "Max. warps per multiproc:   " << SelectedDeviceProperties.maxThreadsPerMultiProcessor / SelectedDeviceProperties.warpSize << std::endl;
	std::cout << "Max. threads per multiproc: " << SelectedDeviceProperties.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "Max. threads per block:     " << SelectedDeviceProperties.maxThreadsPerBlock << std::endl;
	std::cout << "Max. block dimensions:      " << SelectedDeviceProperties.maxThreadsDim[0] << " * " << SelectedDeviceProperties.maxThreadsDim[1] << " * " << SelectedDeviceProperties.maxThreadsDim[2] << std::endl;
	std::cout << "Max. grid dimensions:       " << SelectedDeviceProperties.maxGridSize[0] << " * " << SelectedDeviceProperties.maxGridSize[1] << " * " << SelectedDeviceProperties.maxGridSize[2] << std::endl;
	std::cout << std::endl;
	std::cout << "Concurrent memory copy:     " << SelectedDeviceProperties.deviceOverlap << std::endl;
	std::cout << "Execution multiple kernels: " << SelectedDeviceProperties.concurrentKernels << std::endl;
	std::cout << "ECC support turned on:      " << SelectedDeviceProperties.ECCEnabled << std::endl << std::endl;
	
	std::cout << std::endl;
}