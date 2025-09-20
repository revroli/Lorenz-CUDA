#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>

#include "DeviceProperties.cuh"

using namespace std;

__device__ void Lorenz(float*, float*, float);
__global__ void RungeKutta4(float*, float*, int);

void Linspace(float*, float, float, int);

int main()
{
	
	int Resolution      = 1536*64*100; // Threads per SM * number of SMs * 3
	int BlockSize; //      = 128;
	
	ListCUDADevices();
	
	int MajorRevision  = 8;
	int MinorRevision  = 6;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	cudaSetDevice(SelectedDevice);
	
	float* h_State      = (float*)aligned_alloc(64, 3*Resolution * sizeof(float));
	float* h_Parameters = (float*)aligned_alloc(64,   Resolution * sizeof(float));
	

	float* d_State;
	float* d_Parameters;
	cudaMalloc((void**)&d_State,      3*Resolution * sizeof(float));
	cudaMalloc((void**)&d_Parameters,   Resolution * sizeof(float));


	// Initialisation
	Linspace(h_Parameters, 0.0, 40.0, Resolution);
	for (int i = 1; i<Resolution; i++)
	{
		h_State[i]              = -5.0;
		h_State[i+Resolution]   =  0.0;
		h_State[i+2*Resolution] =  0.0;
	}	

	cudaMemcpy(d_State, h_State, 3*sizeof(float)*Resolution, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Parameters, h_Parameters, sizeof(float)*Resolution, cudaMemcpyHostToDevice);
	
	// Integration
	int GridSize;// = Resolution/BlockSize + (Resolution % BlockSize == 0 ? 0:1);
	
	//clock_t SimulationStart = clock();
	
/*     BlockSize = 64;
    GridSize = Resolution/BlockSize + (Resolution % BlockSize == 0 ? 0:1);
	
	RungeKutta4<<<GridSize, BlockSize>>> (d_State, d_Parameters, Resolution);
	cudaDeviceSynchronize(); */
    
    BlockSize = 128;
    GridSize = Resolution/BlockSize + (Resolution % BlockSize == 0 ? 0:1);
	
	RungeKutta4<<<GridSize, BlockSize>>> (d_State, d_Parameters, Resolution);
	cudaDeviceSynchronize();
	
/*     BlockSize = 192;
    GridSize = Resolution/BlockSize + (Resolution % BlockSize == 0 ? 0:1);
	
    RungeKutta4<<<GridSize, BlockSize>>> (d_State, d_Parameters, Resolution);
	cudaDeviceSynchronize();
	
    BlockSize = 256;
    GridSize = Resolution/BlockSize + (Resolution % BlockSize == 0 ? 0:1);
	
    RungeKutta4<<<GridSize, BlockSize>>> (d_State, d_Parameters, Resolution);
	cudaDeviceSynchronize(); */

	
	//clock_t SimulationEnd = clock();
	//cout << "Simulation time: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
	
	cudaMemcpy(h_State, d_State, 3*sizeof(float)*Resolution, cudaMemcpyDeviceToHost);
	
	ofstream outfile("output_origi.txt");
	outfile << std::setprecision(8) << std::fixed;
	for (int i = 0; i < Resolution; ++i) {
		outfile << h_Parameters[i] << " "
				<< h_State[i] << " "
				<< h_State[i + Resolution] << " "
				<< h_State[i + 2 * Resolution] << "\n";
	}
	outfile.close();

	//for (int i=0; i<NumberOfProblems; i++)
	//	cout << "P: " << h_Parameters[i] << "   Sates: " << h_State[i] << ", " << h_State[i+NumberOfProblems] << ", " << h_State[i+2*NumberOfProblems] << endl;
}

__forceinline__ __device__ void Lorenz(float* F, float* X, float P)
{
	F[0] = float(10)*(X[1] - X[0]);
	F[1] = P*X[0] - X[1] - X[0]*X[2];
	F[2] = X[0]*X[1] - float(2.666) * X[2];
}

/*__global__ void RungeKutta4Long(float* d_State, float* d_Parameters, int N)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (tid < N)
	{
		float X[3];
		float P;
		
		float k1[3];
		float k2[3];
		float k3[3];
		float k4[3];
		float x[3];
		
		float T    = 0; 
		float dT   = 1e-3;
		float dTp2 = 0.5*dT;
		float dTp6 = dT * (float(1)/float(6));
		
		X[0] = d_State[tid];
		X[1] = d_State[tid + N];
		X[2] = d_State[tid + 2*N];
		
		P = d_Parameters[tid];
		
		for (int i=0; i<10000; i++)
		{
			Lorenz(k1, X, P);
			
			#pragma unroll
			for (int j=0; j<3; j++)
				x[j] = X[j] + dTp2*k1[j];
			
			Lorenz(k2, x, P);
			
			#pragma unroll
			for (int j=0; j<3; j++)
				x[j] = X[j] + dTp2*k2[j];
			
			Lorenz(k3, x, P);
			
			#pragma unroll
			for (int j=0; j<3; j++)
				x[j] = X[j] + dT*k3[j];
			
			Lorenz(k4, x, P);
			
			// Update state
			#pragma unroll
			for (int j=0; j<3; j++)
				X[j] = X[j] + dTp6*( k1[j] + 2*k2[j] + 2*k3[j] + k4[j] );
			
			T += dT;
		}
		
		d_State[tid] = X[0];
		d_State[tid + N] = X[1];
		d_State[tid + 2*N] = X[2];
	}
}*/


__global__ void RungeKutta4(float* d_State, float* d_Parameters, int N)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (tid < N)
	{
		float X[3] = {d_State[tid],d_State[tid+N],d_State[tid+2*N]};

		//X[0] = d_State[tid];
		//X[1] = d_State[tid + N];
		//X[2] = d_State[tid + 2*N];

		float P = d_Parameters[tid];
		
		// van egy k vector
		//implicitet nem lehet így kiszámolni, úgyhogy csak az explicitet számoljuk


		float k1[3];
		float k2[3];
		float k3[3];
		float k4[3];
		float x[3];
		
		float T    = 0; 
		float dT   = 0.001;
		float dTp2 = 0.0005;  //0.5*dT
		float dTp6 = dT * (float(1)/6);
		
		for (int i=0; i<10000; i++)
		{
			Lorenz(k1, X, P);
			
			#pragma unroll
			for (int j=0; j<3; j++)
				x[j] = X[j] + dTp2*k1[j];
			
			Lorenz(k2, x, P);
			
			#pragma unroll
			for (int j=0; j<3; j++)
				x[j] = X[j] + dTp2*k2[j];
			
			Lorenz(k3, x, P);
			
			#pragma unroll
			for (int j=0; j<3; j++)
				x[j] = X[j] + dT*k3[j];
			
			Lorenz(k4, x, P);
			
			// Update state
			#pragma unroll
			for (int j=0; j<3; j++)
				X[j] = X[j] + dTp6*( k1[j] + 2*k2[j] + 2*k3[j] + k4[j] );
			
			T += dT;
		}
		
		d_State[tid] = X[0];
		d_State[tid + N] = X[1];
		d_State[tid + 2*N] = X[2];
	}
}

void Linspace(float* x, float B, float E, int N)
{
    float Increment;
	
	x[0] = B;
	
	if ( N>1 )
	{
		x[N-1] = E;
		Increment = (E-B)/(N-1);
		
		for (int i=1; i<N-1; i++)
		{
			x[i] = B + i*Increment;
		}
	}
}