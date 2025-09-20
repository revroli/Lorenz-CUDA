#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>

#include "DeviceProperties.cuh"

using namespace std;

__device__ void Lorenz(double*, double*, double);
__global__ void RungeKutta4(double*, double*, int);

void Linspace(double*, double, double, int);

int main()
{
// INITIAL SETUP ----------------------------------------------------------------------------------
	
	int Resolution      = 15360; // Equal to the number of threads
	int BlockSize       = 128;
	
	ListCUDADevices();
	
	int MajorRevision  = 11;
	int MinorRevision  = 6;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	cudaSetDevice(SelectedDevice);
	
	double* h_State      = (double*)aligned_alloc(64, 3*Resolution * sizeof(double));
	double* h_Parameters = (double*)aligned_alloc(64,   Resolution * sizeof(double));
	double* d_State;
	double* d_Parameters;
	cudaMalloc((void**)&d_State,      3*Resolution * sizeof(double));
	cudaMalloc((void**)&d_Parameters,   Resolution * sizeof(double));
	
	// Initialisation
	Linspace(h_Parameters, 0.0, 40.0, Resolution);
	for (int i = 1; i<Resolution; i++)
	{
		h_State[i]              = -5.0;
		h_State[i+Resolution]   =  0.0;
		h_State[i+2*Resolution] =  0.0;
	}	
	
	cudaMemcpy(d_State, h_State, 3*sizeof(double)*Resolution, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Parameters, h_Parameters, sizeof(double)*Resolution, cudaMemcpyHostToDevice);
	
	// Integration
	int GridSize = Resolution/BlockSize + (Resolution % BlockSize == 0 ? 0:1);
	
	clock_t SimulationStart = clock();
	
	RungeKutta4<<<GridSize, BlockSize>>> (d_State, d_Parameters, Resolution);
	cudaDeviceSynchronize();
	
	clock_t SimulationEnd = clock();
	cout << "Simulation time: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
	
	cudaMemcpy(h_State, d_State, 3*sizeof(double)*Resolution, cudaMemcpyDeviceToHost);
	
	
	//for (int i=0; i<NumberOfProblems; i++)
	//	cout << "P: " << h_Parameters[i] << "   Sates: " << h_State[i] << ", " << h_State[i+NumberOfProblems] << ", " << h_State[i+2*NumberOfProblems] << endl;
}

__forceinline__ __device__ void Lorenz(double* F, double* X, double P)
{
	F[0] = 10.0*(X[1] - X[0]);
	F[1] = P*X[0] - X[1] - X[0]*X[2];
	F[2] = X[0]*X[1] - 2.666 * X[2];
}

__global__ void RungeKutta4(double* d_State, double* d_Parameters, int N)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (tid < N)
	{
		double X[3];
		double P;
		
		double k1[3];
		double k2[3];
		double k3[3];
		double k4[3];
		double x[3];
		
		double T    = 0.0; 
		double dT   = 1e-3;
		double dTp2 = 0.5*dT;
		double dTp6 = dT * (1.0/6.0);
		
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
}

void Linspace(double* x, double B, double E, int N)
{
    double Increment;
	
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