#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>

#include "DeviceProperties.cuh"

#define RK_ORDER 4
#define BUTCHER_SIZE RK_ORDER + 1

using namespace std;

__device__ void Lorenz(float*, float*, float);
__global__ void RungeKutta4(float*, float*, int);

void Linspace(float*, float, float, int);

__constant__ float const_d_A[(RK_ORDER-1)*(RK_ORDER-1)];
__constant__ float const_d_B[RK_ORDER];

int main()
{

	int Resolution = 1536*64*100; // Threads per SM * number of SMs * 3
	int BlockSize  = 128;
	int GridSize = Resolution/BlockSize + (Resolution % BlockSize == 0 ? 0:1);
	
	ListCUDADevices();
	
	int MajorRevision  = 8;
	int MinorRevision  = 6;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	cudaSetDevice(SelectedDevice);
	
	float* h_State      = (float*)aligned_alloc(64, 3*Resolution * sizeof(float));
	float* h_Parameters = (float*)aligned_alloc(64,   Resolution * sizeof(float));
	float* h_A = (float*)aligned_alloc(64,   (RK_ORDER - 1) * (RK_ORDER - 1) * sizeof(float));
	float* h_B = (float*)aligned_alloc(64,   RK_ORDER * sizeof(float));

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

	h_A[0] = 0.5f; h_A[1] = 0.0f; h_A[2] = 0.0f;
	h_A[3] = 0.0f; h_A[4] = 0.5f; h_A[5] = 0.0f;
	h_A[6] = 0.0f; h_A[7] = 0.0f; h_A[8] = 1.0f;

	h_B[0] = 1.0f/6.0f;
	h_B[1] = 1.0f/3.0f;
	h_B[2] = 1.0f/3.0f;
	h_B[3] = 1.0f/6.0f;


	cudaMemcpy(d_State, h_State, 3*sizeof(float)*Resolution, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Parameters, h_Parameters, sizeof(float)*Resolution, cudaMemcpyHostToDevice);
	
	//Butcher Tableau to constant memory
	cudaMemcpyToSymbol(const_d_A, h_A, (RK_ORDER - 1) * (RK_ORDER - 1) * sizeof(float));
	cudaMemcpyToSymbol(const_d_B, h_B, sizeof(float) * RK_ORDER);

	//Kernel run
	RungeKutta4<<<GridSize, BlockSize>>> (d_State, d_Parameters, Resolution);
	cudaDeviceSynchronize();
	
	//Save the products
	cudaMemcpy(h_State, d_State, 3*sizeof(float)*Resolution, cudaMemcpyDeviceToHost);
	

	ofstream outfile("output_files/main_output.txt");
	outfile << std::setprecision(8) << std::fixed;
	outfile << "# Lorenz System CUDA Simulation Output\n";
	outfile << "# Columns: Parameter X Y Z\n";
	outfile << "# Resolution: " << Resolution << "\n";
	outfile << "# Each row: <parameter> <X> <Y> <Z>\n";
	for (int i = 0; i < Resolution; ++i) {
		outfile << h_Parameters[i] << " "
				<< h_State[i] << " "
				<< h_State[i + Resolution] << " "
				<< h_State[i + 2 * Resolution] << "\n";
	}
	outfile.close();
}

__forceinline__ __device__ void Lorenz(float* F, float* X, float P)
{
	F[0] = float(10)*(X[1] - X[0]);
	F[1] = P*X[0] - X[1] - X[0]*X[2];
	F[2] = X[0]*X[1] - float(2.666) * X[2];
}

__global__ void RungeKutta4(float* d_State, float* d_Parameters, int N)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (tid < N)
	{
		float X[3] = {d_State[tid], d_State[tid+N], d_State[tid+2*N]};

		float P = d_Parameters[tid];
		
		// van egy k vector
		//implicitet nem lehet így kiszámolni, úgyhogy csak az explicitet számoljuk

		float k[RK_ORDER * 3];		//hogyan rendezem? legyen [iteráció][x-dimenzió]
		float x[3];
		float intersum;
		
		float T = 0;
		float h = 0.001; //DT

		int i_minus;
		
		for (int n=0; n<10000; n++)
		{
			Lorenz(k, X, P);		//kn1

			#pragma unroll
			for (int i = 1; i < RK_ORDER; i++){
				
				i_minus = i-1;

				#pragma unroll
				for (int k_iter = 0; k_iter < 3; k_iter++){

					intersum = 0;

					#pragma unroll
					for (int j=0; j < i; j++){
						intersum += k[j*3 + k_iter] * const_d_A[(i_minus) * 3 + j];	//a a 00-ból kell induljon 
																//ezt átírni valahogy 1 MA-ra?
																// unrollal biztos kijön
																//(i-1)*-at elég lehet csak 1-szer kiszámolni
					}
					
					x[k_iter] = X[k_iter] + h  * intersum;
				}	
				
				Lorenz(k + 3*i, x, P);
			}

			#pragma unroll
			for (int i = 0; i < 3; i++){
				intersum = 0;

				#pragma unroll
				for (int j = 0; j < RK_ORDER; j++){
					intersum += const_d_B[j] * k[3*j + i]; 
				}
				
				X[i] = X[i] + h * intersum;
			}

			T += h; //kihagyható amúgy
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