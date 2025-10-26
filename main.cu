#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>

#include "DeviceProperties.cuh"

#define RK_ORDER 4
#define BUTCHER_SIZE RK_ORDER + 1

#define H 0.001f // [s]
#define ITERATIONS 10000

using namespace std;

__device__ void Lorenz(float*, float*, float);
__global__ void RungeKutta_Butcher_nounroll(float*, float*, int);
__global__ void RungeKutta_Butcher_unrolled(float*, float*, int);
__global__ void RungeKutta_Butcher_half_unrolled(float*, float*, int);
__global__ void RungeKutta_Baseline(float*, float*, int);
__global__ void RungeKutta_Baseline_with_zeros(float*, float*, int);

void Linspace(float*, float, float, int);

__constant__ float const_d_A[(RK_ORDER-1)*(RK_ORDER-1)];
__constant__ float const_d_B[RK_ORDER];

int main()
{

	int Resolution = 1536*64*100; // Threads per SM * number of SMs * 3
	int BlockSize  = 128;
	int GridSize = Resolution/BlockSize + (Resolution % BlockSize == 0 ? 0:1);
	

	//Choose GPU
	ListCUDADevices();
	
	int MajorRevision  = 8;
	int MinorRevision  = 6;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	cudaSetDevice(SelectedDevice);
	
	//Allocate CPU memory
	float* h_State      = (float*)aligned_alloc(64, 3*Resolution * sizeof(float));
	float* h_Parameters = (float*)aligned_alloc(64,   Resolution * sizeof(float));
	float* h_A = (float*)aligned_alloc(64,   (RK_ORDER - 1) * (RK_ORDER - 1) * sizeof(float));
	float* h_B = (float*)aligned_alloc(64,   RK_ORDER * sizeof(float));

	//Allocate GPU memory (constant memory doesn't need allocation)
	float* d_State;
	float* d_Parameters;
	cudaMalloc((void**)&d_State,      3*Resolution * sizeof(float));
	cudaMalloc((void**)&d_Parameters,   Resolution * sizeof(float));

	// Initialization
	Linspace(h_Parameters, 0.0, 40.0, Resolution);
	for (int i = 1; i<Resolution; i++)
	{
		h_State[i]              = -5.0;
		h_State[i+Resolution]   =  0.0;
		h_State[i+2*Resolution] =  0.0;
	}	

	//RK4 method parameters. Can be changed
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
	RungeKutta_Baseline_with_zeros<<<GridSize, BlockSize>>> (d_State, d_Parameters, Resolution); // függvény nevet változtatni
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
	F[0] = float(10)*(X[1] - X[0]);			//FMUL, FADD
	F[1] = P*X[0] - X[1] - X[0]*X[2];		//FMA, FMA
	F[2] = X[0]*X[1] - float(2.666) * X[2];	//FMUL, FMA
}

__global__ void RungeKutta_Butcher_nounroll(float* d_State, float* d_Parameters, int N){
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
		//float h = 0.001; //DT

		int i_minus;
		int i = 0;
		
		for (int n=0; n<ITERATIONS; n++) // több időlépés?; belső időmérések?
		{
			Lorenz(k, X, P);		//kn1

			// másik kernelbe átírni

			#pragma unroll
			for (int i = 1; i < RK_ORDER; i++){
				
				i_minus = i-1;

				#pragma unroll
				for (int k_iter = 0; k_iter < 3; k_iter++){

					intersum = 0;

					#pragma unroll  //enélkül van-e különbség. Fordítási időben ismert indexek -> registerbe bent maradjon -> látni a registerhasználat változását
									//belső chrono időmérés
									//fordítási opciók
									//kiírt dolgokat textfájlba kimenteni
					for (int j=0; j < i; j++){
						intersum += k[j*3 + k_iter] * const_d_A[(i_minus) * 3 + j];	//a a 00-ból kell induljon 
																//ezt átírni valahogy 1 MA-ra?
																// unrollal biztos kijön
																//(i-1)*-at elég lehet csak 1-szer kiszámolni
					}
					
					x[k_iter] = X[k_iter] + H  * intersum;
				}	
				
				Lorenz(k + 3*i, x, P);
			}
			

			#pragma unroll
			for (i = 0; i < 3; i++){
				X[i] += H * (const_d_B[0] * k[i] + const_d_B[1] * k[3 + i] + const_d_B[2] * k[6 + i] + const_d_B[3] * k[9 + i]); 
			}


			T += H; //kihagyható amúgy
		}
		
		d_State[tid] = X[0];
		d_State[tid + N] = X[1];
		d_State[tid + 2*N] = X[2];
	}
}

__global__ void RungeKutta_Butcher_unrolled(float* d_State, float* d_Parameters, int N){
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (tid < N)
	{
		float X[3] = {d_State[tid], d_State[tid+N], d_State[tid+2*N]};

		float P = d_Parameters[tid];
		
		// van egy k vector
		//implicitet nem lehet így kiszámolni, úgyhogy csak az explicitet számoljuk

		float k[RK_ORDER * 3];		//hogyan rendezem? legyen [iteráció][x-dimenzió]
		float x[3];
		//float intersum;
		
		float T = 0;
		//float h = 0.001; //DT

		//int i_minus;
		int i = 0;
		
		for (int n=0; n<ITERATIONS; n++) // több időlépés?; belső időmérések?
		{
			Lorenz(k, X, P);		//kn1
			
			#pragma unroll
			for (i = 0; i<3; i++)
			{
				x[i] = X[i] + H * (k[i] * const_d_A[0]);
			}

			Lorenz(k + 3, x, P);
			
			#pragma unroll
			for (i = 0; i<3; i++)
			{
				x[i] = X[i] + H * (k[0] * const_d_A[0] + k[3+i] * const_d_A[4]);
			}

			Lorenz(k + 6, x, P);
			
			#pragma unroll
			for (i = 0; i<3; i++)
			{
				x[i] = X[i] + H * (k[i] * const_d_A[0] + k[3+i] * const_d_A[4] + k[6 + i] * const_d_A[8]);
			}

			Lorenz(k + 9, x, P);

			//unroll ended

			#pragma unroll
			for (i = 0; i < 3; i++){

				X[i] += H * (const_d_B[0] * k[i] + const_d_B[1] * k[3 + i] + const_d_B[2] * k[6 + i] + const_d_B[3] * k[9 + i]); 
			}

 			X[0] += H * (const_d_B[0] * k[0] + const_d_B[1] * k[3] + const_d_B[2] * k[6] + const_d_B[3] * k[9]); 
			X[1] += H * (const_d_B[0] * k[1] + const_d_B[1] * k[4] + const_d_B[2] * k[7] + const_d_B[3] * k[10]); 
			X[2] += H * (const_d_B[0] * k[2] + const_d_B[1] * k[5] + const_d_B[2] * k[8] + const_d_B[3] * k[11]);

			T += H; //kihagyható amúgy
		}
		
		d_State[tid] = X[0];
		d_State[tid + N] = X[1];
		d_State[tid + 2*N] = X[2];
	}
}

__global__ void RungeKutta_Butcher_half_unrolled(float* d_State, float* d_Parameters, int N){
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (tid < N)
	{
		float X[3] = {d_State[tid], d_State[tid+N], d_State[tid+2*N]};

		float P = d_Parameters[tid];
		
		// van egy k vector
		//implicitet nem lehet így kiszámolni, úgyhogy csak az explicitet számoljuk

		float k[RK_ORDER * 3];		//hogyan rendezem? legyen [iteráció][x-dimenzió]
		float x[3];
		//float intersum;
		
		float T = 0;
		//float h = 0.001; //DT

		//int i_minus;
		int i = 0;
		
		for (int n=0; n<ITERATIONS; n++) // több időlépés?; belső időmérések?
		{
			Lorenz(k, X, P);		//kn1
			
			#pragma unroll
			for (i = 0; i<3; i++)
			{
				x[i] = X[i] + H * (k[i] * const_d_A[0]);	//FMA, FMUL 
			}

			Lorenz(k + 3, x, P);
			
			#pragma unroll
			for (i = 0; i<3; i++)
			{
				x[i] = X[i] + H * (k[0] * const_d_A[0] + k[3+i] * const_d_A[4]);	//FMA, FMUL, FADD, FMA
			}

			Lorenz(k + 6, x, P);
			
			#pragma unroll
			for (i = 0; i<3; i++)
			{
				x[i] = X[i] + H * (k[i] * const_d_A[0] + k[3+i] * const_d_A[4] + k[6 + i] * const_d_A[8]);	//FMA, FMUL, FADD, FMA, FADD, FMA
			}

			Lorenz(k + 9, x, P);

			#pragma unroll
			for (i = 0; i < 3; i++){
				X[i] += H * (const_d_B[0] * k[i] + const_d_B[1] * k[3 + i] + const_d_B[2] * k[6 + i] + const_d_B[3] * k[9 + i]); //FMA, FMUL, FADD, FMA, FADD, FMA, FADD, FMA
			}


			T += H; //kihagyható amúgy	//FADD
		}
		
		d_State[tid] = X[0];
		d_State[tid + N] = X[1];
		d_State[tid + 2*N] = X[2];
	}
}

__global__ void RungeKutta_Baseline(float* d_State, float* d_Parameters, int N){
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (tid < N)
	{
		float X[3] = {d_State[tid], d_State[tid+N], d_State[tid+2*N]};

		float P = d_Parameters[tid];
		
		float k1[3];
		float k2[3];
		float k3[3];
		float k4[3];
		float x[3];
		
		float T    = 0; 
		float dT   = 0.001;
		float dTp2 = 0.0005;  //0.5*dT
		float dTp6 = dT * (float(1)/6);
		
		for (int i=0; i<ITERATIONS; i++)
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

__global__ void RungeKutta_Baseline_with_zeros(float* d_State, float* d_Parameters, int N){
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (tid < N)
	{
		float X[3] = {d_State[tid], d_State[tid+N], d_State[tid+2*N]};

		float P = d_Parameters[tid];
		
		float k1[3];
		float k2[3];
		float k3[3];
		float k4[3];
		float x[3];
		
		float T    = 0; 
		//float h   = 0.001;
		//float dTp2 = 0.0005;  //0.5*dT
		//float dTp6 = dT * (float(1)/6);
		
		for (int i=0; i<ITERATIONS; i++)
		{
			Lorenz(k1, X, P);
			
			#pragma unroll
			for (int j=0; j<3; j++)
				x[j] = X[j] + float(0.5)*H*k1[j]; //dTp2 = a21*h
			
			Lorenz(k2, x, P);
			
			#pragma unroll
			for (int j=0; j<3; j++)
				x[j] = X[j] + H*(float(0)*k1[j] + float(0.5)*k2[j]); //dTp2 = a32*h; a31 = 0
			
			Lorenz(k3, x, P);
			
			#pragma unroll
			for (int j=0; j<3; j++)
				x[j] = X[j] + H*(float(0)*k1[j] + float(0)*k2[j] + float(1)*k3[j]); //dT = a43 * h; a41, a42 = 0
			
			Lorenz(k4, x, P);
			
			// Update state
			#pragma unroll
			for (int j=0; j<3; j++)
				X[j] = X[j] + H*(float(0.16161616)*k1[j] + float(0.33333333)*k2[j] + float(0.33333333)*k3[j] + float(0.16161616)*k4[j]);
			
			T += H;
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