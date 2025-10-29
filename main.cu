#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>

#include "DeviceProperties.cuh"

#define RK_STAGE 4
#define BUTCHER_SIZE RK_STAGE + 1
#define SYS_DIM 3

#define H 0.001f // [s]
#define ITERATIONS 10000

using namespace std;

__device__ void Lorenz(float*, float*, float);
__global__ void RungeKutta_Butcher(float*, float*, int);
__global__ void RungeKutta_Baseline(float*, float*, int);

void Linspace(float*, float, float, int);

__constant__ float const_d_A[RK_STAGE][RK_STAGE];
__constant__ float const_d_B[RK_STAGE];

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
	float h_A[RK_STAGE][RK_STAGE];
	//float* h_A = (float*)aligned_alloc(64,   (RK_STAGE * RK_STAGE) * sizeof(float));
	float* h_B = (float*)aligned_alloc(64,   RK_STAGE * sizeof(float));

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
	h_A[1][0] = H*0.5f; h_A[1][1] = H*0.0f; h_A[1][2] = H*0.0f;
	h_A[2][0] = H*0.0f; h_A[2][1] = H*0.5f; h_A[2][2] = H*0.0f;
	h_A[3][0] = H*0.0f; h_A[3][1] = H*0.0f; h_A[3][2] = H*1.0f;

	h_B[0] = H*1.0f/6.0f;
	h_B[1] = H*1.0f/3.0f;
	h_B[2] = H*1.0f/3.0f;
	h_B[3] = H*1.0f/6.0f;


	cudaMemcpy(d_State, h_State, 3*sizeof(float)*Resolution, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Parameters, h_Parameters, sizeof(float)*Resolution, cudaMemcpyHostToDevice);
	
	//Butcher Tableau to constant memory
	cudaMemcpyToSymbol(const_d_A, h_A, RK_STAGE * RK_STAGE * sizeof(float));
	cudaMemcpyToSymbol(const_d_B, h_B, RK_STAGE * sizeof(float));

	//Kernel run
	RungeKutta_Butcher_nounroll<<<GridSize, BlockSize>>> (d_State, d_Parameters, Resolution); // függvény nevet változtatni
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

__global__ void RungeKutta_Butcher(float* d_State, float* d_Parameters, int N){
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (tid < N)
	{
		float X[3] = {d_State[tid], d_State[tid+N], d_State[tid+2*N]};

		float P = d_Parameters[tid];
		
		// van egy k vector
		//implicitet nem lehet így kiszámolni, úgyhogy csak az explicitet számoljuk

		float k[RK_STAGE][SYS_DIM];
		float x[3]; 
		
		float T = 0;
		
		for (int n=0; n<ITERATIONS; n++) // több időlépés?; belső időmérések?
		{
			#pragma unroll
			for (int i = 0; i < RK_STAGE; i++){ //Azért 1-től kezdődik, mert a 0-dik sorát már előbb kiszámoiltuk
				
				#pragma unroll	
				for (int k_iter = 0; k_iter < SYS_DIM; k_iter++){		//gyorsabb különszedve a 2 for loop

					x[k_iter] = X[k_iter];
				}

				#pragma unroll
				for (int k_iter = 0; k_iter < SYS_DIM; k_iter++){

					#pragma unroll
					for (int j=0; j < i; j++){	//indexelést kijavítani mátrixosra akár
						x[k_iter] = x[k_iter] + k[j][k_iter] * const_d_A[i][j];
					}
				}	
				
				Lorenz(k[i], x, P);
			}
			
				
			#pragma unroll
			for (int j = 0; j < RK_STAGE; j++){
				#pragma unroll
				for (int i = 0; i < SYS_DIM; i++){
					X[i] = X[i] + const_d_B[j] * k[j][i];
				}
			}

			/*for (int i = 0; i < 3; i++){
				X[i] = X[i] + const_d_B[0] * k[0][i] + const_d_B[1] * k[1][i] + const_d_B[2] * k[2][i] + const_d_B[3] * k[3][i]; 
			}*/


			T += H; //kihagyható amúgy
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
		
		float T    = 0.0f; 
		float dT   = 0.001f;
		float dTp2 = 0.0005f;  //0.5*dT
		float dTp6 = dT * (1.0f/6.0f);
		
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
				X[j] = X[j] + dTp6*( k1[j] + 2.0f*k2[j] + 2.0f*k3[j] + k4[j] );
			
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