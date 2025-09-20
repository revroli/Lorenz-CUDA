#define X10 0
#define X20 0
#define X30 0
#define X1END 4
#define X2END 4
#define X3END 4

#define SPDIM 3
#define RESDIM 1

#define SIGMA 10.0
#define BETA 2.666


__global__ void Lorenz(float* x1, float* x2, float* x3, int N, int I, float* sigma, float* rho, float* beta)
__global__ void RungeKutta4(float* d_State, float* rho, int N)

int main(){

    int blocksize = 64 //minimum 64 kéne
    int Resolution = 15*2048*3;
    int StartingPoints = 
    int Transients = 128;

    float* h_x1 = new float[StartingPoints];
    float* h_x2 = new float[StartingPoints];
    float* h_x3 = new float[StartingPoints];
    float* h_sigma = new float[Resolution];
    float* h_rho = new float;
    float* h_beta = new float;


    h_x1[0] = X10;
    h_x2[0] = X20;
    h_x3[0] = X30;

	float dr = X1END/(StartingPoints-1.0); // Only 1 type conversion!
    float dr1, dr2, dr3;
    dr1 = dr2 = dr3 = dr;
    int initidx = 0;

    for (int i=1; i<StartingPoints^3; i++){
        if ((i%(StartingPoints*StartingPoints))==0){
            h_x1[i] = h_x1[i-1]+dr1;
            h_x2[i] = X20;
            h_x3[i] = X30;
        }
        else if (i%(StartingPoints)==0){
            h_x1[i] = h_x1[i-1];
            h_x2[i] = h_x2[i-1]+dr2;
            h_x3[i] = X30;
        }
        else{
            h_x1[i] = h_x1[i-1];
            h_x2[i] = h_x2[i-1];
            h_x3[i] = h_x3[i-1]+dr3;
        }
    }

    int GridSize = StartingPoints^3/BlockSize + (StartingPoints^3 % BlockSize == 0 ? 0:1);

    // GPU selection
	ListCUDADevices();
	
	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	cudaSetDevice(SelectedDevice);

    float* d_x1;
    float* d_x2;
    float* d_x3;
    float* d_sigma;
    float* d_rho;
    float* d_beta;
    cudaMalloc((void**)&d_x1, StartingPoints*sizeof(float));
    cudaMalloc((void**)&d_x2, StartingPoints*sizeof(float));
    cudaMalloc((void**)&d_x3, StartingPoints*sizeof(float));
    cudaMalloc((void**)&d_sigma, Resolution*sizeof(float));
    cudaMalloc((void**)&d_rho, sizeof(float));
    cudaMalloc((void**)&d_beta, sizeof(float));

    cudaMemcpy(d_x1, h_x1, StartingPoints * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, h_x2, StartingPoints * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x3, h_x3, StartingPoints * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma, h_sigma, Resolution * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, h_rho, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, sizeof(float), cudaMemcpyHostToDevice);

    int Calculations = StartingPoints^SPDIM*Resolution^RESDIM;
    int GridSize = Calculations/BlockSize + (Calculations%BlockSize == 0 ? 0:1);

    //cout << "Number of Blocks per Grid: " << GridSize << ", Number of Threads per Block: " << BlockSize << endl << endl;
    
    //clock_t SimulationStart = clock();

    Lorenz<<<GridSize, BlockSize>>>();
	
    cudaDeviceSynchronize();

    cudaMemcpy(h_x1, d_x1, StartingPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x2, d_x2, StartingPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x3, d_x3, StartingPoints * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_x3);
    cudaFree(d_sigma);
    cudaFree(d_rho);
    cudaFree(d_beta);

    delete[] h_x1;
    delete[] h_x2;
    delete[] h_x3;
    delete[] h_sigma;
    delete h_rho;
    delete h_beta;
}

__global__ void Lorenz(float* F, float* x, float* rho){
    F[0] = SIGMA*x[1]-SIGMA*x[0];
    F[1] = rho*x[0]-x[1]-x[0]*x[2];
    F[2] = x[0]*x[1]-BETA*x[2];
}

__global__ void RungeKutta4(float* d_State, float* rho, int N){

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid<N)
    {

        float X[3];
		float P;
		
		float k1[3];
		float k2[3];
		float k3[3];
		float k4[3];
		float x[3];
		
		float T    = 0.0; 
		float dT   = 1e-3;
		float dTp2 = 0.5*dT;
		float dTp6 = dT * (1.0/6.0);
		
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

/*
__global__ void Lorenz(float* x1, float* x2, float* x3, float* sigma, float* rho, float* beta, int N, int I){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    //*
    int xidx = threadIdx.x + blockIdx.x * blockDim.x;
    int yidx = threadIdx.y + blockIdx.y * blockDim.y;
    int zidx = threadIdx.z + blockIdx.z * blockDim.z;
    //
    //végső esetben mind a három paraméter lehet egy tömb és akkor az x1, x2, x3 3D-s tömb lesz

    if (idx<N)
    {
        //Előszőr ezek nélkül kéne megnézni
        float X1 = x1[idx];
        float X2 = x2[idx];
        float X3 = x3[idx];

        float Sigma = sigma[idx];
        float Rho = *rho;
        float Beta = *beta;

        //*


        float X1 = x1[xidx][yidx][zidx];
        float X2 = x2[xidx][yidx][zidx];
        float X3 = x3[xidx][yidx][zidx];

        float Sigma = sigma[xidx];
        float Rho = rho[yidx];
        float Beta = Beta[zidx]
        //


        fot (int iter = 0; i<I; i++){
            //Rungekutta
        }
    }


}*/