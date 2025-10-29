// Lorenz_RK4_scan.cu
// CUDA implementation of the Lorenz system using RK4, parallelized over rho

#include <stdio.h>
#include <cuda_runtime.h>

#define SIGMA 10.0f
#define BETA 2.666f

// Lorenz system derivatives
device void lorenz(float x, float y, float z, float rho, float &dx, float &dy, float &dz) {
    dx = SIGMA * (y - x);
    dy = x * (rho - z) - y;
    dz = x * y - BETA * z;
}

// RK4 step for Lorenz system
device void rk4_step(float &x, float &y, float &z, float rho, float dt) {
    float dx1, dy1, dz1;
    float dx2, dy2, dz2;
    float dx3, dy3, dz3;
    float dx4, dy4, dz4;
    float x1, y1, z1;
    float x2, y2, z2;
    float x3, y3, z3;

    lorenz(x, y, z, rho, dx1, dy1, dz1);
    x1 = x + 0.5f * dt * dx1;
    y1 = y + 0.5f * dt * dy1;
    z1 = z + 0.5f * dt * dz1;
    lorenz(x1, y1, z1, rho, dx2, dy2, dz2);
    x2 = x + 0.5f * dt * dx2;
    y2 = y + 0.5f * dt * dy2;
    z2 = z + 0.5f * dt * dz2;
    lorenz(x2, y2, z2, rho, dx3, dy3, dz3);
    x3 = x + dt * dx3;
    y3 = y + dt * dy3;
    z3 = z + dt * dz3;
    lorenz(x3, y3, z3, rho, dx4, dy4, dz4);

    x += (dt / 6.0f) * (dx1 + 2.0f * dx2 + 2.0f * dx3 + dx4);
    y += (dt / 6.0f) * (dy1 + 2.0f * dy2 + 2.0f * dy3 + dy4);
    z += (dt / 6.0f) * (dz1 + 2.0f * dz2 + 2.0f * dz3 + dz4);
}

// Kernel: each thread integrates for a different rho
__global__ void lorenz_scan_kernel(float *rho_values, float *x_out, float *y_out, float *z_out, int nsteps, float dt, float x0, float y0, float z0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float rho = rho_values[idx];
    float x = x0, y = y0, z = z0;
    for (int i = 0; i < nsteps; ++i) {
        rk4_step(x, y, z, rho, dt);
    }
    x_out[idx] = x;
    y_out[idx] = y;
    z_out[idx] = z;
}

int main() {
    // Parameter scan settings
    int n_rho = 1024; // Number of rho values (can be changed)
    float rho_min = 0.0f, rho_max = 40.0f;
    float drho = (rho_max - rho_min) / (n_rho - 1);

    // Integration settings
    int nsteps = 10000;
    float dt = 0.01f;
    float x0 = 1.0f, y0 = 1.0f, z0 = 1.0f;

    // Allocate host memory
    float *h_rho = (float*)malloc(n_rho * sizeof(float));
    float *h_x = (float*)malloc(n_rho * sizeof(float));
    float *h_y = (float*)malloc(n_rho * sizeof(float));
    float *h_z = (float*)malloc(n_rho * sizeof(float));

    // Fill rho values
    for (int i = 0; i < n_rho; ++i) {
        h_rho[i] = rho_min + i * drho;
    }

    // Allocate device memory
    float *d_rho, *d_x, *d_y, *d_z;
    cudaMalloc(&d_rho, n_rho * sizeof(float));
    cudaMalloc(&d_x, n_rho * sizeof(float));
    cudaMalloc(&d_y, n_rho * sizeof(float));
    cudaMalloc(&d_z, n_rho * sizeof(float));

    cudaMemcpy(d_rho, h_rho, n_rho * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (n_rho + blockSize - 1) / blockSize;
    lorenz_scan_kernel<<<gridSize, blockSize>>>(d_rho, d_x, d_y, d_z, nsteps, dt, x0, y0, z0);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_x, d_x, n_rho * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, n_rho * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z, d_z, n_rho * sizeof(float), cudaMemcpyDeviceToHost);

    // Output results
    FILE *fout = fopen("lorenz_scan_output.txt", "w");
    fprintf(fout, "# rho x y z\n");
    for (int i = 0; i < n_rho; ++i) {
        fprintf(fout, "%f %f %f %f\n", h_rho[i], h_x[i], h_y[i], h_z[i]);
    }
    fclose(fout);

    // Free memory
    cudaFree(d_rho); cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    free(h_rho); free(h_x); free(h_y); free(h_z);
    return 0;
}
