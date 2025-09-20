# Lorenz-CUDA
This repository contains the CUDA code I used in my thesis work. The main goal is to create a general solver for the Lorenz equation using Butcher tableau, profile it and make it fully optimized.


## Running

To compile the code, run in the terminal:

```
make SRC=your_source_file.cu
```

If SRC is not specified, it will compile `main.cu` by default.