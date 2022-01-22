all: nbody.g knl_nbody.g

nbody.g: nbody.c
	gcc -march=native -mfma -mavx512f -mavx512er -mavx512cd -Ofast -ftree-vectorize -funroll-all-loops -fassociative-math -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp

nbody.i: nbody.c
	icc -xhost -Ofast -qopt-report $< -o $@ -qmkl -qopenmp

clean: 
	rm -Rf *~ *.g nbody.i *.optrpt *.gcc

