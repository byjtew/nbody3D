all: nbody.g

nbody.g: nbody.c
	gcc -march=native -mavx2 -Ofast -ftree-vectorize -funroll-all-loops -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp

nbody.i: nbody.c
	icc -xhost -O3 -qopt-report $< -o $@ -qmkl -qopenmp

run_nbody.g: nbody.g
	taskset -c 3 ./nbody.g

clean:
	rm -Rf *~ nbody.g nbody.i *.optrpt

