all: nbody.g

nbody.g: nbody.c
	gcc -march=native -fprofile-use=profile.gcc -fprofile-generate=profile.gcc -mavx2 -mrecip=sqrt -mrecip=vec-sqrt -Ofast -ftree-vectorize -funroll-all-loops -fassociative-math -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp

nbody.i: nbody.c
	icc -xhost -Ofast -qopt-report $< -o $@ -qmkl -qopenmp

run_nbody.g: nbody.g
	taskset -c 3 ./nbody.g

clean: 
	rm -Rf *~ nbody.g nbody.i *.optrpt

