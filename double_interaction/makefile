all: nbody.g

nbody.g: nbody.c
	gcc -march=native -mavx2 -Ofast -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp

nbody.i: nbody.c
	icc -xhost -ipo -qopt-prefetch=5 -Ofast -qopt-report $< -o $@ -qmkl -qopenmp


clean:
	rm -Rf *~ *.g *.i *.optrpt *.dat

run: nbody.i
	taskset -c 12 ./nbody.i