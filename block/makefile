all: nbody.g

nbody.g: nbody.c
	gcc -march=native -mavx2 -Ofast -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp

nbody.i: nbody.c
	icc -xhost -Ofast -qopt-report $< -o $@ -qmkl -qopenmp


nbody_gen.g: nbody.c
	gcc -march=native -fprofile-generate=profile.gcc -mavx2 -Ofast -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp

nbody_gen.i: nbody.c
	icc -xhost -prof-gen -Ofast -qopt-report $< -o $@ -qmkl -qopenmp


nbody_use.g: nbody.c
	gcc -march=native -fprofile-use=profile.gcc -mavx2 -Ofast -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp

nbody_use.i: nbody.c
	icc -xhost -prof-use -Ofast -qopt-report $< -o $@ -qmkl -qopenmp


clean:
	rm -Rf *~ *.g *.i *.optrpt *.dat *.dyn *.dpi *.lock *.gcc

run: nbody.i
	taskset -c 12 ./nbody.i