all: nbody.g snb_nbody.g knl_nbody.g

debug_nbody.g: nbody.c
	gcc -mavx2 -mfma -O0  $< -o $@ -lm -fopenmp

nbody.g: nbody.c
	gcc -march=native -mtune=native -fprofile-use=profile.gcc -fprofile-generate=profile.gcc -Ofast -ftree-vectorize -funroll-all-loops -fassociative-math -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp
snb_nbody.g: nbody.c
	gcc -march=sandybridge -mtune=sandybridge -fprofile-use=profile_snb.gcc -fprofile-generate=profile_snb.gcc -mfma -mavx -mavx2 -Ofast -ftree-vectorize -funroll-all-loops -fassociative-math -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp
knl_nbody.g: nbody.c
	gcc -march=knl -mtune=knl -fprofile-use=profile_knl.gcc -fprofile-generate=profile_knl.gcc -mfma -mavx -mavx2 -Ofast -ftree-vectorize -funroll-all-loops -fassociative-math -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp

nbody.i: nbody.c
	icc -xhost -Ofast -qopt-report $< -o $@ -qmkl -qopenmp

run_nbody.g: nbody.g
	taskset -c 3 ./nbody.g

clean: 
	rm -Rf *~ debug_nbody.g nbody.g snb_nbody.g knl_nbody.g nbody.i *.optrpt

