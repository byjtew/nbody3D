all: nbody.g snb_nbody.g knl_nbody.g

nbody.g: nbody.c
	gcc -march=native -mtune=native -fprofile-use=profile.gcc -fprofile-generate=profile.gcc -mno-avx -mno-aes -mfma -mavx -mavx2 -mavx512f -Ofast -ftree-vectorize -funroll-all-loops -fassociative-math -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp
snb_nbody.g: nbody.c
	gcc -march=sandybridge -mtune=sandybridge -fprofile-use=profile_snb.gcc -fprofile-generate=profile_snb.gcc -mfma -mavx512f -Ofast -ftree-vectorize -funroll-all-loops -fassociative-math -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp
knl_nbody.g: nbody.c
	gcc -march=knl -mtune=knl -fprofile-use=profile_knl.gcc -fprofile-generate=profile_knl.gcc -mavx512f -Ofast -ftree-vectorize -funroll-all-loops -fassociative-math -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp

nbody.i: nbody.c
	icc -xhost -Ofast -qopt-report $< -o $@ -qmkl -qopenmp

run_nbody.g: nbody.g
	taskset -c 3 ./snb_nbody.g
	taskset -c 3 ./knl_nbody.g

clean: 
	rm -Rf *~ nbody.g snb_nbody.g knl_nbody.g nbody.i *.optrpt

