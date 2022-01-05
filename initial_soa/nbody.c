//
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define DEBUG 0
//
typedef float f32;
typedef double f64;
typedef unsigned long long u64;

//
typedef struct particles_s {
    f32 *x, *y, *z;
    f32 *vx, *vy, *vz;
} particles_t;

void initialize_dumping() {
    FILE *fp = fopen("results.dat", "wb");
    fclose(fp);
}

void dump_values(particles_t p, u64 n) {
    FILE *fp = fopen("results.dat", "ab");
    for (u64 i = 0; i < n; i++)
        fprintf(fp, "%f %f %f\n", p.x[i], p.y[i], p.z[i]);
    // fprintf(fp, "%f %f %f %f %f %f\n", p.x[i], p.y[i], p.z[i], p.vx[i],
    //         p.vy[i], p.vz[i]);
    fclose(fp);
}

//
void init(particles_t p, u64 n) {
    srand(12);

    for (u64 i = 0; i < n; i++) {
        //
        u64 r1 = (u64)rand();
        u64 r2 = (u64)rand();
        f32 sign = (r1 > r2) ? 1 : -1;

        //
        p.x[i] = sign * (f32)rand() / (f32)RAND_MAX;
        p.y[i] = (f32)rand() / (f32)RAND_MAX;
        p.z[i] = sign * (f32)rand() / (f32)RAND_MAX;

        //
        p.vx[i] = (f32)rand() / (f32)RAND_MAX;
        p.vy[i] = sign * (f32)rand() / (f32)RAND_MAX;
        p.vz[i] = (f32)rand() / (f32)RAND_MAX;
    }
}

//
void move_particles(particles_t p, const f32 dt, u64 n) {
    //
    const f32 softening = 1e-20;

    //
    for (u64 i = 0; i < n; i++) {
        //
        f32 fx = 0.0;
        f32 fy = 0.0;
        f32 fz = 0.0;

        if (DEBUG) printf("\npx_i: %f\n", p.x[i]);
        if (DEBUG) printf("py_i: %f\n", p.y[i]);
        if (DEBUG) printf("pz_i: %f\n", p.z[i]);

        // 23 floating-point operations
        for (u64 j = 0; j < n; j++) {
            // Newton's law
            if (DEBUG) printf("p.x[j]: %f\n", p.x[j]);
            const f32 dx = p.x[j] - p.x[i];  // 1
            if (DEBUG) printf("dx: %f\n", dx);
            const f32 dy = p.y[j] - p.y[i];  // 2
            if (DEBUG) printf("dy: %f\n", dy);
            const f32 dz = p.z[j] - p.z[i];  // 3
            if (DEBUG) printf("dz: %f\n", dz);
            const f32 d_2 = (dx * dx) + (dy * dy) + (dz * dz) + softening;  // 9
            if (DEBUG) printf("d_2: %f\n", d_2);
            const f32 sqrt_d_2 = sqrtf(d_2);
            if (DEBUG) printf("sqrt_d_2: %f\n", sqrt_d_2);
            const f32 d_3_over_2 = 1.0f / (d_2 * sqrt_d_2);  // 11
            if (DEBUG) printf("d_3_over_2: %f\n", d_3_over_2);

            if (DEBUG) printf("fx: %f\n", fx);
            // Net force
            fx += dx * d_3_over_2;  // 13
            fy += dy * d_3_over_2;  // 15
            fz += dz * d_3_over_2;  // 17

            if (DEBUG) printf("fx: %f\n--\n", fx);
        }

        //
        p.vx[i] += dt * fx;  // 19
        p.vy[i] += dt * fy;  // 21
        p.vz[i] += dt * fz;  // 23

        if (DEBUG) printf("p.vx[i]: %f\n", p.vx[i]);
        if (DEBUG) printf("p.vy[i]: %f\n", p.vy[i]);
        if (DEBUG) printf("p.vz[i]: %f\n", p.vz[i]);
    }

    // 3 floating-point operations
    for (u64 i = 0; i < n; i++) {
        p.x[i] += dt * p.vx[i];
        p.y[i] += dt * p.vy[i];
        p.z[i] += dt * p.vz[i];
    }
}

//
int main(int argc, char **argv) {
    //
    const u64 n = (argc > 1) ? atoll(argv[1]) : 16384;
    const u64 steps = (argc > 2) ? atoll(argv[2]) : (10);
    const f32 dt = 0.01;

    //
    f64 rate = 0.0, drate = 0.0;

    // Steps to skip for warm up
    const u64 warmup = 3;

    //
    particles_t p;
    p.x = malloc(sizeof(f32) * n);
    p.y = malloc(sizeof(f32) * n);
    p.z = malloc(sizeof(f32) * n);

    p.vx = malloc(sizeof(f32) * n);
    p.vy = malloc(sizeof(f32) * n);
    p.vz = malloc(sizeof(f32) * n);

    //
    init(p, n);
    initialize_dumping();
    dump_values(p, n);

    const u64 s = sizeof(f32) * n * 6;

    printf("\n\033[1mTotal memory size:\033[0m %llu B, %llu KiB, %llu MiB\n\n",
           s, s >> 10, s >> 20);

    //
    printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s",
           "GFLOP/s");
    fflush(stdout);

    //
    for (u64 i = 0; i < steps; i++) {
        // Measure
        const f64 start = omp_get_wtime();

        move_particles(p, dt, n);

        const f64 end = omp_get_wtime();

        // Number of interactions/iterations
        const f32 h1 = (f32)(n) * (f32)(n - 1);

        // GFLOPS
        const f32 h2 = (23.0 * h1 + 3.0 * (f32)n) * 1e-9;

        if (i >= warmup) {
            rate += h2 / (end - start);
            drate += (h2 * h2) / ((end - start) * (end - start));
        }

        //
        printf("%5llu %10.3e %10.3e %8.1f %s\n", i, (end - start),
               h1 / (end - start), h2 / (end - start), (i < warmup) ? "*" : "");

        fflush(stdout);
    }
    dump_values(p, n);

    //
    rate /= (f64)(steps - warmup);
    drate = sqrt(drate / (f64)(steps - warmup) - (rate * rate));

    printf("-----------------------------------------------------\n");
    printf("\033[1m%s %4s \033[42m%10.1lf +- %.1lf GFLOP/s\033[0m\n",
           "Average performance:", "", rate, drate);
    printf("-----------------------------------------------------\n");

    //
    free(p.x);
    free(p.y);
    free(p.z);
    free(p.vx);
    free(p.vy);
    free(p.vz);

    //
    return 0;
}
