//
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
//
typedef float f32;
typedef double f64;
typedef unsigned long long u64;

__m128 one_vector;

//
typedef struct particles_s {
    f32 *x, *y, *z;
    f32 *vx, *vy, *vz;
} particles_t;

static inline void reciprocal_sqrt(__m128 *num) { *num = _mm_rsqrt_ps(*num); }

static inline __m128 create__m128_from_elements(float a, float b, float c,
                                                float d) {
    // _mm_loadu_ps(values);
    return _mm_setr_ps(a, b, c, d);
}

static inline __m128 create__m128_from_ptr(float *values) {
    return _mm_loadu_ps(values);
    // return _mm_setr_ps(a, b, c, d);
}

//
void init(particles_t p, u64 n) {
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
    f32 px_i, py_i, pz_i;
    __m128 dxs, dys, dzs;
    float d_2_array[4];
    float *d_2_vector = d_2_array;

    //
    for (u64 i = 0; i < n; i++) {
        //
        f32 fx = 0.0;
        f32 fy = 0.0;
        f32 fz = 0.0;

        px_i = p.x[i];
        py_i = p.y[i];
        pz_i = p.z[i];

        for (u64 j = 0; j < n; j += 4) {
            f32 dx_0 = p.x[j + 0] - px_i;
            f32 dy_0 = p.y[j + 0] - py_i;
            f32 dz_0 = p.z[j + 0] - pz_i;
            // f32 d_2_0 =                (dx_0 * dx_0) + (dy_0 * dy_0) + (dz_0
            // * dz_0) + softening;
            //  d_2_vector[0] = d_2_0;

            f32 dx_1 = p.x[j + 1] - px_i;
            f32 dy_1 = p.y[j + 1] - py_i;
            f32 dz_1 = p.z[j + 1] - pz_i;
            // f32 d_2_1 =                (dx_1 * dx_1) + (dy_1 * dy_1) + (dz_1
            // * dz_1) + softening;
            //  d_2_vector[1] = d_2_1;

            f32 dx_2 = p.x[j + 2] - px_i;
            f32 dy_2 = p.y[j + 2] - py_i;
            f32 dz_2 = p.z[j + 2] - pz_i;
            // f32 d_2_2 = (dx_2 * dx_2) + (dy_2 * dy_2) + (dz_2 * dz_2) +
            // softening;
            //  d_2_vector[2] = d_2_2;

            f32 dx_3 = p.x[j + 3] - px_i;
            f32 dy_3 = p.y[j + 3] - py_i;
            f32 dz_3 = p.z[j + 3] - pz_i;
            // f32 d_2_3 = (dx_3 * dx_3) + (dy_3 * dy_3) + (dz_3 * dz_3) +
            // softening;
            //  d_2_vector[3] = d_2_3;

            __m128 d_2_packed = create__m128_from_elements(
                (dx_0 * dx_0) + (dy_0 * dy_0) + (dz_0 * dz_0) + softening,
                (dx_1 * dx_1) + (dy_1 * dy_1) + (dz_1 * dz_1) + softening,
                (dx_2 * dx_2) + (dy_2 * dy_2) + (dz_2 * dz_2) + softening,
                (dx_3 * dx_3) + (dy_3 * dy_3) + (dz_3 * dz_3) + softening);

            dxs = create__m128_from_elements(dx_0, dx_1, dx_2, dx_3);
            dys = create__m128_from_elements(dy_0, dy_1, dy_2, dy_3);
            dzs = create__m128_from_elements(dz_0, dz_1, dz_2, dz_3);

            //__m128 d_2_packed = create__m128_from_ptr(d_2_vector);
            __m128 inv_d2_packed = _mm_div_ps(one_vector, d_2_packed);
            reciprocal_sqrt(&d_2_packed);

            __m128 results = _mm_mul_ps(inv_d2_packed,
                                        d_2_packed);  // [ 1.0f/d_2, ...]{4}
            // float *results_ptr = (float *)&results;

            results = _mm_mul_ps(dxs, results);
            float *d_dot_results_ptr = (float *)&results;
            fx += d_dot_results_ptr[0] + d_dot_results_ptr[1] +
                  d_dot_results_ptr[2] + d_dot_results_ptr[3];

            results = _mm_mul_ps(dys, results);
            fy += d_dot_results_ptr[0] + d_dot_results_ptr[1] +
                  d_dot_results_ptr[2] + d_dot_results_ptr[3];

            results = _mm_mul_ps(dzs, results);
            fz += d_dot_results_ptr[0] + d_dot_results_ptr[1] +
                  d_dot_results_ptr[2] + d_dot_results_ptr[3];

            /*
            fx += dx_0 * (results_ptr[0]);
            fx += dx_1 * (results_ptr[1]);
            fx += dx_2 * (results_ptr[2]);
            fx += dx_3 * (results_ptr[3]);

            fy += dy_0 * (results_ptr[0]);
            fy += dy_1 * (results_ptr[1]);
            fy += dy_2 * (results_ptr[2]);
            fy += dy_3 * (results_ptr[3]);

            fz += dz_0 * (results_ptr[0]);
            fz += dz_1 * (results_ptr[1]);
            fz += dz_2 * (results_ptr[2]);
            fz += dz_3 * (results_ptr[3]);
            */
        }
        //
        p.vx[i] += dt * fx;  // 19
        p.vy[i] += dt * fy;  // 21
        p.vz[i] += dt * fz;  // 23
    }

    // 3 floating-point operations
    for (u64 i = 0; i < n; i++) p.x[i] += dt * p.vx[i];
    for (u64 i = 0; i < n; i++) p.y[i] += dt * p.vy[i];
    for (u64 i = 0; i < n; i++) p.z[i] += dt * p.vz[i];
}

//
int main(int argc, char **argv) {
    //
    const u64 n = (argc > 1) ? atoll(argv[1]) : 16384;
    const u64 steps = 10;
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

    const u64 s = sizeof(f32) * n * 6;

    printf("\n\033[1mTotal memory size:\033[0m %llu B, %llu KiB, %llu MiB\n\n",
           s, s >> 10, s >> 20);

    //
    printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s",
           "GFLOP/s");
    fflush(stdout);

    one_vector = create__m128_from_elements(1.0f, 1.0f, 1.0f, 1.0f);

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
