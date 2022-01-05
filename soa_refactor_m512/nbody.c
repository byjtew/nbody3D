//
#include <immintrin.h>
#include <execinfo.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
//
typedef float f32;
typedef double f64;
typedef unsigned long long u64;

typedef __m512 pd_float;

const f32 softening = 1e-20;

pd_float one_vector, softening_pd_vector;

void print_trace() {
    void *array[10];
    char **strings;
    int size, i;

    size = backtrace(array, 10);
    strings = backtrace_symbols(array, size);
    if (strings != NULL) {
        printf("Obtained %d stack frames.\n", size);
        for (i = 0; i < size; i++) printf("%s\n", strings[i]);
    }

    free(strings);
}

//
typedef struct particles_s {
    pd_float *x, *y, *z;
    pd_float *vx, *vy, *vz;
} particles_t;

static inline void reciprocal_sqrt(pd_float *num) {
    print_trace();
    *num = _mm512_rsqrt28_ps(*num);
    print_trace();
}

static inline pd_float createpd_float_from_ptr(float *values) {
    return _mm512_loadu_ps(values);
    // return _mm_setr_ps(a, b, c, d);
}

f32 randrealf() {
    f32 sign = ((u64)rand() > (u64)rand()) ? 1 : -1;
    return sign * (f32)rand() / (f32)RAND_MAX;
}

pd_float random__m512() {
    // f32 *values = aligned_alloc(64, 16 * sizeof(f32));
    // for (u64 j = 0; j < 16; j++) values[j] = randrealf();
    return _mm512_set1_ps(randrealf());  // _mm512_loadu_ps(values_ptr);
    // free(values);
}

//
void init(particles_t p, u64 n) {
    f32 px[16], py[16], pz[16], pvx[16], pvy[16], pvz[16];
    for (u64 i = 0; i < n; i++) {
        p.x[i] = random__m512();
        p.y[i] = random__m512();
        p.z[i] = random__m512();

        p.vx[i] = random__m512();
        p.vy[i] = random__m512();
        p.vz[i] = random__m512();
    }
}

//
void move_particles(particles_t p, const pd_float dt_vector, u64 n) {
    //
    pd_float px_i, py_i, pz_i;
    pd_float dxs, dys, dzs;

    //
    for (u64 i = 0; i < n; i++) {
        //
        pd_float fx = _mm512_setzero_ps();
        pd_float fy = _mm512_setzero_ps();
        pd_float fz = _mm512_setzero_ps();

        px_i = p.x[i];
        py_i = p.y[i];
        pz_i = p.z[i];

        for (u64 j = 0; j < n; j++) {
            /*
            const f32 dx = p.x[j] - p.x[i];
            const f32 dy = p.y[j] - p.y[i];
            const f32 dz = p.z[j] - p.z[i];
            const f32 d_2 = (dx * dx) + (dy * dy) + (dz * dz) + softening;
            const f32 d_3_over_2 = pow(d_2, 3.0 / 2.0);

            // Net force
            fx += dx / d_3_over_2;  // 13
            fy += dy / d_3_over_2;  // 15
            fz += dz / d_3_over_2;  // 17
            */

            pd_float dx = _mm512_sub_ps(p.x[j], px_i);
            pd_float dy = _mm512_sub_ps(p.y[j], py_i);
            pd_float dz = _mm512_sub_ps(p.z[j], pz_i);
            pd_float d_2 = _mm512_fmadd_ps(
                dx, dx, _mm512_fmadd_ps(dy, dy, _mm512_mul_ps(dz, dz)));
            pd_float d_2_inv = _mm512_div_ps(one_vector, d_2);
            pd_float d_3_over_2 =
                _mm512_mul_ps(_mm512_rsqrt14_ps(d_2), d_2_inv);

            fx = _mm512_fmadd_ps(dx, d_3_over_2, fx);
            fy = _mm512_fmadd_ps(dy, d_3_over_2, fy);
            fz = _mm512_fmadd_ps(dz, d_3_over_2, fz);
        }

        // p.vx[i] += dt * fx;  // 19
        // p.vy[i] += dt * fy;  // 21
        // p.vz[i] += dt * fz;  // 23

        p.vx[i] = _mm512_fmadd_ps(dt_vector, fx, p.vx[i]);
        p.vy[i] = _mm512_fmadd_ps(dt_vector, fy, p.vy[i]);
        p.vz[i] = _mm512_fmadd_ps(dt_vector, fz, p.vz[i]);
    }

    // 3 floating-point operations

    for (u64 i = 0; i < n; i++)
        p.x[i] = _mm512_fmadd_ps(p.vx[i], dt_vector,
                                 p.x[i]);  // p.x[i] += dt * p.vx[i];
    for (u64 i = 0; i < n; i++)
        p.y[i] = _mm512_fmadd_ps(p.vy[i], dt_vector,
                                 p.y[i]);  // p.y[i] += dt * p.vy[i];
    for (u64 i = 0; i < n; i++)
        p.z[i] = _mm512_fmadd_ps(p.vz[i], dt_vector,
                                 p.z[i]);  // p.z[i] += dt * p.vz[i];
}

//
int main(int argc, char **argv) {
    //
    u64 ntemp = (argc > 1) ? atoll(argv[1]) : (16384);
    const u64 n = (ntemp - (ntemp % 16)) / 16;  // AVX 512 <=> 16 packed floats
    const u64 steps = 10;
    const pd_float dt = _mm512_set1_ps(0.01f);

    //
    f64 rate = 0.0, drate = 0.0;

    // Steps to skip for warm up
    const u64 warmup = 3;

    //
    particles_t p;
    p.x = (pd_float *)malloc(sizeof(pd_float) * n);
    p.y = (pd_float *)malloc(sizeof(pd_float) * n);
    p.z = (pd_float *)malloc(sizeof(pd_float) * n);

    p.vx = (pd_float *)malloc(sizeof(pd_float) * n);
    p.vy = (pd_float *)malloc(sizeof(pd_float) * n);
    p.vz = (pd_float *)malloc(sizeof(pd_float) * n);

    //
    init(p, n);

    const u64 s = sizeof(f32) * n * 6;

    printf("\n\033[1mTotal memory size:\033[0m %llu B, %llu KiB, %llu MiB\n\n",
           s, s >> 10, s >> 20);

    //
    printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s",
           "GFLOP/s");
    fflush(stdout);

    one_vector = _mm512_set1_ps(1.0f);

    softening_pd_vector = _mm512_set1_ps(softening);

    //
    for (u64 i = 0; i < steps; i++) {
        // Measure
        const f64 start = omp_get_wtime();

        move_particles(p, dt, n);

        const f64 end = omp_get_wtime();

        // Number of interactions/iterations
        const f32 h1 = 16 * (f32)(n)*16 * (f32)(n - 1);

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
