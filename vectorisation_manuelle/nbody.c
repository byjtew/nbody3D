//
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
//
typedef float f32;
typedef double f64;
typedef unsigned long long u64;

typedef __m512 ps_float;

const f32 softening = 1e-20f;
const u64 ps_float_nb_packed = 16;

ps_float one_vector, softening_ps_vector;

//
typedef struct particles_s {
    ps_float *x, *y, *z;
    ps_float *vx, *vy, *vz;
} particles_t;

static inline f32 randrealf(int use_sign) {
    u64 r1 = (u64)rand();
    u64 r2 = (u64)rand();
    f32 sign = (r1 > r2) ? 1 : -1;
    return (use_sign ? sign : 1.0) * (f32)rand() / (f32)RAND_MAX;
}

static inline void random__m512(ps_float *v, int use_sign) {
    *v = _mm512_set_ps(
        randrealf(use_sign), randrealf(use_sign), randrealf(use_sign),
        randrealf(use_sign), randrealf(use_sign), randrealf(use_sign),
        randrealf(use_sign), randrealf(use_sign), randrealf(use_sign),
        randrealf(use_sign), randrealf(use_sign), randrealf(use_sign),
        randrealf(use_sign), randrealf(use_sign), randrealf(use_sign),
        randrealf(use_sign));
}

//
void init(particles_t *p, u64 n) {
    for (u64 i = 0; i < n; i++) random__m512(&(p->x[i]), 1);
    for (u64 i = 0; i < n; i++) random__m512(&(p->y[i]), 0);
    for (u64 i = 0; i < n; i++) random__m512(&(p->z[i]), 1);
    for (u64 i = 0; i < n; i++) random__m512(&(p->vx[i]), 1);
    for (u64 i = 0; i < n; i++) random__m512(&(p->vy[i]), 0);
    for (u64 i = 0; i < n; i++) random__m512(&(p->vz[i]), 1);
}

static inline f32 cumsum(ps_float *v) { return _mm512_reduce_add_ps(*v); }

//
void move_particles(particles_t p, const f32 dt, const ps_float dt_vector,
                    u64 n) {
    ps_float px_i, py_i, pz_i;
    ps_float fx, fy, fz;

    for (u64 bi = 0; bi < n; bi++) {
        f32 *px_ptr = (f32 *)(p.x + bi);
        f32 *py_ptr = (f32 *)(p.y + bi);
        f32 *pz_ptr = (f32 *)(p.z + bi);

        for (u64 i = 0; i < ps_float_nb_packed; i++) {
            fx = _mm512_setzero_ps();
            fy = _mm512_setzero_ps();
            fz = _mm512_setzero_ps();

            px_i = _mm512_set1_ps(px_ptr[i]);
            py_i = _mm512_set1_ps(py_ptr[i]);
            pz_i = _mm512_set1_ps(pz_ptr[i]);

            for (u64 j = 0; j < n; j++) {
                ps_float dx = _mm512_sub_ps(p.x[j], px_i);

                ps_float dy = _mm512_sub_ps(p.y[j], py_i);

                ps_float dz = _mm512_sub_ps(p.z[j], pz_i);

                ps_float d_2 = _mm512_fmadd_ps(
                    dx, dx,
                    _mm512_fmadd_ps(
                        dy, dy, _mm512_fmadd_ps(dz, dz, softening_ps_vector)));

                d_2 =
                    _mm512_mul_ps(_mm512_rcp14_ps(d_2), _mm512_rsqrt14_ps(d_2));

                fx = _mm512_fmadd_ps(dx, d_2, fx);
                fy = _mm512_fmadd_ps(dy, d_2, fy);
                fz = _mm512_fmadd_ps(dz, d_2, fz);
            }

            ((f32 *)(&p.vx[bi]))[i] += dt * cumsum(&fx);
            ((f32 *)(&p.vy[bi]))[i] += dt * cumsum(&fy);
            ((f32 *)(&p.vz[bi]))[i] += dt * cumsum(&fz);
        }
    }

    for (u64 i = 0; i < n; i++) {
        p.x[i] = _mm512_fmadd_ps(p.vx[i], dt_vector, p.x[i]);
        p.y[i] = _mm512_fmadd_ps(p.vy[i], dt_vector, p.y[i]);
        p.z[i] = _mm512_fmadd_ps(p.vz[i], dt_vector, p.z[i]);
    }
}

//
int main(int argc, char **argv) {
    //
    u64 ntemp = (argc > 1) ? atoll(argv[1]) : (16384);
    const u64 n = (ntemp - (ntemp % 16)) / 16;  // AVX 512 <=> 16 packed floats
    printf("== Init with %lld __m512 [%ld] values==\n", n, sizeof(__m256));

    const u64 steps = (argc > 2) ? atoll(argv[2]) : (10);
    const f32 dt = 0.01f;
    const ps_float dt_ps_vec = _mm512_set1_ps(dt);

    //
    f64 rate = 0.0, drate = 0.0, total_time = .0;

    // Steps to skip for warm up
    const u64 warmup = 3;

    const size_t alignment = 256;
    //
    particles_t p;
    p.x = (ps_float *)aligned_alloc(alignment,
                                    n * sizeof(ps_float) * sizeof(f32));
    p.y = (ps_float *)aligned_alloc(alignment,
                                    n * sizeof(ps_float) * sizeof(f32));
    p.z = (ps_float *)aligned_alloc(alignment,
                                    n * sizeof(ps_float) * sizeof(f32));

    p.vx = (ps_float *)aligned_alloc(alignment,
                                     n * sizeof(ps_float) * sizeof(f32));
    p.vy = (ps_float *)aligned_alloc(alignment,
                                     n * sizeof(ps_float) * sizeof(f32));
    p.vz = (ps_float *)aligned_alloc(alignment,
                                     n * sizeof(ps_float) * sizeof(f32));

    //
    init(&p, n);

    const u64 s = sizeof(f32) * n * 6;

    printf("\n\033[1mTotal memory size:\033[0m %llu B, %llu KiB, %llu MiB\n\n",
           s, s >> 10, s >> 20);

    //
    printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s",
           "GFLOP/s");
    fflush(stdout);

    one_vector = _mm512_set1_ps(1.0f);

    softening_ps_vector = _mm512_set1_ps(softening);

    //
    for (u64 i = 0; i < steps; i++) {
        // Measure
        const f64 start = omp_get_wtime();

        move_particles(p, dt, dt_ps_vec, n);

        const f64 end = omp_get_wtime();

        // Number of interactions/iterations
        const f32 h1 = 16 * (f32)(n)*16 * (f32)(n - 1);

        // GFLOPS
        const f32 h2 = (23.0 * h1 + 3.0 * (f32)n) * 1e-9;

        if (i >= warmup) {
            total_time += (end - start);
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
    total_time /= (f64)(steps - warmup);
    drate = sqrt(drate / (f64)(steps - warmup) - (rate * rate));

    printf("-----------------------------------------------------\n");
    printf(
        "\033[1m%s %4s \033[42m%2.4f msec | %10.1lf +- %.1lf GFLOP/s\033[0m\n",
        "Average performance:", "", total_time*1000.0f, rate, drate);
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
