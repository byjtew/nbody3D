//
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define DEBUG 0

//
typedef float f32;
typedef double f64;
typedef unsigned long long u64;

// To define:
const u64 ps_float_nb_packed = 8;
typedef __m256 ps_float;

const f32 softening = 1e-20;

ps_float one_vector, softening_pd_vector;

//
typedef struct {
    ps_float *x;
    ps_float *y;
    ps_float *z;
    ps_float *vx;
    ps_float *vy;
    ps_float *vz;
} particles_t;

static inline f32 randrealf(f32 sign) {
    return (f32)(sign * (f32)rand() / (f32)RAND_MAX);
}

void initialize_dumping() {
    FILE *fp = fopen("results.dat", "wb");
    fclose(fp);
}

void fprintf_ps_float(FILE *fp, ps_float v) {
    float *ptr = (float *)&v;
    for (u64 j = 0; j < ps_float_nb_packed; j++) fprintf(fp, "%f ", ptr[j]);
    fprintf(fp, "\n");
}

void fprintf_current_status(particles_t p, FILE *fp, u64 i) {
    for (u64 j = 0; j < ps_float_nb_packed; j++)
        fprintf(fp, "%f %f %f\n", ((float *)p.x + i)[j], ((float *)p.y + i)[j],
                ((float *)p.z + i)[j]);
    // fprintf(fp, "%f %f %f %f %f %f\n", ((float *)p.x + i)[j],
    //         ((float *)p.y + i)[j], ((float *)p.z + i)[j],
    //         ((float *)p.vx + i)[j], ((float *)p.vy + i)[j],
    //         ((float *)p.vz + i)[j]);
}

void dump_values(particles_t p, u64 n) {
    FILE *fp = fopen("results.dat", "ab");
    for (u64 i = 0; i < n; i++) fprintf_current_status(p, fp, i);
    fclose(fp);
}

//
void init(particles_t *p, u64 n) {
    srand(12);
    f32 *rd_values =
        (f32 *)aligned_alloc(64, sizeof(f32) * 6 * ps_float_nb_packed);

    for (u64 i = 0; i < n; i++) {
        for (u64 j = 0; j < ps_float_nb_packed; j++) {
            f32 sign = ((u64)rand() > (u64)rand()) ? 1 : -1;
            for (u64 k = 0; k < 6; k++)
                if (k == 1 || k == 3 || k == 5)
                    rd_values[(ps_float_nb_packed * k + j)] = randrealf(1.0f);
                else
                    rd_values[(ps_float_nb_packed * k + j)] = randrealf(sign);
        }

        p->x[i] = _mm256_load_ps(rd_values + 0);
        p->y[i] = _mm256_load_ps(rd_values + (ps_float_nb_packed * 1));
        p->z[i] = _mm256_load_ps(rd_values + (ps_float_nb_packed * 2));

        p->vx[i] = _mm256_load_ps(rd_values + (ps_float_nb_packed * 3));
        p->vy[i] = _mm256_load_ps(rd_values + (ps_float_nb_packed * 4));
        p->vz[i] = _mm256_load_ps(rd_values + (ps_float_nb_packed * 5));
    }
    free(rd_values);
}

//
void move_particles(particles_t p, const ps_float dt_vector, u64 n) {
    //
    ps_float px_i, py_i, pz_i;

    for (u64 bi = 0; bi < n; bi++) {
        f32 *px_ptr = (f32 *)(p.x + bi);
        f32 *py_ptr = (f32 *)(p.y + bi);
        f32 *pz_ptr = (f32 *)(p.z + bi);

        for (u64 i = 0; i < ps_float_nb_packed; i++) {
            // printf("<- i: %lld ->\n", i);
            //
            ps_float fx = _mm256_setzero_ps();
            ps_float fy = _mm256_setzero_ps();
            ps_float fz = _mm256_setzero_ps();

            px_i = _mm256_set1_ps(px_ptr[i]);
            py_i = _mm256_set1_ps(
                py_ptr[i]);  // _mm256_set1_ps(((f32 *)p.y + bi)[i]);
            pz_i = _mm256_set1_ps(
                pz_ptr[i]);  // _mm256_set1_ps(((f32 *)p.z + bi)[i]);
            if (DEBUG) {
                printf("\npx_i:\n");
                fprintf_ps_float(stdout, px_i);
                printf("py_i:\n");
                fprintf_ps_float(stdout, py_i);
                printf("pz_i:\n");
                fprintf_ps_float(stdout, pz_i);
            }

            for (u64 j = 0; j < n; j++) {
                // printf("<- j: %lld ->\n", j);
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

                if (DEBUG) printf("p.x[j]:\n");
                if (DEBUG) fprintf_ps_float(stdout, p.x[j]);
                if (DEBUG) printf("p.y[j]:\n");
                if (DEBUG) fprintf_ps_float(stdout, p.y[j]);
                if (DEBUG) printf("p.z[j]:\n");
                if (DEBUG) fprintf_ps_float(stdout, p.z[j]);

                ps_float dx = _mm256_sub_ps(p.x[j], px_i);

                if (DEBUG) printf("dx:\n");
                if (DEBUG) fprintf_ps_float(stdout, dx);

                ps_float dy = _mm256_sub_ps(py_i, p.y[j]);

                if (DEBUG) printf("dy:\n");
                if (DEBUG) fprintf_ps_float(stdout, dy);

                ps_float dz = _mm256_sub_ps(p.z[j], pz_i);

                if (DEBUG) printf("dz:\n");
                if (DEBUG) fprintf_ps_float(stdout, dz);

                ps_float d_2 = _mm256_fmadd_ps(
                    dx, dx,
                    _mm256_fmadd_ps(
                        dy, dy, _mm256_fmadd_ps(dz, dz, softening_pd_vector)));

                if (DEBUG) printf("d_2:\n");
                if (DEBUG) fprintf_ps_float(stdout, d_2);

                ps_float sqrt_d_2 = _mm256_sqrt_ps(d_2);

                if (DEBUG) printf("sqrt_d_2:\n");
                if (DEBUG) fprintf_ps_float(stdout, sqrt_d_2);

                ps_float d_3_over_2 =
                    _mm256_rcp_ps(_mm256_mul_ps(sqrt_d_2, d_2));

                if (DEBUG) printf("d_3_over_2:\n");
                if (DEBUG) fprintf_ps_float(stdout, d_3_over_2);

                if (DEBUG) printf("fx:\n");
                if (DEBUG) fprintf_ps_float(stdout, fx);

                fx = _mm256_add_ps(fx, _mm256_mul_ps(dx, d_3_over_2));
                fy = _mm256_fmadd_ps(dy, d_3_over_2, fy);
                fz = _mm256_fmadd_ps(dz, d_3_over_2, fz);

                if (DEBUG) printf("fx:\n");
                if (DEBUG) fprintf_ps_float(stdout, fx);
                if (DEBUG) printf("\n\n");
            }

            // p.vx[i] += dt * fx;  // 19
            // p.vy[i] += dt * fy;  // 21
            // p.vz[i] += dt * fz;  // 23
            if (0 && DEBUG) {
                printf("BFORE p.vx[bi]:\n");
                fprintf_ps_float(stdout, p.vx[bi]);
                printf("BFORE p.vy[bi]:\n");
                fprintf_ps_float(stdout, p.vy[bi]);
                printf("BFORE p.vz[bi]:\n");
                fprintf_ps_float(stdout, p.vz[bi]);
            }

            p.vx[bi] = _mm256_fmadd_ps(dt_vector, fx, p.vx[bi]);
            p.vy[bi] = _mm256_fmadd_ps(dt_vector, fy, p.vy[bi]);
            p.vz[bi] = _mm256_fmadd_ps(dt_vector, fz, p.vz[bi]);
            if (DEBUG) {
                printf("p.vx[bi]:\n");
                fprintf_ps_float(stdout, p.vx[bi]);
                printf("p.vy[bi]:\n");
                fprintf_ps_float(stdout, p.vy[bi]);
                printf("p.vz[bi]:\n");
                fprintf_ps_float(stdout, p.vz[bi]);
            }
        }
    }

    // 3 floating-point operations

    for (u64 i = 0; i < n; i++)
        p.x[i] = _mm256_fmadd_ps(p.vx[i], dt_vector,
                                 p.x[i]);  // p.x[i] += dt * p.vx[i];
    for (u64 i = 0; i < n; i++)
        p.y[i] = _mm256_fmadd_ps(p.vy[i], dt_vector,
                                 p.y[i]);  // p.y[i] += dt * p.vy[i];
    for (u64 i = 0; i < n; i++)
        p.z[i] = _mm256_fmadd_ps(p.vz[i], dt_vector,
                                 p.z[i]);  // p.z[i] += dt * p.vz[i];
}

//
int main(int argc, char **argv) {
    //
    u64 ntemp = (argc > 1) ? atoll(argv[1]) : (16384);
    const u64 n = (u64)((ntemp - (ntemp % ps_float_nb_packed)) /
                        ps_float_nb_packed);  // AVX 512 <=> 16 packed floats
    const u64 steps = (argc > 2) ? atoll(argv[2]) : (10);
    const ps_float dt = _mm256_set1_ps(0.01f);

    //
    f64 rate = 0.0, drate = 0.0;

    // Steps to skip for warm up
    const u64 warmup = 3;

    //
    particles_t p;
    p.x = aligned_alloc(64, n * sizeof(__m256));
    p.y = aligned_alloc(64, n * sizeof(__m256));
    p.z = aligned_alloc(64, n * sizeof(__m256));

    p.vx = aligned_alloc(64, n * sizeof(__m256));
    p.vy = aligned_alloc(64, n * sizeof(__m256));
    p.vz = aligned_alloc(64, n * sizeof(__m256));
    printf("== Init with %lld (__m256 [%ld]) values ==\n",
           n * ps_float_nb_packed, sizeof(__m256));
    //
    init(&p, n);
    initialize_dumping();
    dump_values(p, n);

    const u64 s = sizeof(f32) * n * 6;

    printf("\n\033[1mTotal memory size:\033[0m %llu B, %llu KiB, %llu MiB\n\n",
           s, s >> 10, s >> 20);

    //
    printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s",
           "GFLOP/s");
    fflush(stdout);

    one_vector = _mm256_set1_ps(1.0f);

    softening_pd_vector = _mm256_set1_ps(softening);

    //
    for (u64 i = 0; i < steps; i++) {
        // Measure
        const f64 start = omp_get_wtime();

        move_particles(p, dt, n);

        // fprintf_current_status(p, stdout, 0);

        const f64 end = omp_get_wtime();

        // Number of interactions/iterations
        const f32 h1 =
            ps_float_nb_packed * (f32)(n)*ps_float_nb_packed * (f32)(n - 1);

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
