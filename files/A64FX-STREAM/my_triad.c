// Compile: gcc -O3 -fopenmp -march=armv8.2-a+sve my_triad.c -o my_triad.exe

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>
#include <arm_sve.h>

#define STREAM_TYPE double

#ifndef STREAM_ARRAY_SIZE
#define STREAM_ARRAY_SIZE 160000000
#endif

#ifndef NTIMES
#define NTIMES 10
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

double get_wtime_sec()
{
    double sec;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    sec = tv.tv_sec + (double) tv.tv_usec / 1000000.0;
    return sec;
}


static STREAM_TYPE a[STREAM_ARRAY_SIZE];
static STREAM_TYPE b[STREAM_ARRAY_SIZE];
static STREAM_TYPE c[STREAM_ARRAY_SIZE];
static STREAM_TYPE scalar = 2;
size_t *thread_displs;

#define NFUNCTION 3
static double max_time[NFUNCTION], min_time[NFUNCTION], avg_time[NFUNCTION];
static char *label[NFUNCTION] = {
    "STREAM TRIAD, auto-vectorization ",
    "STREAM TRIAD, SVE-VLA no unroll  ",
    "STREAM TRIAD, SVE-512 unroll-4   "
};

void triad_autovec()
{
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const size_t start_idx = thread_displs[tid];
        const size_t end_idx   = thread_displs[tid + 1];
        #pragma omp simd
        for (size_t i = start_idx; i < end_idx; i++) a[i] = b[i] + scalar * c[i];
    }
}

void triad_sve()
{
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const size_t start_idx = thread_displs[tid];
        const size_t end_idx   = thread_displs[tid + 1];
        size_t idx = start_idx;
        svbool_t pg = svwhilelt_b64(idx, end_idx);
        svfloat64_t vec_scalar = svdup_f64_z(pg, scalar);
        svfloat64_t vec_a, vec_b, vec_c;
        do
        {
            vec_b = svldnt1(pg, b + idx);
            vec_c = svldnt1(pg, c + idx);
            vec_a = svmad_f64_z(pg, vec_c, vec_scalar, vec_b);
            svstnt1(pg, a + idx, vec_a);
            idx += svcntd();
            pg = svwhilelt_b64(idx, end_idx);
        } while (svptest_any(svptrue_b64(), pg));
    }
}

static inline void zfill(double *a)
{
    asm volatile("dc zva, %0": : "r"(a));
}

void triad_sve512_unroll4_st(const int n, const double scalar, double *a, const double *b, const double *c)
{
    // a[0 : n0-1] are in an incomplete cache line
    size_t a_addr = (size_t) a;
    size_t a_addr_cl = (a_addr + 256 - 1) / 256 * 256;
    int n0 = (a_addr_cl - a_addr) / sizeof(double);

    // a[n0 : n1-1] are in complete cache lines
    int ncl = sizeof(double) * (n - n0) / 256;
    int n1  = n0 + ncl * 256 / sizeof(double);

    // Handle the first part and the last part
    #pragma omp simd
    for (int i = 0; i < n0; i++) a[i] = b[i] + scalar * c[i];
    #pragma omp simd
    for (int i = n1; i < n; i++) a[i] = b[i] + scalar * c[i];

    // Handle complete cache lines
    a += n0;
    b += n0;
    c += n0;
    svbool_t ptrue64b = svptrue_b64();
    svfloat64_t vec_scalar = svdup_f64_z(ptrue64b, scalar);
    const int zfill_distance = 12;
    for (int icl = 0; icl < ncl; icl++)
    {
        if (icl < ncl - zfill_distance) zfill(a + zfill_distance * 32);
        svfloat64_t vec_b0 = svld1(ptrue64b, b + 8 * 0);
        svfloat64_t vec_b1 = svld1(ptrue64b, b + 8 * 1);
        svfloat64_t vec_b2 = svld1(ptrue64b, b + 8 * 2);
        svfloat64_t vec_b3 = svld1(ptrue64b, b + 8 * 3);
        svfloat64_t vec_c0 = svld1(ptrue64b, c + 8 * 0);
        svfloat64_t vec_c1 = svld1(ptrue64b, c + 8 * 1);
        svfloat64_t vec_c2 = svld1(ptrue64b, c + 8 * 2);
        svfloat64_t vec_c3 = svld1(ptrue64b, c + 8 * 3);
        svfloat64_t vec_a0 = svmad_f64_z(ptrue64b, vec_scalar, vec_c0, vec_b0);
        svfloat64_t vec_a1 = svmad_f64_z(ptrue64b, vec_scalar, vec_c1, vec_b1);
        svfloat64_t vec_a2 = svmad_f64_z(ptrue64b, vec_scalar, vec_c2, vec_b2);
        svfloat64_t vec_a3 = svmad_f64_z(ptrue64b, vec_scalar, vec_c3, vec_b3);
        svst1(ptrue64b, a + 8 * 0, vec_a0);
        svst1(ptrue64b, a + 8 * 1, vec_a1);
        svst1(ptrue64b, a + 8 * 2, vec_a2);
        svst1(ptrue64b, a + 8 * 3, vec_a3);
        a += 32;
        b += 32;
        c += 32;
    }
}

void triad_sve512_unroll4()
{
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const size_t start_idx = thread_displs[tid];
        const size_t end_idx   = thread_displs[tid + 1];
        triad_sve512_unroll4_st(end_idx - start_idx, scalar, a + start_idx, b + start_idx, c + start_idx);
    }
}

int main(int argc, char **argv)
{
    int nthread = omp_get_max_threads();
    printf("Size of test arrays      : %zu\n", (size_t) STREAM_ARRAY_SIZE);
    printf("Bytes per array element  : %zu\n", sizeof(STREAM_TYPE));
    printf("Number of threads to use : %d\n", nthread);
    printf("Number of tests to run   : %d\n", NTIMES);
    double MB = (double) STREAM_ARRAY_SIZE * (double) sizeof(STREAM_TYPE) / 1024.0 / 1024.0;
    double GB = MB / 1024.0;
    printf("Memory per array         : %.1f MiB (= %.1f GiB)\n", MB, GB);

    thread_displs = (size_t *) malloc(sizeof(size_t) * (nthread + 1));
    size_t rem = STREAM_ARRAY_SIZE % nthread;
    size_t bs0 = STREAM_ARRAY_SIZE / nthread;
    size_t bs1 = bs0 + 1;
    for (size_t i = 0; i <= nthread; i++)
        thread_displs[i] = (i < rem) ? (bs1 * i) : (bs0 * i + rem);

    // NUMA first touch initialization
    #pragma omp parallel num_threads(nthread)
    {
        int tid = omp_get_thread_num();
        for (size_t i = thread_displs[tid]; i < thread_displs[tid + 1]; i++)
        {
            a[i] = 1.0;
            b[i] = 2.0;
            c[i] = 0.0;
        }
    }

    // Main timing tests
    double start_t, stop_t, used_sec;
    for (int j = 0; j < NFUNCTION; j++)
    {
        max_time[j] = 0.0;
        min_time[j] = 19241112.0;
        avg_time[j] = 0.0;
    }
    for (int k = 0; k <= NTIMES; k++)
    {
        start_t  = get_wtime_sec();
        triad_autovec();
        stop_t   = get_wtime_sec();
        used_sec = stop_t - start_t;
        if (k > 0)
        {
            max_time[0] = MAX(max_time[0], used_sec);
            min_time[0] = MIN(min_time[0], used_sec);
            avg_time[0] += used_sec;
        }

        start_t  = get_wtime_sec();
        triad_sve();
        stop_t   = get_wtime_sec();
        used_sec = stop_t - start_t;
        if (k > 0)
        {
            max_time[1] = MAX(max_time[1], used_sec);
            min_time[1] = MIN(min_time[1], used_sec);
            avg_time[1] += used_sec;
        }

        start_t  = get_wtime_sec();
        triad_sve512_unroll4();
        stop_t   = get_wtime_sec();
        used_sec = stop_t - start_t;
        if (k > 0)
        {
            max_time[2] = MAX(max_time[2], used_sec);
            min_time[2] = MIN(min_time[2], used_sec);
            avg_time[2] += used_sec;
        }

    }

    // Print results
    double bytes[NFUNCTION];
    bytes[0] = 3.0 * (double) sizeof(STREAM_TYPE) * (double) STREAM_ARRAY_SIZE;
    bytes[1] = 3.0 * (double) sizeof(STREAM_TYPE) * (double) STREAM_ARRAY_SIZE;
    bytes[2] = 3.0 * (double) sizeof(STREAM_TYPE) * (double) STREAM_ARRAY_SIZE;
    printf("\nFunction                             Best MB/s    Avg time     Min time     Max time\n");
    for (int j = 0; j < NFUNCTION; j++)
    {
        avg_time[j] /= (double) NTIMES;
        printf(
            "%s%12.1f  %11.6f  %11.6f  %11.6f\n", label[j],
            1.0E-06 * bytes[j] / min_time[j], avg_time[j], min_time[j], max_time[j]
        );
    }
    printf("\n");

    free(thread_displs);
    return 0;
}