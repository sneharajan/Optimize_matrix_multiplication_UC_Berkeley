/*
 * Project 1: Tuning Matrix Multiplication
 * Done By: Sneha Rajan - U95134515
 *          Subramanian Viswanathan - U16797734
 * File Name: dgemm-blocked.c
 */

#include <stdio.h>
#include <stdlib.h>
#include "emmintrin.h"
#include "immintrin.h"
#include "mm_malloc.h"

/* 
 *  Please include compiler name below (you may also include any other modules you would like to be loaded)
 *
 *   COMPILER= gnu
 *
 *    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 *     
 *      CC = icc 
 *      OPT = -O3
 *      CFLAGS = -axSSE4.1 -fast -march=xeon -fomit-frame-pointer -funroll-loops -Wall -std=gnu99 $(OPT)
 *      MKLROOT = /opt/apps/intel/15/composer_xe_2015.2.164/mkl
 *      LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm
 *
 *           */

const char* dgemm_desc = "Simple blocked dgemm.";

/* Performance enhancement technique: Block size and register block size */
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 128
#define REGISTER_BLOCK_SIZE 16
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C) {

  /* Performance enhancement technique: alignment  */
  static double CPrime[BLOCK_SIZE * REGISTER_BLOCK_SIZE] __attribute__((aligned(16)));

  /* Performance enhancement technique: SSE2 SIMD intrinsics */
  __m128d original, temporary = _mm_setzero_pd();
  
  /* Performance enhancement technique: Loop exchange */  
  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < N; ++j) {
      double bLoopExchanged = B[k + j * lda];

      /* Performance enhancement technique: Loop unrolling */
      for (int i = 0; i < M - 7; i += 8) {
        C[i + j * lda] += A[i + k * lda] * bLoopExchanged;
        C[i + j * lda + 1] += A[i + k * lda + 1] * bLoopExchanged;
        C[i + j * lda + 2] += A[i + k * lda + 2] * bLoopExchanged;
        C[i + j * lda + 3] += A[i + k * lda + 3] * bLoopExchanged;
        C[i + j * lda + 4] += A[i + k * lda + 4] * bLoopExchanged;
        C[i + j * lda + 5] += A[i + k * lda + 5] * bLoopExchanged;
        C[i + j * lda + 6] += A[i + k * lda + 6] * bLoopExchanged;
        C[i + j * lda + 7] += A[i + k * lda + 7] * bLoopExchanged;
      }
      /* Unrolling loops for 7 iterations. Do something else for the 8th iteration on */
      if (M % 8 != 0) {
        for (int i = (M - (M % 8)); i < M; i++) {
          C[i + j * lda] += A[i + k * lda] * bLoopExchanged;
        }
      }
    }
  }

  /* Performance enhancement technique: Register block size */
  /* Commenting due to seg fault */
  /* for (int k = 0; k < K; ++k) {
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; i++) { */
        /* Performance enhancement technique: Block size and register block size */
        /* original = _mm_loadu_pd(&C[((i + j) * lda) + k]);
        temporary = _mm_load_pd(&CPrime[j * BLOCK_SIZE + k]);
        temporary = _mm_add_pd(temporary, original);

        _mm_storeu_pd(&C[((i + j) * lda) + k], temporary);
      }
    }
  } */
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm(int lda, double* restrict A, double* restrict B,
    double* restrict C) {
  
  /* Performance enhancement technique: Buffer allocation */
  double *buffer = (double*) _mm_malloc(lda * BLOCK_SIZE * sizeof(double),
      16);
  /* For each row of A, for each block */
  for (int i = 0; i < lda; i += BLOCK_SIZE) {
    int M = min(BLOCK_SIZE, lda - i); /* To keep the matrix calculation within bounds */

    /* For each column of B, for each block */
    for (int j = 0; j < lda; j += BLOCK_SIZE) {
      int N = min(BLOCK_SIZE, lda - j); /* To keep the matrix calculation within bounds */
      
      /* Call block program to do matmul in blocked fashion */
      for (int k = 0; k < lda; k += BLOCK_SIZE) {
        int K = min(BLOCK_SIZE, lda - k);

        /* Perform individual block dgemm */
        do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda,
            C + i + j * lda);
      }
    }
  }
  /* Free the buffer back */
  _mm_free(buffer);
}

