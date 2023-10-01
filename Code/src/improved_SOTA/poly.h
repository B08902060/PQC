#ifndef POLY_N
#define POLY_N 864
#endif

#include "params.h"

#include <stddef.h>
#include <stdint.h>

#define NTRU_N_32 832
#define NTRU_N_PAD 864

typedef struct{
  uint16_t coeffs[NTRU_N_PAD];
} poly;

void tc3_evaluate_neon_SB1(uint16_t *restrict w[5], uint16_t *restrict _poly);
void tc3_evaluate_neon_combine(uint16_t *restrict w, uint16_t *restrict _poly);
void tc3_interpolate_neon_SB3(uint16_t *restrict _poly, uint16_t *restrict w);
void tc3_interpolate_neon_SB2(uint16_t *restrict _poly, uint16_t *restrict w);
void tc3_interpolate_neon_SB1(uint16_t *restrict _poly, uint16_t *restrict w);
void neon_toom_cook_333_combine(uint16_t *restrict _polyC, uint16_t *restrict polyA, uint16_t *restrict polyB);
void _poly_neon_reduction(uint16_t *poly, uint16_t *tmp);
void karat_neon_evaluate_SB0(uint16_t *restrict w[3], uint16_t *restrict  _poly);
void karat_neon_interpolate_SB0(uint16_t *restrict _poly, uint16_t *restrict w[3]);
void _poly_mul_neon(uint16_t *restrict polyC, uint16_t *restrict polyA, uint16_t *restrict polyB);
void poly_Rq_mul_small(poly *r, poly *a, poly *b);
