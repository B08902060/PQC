#include <arm_neon.h>
#include <stdint.h>

void batch_mul(uint16_t* restrict c_in_mem, uint16_t *restrict a_in_mem, uint16_t *restrict b_in_mem);
void tran_mul_tran(uint16_t src_a[64], uint16_t src_b[64], uint16_t dst[128]);