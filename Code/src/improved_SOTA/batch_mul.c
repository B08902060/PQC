#include"batch_mul.h"
#include"tmt.h"
#include<stdio.h>
#define vtrn1u16(x,y) vtrn1q_u16(x,y)
#define vtrn2u16(x,y) vtrn2q_u16(x,y)
#define vtrn1u16x2(x,y) ((uint16x8_t)vtrn1q_u32((uint32x4_t)x, (uint32x4_t)y))
#define vtrn2u16x2(x,y) ((uint16x8_t)vtrn2q_u32((uint32x4_t)x, (uint32x4_t)y))
#define vtrn1u16x4(x,y) ((uint16x8_t)vtrn1q_u64((uint64x2_t)x, (uint64x2_t)y))
#define vtrn2u16x4(x,y) ((uint16x8_t)vtrn2q_u64((uint64x2_t)x, (uint64x2_t)y))
// c = aa * bb
#define sb_vmul(c, aa, bb) c = vmulq_u16(aa, bb);
// c = aa - bb
#define sb_vsub(c, aa, bb) c = vsubq_u16(aa,bb);
// c += aa*bb
#define sb_vmla(c, aa, bb) c = vmlaq_u16(c, aa, bb);
// c = aa ^ bb
#define sb_vxor(c, aa, bb) c = veorq_u16(aa, bb);







#define vld(reg,addr) reg=vld1q_u16(addr)
#define vst(reg,addr) vst1q_u16(addr, reg)
#define vadd(c, a, b) c=vaddq_u16(a,b)
#define vsub(c, a, b) c=vsubq_u16(a,b)
#define ROUND 16

#ifdef DEBUG
void print_array(char s[], uint16_t a[], int n){
    printf("%s: ", s);
    for(int i=0; i<n; i++)
        printf("%d ",a[i]);
    printf("\n");
    printf("\n");
}
#endif

void batch_mul(uint16_t* restrict c_in_mem, uint16_t *restrict a_in_mem, uint16_t *restrict b_in_mem){
    uint16_t* c = c_in_mem,*a = a_in_mem,*b = b_in_mem;
    uint16x8_t a0[8],b0[8],ainf[8],binf[8],c0[16],cinf[16];
    uint16x8_t t[8];
    uint16x8_t a1[8],b1[8],c1[16];
    for(int _i=0; _i<ROUND; _i++){

        //evaluate c(0)
        vld(a0[0], &a[0*16]);
        vld(a0[1], &a[1*16]);
        vld(a0[2], &a[2*16]);
        vld(a0[3], &a[3*16]);
        vld(a0[4], &a[4*16]);
        vld(a0[5], &a[5*16]);
        vld(a0[6], &a[6*16]);
        vld(a0[7], &a[7*16]);

        tran(a0[0],a0[1],a0[2],a0[3],a0[4],a0[5],a0[6],a0[7],t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]);

        vld(b0[0], &b[0*16]);
        vld(b0[1], &b[1*16]);
        vld(b0[2], &b[2*16]);
        vld(b0[3], &b[3*16]);
        vld(b0[4], &b[4*16]);
        vld(b0[5], &b[5*16]);
        vld(b0[6], &b[6*16]);
        vld(b0[7], &b[7*16]);

        tran(b0[0],b0[1],b0[2],b0[3],b0[4],b0[5],b0[6],b0[7],t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]);

        mul8x8( a0[0],a0[1],a0[2],a0[3],a0[4],a0[5],a0[6],a0[7],
                b0[0],b0[1],b0[2],b0[3],b0[4],b0[5],b0[6],b0[7],
                c0[0],c0[1],c0[2],c0[3],c0[4],c0[5],c0[6],c0[7],
                c0[8],c0[9],c0[10],c0[11],c0[12],c0[13],c0[14],c0[15]);

        tran(c0[0],c0[1],c0[2],c0[3],c0[4],c0[5],c0[6],c0[7],t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]);
        vst(c0[0], &c[0*32]);
        vst(c0[1], &c[1*32]);
        vst(c0[2], &c[2*32]);
        vst(c0[3], &c[3*32]);
        vst(c0[4], &c[4*32]);
        vst(c0[5], &c[5*32]);
        vst(c0[6], &c[6*32]);
        vst(c0[7], &c[7*32]);
        tran(c0[8],c0[9],c0[10],c0[11],c0[12],c0[13],c0[14],c0[15],t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]);


        //evaluate c(inf)
        vld(ainf[0], &a[0*16+8]);
        vld(ainf[1], &a[1*16+8]);
        vld(ainf[2], &a[2*16+8]);
        vld(ainf[3], &a[3*16+8]);
        vld(ainf[4], &a[4*16+8]);
        vld(ainf[5], &a[5*16+8]);
        vld(ainf[6], &a[6*16+8]);
        vld(ainf[7], &a[7*16+8]);

        tran(ainf[0],ainf[1],ainf[2],ainf[3],ainf[4],ainf[5],ainf[6],ainf[7],t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]);

        vld(binf[0], &b[0*16+8]);
        vld(binf[1], &b[1*16+8]);
        vld(binf[2], &b[2*16+8]);
        vld(binf[3], &b[3*16+8]);
        vld(binf[4], &b[4*16+8]);
        vld(binf[5], &b[5*16+8]);
        vld(binf[6], &b[6*16+8]);
        vld(binf[7], &b[7*16+8]);

        tran(binf[0],binf[1],binf[2],binf[3],binf[4],binf[5],binf[6],binf[7],t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]);

        mul8x8( ainf[0],ainf[1],ainf[2],ainf[3],ainf[4],ainf[5],ainf[6],ainf[7],
                binf[0],binf[1],binf[2],binf[3],binf[4],binf[5],binf[6],binf[7],
                cinf[0],cinf[1],cinf[2],cinf[3],cinf[4],cinf[5],cinf[6],cinf[7],
                cinf[8],cinf[9],cinf[10],cinf[11],cinf[12],cinf[13],cinf[14],cinf[15]);

        tran(cinf[8],cinf[9],cinf[10],cinf[11],cinf[12],cinf[13],cinf[14],cinf[15],t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]);
        vst(cinf[8], &c[0*32+24]);
        vst(cinf[9], &c[1*32+24]);
        vst(cinf[10], &c[2*32+24]);
        vst(cinf[11], &c[3*32+24]);
        vst(cinf[12], &c[4*32+24]);
        vst(cinf[13], &c[5*32+24]);
        vst(cinf[14], &c[6*32+24]);
        vst(cinf[15], &c[7*32+24]);
        tran(cinf[0],cinf[1],cinf[2],cinf[3],cinf[4],cinf[5],cinf[6],cinf[7],t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]);


        //evaluate c(1)
        vld(a1[0], &a[0*16]);
        vld(a1[1], &a[1*16]);
        vld(a1[2], &a[2*16]);
        vld(a1[3], &a[3*16]);
        vld(a1[4], &a[4*16]);
        vld(a1[5], &a[5*16]);
        vld(a1[6], &a[6*16]);
        vld(a1[7], &a[7*16]);

        vld(t[0], &a[0*16+8]);
        vld(t[1], &a[1*16+8]);
        vld(t[2], &a[2*16+8]);
        vld(t[3], &a[3*16+8]);
        vld(t[4], &a[4*16+8]);
        vld(t[5], &a[5*16+8]);
        vld(t[6], &a[6*16+8]);
        vld(t[7], &a[7*16+8]);
        for(int i=0; i<8; i++)
            vadd(a1[i],a1[i],t[i]);
        tran(a1[0],a1[1],a1[2],a1[3],a1[4],a1[5],a1[6],a1[7],t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]);
        
        vld(b1[0], &b[0*16]);
        vld(b1[1], &b[1*16]);
        vld(b1[2], &b[2*16]);
        vld(b1[3], &b[3*16]);
        vld(b1[4], &b[4*16]);
        vld(b1[5], &b[5*16]);
        vld(b1[6], &b[6*16]);
        vld(b1[7], &b[7*16]);

        vld(t[0], &b[0*16+8]);
        vld(t[1], &b[1*16+8]);
        vld(t[2], &b[2*16+8]);
        vld(t[3], &b[3*16+8]);
        vld(t[4], &b[4*16+8]);
        vld(t[5], &b[5*16+8]);
        vld(t[6], &b[6*16+8]);
        vld(t[7], &b[7*16+8]);
        for(int i=0; i<8; i++)
            vadd(b1[i],b1[i],t[i]);
        tran(b1[0],b1[1],b1[2],b1[3],b1[4],b1[5],b1[6],b1[7],t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]);

        mul8x8( a1[0],a1[1],a1[2],a1[3],a1[4],a1[5],a1[6],a1[7],
                b1[0],b1[1],b1[2],b1[3],b1[4],b1[5],b1[6],b1[7],
                c1[0],c1[1],c1[2],c1[3],c1[4],c1[5],c1[6],c1[7],
                c1[8],c1[9],c1[10],c1[11],c1[12],c1[13],c1[14],c1[15]);
        tran(c1[0],c1[1],c1[2],c1[3],c1[4],c1[5],c1[6],c1[7],t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]);
        tran(c1[8],c1[9],c1[10],c1[11],c1[12],c1[13],c1[14],c1[15],t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]);
        for(int i=0; i<16; i++){
            vsub(c1[i],c1[i],c0[i]);
            vsub(c1[i],c1[i],cinf[i]);
        }
        for(int i=0; i<8; i++){
            vadd(c1[i],c1[i],c0[i+8]);
            vadd(c1[i+8],c1[i+8],cinf[i]);
        }
        for(int i=0; i<8; i++){
            vst(c1[i], &c[32*i+8]);
            vst(c1[i+8], &c[32*i+16]);
        }


        a += 128;
        b += 128;
        c += 256;
    }
    return;
}

#ifdef DEBUG
int main(){
    uint16_t a[128],b[128],c[256];
    for(int i=0; i<128; i++)
        a[i] = 1;
    for(int i=0; i<128; i++)
        b[i] = (i%16 == 0)? 1:0;
    batch_mul(c, a, b);
    for(int i=0; i<8; i++){
        for(int j=0; j<32; j++){
            printf("%d ",c[i*32+j]);
        }
        printf("\n");
    }
}
#endif