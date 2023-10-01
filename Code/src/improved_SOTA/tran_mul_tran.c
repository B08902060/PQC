#include<arm_neon.h>
#include "poly.h"
#include "batch_mul.h"
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


void tran_mul_tran(uint16_t src_a[64], uint16_t src_b[64], uint16_t dst[128]){
    uint16x8_t s0,s1,s2,s3,s4,s5,s6,s7;
    uint16x8_t t0,t1,t2,t3,t4,t5,t6,t7;
    uint16x8_t u0,u1,u2,u3,u4,u5,u6,u7;
    //s:src_a,t:tmp,u:src_b
    //SRC_A:trans 1
    s0=vld1q_u16(src_a+0);
    s1=vld1q_u16(src_a+8);
    s2=vld1q_u16(src_a+16);
    s3=vld1q_u16(src_a+24);
    s4=vld1q_u16(src_a+32);
    s5=vld1q_u16(src_a+40);
    s6=vld1q_u16(src_a+48);
    s7=vld1q_u16(src_a+56);

    t0 = vtrn1u16(s0,s1);
    t1 = vtrn2u16(s0,s1);
    t2 = vtrn1u16(s2,s3);
    t3 = vtrn2u16(s2,s3);
    t4 = vtrn1u16(s4,s5);
    t5 = vtrn2u16(s4,s5);
    t6 = vtrn1u16(s6,s7);
    t7 = vtrn2u16(s6,s7);

    s0 = vtrn1u16x2(t0,t1);
    s1 = vtrn2u16x2(t0,t1);
    s2 = vtrn1u16x2(t2,t3);
    s3 = vtrn2u16x2(t2,t3);
    s4 = vtrn1u16x2(t4,t5);
    s5 = vtrn2u16x2(t4,t5);
    s6 = vtrn1u16x2(t6,t7);
    s7 = vtrn2u16x2(t6,t7);

    t0 = vtrn1u16x2(s0,s2);
    t1 = vtrn1u16x2(s1,s3);
    t2 = vtrn2u16x2(s0,s2);
    t3 = vtrn2u16x2(s1,s3);
    t4 = vtrn1u16x2(s4,s6);
    t5 = vtrn1u16x2(s5,s7);
    t6 = vtrn2u16x2(s4,s6);
    t7 = vtrn2u16x2(s5,s7);

    s0 = vtrn1u16x4(t0,t4);
    s1 = vtrn1u16x4(t2,t6);
    s2 = vtrn1u16x4(t1,t5);
    s3 = vtrn1u16x4(t3,t7);
    s4 = vtrn2u16x4(t0,t4);
    s5 = vtrn2u16x4(t2,t6);
    s6 = vtrn2u16x4(t1,t5);
    s7 = vtrn2u16x4(t3,t7);
    //SRC_B:trans 2
    u0=vld1q_u16(src_b+0);
    u1=vld1q_u16(src_b+8);
    u2=vld1q_u16(src_b+16);
    u3=vld1q_u16(src_b+24);
    u4=vld1q_u16(src_b+32);
    u5=vld1q_u16(src_b+40);
    u6=vld1q_u16(src_b+48);
    u7=vld1q_u16(src_b+56);

    t0 = vtrn1u16(u0,u1);
    t1 = vtrn2u16(u0,u1);
    t2 = vtrn1u16(u2,u3);
    t3 = vtrn2u16(u2,u3);
    t4 = vtrn1u16(u4,u5);
    t5 = vtrn2u16(u4,u5);
    t6 = vtrn1u16(u6,u7);
    t7 = vtrn2u16(u6,u7);

    u0 = vtrn1u16x2(t0,t1);
    u1 = vtrn2u16x2(t0,t1);
    u2 = vtrn1u16x2(t2,t3);
    u3 = vtrn2u16x2(t2,t3);
    u4 = vtrn1u16x2(t4,t5);
    u5 = vtrn2u16x2(t4,t5);
    u6 = vtrn1u16x2(t6,t7);
    u7 = vtrn2u16x2(t6,t7);

    t0 = vtrn1u16x2(u0,u2);
    t1 = vtrn1u16x2(u1,u3);
    t2 = vtrn2u16x2(u0,u2);
    t3 = vtrn2u16x2(u1,u3);
    t4 = vtrn1u16x2(u4,u6);
    t5 = vtrn1u16x2(u5,u7);
    t6 = vtrn2u16x2(u4,u6);
    t7 = vtrn2u16x2(u5,u7);

    u0 = vtrn1u16x4(t0,t4);
    u1 = vtrn1u16x4(t2,t6);
    u2 = vtrn1u16x4(t1,t5);
    u3 = vtrn1u16x4(t3,t7);
    u4 = vtrn2u16x4(t0,t4);
    u5 = vtrn2u16x4(t2,t6);
    u6 = vtrn2u16x4(t1,t5);
    u7 = vtrn2u16x4(t3,t7);
    //MUL
    uint16x8_t c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15;
    //c0:0
    sb_vmul(c0,s0,u0);
    //c1:1
    sb_vmul(c1, s0, u1);
    sb_vmla(c1, s1, u0);
    //c2:2
    sb_vmul(c2, s0, u2);
    sb_vmla(c2, s1, u1); 
    sb_vmla(c2, s2, u0);
    //c3:3
    sb_vmul(c3, s0, u3);
    sb_vmla(c3, s1, u2); 
    sb_vmla(c3, s2, u1); 
    sb_vmla(c3, s3, u0);
    //c4:4
    sb_vmul(c4, s0, u4);
    sb_vmla(c4, s1, u3); 
    sb_vmla(c4, s2, u2); 
    sb_vmla(c4, s3, u1); 
    sb_vmla(c4, s4, u0);
    //c5:5
    sb_vmul(c5, s0, u5);
    sb_vmla(c5, s1, u4);
    sb_vmla(c5, s2, u3); 
    sb_vmla(c5, s3, u2); 
    sb_vmla(c5, s4, u1); 
    sb_vmla(c5, s5, u0);
    //c6:6
    sb_vmul(c6, s0, u6);
    sb_vmla(c6, s1, u5);
    sb_vmla(c6, s2, u4); 
    sb_vmla(c6, s3, u3); 
    sb_vmla(c6, s4, u2); 
    sb_vmla(c6, s5, u1); 
    sb_vmla(c6, s6, u0);
    //c7:7
    sb_vmul(c7, s0, u7);
    sb_vmla(c7, s1, u6);
    sb_vmla(c7, s2, u5); 
    sb_vmla(c7, s3, u4); 
    sb_vmla(c7, s4, u3); 
    sb_vmla(c7, s5, u2); 
    sb_vmla(c7, s6, u1); 
    sb_vmla(c7, s7, u0);
    //c8:8
    sb_vmul(c8, s1, u7);
    sb_vmla(c8, s2, u6); 
    sb_vmla(c8, s3, u5); 
    sb_vmla(c8, s4, u4); 
    sb_vmla(c8, s5, u3); 
    sb_vmla(c8, s6, u2); 
    sb_vmla(c8, s7, u1);
    //c9:9
    sb_vmul(c9, s2, u7);
    sb_vmla(c9, s3, u6); 
    sb_vmla(c9, s4, u5); 
    sb_vmla(c9, s5, u4); 
    sb_vmla(c9, s6, u3); 
    sb_vmla(c9, s7, u2);
    //c10:10
    sb_vmul(c10, s3, u7);
    sb_vmla(c10, s4, u6); 
    sb_vmla(c10, s5, u5); 
    sb_vmla(c10, s6, u4); 
    sb_vmla(c10, s7, u3);
    //c11:11
    sb_vmul(c11, s4, u7);
    sb_vmla(c11, s5, u6); 
    sb_vmla(c11, s6, u5); 
    sb_vmla(c11, s7, u4);
    //c12:12
    sb_vmul(c12, s5, u7);
    sb_vmla(c12, s6, u6); 
    sb_vmla(c12, s7, u5);
    //c13:13
    sb_vmul(c13, s6, u7); 
    sb_vmla(c13, s7, u6);
    //c14:14
    sb_vmul(c14, s7, u7);
    //c15:reset to zeros
    sb_vsub(c15,c14,c14);
    //Trans 2
    //part1
    t0 = vtrn1u16(c0,c1);
    t1 = vtrn2u16(c0,c1);
    t2 = vtrn1u16(c2,c3);
    t3 = vtrn2u16(c2,c3);
    t4 = vtrn1u16(c4,c5);
    t5 = vtrn2u16(c4,c5);
    t6 = vtrn1u16(c6,c7);
    t7 = vtrn2u16(c6,c7);

    c0 = vtrn1u16x2(t0,t1);
    c1 = vtrn2u16x2(t0,t1);
    c2 = vtrn1u16x2(t2,t3);
    c3 = vtrn2u16x2(t2,t3);
    c4 = vtrn1u16x2(t4,t5);
    c5 = vtrn2u16x2(t4,t5);
    c6 = vtrn1u16x2(t6,t7);
    c7 = vtrn2u16x2(t6,t7);

    t0 = vtrn1u16x2(c0,c2);
    t1 = vtrn1u16x2(c1,c3);
    t2 = vtrn2u16x2(c0,c2);
    t3 = vtrn2u16x2(c1,c3);
    t4 = vtrn1u16x2(c4,c6);
    t5 = vtrn1u16x2(c5,c7);
    t6 = vtrn2u16x2(c4,c6);
    t7 = vtrn2u16x2(c5,c7);

    c0 = vtrn1u16x4(t0,t4);
    c1 = vtrn1u16x4(t2,t6);
    c2 = vtrn1u16x4(t1,t5);
    c3 = vtrn1u16x4(t3,t7);
    c4 = vtrn2u16x4(t0,t4);
    c5 = vtrn2u16x4(t2,t6);
    c6 = vtrn2u16x4(t1,t5);
    c7 = vtrn2u16x4(t3,t7);
    
    vst1q_u16(&dst[0],c0);
    vst1q_u16(&dst[16],c1);
    vst1q_u16(&dst[32],c2);
    vst1q_u16(&dst[48],c3);
    vst1q_u16(&dst[64],c4);
    vst1q_u16(&dst[80],c5);
    vst1q_u16(&dst[96],c6);
    vst1q_u16(&dst[112],c7);
    
    //part2
    t0 = vtrn1u16(c8,c9);
    t1 = vtrn2u16(c8,c9);
    t2 = vtrn1u16(c10,c11);
    t3 = vtrn2u16(c10,c11);
    t4 = vtrn1u16(c12,c13);
    t5 = vtrn2u16(c12,c13);
    t6 = vtrn1u16(c14,c15);
    t7 = vtrn2u16(c14,c15);

    c8 = vtrn1u16x2(t0,t1);
    c9 = vtrn2u16x2(t0,t1);
    c10 = vtrn1u16x2(t2,t3);
    c11 = vtrn2u16x2(t2,t3);
    c12 = vtrn1u16x2(t4,t5);
    c13 = vtrn2u16x2(t4,t5);
    c14 = vtrn1u16x2(t6,t7);
    c15 = vtrn2u16x2(t6,t7);

    t0 = vtrn1u16x2(c8,c10);
    t1 = vtrn1u16x2(c9,c11);
    t2 = vtrn2u16x2(c8,c10);
    t3 = vtrn2u16x2(c9,c11);
    t4 = vtrn1u16x2(c12,c14);
    t5 = vtrn1u16x2(c13,c15);
    t6 = vtrn2u16x2(c12,c14);
    t7 = vtrn2u16x2(c13,c15);

    c8 = vtrn1u16x4(t0,t4);
    c9 = vtrn1u16x4(t2,t6);
    c10 = vtrn1u16x4(t1,t5);
    c11 = vtrn1u16x4(t3,t7);
    c12 = vtrn2u16x4(t0,t4);
    c13 = vtrn2u16x4(t2,t6);
    c14 = vtrn2u16x4(t1,t5);
    c15 = vtrn2u16x4(t3,t7);

    vst1q_u16(&dst[8],c8);
    vst1q_u16(&dst[24],c9);
    vst1q_u16(&dst[40],c10);
    vst1q_u16(&dst[56],c11);
    vst1q_u16(&dst[72],c12);
    vst1q_u16(&dst[88],c13);
    vst1q_u16(&dst[104],c14);
    vst1q_u16(&dst[120],c15);

    return;
}
