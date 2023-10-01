
include make_opt.mk

COMMON_HEADERs = hal.h params.h
COMMON_SOURCEs = hal-cortexa.c

OPT_HEADERs = $(C_HEADERs) $(ASM_HEADERs)
OPT_SOURCEs = $(C_SOURCEs) $(ASM_SOURCEs)

HEADERs = $(COMMON_HEADERs) $(OPT_HEADERs)
SOURCEs = $(COMMON_SOURCEs) $(OPT_SOURCEs)

all: test speed mod_inverse.s

test: $(HEADERs) $(SOURCEs) test.c
	$(CC) $(CFLAGS) -o test test.c $(SOURCEs)

speed: $(HEADERs) $(SOURCEs) speed.c
	$(CC) $(CFLAGS) -o speed speed.c $(SOURCEs)
mod_inverse.s: $(HEADERs) $(SOURCEs) mod_inverse.c
	$(CC) $(CFLAGS) -o mod_inverse.s -S mod_inverse.c


include ./make.mk

.PHONY: clean

clean:
	rm -f test
	rm -f speed


