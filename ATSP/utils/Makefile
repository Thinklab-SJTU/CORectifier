CC ?= cc
CFLAGS ?= -O3 -fPIC

libtsp.so: base_methods.c
	$(CC) $(CFLAGS) -shared -o $@ $< -lm

.PHONY: clean
clean:
	rm -f libtsp.so
