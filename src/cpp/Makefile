# 编 C 预处理为 Python 可加载的 .so(Linux/macOS;Android 端走 jni/CMakeLists)。
# Python 用 ctypes 加载,无需 pybind11 / Cython 等额外依赖。

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -fPIC
LIB     := libhccr_preprocess.so

.PHONY: all clean

all: $(LIB)

$(LIB): preprocess.c preprocess.h
	$(CC) $(CFLAGS) -shared -o $@ preprocess.c

clean:
	rm -f $(LIB) *.o
