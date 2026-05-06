CC = gcc
CFLAGS = -std=c17 -O2 -Wall -Wextra -pedantic -ffast-math
LDFLAGS = -lm -lrt

SRC = src/warp-room.c
OBJ = warp-room.o
TARGET = warp-room

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGET) *.o

test: $(TARGET)
	./$(TARGET) --infer "GPU temperature 48C on Jetson Orin Nano"
	./$(TARGET) --infer "research findings embedding engine"
	./$(TARGET) --infer "oracle forgemaster matrix bridge heartbeat"

.PHONY: all clean test
