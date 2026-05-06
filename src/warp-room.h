#ifndef WARP_ROOM_H
#define WARP_ROOM_H

#include <stdint.h>
#include <stddef.h>

/* ---- Room profiles ---- */

#define NUM_ROOMS    4
#define VOCAB_DIM    97  /* unigrams + bigrams */

/* Room IDs */
enum room_id {
    ROOM_EDGE     = 0,
    ROOM_RESEARCH = 1,
    ROOM_FLEET    = 2,
    ROOM_JC1      = 3,
};

/* A room's profile: vector + handler */
struct room {
    enum room_id id;
    char         name[32];
    float        vector[VOCAB_DIM]; /* normalized feature vector */
    void         (*handler)(const char *query, char *reply, size_t rlen);
};

/* Shared memory header (mmap'd) */
struct room_table {
    volatile uint32_t version; /* incremented on update */
    uint32_t          num_rooms;
    struct room       rooms[NUM_ROOMS];
};

/* ---- API ---- */
void        wr_init(void);          /* open mmap segment or create */
void        wr_train(const char *text, enum room_id room);
enum room_id wr_classify(const char *text, float *confidence);
void        wr_stats(void);         /* print to stdout */

/* ---- NEON SIMD dispatch (when compiled for ARM64) ---- */
#ifdef __aarch64__
float      wr_dot_neon(const float *a, const float *b, int n);
enum room_id wr_classify_neon(const float *vec);
#endif

#endif /* WARP_ROOM_H */
