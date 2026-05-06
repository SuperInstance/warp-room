/* warp-room.c — Subroutine-threaded tile classifier */
#define _GNU_SOURCE
#include "warp-room.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ---- Shared memory segment ---- */
static struct room_table *g_table = NULL;
static int g_shm_fd = -1;

/* ---- Bag-of-features extraction ---- */
static void text_features(const char *text, float *vec) {
    /* Simplified: counts keyword occurrences, sqrt-weighted */
    memset(vec, 0, VOCAB_DIM * sizeof(float));
    /* Keyword-to-index mapping would go here */
    /* For now: placeholder counting */
    (void)text;
}

/* ---- Training: online learning via EMA ---- */
void wr_train(const char *text, enum room_id room) {
    if (!g_table || room >= NUM_ROOMS) return;
    float vec[VOCAB_DIM];
    text_features(text, vec);

    /* Normalize */
    float norm = 0.0f;
    for (int i = 0; i < VOCAB_DIM; i++) norm += vec[i] * vec[i];
    if (norm < 1e-8f) return;
    norm = sqrtf(norm);
    for (int i = 0; i < VOCAB_DIM; i++) vec[i] /= norm;

    /* EMA update to room vector */
    const float lr = 0.3f;
    struct room *r = &g_table->rooms[room];
    for (int i = 0; i < VOCAB_DIM; i++)
        r->vector[i] = (1.0f - lr) * r->vector[i] + lr * vec[i];

    /* Re-normalize */
    norm = 0.0f;
    for (int i = 0; i < VOCAB_DIM; i++) norm += r->vector[i] * r->vector[i];
    norm = sqrtf(norm);
    if (norm > 1e-8f)
        for (int i = 0; i < VOCAB_DIM; i++) r->vector[i] /= norm;

    __sync_fetch_and_add(&g_table->version, 1);
}

/* ---- Classification ---- */
enum room_id wr_classify(const char *text, float *confidence) {
    if (!g_table) return ROOM_EDGE;

    float vec[VOCAB_DIM];
    text_features(text, vec);

    float norm = 0.0f;
    for (int i = 0; i < VOCAB_DIM; i++) norm += vec[i] * vec[i];
    if (norm < 1e-8f) return ROOM_EDGE;
    norm = sqrtf(norm);
    for (int i = 0; i < VOCAB_DIM; i++) vec[i] /= norm;

    enum room_id best = ROOM_EDGE;
    float best_score = -1.0f;
    for (int i = 0; i < NUM_ROOMS; i++) {
        float dot = 0.0f;
        for (int j = 0; j < VOCAB_DIM; j++)
            dot += vec[j] * g_table->rooms[i].vector[j];
        if (dot > best_score) {
            best_score = dot;
            best = (enum room_id)i;
        }
    }
    if (confidence) *confidence = best_score;
    return best;
}

/* ---- init: create or open shared memory ---- */
void wr_init(void) {
    g_shm_fd = shm_open("/warp-room-vectors", O_CREAT | O_RDWR, 0666);
    if (g_shm_fd < 0) { perror("shm_open"); return; }
    ftruncate(g_shm_fd, sizeof(struct room_table));

    g_table = mmap(NULL, sizeof(struct room_table),
                   PROT_READ | PROT_WRITE, MAP_SHARED, g_shm_fd, 0);
    if (g_table == MAP_FAILED) { perror("mmap"); g_table = NULL; return; }

    /* Initialize if fresh (version == 0) */
    if (g_table->version == 0) {
        g_table->num_rooms = NUM_ROOMS;

        strcpy(g_table->rooms[ROOM_EDGE].name,     "edge");
        strcpy(g_table->rooms[ROOM_RESEARCH].name, "research");
        strcpy(g_table->rooms[ROOM_FLEET].name,    "fleet");
        strcpy(g_table->rooms[ROOM_JC1].name,      "jc1");

        /* Seed with keyword vectors from the Python prototype */
        /* ... initialized from embedded keyword profiles ... */

        g_table->version = 1;
        printf("warp-room: created fresh shared memory (version 1)\n");
    } else {
        printf("warp-room: attached to existing shared memory (version %u)\n",
               g_table->version);
    }
}

/* ---- CLI ---- */
int main(int argc, char **argv) {
    (void)argc; (void)argv;
    wr_init();

    if (argc >= 3 && strcmp(argv[1], "--infer") == 0) {
        float conf;
        enum room_id r = wr_classify(argv[2], &conf);
        printf("{\"room\": \"%s\", \"confidence\": %.4f}\n",
               g_table ? g_table->rooms[r].name : "unknown", conf);
    } else if (argc >= 3 && strcmp(argv[1], "--train") == 0) {
        /* Usage: --train <text> <room_name> */
        if (argc < 4) { fprintf(stderr, "usage: --train <text> <room>\n"); return 1; }
        enum room_id id = ROOM_EDGE;
        for (int i = 0; i < NUM_ROOMS; i++)
            if (strcmp(argv[3], g_table->rooms[i].name) == 0) id = (enum room_id)i;
        wr_train(argv[2], id);
        printf("trained on room %s\n", argv[3]);
    } else {
        printf("Usage: warp-room --infer <text> | --train <text> <room>\n");
    }
    return 0;
}
