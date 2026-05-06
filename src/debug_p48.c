#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "warp-room.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* Copy of p48_dist_sq for diagnostic */
static int p48_dist_sq_dbg(const uint64_t *a, const uint64_t *b, int n) {
    int sum = 0;
    for (int v = 0; v < n; v++) {
        uint64_t pa = a[v], pb = b[v];
        printf("  vec[%d]: a=0x%016lx b=0x%016lx\n", v, pa, pb);
        for (int i = 0; i < 8; i++) {
            int ca = (pa >> (6 * i)) & 0x3F;
            int cb = (pb >> (6 * i)) & 0x3F;
            if (ca || cb) {
                int idx = v * 8 + i;
                int d = ca - cb;
                printf("    [%d] %3d - %3d = %3d, sq=%d\n", idx, ca, cb, d, d*d);
            }
            int ca2 = (pa >> (6 * i)) & 0x3F;
            int cb2 = (pb >> (6 * i)) & 0x3F;
            int d2 = ca2 - cb2;
            sum += d2 * d2;
        }
    }
    return sum;
}

int main(void) {
    /* Create local room table */
    struct room_table tbl;
    memset(&tbl, 0, sizeof(tbl));
    tbl.num_rooms = NUM_ROOMS;
    tbl.version = 1;
    
    strcpy(tbl.rooms[ROOM_EDGE].name, "edge");
    strcpy(tbl.rooms[ROOM_RESEARCH].name, "research");
    strcpy(tbl.rooms[ROOM_FLEET].name, "fleet");
    strcpy(tbl.rooms[ROOM_JC1].name, "jc1");
    
    /* Edge: 26 keywords at 1/sqrt(26) */
    float edge_w = 1.0f / sqrtf(26.0f);
    for (int i = 0; i < 26; i++) {
        int v = (int)(edge_w * 63.0f + 0.5f);
        int vi = i / 8, bi = (i % 8) * 6;
        tbl.rooms[ROOM_EDGE].vector[vi] |= ((uint64_t)(v & 0x3F)) << bi;
    }
    
    /* JC1: 13 keywords at 1/sqrt(13) */
    float jc1_w = 1.0f / sqrtf(13.0f);
    for (int i = 0; i < 13; i++) {
        int idx = 77 + i;
        int v = (int)(jc1_w * 63.0f + 0.5f);
        int vi = idx / 8, bi = (idx % 8) * 6;
        tbl.rooms[ROOM_JC1].vector[vi] |= ((uint64_t)(v & 0x3F)) << bi;
    }
    
    printf("Edge weight (1/sqrt(26)) = %.4f -> P48: %d\n", edge_w, (int)(edge_w * 63 + 0.5f));
    printf("JC1 weight (1/sqrt(13)) = %.4f -> P48: %d\n", jc1_w, (int)(jc1_w * 63 + 0.5f));
    printf("\nEdge room vector[0]: 0x%016lx\n", tbl.rooms[ROOM_EDGE].vector[0]);
    printf("JC1 room vector[9]: 0x%016lx\n", tbl.rooms[ROOM_JC1].vector[9]);
    
    /* Query: "edge gpu" — keyword at indices 2, 11 */
    float q[VOCAB_DIM] = {0};
    q[2] = 1.0f; q[11] = 1.0f;  /* "gpu", "nvidia" indices in edge_kw */
    float norm = 0;
    for (int i = 0; i < VOCAB_DIM; i++) norm += q[i]*q[i];
    norm = sqrtf(norm);
    for (int i = 0; i < VOCAB_DIM; i++) q[i] /= norm;
    
    printf("\nQuery [2]=%.4f [11]=%.4f (norm=%f)\n", q[2], q[11], norm);
    
    /* Quantize query */
    uint64_t qv[P48_DIMS] = {0};
    for (int i = 0; i < VOCAB_DIM; i++) {
        int v = (int)(q[i] * 63.0f + 0.5f);
        if (v < 0) v = 0; if (v > 63) v = 63;
        int vi = i / 8, bi = (i % 8) * 6;
        qv[vi] &= ~((uint64_t)0x3F << bi);
        qv[vi] |= ((uint64_t)(v & 0x3F)) << bi;
    }
    
    printf("Query P48: ");
    for (int vi = 0; vi < P48_DIMS; vi++) {
        if (qv[vi]) printf("[%d]=0x%016lx ", vi, qv[vi]);
    }
    printf("\n\n=== P48 distance to each room ===\n");
    
    for (int ri = 0; ri < NUM_ROOMS; ri++) {
        printf("\n== %s ==\n", tbl.rooms[ri].name);
        int d = p48_dist_sq_dbg(qv, tbl.rooms[ri].vector, P48_DIMS);
        printf("  TOTAL DISTANCE: %d\n", d);
    }
    
    return 0;
}
