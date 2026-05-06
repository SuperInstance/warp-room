#ifndef WARP_CONSTRAINTS_H
#define WARP_CONSTRAINTS_H

/*
 * FLUX Constraint Boundaries for Warp-Room Classification
 *
 * INT8 saturated constraint checks for the warp-room classifier.
 * These validate that classification scores are within acceptable bounds
 * before routing queries to rooms. Zero floating point in the hot path.
 *
 * Author: Forgemaster ⚒️
 * Uses: INT8 saturated [-127, 127] arithmetic, same as CUDA production kernel
 */

#include <stdint.h>

/* ---- INT8 Saturation ---- */
static inline int32_t sat8(int32_t v) {
    return v < -127 ? -127 : (v > 127 ? 127 : v);
}

/* ---- Classification Constraint Bounds ----
 *
 * P48 dot products produce integer scores. We constrain them:
 * - MIN_DOT: reject if score too low (garbage input)
 * - MIN_CONFIDENCE_GAP: top-1 must beat top-2 by this margin
 * - MAX_AMBIGUITY: if top-2 rooms are too close, flag as ambiguous
 *
 * All bounds are INT8-safe: [-127, 127]
 */

#define WR_MIN_DOT_SCORE       10   /* minimum valid dot product score */
#define WR_MIN_CONFIDENCE_GAP   5   /* top-1 must beat top-2 by ≥5 */
#define WR_MAX_AMBIGUITY_SCORE  3   /* gap ≤ 3 means ambiguous */
#define WR_MAX_TOTAL_SCORE    127   /* saturation ceiling */

/* ---- Constraint Check Result ---- */
typedef struct {
    int32_t pass;           /* 1 = all constraints satisfied, 0 = violated */
    int32_t error_mask;     /* bit mask: which constraints failed */
    int32_t confidence;     /* sat8(dot_top1 - dot_top2) */
    int32_t quality;        /* sat8(dot_top1) */
} wr_constraint_result_t;

/* Error mask bits */
#define WR_ERR_MIN_DOT        (1 << 0)  /* score below minimum */
#define WR_ERR_CONFIDENCE_GAP (1 << 1)  /* insufficient gap between top-1/top-2 */
#define WR_ERR_AMBIGUITY      (1 << 2)  /* top-2 rooms too similar */
#define WR_ERR_SATURATION     (1 << 3)  /* value saturated (information loss) */

/* ---- Batch Constraint Check (hot path) ----
 *
 * For each classification result, check:
 * 1. Top score meets minimum threshold
 * 2. Gap between top-1 and top-2 is sufficient
 * 3. Result is not ambiguous
 * 4. No saturation occurred
 */
static inline wr_constraint_result_t wr_check_classification(
    int32_t dot_scores[4],      /* P48 dot products for each room */
    int top1_room,              /* index of highest-scoring room */
    int top2_room               /* index of second-highest room */
) {
    wr_constraint_result_t r = {1, 0, 0, 0};

    int32_t score_top1 = sat8(dot_scores[top1_room]);
    int32_t score_top2 = sat8(dot_scores[top2_room]);
    int32_t gap = sat8(score_top1 - score_top2);

    r.confidence = gap;
    r.quality = score_top1;

    /* Check 1: minimum dot score */
    if (score_top1 < WR_MIN_DOT_SCORE) {
        r.pass = 0;
        r.error_mask |= WR_ERR_MIN_DOT;
    }

    /* Check 2: confidence gap */
    if (gap < WR_MIN_CONFIDENCE_GAP) {
        r.pass = 0;
        r.error_mask |= WR_ERR_CONFIDENCE_GAP;
    }

    /* Check 3: ambiguity detection */
    if (gap <= WR_MAX_AMBIGUITY_SCORE) {
        r.error_mask |= WR_ERR_AMBIGUITY;
        /* Note: ambiguity is informational, doesn't fail the check alone */
    }

    /* Check 4: saturation warning */
    if (dot_scores[top1_room] != score_top1 || dot_scores[top2_room] != score_top2) {
        r.error_mask |= WR_ERR_SATURATION;
    }

    return r;
}

/* ---- Batch Check (vectorized for multiple queries) ---- */
static inline void wr_check_batch(
    int32_t (*dot_scores)[4],   /* array of dot score quads */
    int *top1_rooms,
    int *top2_rooms,
    wr_constraint_result_t *results,
    int n_queries
) {
    for (int i = 0; i < n_queries; i++) {
        results[i] = wr_check_classification(
            dot_scores[i], top1_rooms[i], top2_rooms[i]
        );
    }
}

#endif /* WARP_CONSTRAINTS_H */
