/*
 * Test: FLUX Constraint Boundary Checking for Warp-Room Classification
 *
 * Verifies that classification constraint checking works correctly
 * with INT8 saturated arithmetic.
 *
 * Author: Forgemaster ⚒️
 */

#include "src/warp-constraints.h"
#include <stdio.h>
#include <assert.h>

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { tests_total++; printf("  %-50s", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); } while(0)

int main(void) {
    printf("Warp-Room Constraint Boundary Tests\n");
    printf("====================================\n\n");

    /* Test 1: Clear pass — high score, big gap */
    TEST("clear pass (score=80, gap=40)");
    {
        int32_t scores[4] = {80, 40, 20, 10};
        wr_constraint_result_t r = wr_check_classification(scores, 0, 1);
        assert(r.pass == 1);
        assert(r.error_mask == 0);
        assert(r.confidence == 40);
        assert(r.quality == 80);
        PASS();
    }

    /* Test 2: Fail on minimum dot score */
    TEST("fail: score below minimum (score=5)");
    {
        int32_t scores[4] = {5, 3, 1, 0};
        wr_constraint_result_t r = wr_check_classification(scores, 0, 1);
        assert(r.pass == 0);
        assert(r.error_mask & WR_ERR_MIN_DOT);
        PASS();
    }

    /* Test 3: Fail on confidence gap */
    TEST("fail: insufficient gap (gap=3)");
    {
        int32_t scores[4] = {50, 47, 20, 10};
        wr_constraint_result_t r = wr_check_classification(scores, 0, 1);
        assert(r.pass == 0);
        assert(r.error_mask & WR_ERR_CONFIDENCE_GAP);
        assert(r.confidence == 3);
        PASS();
    }

    /* Test 4: Ambiguity flag but still passes */
    TEST("ambiguity warning (gap=6, passes with flag)");
    {
        int32_t scores[4] = {60, 54, 20, 10};
        wr_constraint_result_t r = wr_check_classification(scores, 0, 1);
        assert(r.pass == 1);
        assert(!(r.error_mask & WR_ERR_AMBIGUITY));  /* gap=6 > 3, no ambiguity */
        assert(r.confidence == 6);
        PASS();
    }

    /* Test 5: Saturation detection */
    TEST("saturation detection (score=200 → 127)");
    {
        int32_t scores[4] = {200, 150, 100, 50};
        wr_constraint_result_t r = wr_check_classification(scores, 0, 1);
        assert(r.quality == 127);  /* saturated */
        assert(r.error_mask & WR_ERR_SATURATION);
        PASS();
    }

    /* Test 6: Multiple failures */
    TEST("multiple failures (low score + no gap)");
    {
        int32_t scores[4] = {8, 7, 5, 2};
        wr_constraint_result_t r = wr_check_classification(scores, 0, 1);
        assert(r.pass == 0);
        assert(r.error_mask & WR_ERR_MIN_DOT);
        assert(r.error_mask & WR_ERR_CONFIDENCE_GAP);
        PASS();
    }

    /* Test 7: Exact boundary — minimum dot score */
    TEST("boundary: exact min dot (score=10)");
    {
        int32_t scores[4] = {10, 2, 1, 0};
        wr_constraint_result_t r = wr_check_classification(scores, 0, 1);
        assert(r.pass == 1);  /* exactly at boundary = pass */
        assert(!(r.error_mask & WR_ERR_MIN_DOT));
        PASS();
    }

    /* Test 8: Exact boundary — confidence gap */
    TEST("boundary: exact min gap (gap=5)");
    {
        int32_t scores[4] = {50, 45, 20, 10};
        wr_constraint_result_t r = wr_check_classification(scores, 0, 1);
        assert(r.pass == 1);
        assert(r.confidence == 5);
        PASS();
    }

    /* Test 9: Batch check */
    TEST("batch check (3 queries)");
    {
        int32_t scores[3][4] = {
            {80, 40, 20, 10},  /* pass */
            {8, 7, 5, 2},      /* fail: low score + no gap */
            {60, 30, 15, 5},   /* pass */
        };
        int top1[3] = {0, 0, 0};
        int top2[3] = {1, 1, 1};
        wr_constraint_result_t results[3];

        wr_check_batch(scores, top1, top2, results, 3);

        assert(results[0].pass == 1);
        assert(results[1].pass == 0);
        assert(results[2].pass == 1);
        PASS();
    }

    /* Test 10: INT8 saturation edge cases */
    TEST("INT8 saturation symmetry");
    {
        assert(sat8(-128) == -127);
        assert(sat8(128) == 127);
        assert(sat8(-127) == -127);
        assert(sat8(127) == 127);
        assert(sat8(0) == 0);
        assert(sat8(-200) == -127);
        assert(sat8(200) == 127);
        PASS();
    }

    /* Test 11: Negative dot products */
    TEST("negative scores (anti-correlation)");
    {
        int32_t scores[4] = {-5, -10, -20, -30};
        wr_constraint_result_t r = wr_check_classification(scores, 0, 1);
        assert(r.pass == 0);
        assert(r.error_mask & WR_ERR_MIN_DOT);
        PASS();
    }

    printf("\n  Results: %d/%d passed\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
