package com.benodb.trino;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Arithmetic coverage for the shared helpers behind the scalar functions.
 *
 * <p>The SQL entry points take Trino {@code Block}s and are exercised end to end
 * in {@code TestBenoDBQueries}, which asserts registration, resolution,
 * and results for {@code l2_distance}, {@code vector_distance},
 * {@code sparse_dot_product}, {@code hybrid_score} and
 * {@code reciprocal_rank_fusion}. Keeping the pure arithmetic here means a
 * formula change and a wiring change fail in different places.</p>
 */
public class TestBenoDBScalarFunctions {

    @Test
    void cosineDistanceMatchesSpark() {
        assertEquals(0.0, BenoDBScalarFunctions.cosineDistance(List.of(1.0, 0.0), List.of(1.0, 0.0)), 1e-12);
        assertEquals(1.0, BenoDBScalarFunctions.cosineDistance(List.of(1.0, 0.0), List.of(0.0, 1.0)), 1e-12);
        // Spark's `.max(0.0)` only floors the result at zero; it does not cap it
        // at 1, so opposing vectors legitimately yield 2.0.
        assertEquals(2.0, BenoDBScalarFunctions.cosineDistance(List.of(1.0, 0.0), List.of(-1.0, 0.0)), 1e-12);
        // Zero vectors return 0 rather than NaN.
        assertEquals(0.0, BenoDBScalarFunctions.cosineDistance(List.of(0.0), List.of(0.0)), 1e-12);
    }

    @Test
    void l2AndDotMatchSpark() {
        assertEquals(5.0, BenoDBScalarFunctions.l2Distance(List.of(0.0, 0.0), List.of(3.0, 4.0)), 1e-12);
        assertEquals(11.0, BenoDBScalarFunctions.dotProduct(List.of(1.0, 2.0), List.of(3.0, 4.0)), 1e-12);
    }

    @Test
    void hybridScoreClampsAlpha() {
        assertEquals(0.25, BenoDBScalarFunctions.hybridScore(1.0, 0.0, 0.25), 1e-12);
        assertEquals(1.0, BenoDBScalarFunctions.hybridScore(1.0, 0.0, 5.0), 1e-12);
        assertEquals(0.0, BenoDBScalarFunctions.hybridScore(1.0, 0.0, -2.0), 1e-12);
    }

    @Test
    void vectorNormAndNormalizeAreTotal() {
        assertEquals(5.0, BenoDBScalarFunctions.vectorNorm(List.of(3.0, 4.0)), 1e-12);
        assertEquals(List.of(0.6, 0.8), BenoDBScalarFunctions.l2Normalize(List.of(3.0, 4.0)));
        // A zero vector must not divide by zero (NaN/Infinity) — it normalizes to zeros.
        assertEquals(List.of(0.0), BenoDBScalarFunctions.l2Normalize(List.of(0.0)));
        assertEquals(List.of(), BenoDBScalarFunctions.l2Normalize(List.of()));
    }

    @Test
    void subvectorClampsInsteadOfThrowing() {
        List<Double> vector = List.of(10.0, 20.0, 30.0);
        // 1-based offset, like SQL slice().
        assertEquals(List.of(20.0, 30.0), BenoDBScalarFunctions.subvector(vector, 2, 2));
        // Out-of-range and negative bounds are clamped, never an exception.
        assertEquals(List.of(10.0, 20.0, 30.0), BenoDBScalarFunctions.subvector(vector, -5, 100));
        assertEquals(List.of(), BenoDBScalarFunctions.subvector(vector, 99, 2));
        assertEquals(List.of(), BenoDBScalarFunctions.subvector(vector, 1, 0));
        assertEquals(List.of(), BenoDBScalarFunctions.subvector(vector, 1, -3));
        assertEquals(List.of(), BenoDBScalarFunctions.subvector(List.of(), 1, 2));
    }

    @Test
    void vectorAddAndSubStopAtShorterInput() {
        assertEquals(
                List.of(4.0, 6.0),
                BenoDBScalarFunctions.vectorAdd(List.of(1.0, 2.0), List.of(3.0, 4.0)));
        assertEquals(
                List.of(-2.0, -2.0),
                BenoDBScalarFunctions.vectorSub(List.of(1.0, 2.0), List.of(3.0, 4.0)));
        assertEquals(
                List.of(4.0),
                BenoDBScalarFunctions.vectorAdd(List.of(1.0, 2.0), List.of(3.0)));
        assertEquals(List.of(), BenoDBScalarFunctions.vectorAdd(List.of(), List.of(1.0)));
    }
}
