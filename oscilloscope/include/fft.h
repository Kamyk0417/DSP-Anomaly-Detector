/* include/fft.h */

#ifndef FFT_H
#define FFT_H

#include <stdbool.h>
#include <stddef.h>

/* Cached plan for a radix-2 Cooley-Tukey transform of length `n`. */
typedef struct {
    size_t n;
    size_t levels;
    double *cos_table;
    double *sin_table;
    size_t *bit_reversal;
} FftPlan;

/* `n` must be a power of two and at least 2. Twiddle factors are computed and
 * cached here. */
int fft_plan_create(FftPlan *plan, size_t n);
void fft_plan_destroy(FftPlan *plan);
bool fft_plan_is_valid(const FftPlan *plan);

bool fft_is_power_of_two(size_t n);
/* Smallest power of two >= n, or 0 when that cannot be represented. */
size_t fft_next_power_of_two(size_t n);
/* Number of non-redundant bins of a real transform: n / 2 + 1. */
size_t fft_positive_half_size(size_t n);

/* Real input of exactly plan->n samples. `in` may alias `out_re`. */
int fft_forward(const FftPlan *plan, const float *in, float *out_re, float *out_im);

/* Real input of `count` samples zero-padded to plan->n. `in` may alias
 * `out_re` only when count == plan->n. */
int fft_forward_padded(const FftPlan *plan, const float *in, size_t count, float *out_re,
                       float *out_im);

/* In-place complex transform on natural-order data; the inverse is normalized
 * by 1 / n. */
int fft_transform(const FftPlan *plan, float *re, float *im, bool inverse);

/* |X[k]| for k = 0 .. n / 2 of a zero-padded real input. `re` and `im` are
 * caller-provided workspaces of plan->n elements. */
int fft_real_spectrum(const FftPlan *plan, const float *in, size_t count, float *magnitude,
                      float *re, float *im);

/* |X[k]| for k = 0 .. n / 2 from an existing complex spectrum. */
int fft_magnitude(const FftPlan *plan, const float *re, const float *im, float *magnitude);

#endif /* FFT_H */

