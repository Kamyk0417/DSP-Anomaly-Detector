/* src/fft.c */

#include "fft.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define FFT_PI 3.141592653589793238462643383279

bool fft_is_power_of_two(size_t n) {
    return n != 0U && (n & (n - 1U)) == 0U;
}

size_t fft_next_power_of_two(size_t n) {
    if (n <= 1U) {
        return 1U;
    }
    size_t value = 1U;
    while (value < n) {
        if (value > (SIZE_MAX / 2U)) {
            return 0U;
        }
        value <<= 1U;
    }
    return value;
}

size_t fft_positive_half_size(size_t n) {
    if (n == 0U) {
        return 0U;
    }
    return n / 2U + 1U;
}

bool fft_plan_is_valid(const FftPlan *plan) {
    if (plan == NULL || plan->n < 2U || !fft_is_power_of_two(plan->n)) {
        return false;
    }
    return plan->cos_table != NULL && plan->sin_table != NULL && plan->bit_reversal != NULL;
}

int fft_plan_create(FftPlan *plan, size_t n) {
    if (plan == NULL) {
        return -1;
    }
    plan->n = 0;
    plan->levels = 0;
    plan->cos_table = NULL;
    plan->sin_table = NULL;
    plan->bit_reversal = NULL;

    if (!fft_is_power_of_two(n) || n < 2U) {
        return -2;
    }

    const size_t half = n / 2U;
    plan->cos_table = malloc(half * sizeof(double));
    plan->sin_table = malloc(half * sizeof(double));
    plan->bit_reversal = malloc(n * sizeof(size_t));
    if (plan->cos_table == NULL || plan->sin_table == NULL || plan->bit_reversal == NULL) {
        fft_plan_destroy(plan);
        return -3;
    }

    for (size_t k = 0; k < half; ++k) {
        const double angle = 2.0 * FFT_PI * (double)k / (double)n;
        plan->cos_table[k] = cos(angle);
        plan->sin_table[k] = sin(angle);
    }

    size_t levels = 0;
    for (size_t size = n; size > 1U; size >>= 1U) {
        ++levels;
    }
    plan->levels = levels;
    plan->n = n;

    for (size_t i = 0; i < n; ++i) {
        size_t reversed = 0;
        size_t value = i;
        for (size_t bit = 0; bit < levels; ++bit) {
            reversed = (reversed << 1U) | (value & 1U);
            value >>= 1U;
        }
        plan->bit_reversal[i] = reversed;
    }
    return 0;
}

void fft_plan_destroy(FftPlan *plan) {
    if (plan == NULL) {
        return;
    }
    free(plan->cos_table);
    free(plan->sin_table);
    free(plan->bit_reversal);
    plan->n = 0;
    plan->levels = 0;
    plan->cos_table = NULL;
    plan->sin_table = NULL;
    plan->bit_reversal = NULL;
}

static void fft_butterflies(const FftPlan *plan, float *re, float *im, bool inverse) {
    const size_t n = plan->n;
    for (size_t size = 2; size <= n; size <<= 1U) {
        const size_t half = size / 2U;
        const size_t stride = n / size;
        for (size_t start = 0; start < n; start += size) {
            for (size_t k = 0; k < half; ++k) {
                const size_t twiddle = k * stride;
                const double wr = plan->cos_table[twiddle];
                const double wi = inverse ? plan->sin_table[twiddle] : -plan->sin_table[twiddle];
                const size_t a = start + k;
                const size_t b = a + half;

                const double a_re = (double)re[a];
                const double a_im = (double)im[a];
                const double b_re = (double)re[b];
                const double b_im = (double)im[b];
                const double product_re = wr * b_re - wi * b_im;
                const double product_im = wr * b_im + wi * b_re;

                re[a] = (float)(a_re + product_re);
                im[a] = (float)(a_im + product_im);
                re[b] = (float)(a_re - product_re);
                im[b] = (float)(a_im - product_im);
            }
        }
    }
}

static void fft_bit_reverse_complex(const FftPlan *plan, float *re, float *im) {
    for (size_t i = 0; i < plan->n; ++i) {
        const size_t j = plan->bit_reversal[i];
        if (j > i) {
            const float swap_re = re[i];
            const float swap_im = im[i];
            re[i] = re[j];
            im[i] = im[j];
            re[j] = swap_re;
            im[j] = swap_im;
        }
    }
}

static void fft_bit_reverse_real(const FftPlan *plan, float *re) {
    for (size_t i = 0; i < plan->n; ++i) {
        const size_t j = plan->bit_reversal[i];
        if (j > i) {
            const float swap = re[i];
            re[i] = re[j];
            re[j] = swap;
        }
    }
}

int fft_forward_padded(const FftPlan *plan, const float *in, size_t count, float *out_re,
                       float *out_im) {
    if (!fft_plan_is_valid(plan) || out_re == NULL || out_im == NULL) {
        return -1;
    }
    if (count > plan->n || (count > 0U && in == NULL)) {
        return -2;
    }
    if (in == out_re && count != plan->n) {
        return -3;
    }

    if (in == out_re) {
        fft_bit_reverse_real(plan, out_re);
    } else {
        for (size_t i = 0; i < plan->n; ++i) {
            out_re[plan->bit_reversal[i]] = (i < count) ? in[i] : 0.0f;
        }
    }
    memset(out_im, 0, plan->n * sizeof(float));

    fft_butterflies(plan, out_re, out_im, false);
    return 0;
}

int fft_forward(const FftPlan *plan, const float *in, float *out_re, float *out_im) {
    if (!fft_plan_is_valid(plan)) {
        return -1;
    }
    return fft_forward_padded(plan, in, plan->n, out_re, out_im);
}

int fft_transform(const FftPlan *plan, float *re, float *im, bool inverse) {
    if (!fft_plan_is_valid(plan) || re == NULL || im == NULL) {
        return -1;
    }

    fft_bit_reverse_complex(plan, re, im);
    fft_butterflies(plan, re, im, inverse);

    if (inverse) {
        const float scale = 1.0f / (float)plan->n;
        for (size_t i = 0; i < plan->n; ++i) {
            re[i] *= scale;
            im[i] *= scale;
        }
    }
    return 0;
}

int fft_magnitude(const FftPlan *plan, const float *re, const float *im, float *magnitude) {
    if (!fft_plan_is_valid(plan) || re == NULL || im == NULL || magnitude == NULL) {
        return -1;
    }

    const size_t bins = fft_positive_half_size(plan->n);
    for (size_t k = 0; k < bins; ++k) {
        const double re_value = (double)re[k];
        const double im_value = (double)im[k];
        magnitude[k] = (float)sqrt(re_value * re_value + im_value * im_value);
    }
    return 0;
}

int fft_real_spectrum(const FftPlan *plan, const float *in, size_t count, float *magnitude,
                      float *re, float *im) {
    if (magnitude == NULL || re == NULL || im == NULL) {
        return -1;
    }
    if (magnitude == re || magnitude == im) {
        return -2;
    }

    const int status = fft_forward_padded(plan, in, count, re, im);
    if (status != 0) {
        return status;
    }
    return fft_magnitude(plan, re, im, magnitude);
}

