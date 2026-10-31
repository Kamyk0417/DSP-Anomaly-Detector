/* src/filter.c */

#include "filter.h"

#include <math.h>
#include <string.h>

#define FILTER_PI 3.141592653589793238462643383279

static bool filter_cutoff_is_valid(double cutoff_hz, double sample_rate_hz) {
    if (!isfinite(cutoff_hz) || !isfinite(sample_rate_hz) || !(sample_rate_hz > 0.0)) {
        return false;
    }
    return cutoff_hz > 0.0 && cutoff_hz < 0.5 * sample_rate_hz;
}

/* Bilinear transform s = (1 - z^-1) / (1 + z^-1) of an analog lowpass biquad. */
static int filter_bilinear_biquad(double b2, double b1, double b0, double a2, double a1,
                                  double a0, BiquadCoeffs *out) {
    const double denominator = a2 + a1 + a0;
    if (!isfinite(denominator) || fabs(denominator) < 1.0e-12) {
        return -1;
    }

    out->b0 = (b2 + b1 + b0) / denominator;
    out->b1 = 2.0 * (b0 - b2) / denominator;
    out->b2 = (b2 - b1 + b0) / denominator;
    out->a1 = 2.0 * (a0 - a2) / denominator;
    out->a2 = (a2 - a1 + a0) / denominator;
    return 0;
}

/* Bilinear transform of an analog lowpass first-order section. */
static int filter_bilinear_first_order(double b1, double b0, double a1, double a0,
                                       BiquadCoeffs *out) {
    const double denominator = a1 + a0;
    if (!isfinite(denominator) || fabs(denominator) < 1.0e-12) {
        return -1;
    }

    out->b0 = (b0 + b1) / denominator;
    out->b1 = (b0 - b1) / denominator;
    out->b2 = 0.0;
    out->a1 = (a0 - a1) / denominator;
    out->a2 = 0.0;
    return 0;
}

/* One conjugate pole pair of the unit-cutoff analog prototype, mapped to a
 * digital biquad. Chebyshev II contributes a conjugate zero pair as well. */
static int filter_design_pair(FilterIirType type, size_t order, double chebyshev_v,
                              size_t pair_index, double warped_cutoff, BiquadCoeffs *out) {
    const double theta = FILTER_PI * (2.0 * (double)pair_index + 1.0) / (2.0 * (double)order);

    double scale_sinh = 1.0;
    double scale_cosh = 1.0;
    if (type != FILTER_IIR_BUTTERWORTH) {
        scale_sinh = sinh(chebyshev_v);
        scale_cosh = cosh(chebyshev_v);
    }

    double pole_re = -scale_sinh * sin(theta);
    double pole_im = scale_cosh * cos(theta);
    if (type == FILTER_IIR_CHEBYSHEV2) {
        const double magnitude = pole_re * pole_re + pole_im * pole_im;
        if (!(magnitude > 0.0)) {
            return -1;
        }
        pole_re = pole_re / magnitude;
        pole_im = -pole_im / magnitude;
    }
    pole_re *= warped_cutoff;
    pole_im *= warped_cutoff;

    double b2 = 0.0;
    double b1 = 0.0;
    double b0 = 1.0;
    if (type == FILTER_IIR_CHEBYSHEV2) {
        const double zero_omega = warped_cutoff / cos(theta);
        b2 = 1.0;
        b0 = zero_omega * zero_omega;
    }

    const double a2 = 1.0;
    const double a1 = -2.0 * pole_re;
    const double a0 = pole_re * pole_re + pole_im * pole_im;
    return filter_bilinear_biquad(b2, b1, b0, a2, a1, a0, out);
}

/* Real pole of an odd-order prototype, mapped to a first-order section. */
static int filter_design_real_pole(FilterIirType type, double chebyshev_v, double warped_cutoff,
                                   BiquadCoeffs *out) {
    double pole = -1.0;
    if (type != FILTER_IIR_BUTTERWORTH) {
        pole = -sinh(chebyshev_v);
    }
    if (type == FILTER_IIR_CHEBYSHEV2) {
        pole = 1.0 / pole;
    }
    pole *= warped_cutoff;

    return filter_bilinear_first_order(0.0, 1.0, 1.0, -pole, out);
}

int filter_design_fir_lowpass(FirCoeffs *coeffs, double cutoff_hz, double sample_rate_hz,
                              size_t order, WindowType window) {
    if (coeffs == NULL) {
        return -1;
    }
    memset(coeffs, 0, sizeof(*coeffs));
    if (!filter_cutoff_is_valid(cutoff_hz, sample_rate_hz)) {
        return -2;
    }
    if (order < 1U || order + 1U > FILTER_MAX_FIR_TAPS) {
        return -3;
    }
    if (!window_type_is_valid(window)) {
        return -4;
    }

    const size_t tap_count = order + 1U;
    const double normalized_cutoff = cutoff_hz / sample_rate_hz;
    const double center = (double)order / 2.0;

    double window_values[FILTER_MAX_FIR_TAPS];
    const int window_status = window_generate(window, window_values, tap_count);
    if (window_status != 0) {
        return window_status;
    }

    double sum = 0.0;
    for (size_t i = 0; i < tap_count; ++i) {
        const double offset = 2.0 * normalized_cutoff * ((double)i - center);
        const double ideal = (fabs(offset) < 1.0e-12)
                                 ? 2.0 * normalized_cutoff
                                 : 2.0 * normalized_cutoff * sin(FILTER_PI * offset) /
                                       (FILTER_PI * offset);
        coeffs->taps[i] = ideal * window_values[i];
        sum += coeffs->taps[i];
    }

    if (!isfinite(sum) || fabs(sum) < 1.0e-12) {
        return -5;
    }
    for (size_t i = 0; i < tap_count; ++i) {
        coeffs->taps[i] /= sum;
    }
    coeffs->tap_count = tap_count;
    return 0;
}

int filter_design_iir_lowpass(FilterCoeffs *coeffs, FilterIirType type, double cutoff_hz,
                              double sample_rate_hz, size_t order, double ripple_db) {
    if (coeffs == NULL) {
        return -1;
    }
    memset(coeffs, 0, sizeof(*coeffs));
    if (!filter_cutoff_is_valid(cutoff_hz, sample_rate_hz)) {
        return -2;
    }
    if ((unsigned)type >= (unsigned)FILTER_IIR_TYPE_COUNT) {
        return -3;
    }
    if (order < 1U || order > FILTER_MAX_IIR_ORDER) {
        return -4;
    }
    if (!isfinite(ripple_db)) {
        return -5;
    }

    double chebyshev_v = 0.0;
    if (type != FILTER_IIR_BUTTERWORTH) {
        if (!(ripple_db > 0.0) || ripple_db > 120.0) {
            return -6;
        }
        const double factor = pow(10.0, ripple_db / 10.0) - 1.0;
        if (!(factor > 0.0)) {
            return -6;
        }
        const double epsilon = (type == FILTER_IIR_CHEBYSHEV1) ? sqrt(factor) : 1.0 / sqrt(factor);
        chebyshev_v = asinh(1.0 / epsilon) / (double)order;
    }

    const double warped_cutoff = tan(FILTER_PI * cutoff_hz / sample_rate_hz);
    if (!isfinite(warped_cutoff) || !(warped_cutoff > 0.0)) {
        return -7;
    }

    size_t section_count = 0;
    for (size_t pair_index = 0; pair_index < order / 2U; ++pair_index) {
        const int status = filter_design_pair(type, order, chebyshev_v, pair_index,
                                              warped_cutoff, &coeffs->sections[section_count]);
        if (status != 0) {
            return status;
        }
        ++section_count;
    }
    if (order % 2U != 0U) {
        const int status = filter_design_real_pole(type, chebyshev_v, warped_cutoff,
                                                   &coeffs->sections[section_count]);
        if (status != 0) {
            return status;
        }
        ++section_count;
    }

    /* Butterworth and Chebyshev II have unity DC gain. Even-order Chebyshev I
     * oscillates between 0 dB and -ripple_db, and its DC point sits on the
     * lower bound, so the passband maximum is normalized to 0 dB instead. */
    double target_dc_gain = 1.0;
    if (type == FILTER_IIR_CHEBYSHEV1 && order % 2U == 0U) {
        target_dc_gain = pow(10.0, -ripple_db / 20.0);
    }

    double dc_gain = 1.0;
    for (size_t i = 0; i < section_count; ++i) {
        const BiquadCoeffs *section = &coeffs->sections[i];
        dc_gain *= (section->b0 + section->b1 + section->b2) / (1.0 + section->a1 + section->a2);
    }
    if (!isfinite(dc_gain) || fabs(dc_gain) < 1.0e-12) {
        return -8;
    }
    const double scale = target_dc_gain / dc_gain;
    coeffs->sections[0].b0 *= scale;
    coeffs->sections[0].b1 *= scale;
    coeffs->sections[0].b2 *= scale;

    coeffs->section_count = section_count;
    return 0;
}

bool filter_fir_is_valid(const FirCoeffs *coeffs) {
    if (coeffs == NULL || coeffs->tap_count == 0U || coeffs->tap_count > FILTER_MAX_FIR_TAPS) {
        return false;
    }
    for (size_t i = 0; i < coeffs->tap_count; ++i) {
        if (!isfinite(coeffs->taps[i])) {
            return false;
        }
    }
    return true;
}

bool filter_iir_is_valid(const FilterCoeffs *coeffs) {
    if (coeffs == NULL || coeffs->section_count == 0U ||
        coeffs->section_count > FILTER_MAX_SECTIONS) {
        return false;
    }
    for (size_t i = 0; i < coeffs->section_count; ++i) {
        const BiquadCoeffs *section = &coeffs->sections[i];
        if (!isfinite(section->b0) || !isfinite(section->b1) || !isfinite(section->b2) ||
            !isfinite(section->a1) || !isfinite(section->a2)) {
            return false;
        }
    }
    return true;
}

int filter_apply_fir(const FirCoeffs *coeffs, const float *in, float *out, size_t count) {
    if (!filter_fir_is_valid(coeffs) || in == NULL || out == NULL || count == 0U) {
        return -1;
    }

    const size_t tap_count = coeffs->tap_count;
    double history[FILTER_MAX_FIR_TAPS];
    memset(history, 0, tap_count * sizeof(double));
    size_t position = 0;

    for (size_t i = 0; i < count; ++i) {
        history[position] = (double)in[i];
        double accumulator = 0.0;
        size_t index = position;
        for (size_t k = 0; k < tap_count; ++k) {
            accumulator += coeffs->taps[k] * history[index];
            index = (index == 0U) ? tap_count - 1U : index - 1U;
        }
        out[i] = (float)accumulator;
        position = (position + 1U) % tap_count;
    }
    return 0;
}

int filter_apply_iir_sos(const FilterCoeffs *coeffs, const float *in, float *out, size_t count) {
    if (!filter_iir_is_valid(coeffs) || in == NULL || out == NULL || count == 0U) {
        return -1;
    }

    for (size_t section_index = 0; section_index < coeffs->section_count; ++section_index) {
        const BiquadCoeffs *section = &coeffs->sections[section_index];
        const float *source = (section_index == 0U) ? in : out;
        double delay1 = 0.0;
        double delay2 = 0.0;

        for (size_t i = 0; i < count; ++i) {
            const double x = (double)source[i];
            const double y = section->b0 * x + delay1;
            delay1 = section->b1 * x - section->a1 * y + delay2;
            delay2 = section->b2 * x - section->a2 * y;
            out[i] = (float)y;
        }
    }
    return 0;
}

