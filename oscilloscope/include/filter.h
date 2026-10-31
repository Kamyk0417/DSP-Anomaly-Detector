/* include/filter.h */

#ifndef FILTER_H
#define FILTER_H

#include <stdbool.h>
#include <stddef.h>

#include "window.h"

#define FILTER_FIR_ORDER_DEFAULT 64U
#define FILTER_IIR_ORDER_DEFAULT 4U
#define FILTER_IIR_RIPPLE_DB_DEFAULT 1.0
#define FILTER_MAX_FIR_TAPS 257U
#define FILTER_MAX_SECTIONS 8U
#define FILTER_MAX_IIR_ORDER (2U * FILTER_MAX_SECTIONS)

typedef enum {
    FILTER_IIR_CHEBYSHEV1 = 0,
    FILTER_IIR_CHEBYSHEV2,
    FILTER_IIR_BUTTERWORTH,
    FILTER_IIR_TYPE_COUNT
} FilterIirType;

#define FILTER_IIR_TYPE_DEFAULT FILTER_IIR_CHEBYSHEV1

/* Symmetric windowed-sinc taps; tap_count = order + 1. */
typedef struct {
    size_t tap_count;
    double taps[FILTER_MAX_FIR_TAPS];
} FirCoeffs;

/* Normalized second-order section (a0 == 1):
 *   y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2] */
typedef struct {
    double b0;
    double b1;
    double b2;
    double a1;
    double a2;
} BiquadCoeffs;

/* Biquad cascade (SOS). A first-order section stores b2 = 0 and a2 = 0. */
typedef struct {
    size_t section_count;
    BiquadCoeffs sections[FILTER_MAX_SECTIONS];
} FilterCoeffs;

/* Windowed-sinc lowpass with unity DC gain; order + 1 symmetric taps. */
int filter_design_fir_lowpass(FirCoeffs *coeffs, double cutoff_hz, double sample_rate_hz,
                              size_t order, WindowType window);

/* ripple_db is the passband ripple for Chebyshev I and the stopband attenuation
 * for Chebyshev II; Butterworth ignores it. The passband maximum is normalized
 * to 0 dB for every type, so even-order Chebyshev I has a DC gain of
 * -ripple_db. */
int filter_design_iir_lowpass(FilterCoeffs *coeffs, FilterIirType type, double cutoff_hz,
                              double sample_rate_hz, size_t order, double ripple_db);

bool filter_fir_is_valid(const FirCoeffs *coeffs);
bool filter_iir_is_valid(const FilterCoeffs *coeffs);

/* Both apply functions keep zero initial state per call and accept in == out. */
int filter_apply_fir(const FirCoeffs *coeffs, const float *in, float *out, size_t count);
int filter_apply_iir_sos(const FilterCoeffs *coeffs, const float *in, float *out, size_t count);

#endif /* FILTER_H */

