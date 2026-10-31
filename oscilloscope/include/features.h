/* include/features.h */

#ifndef FEATURES_H
#define FEATURES_H

#include <stdbool.h>
#include <stddef.h>

#include "fft.h"
#include "signal.h"

#define FEATURES_MAX_BANDS 8U

/* Summary of one signal. `spectrum` is caller-owned storage for the positive
 * half of the magnitude spectrum and may be NULL to skip it. */
typedef struct {
    double rms;
    double peak_to_peak;
    double variance;
    double skewness;
    double kurtosis; /* excess kurtosis: 0 for a normal distribution */
    double dominant_freq_hz;
    size_t band_count;
    double band_energies[FEATURES_MAX_BANDS];
    size_t spectrum_bins;
    double *spectrum;
} SignalFeatures;

/* Owns the FFT plan and the scratch buffers used by features_extract. */
typedef struct {
    FftPlan plan;
    float *scratch_re;
    float *scratch_im;
    float *magnitude;
} FeatureExtractor;

int features_extractor_create(FeatureExtractor *extractor, size_t fft_size);
void features_extractor_destroy(FeatureExtractor *extractor);
bool features_extractor_is_valid(const FeatureExtractor *extractor);

/* Number of magnitude coefficients for the extractor's FFT size: n / 2 + 1. */
size_t features_spectrum_size(const FeatureExtractor *extractor);

/* Fills band_count + 1 ascending edges spanning 0 .. nyquist_hz. */
int features_default_band_edges(double *edges, size_t band_count, double nyquist_hz);

/* Time-domain features use the whole buffer; the spectral part transforms the
 * first min(count, fft_size) samples, zero-padded to the plan length. Band
 * energies are the sums of squared magnitudes whose bin frequency falls inside
 * each band. */
int features_extract(FeatureExtractor *extractor, const SignalBuffer *signal,
                     const double *band_edges_hz, size_t band_count,
                     SignalFeatures *features);

#endif /* FEATURES_H */

