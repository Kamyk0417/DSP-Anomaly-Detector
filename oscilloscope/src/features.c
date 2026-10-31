/* src/features.c */

#include "features.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int features_extractor_create(FeatureExtractor *extractor, size_t fft_size) {
    if (extractor == NULL) {
        return -1;
    }
    extractor->scratch_re = NULL;
    extractor->scratch_im = NULL;
    extractor->magnitude = NULL;
    extractor->plan.n = 0;
    extractor->plan.levels = 0;
    extractor->plan.cos_table = NULL;
    extractor->plan.sin_table = NULL;
    extractor->plan.bit_reversal = NULL;

    if (!fft_is_power_of_two(fft_size) || fft_size < 8U) {
        return -2;
    }
    if (fft_plan_create(&extractor->plan, fft_size) != 0) {
        return -3;
    }

    const size_t bins = fft_positive_half_size(fft_size);
    extractor->scratch_re = malloc(fft_size * sizeof(float));
    extractor->scratch_im = malloc(fft_size * sizeof(float));
    extractor->magnitude = malloc(bins * sizeof(float));
    if (extractor->scratch_re == NULL || extractor->scratch_im == NULL ||
        extractor->magnitude == NULL) {
        features_extractor_destroy(extractor);
        return -4;
    }
    return 0;
}

void features_extractor_destroy(FeatureExtractor *extractor) {
    if (extractor == NULL) {
        return;
    }
    free(extractor->scratch_re);
    free(extractor->scratch_im);
    free(extractor->magnitude);
    extractor->scratch_re = NULL;
    extractor->scratch_im = NULL;
    extractor->magnitude = NULL;
    fft_plan_destroy(&extractor->plan);
}

bool features_extractor_is_valid(const FeatureExtractor *extractor) {
    if (extractor == NULL || !fft_plan_is_valid(&extractor->plan)) {
        return false;
    }
    return extractor->scratch_re != NULL && extractor->scratch_im != NULL &&
           extractor->magnitude != NULL;
}

size_t features_spectrum_size(const FeatureExtractor *extractor) {
    if (!features_extractor_is_valid(extractor)) {
        return 0U;
    }
    return fft_positive_half_size(extractor->plan.n);
}

int features_default_band_edges(double *edges, size_t band_count, double nyquist_hz) {
    if (edges == NULL || band_count == 0U || band_count > FEATURES_MAX_BANDS) {
        return -1;
    }
    if (!isfinite(nyquist_hz) || !(nyquist_hz > 0.0)) {
        return -2;
    }
    for (size_t i = 0; i <= band_count; ++i) {
        edges[i] = (double)i * nyquist_hz / (double)band_count;
    }
    edges[band_count] = nyquist_hz;
    return 0;
}

static void features_time_domain(const SignalBuffer *signal, SignalFeatures *features) {
    const size_t count = signal->count;

    double sum = 0.0;
    float minimum = signal->samples[0];
    float maximum = signal->samples[0];
    for (size_t i = 0; i < count; ++i) {
        const double value = (double)signal->samples[i];
        sum += value;
        if (signal->samples[i] < minimum) {
            minimum = signal->samples[i];
        }
        if (signal->samples[i] > maximum) {
            maximum = signal->samples[i];
        }
    }
    const double mean = sum / (double)count;

    double sum_squares = 0.0;
    double sum_squared_deviations = 0.0;
    double sum_cubed_deviations = 0.0;
    double sum_fourth_deviations = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double value = (double)signal->samples[i];
        const double deviation = value - mean;
        const double squared = deviation * deviation;
        sum_squares += value * value;
        sum_squared_deviations += squared;
        sum_cubed_deviations += squared * deviation;
        sum_fourth_deviations += squared * squared;
    }

    const double variance = sum_squared_deviations / (double)count;
    features->rms = sqrt(sum_squares / (double)count);
    features->peak_to_peak = (double)maximum - (double)minimum;
    features->variance = variance;

    if (variance > 0.0) {
        const double sigma = sqrt(variance);
        features->skewness =
            (sum_cubed_deviations / (double)count) / (sigma * sigma * sigma);
        features->kurtosis =
            (sum_fourth_deviations / (double)count) / (variance * variance) - 3.0;
    }
}

static size_t features_band_index(const double *edges, size_t band_count, double freq_hz) {
    for (size_t band = 0; band < band_count; ++band) {
        if (freq_hz >= edges[band] && freq_hz < edges[band + 1]) {
            return band;
        }
    }
    /* The Nyquist bin belongs to the last band. */
    if (band_count > 0U && freq_hz >= edges[band_count]) {
        return band_count - 1U;
    }
    return band_count;
}

static int features_spectral(FeatureExtractor *extractor, const SignalBuffer *signal,
                             const double *band_edges_hz, size_t band_count,
                             SignalFeatures *features) {
    const size_t fft_size = extractor->plan.n;
    const size_t used = (signal->count < fft_size) ? signal->count : fft_size;

    if (fft_forward_padded(&extractor->plan, signal->samples, used, extractor->scratch_re,
                           extractor->scratch_im) != 0) {
        return -4;
    }
    if (fft_magnitude(&extractor->plan, extractor->scratch_re, extractor->scratch_im,
                      extractor->magnitude) != 0) {
        return -5;
    }

    const size_t bins = fft_positive_half_size(fft_size);
    features->spectrum_bins = bins;
    if (features->spectrum != NULL) {
        for (size_t k = 0; k < bins; ++k) {
            features->spectrum[k] = (double)extractor->magnitude[k];
        }
    }

    double peak = 0.0;
    size_t peak_bin = 0;
    for (size_t k = 1; k < bins; ++k) {
        const double value = (double)extractor->magnitude[k];
        if (value > peak) {
            peak = value;
            peak_bin = k;
        }
    }

    double position = (double)peak_bin;
    if (peak_bin > 0U && peak_bin + 1U < bins) {
        const double left = (double)extractor->magnitude[peak_bin - 1U];
        const double centre = (double)extractor->magnitude[peak_bin];
        const double right = (double)extractor->magnitude[peak_bin + 1U];
        const double denominator = left - 2.0 * centre + right;
        if (fabs(denominator) > 1.0e-12) {
            position += 0.5 * (left - right) / denominator;
        }
    }
    features->dominant_freq_hz = position * signal->sample_rate_hz / (double)fft_size;

    if (band_count > 0U) {
        for (size_t band = 0; band < band_count; ++band) {
            features->band_energies[band] = 0.0;
        }
        for (size_t k = 0; k < bins; ++k) {
            const double freq_hz = (double)k * signal->sample_rate_hz / (double)fft_size;
            const size_t band = features_band_index(band_edges_hz, band_count, freq_hz);
            if (band < band_count) {
                const double value = (double)extractor->magnitude[k];
                features->band_energies[band] += value * value;
            }
        }
    }
    return 0;
}

int features_extract(FeatureExtractor *extractor, const SignalBuffer *signal,
                     const double *band_edges_hz, size_t band_count,
                     SignalFeatures *features) {
    if (features == NULL || !features_extractor_is_valid(extractor) ||
        !signal_buffer_is_valid(signal)) {
        return -1;
    }
    if (band_count > FEATURES_MAX_BANDS) {
        return -2;
    }
    if (band_count > 0U && band_edges_hz == NULL) {
        return -3;
    }

    double *spectrum = features->spectrum;
    memset(features, 0, sizeof(*features));
    features->spectrum = spectrum;

    features_time_domain(signal, features);
    features->band_count = band_count;
    return features_spectral(extractor, signal, band_edges_hz, band_count, features);
}

