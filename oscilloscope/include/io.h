/* include/io.h */

#ifndef IO_H
#define IO_H

#include <stdbool.h>
#include <stddef.h>

#include "signal.h"

#define IO_METADATA_VERSION 1
#define IO_MAX_FILTERS 4
#define IO_MAX_BANDS 8
#define IO_NAME_LENGTH 32

/* Scalar summary of a signal, mirroring the "features" block of schema v1. */
typedef struct {
    double rms;
    double peak_to_peak;
    double variance;
    double skewness;
    double kurtosis;
    double dominant_freq_hz;
    size_t band_count;
    double band_energies[IO_MAX_BANDS];
} SignalFeatureSummary;

/* Everything the metadata file records about how a signal was produced. Plain
 * data only, so io.c stays independent of the generator and filter modules. */
typedef struct {
    char generator_type[IO_NAME_LENGTH];
    double amplitude_v;
    double frequency_hz;
    double phase_deg;
    double offset_v;
    double duty_cycle;
    double snr_db;
    unsigned adc_bits;
    double adc_v_min;
    double adc_v_max;
    size_t filter_count;
    char filters[IO_MAX_FILTERS][IO_NAME_LENGTH];
    bool has_features;
    SignalFeatureSummary features;
} SignalMetadata;

int io_metadata_init(SignalMetadata *metadata);

/* Adds a filter name to the metadata; returns -1 when the list is full. */
int io_metadata_add_filter(SignalMetadata *metadata, const char *name);

/* Writes "<base_path>.bin" (headerless little-endian float32) and
 * "<base_path>.json" (schema version 1). The output directory must exist. */
int io_write_signal(const char *base_path, const SignalBuffer *signal,
                    const SignalMetadata *metadata);

#endif /* IO_H */

