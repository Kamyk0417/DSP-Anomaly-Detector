/* src/io.c */

#include "io.h"

#include <stdio.h>
#include <string.h>

#define IO_PATH_LENGTH 1024

static int io_copy_text(char *destination, size_t size, const char *source) {
    if (destination == NULL || size == 0U || source == NULL) {
        return -1;
    }
    size_t index = 0;
    while (index + 1U < size && source[index] != '\0') {
        destination[index] = source[index];
        ++index;
    }
    destination[index] = '\0';
    return 0;
}

static int io_build_path(char *destination, size_t size, const char *base, const char *suffix) {
    if (destination == NULL || base == NULL || suffix == NULL) {
        return -1;
    }
    const int written = snprintf(destination, size, "%s%s", base, suffix);
    if (written < 0 || (size_t)written >= size) {
        return -1;
    }
    return 0;
}

int io_metadata_init(SignalMetadata *metadata) {
    if (metadata == NULL) {
        return -1;
    }
    memset(metadata, 0, sizeof(*metadata));
    io_copy_text(metadata->generator_type, IO_NAME_LENGTH, "sine");
    metadata->amplitude_v = 0.0;
    metadata->frequency_hz = 0.0;
    metadata->phase_deg = 0.0;
    metadata->offset_v = 0.0;
    metadata->duty_cycle = 0.5;
    metadata->snr_db = 0.0;
    metadata->adc_bits = 0U;
    metadata->adc_v_min = 0.0;
    metadata->adc_v_max = 0.0;
    metadata->filter_count = 0U;
    metadata->has_features = false;
    return 0;
}

int io_metadata_add_filter(SignalMetadata *metadata, const char *name) {
    if (metadata == NULL || name == NULL || metadata->filter_count >= IO_MAX_FILTERS) {
        return -1;
    }
    io_copy_text(metadata->filters[metadata->filter_count], IO_NAME_LENGTH, name);
    ++metadata->filter_count;
    return 0;
}

static int io_write_samples(const char *path, const SignalBuffer *signal) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        return -2;
    }

    const size_t written = fwrite(signal->samples, sizeof(float), signal->count, file);
    const int close_status = fclose(file);
    if (written != signal->count || close_status != 0) {
        return -3;
    }
    return 0;
}

static int io_write_metadata(const char *path, const SignalBuffer *signal,
                             const SignalMetadata *metadata) {
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        return -2;
    }

    const double duration_s = (double)signal->count / signal->sample_rate_hz;

    fputs("{\n", file);
    fprintf(file, "  \"version\": %d,\n", IO_METADATA_VERSION);
    fprintf(file, "  \"samples\": %zu,\n", signal->count);
    fprintf(file, "  \"sample_rate_hz\": %g,\n", signal->sample_rate_hz);
    fprintf(file, "  \"duration_s\": %g,\n", duration_s);
    fputs("  \"dtype\": \"float32\",\n", file);
    fputs("  \"endianness\": \"little\",\n", file);
    fputs("  \"channels\": 1,\n", file);

    fputs("  \"generator\": {\n", file);
    fprintf(file, "    \"type\": \"%s\",\n", metadata->generator_type);
    fprintf(file, "    \"amplitude_v\": %g,\n", metadata->amplitude_v);
    fprintf(file, "    \"frequency_hz\": %g,\n", metadata->frequency_hz);
    fprintf(file, "    \"phase_deg\": %g,\n", metadata->phase_deg);
    fprintf(file, "    \"offset_v\": %g,\n", metadata->offset_v);
    fprintf(file, "    \"duty_cycle\": %g,\n", metadata->duty_cycle);
    fprintf(file, "    \"snr_db\": %g,\n", metadata->snr_db);
    fputs("    \"components\": []\n", file);
    fputs("  },\n", file);

    fputs("  \"adc\": {\n", file);
    fprintf(file, "    \"bits\": %u,\n", metadata->adc_bits);
    fprintf(file, "    \"v_min\": %g,\n", metadata->adc_v_min);
    fprintf(file, "    \"v_max\": %g\n", metadata->adc_v_max);
    fputs("  },\n", file);

    if (metadata->filter_count == 0U) {
        fputs("  \"filters_applied\": [],\n", file);
    } else {
        fputs("  \"filters_applied\": [\n", file);
        for (size_t i = 0; i < metadata->filter_count; ++i) {
            const char *comma = (i + 1U < metadata->filter_count) ? "," : "";
            fprintf(file, "    {\"type\": \"%s\", \"params\": {}}%s\n", metadata->filters[i],
                    comma);
        }
        fputs("  ],\n", file);
    }

    /* Feature extraction lives in features.c; main.c fills the summary before
     * calling io_write_signal. The full magnitude spectrum is deliberately not
     * persisted: it is derivable from the samples and would dominate the file
     * size of long records. */
    const SignalFeatureSummary *summary =
        metadata->has_features ? &metadata->features : NULL;
    fputs("  \"features\": {\n", file);
    fprintf(file, "    \"rms\": %g,\n", summary != NULL ? summary->rms : 0.0);
    fprintf(file, "    \"peak_to_peak\": %g,\n",
            summary != NULL ? summary->peak_to_peak : 0.0);
    fprintf(file, "    \"variance\": %g,\n", summary != NULL ? summary->variance : 0.0);
    fprintf(file, "    \"skewness\": %g,\n", summary != NULL ? summary->skewness : 0.0);
    fprintf(file, "    \"kurtosis\": %g,\n", summary != NULL ? summary->kurtosis : 0.0);
    fprintf(file, "    \"dominant_freq_hz\": %g,\n",
            summary != NULL ? summary->dominant_freq_hz : 0.0);

    const size_t band_count =
        (summary != NULL && summary->band_count <= IO_MAX_BANDS) ? summary->band_count : 0U;
    if (band_count == 0U) {
        fputs("    \"band_energies\": [],\n", file);
    } else {
        fputs("    \"band_energies\": [", file);
        for (size_t i = 0; i < band_count; ++i) {
            fprintf(file, "%s%g", (i == 0U) ? "" : ", ", summary->band_energies[i]);
        }
        fputs("],\n", file);
    }
    fputs("    \"fft_magnitude\": []\n", file);
    fputs("  }\n", file);

    fputs("}\n", file);

    const int close_status = fclose(file);
    if (close_status != 0) {
        return -3;
    }
    return 0;
}

int io_write_signal(const char *base_path, const SignalBuffer *signal,
                    const SignalMetadata *metadata) {
    if (base_path == NULL || metadata == NULL || !signal_buffer_is_valid(signal)) {
        return -1;
    }

    char bin_file[IO_PATH_LENGTH];
    char json_file[IO_PATH_LENGTH];
    if (io_build_path(bin_file, sizeof(bin_file), base_path, ".bin") != 0 ||
        io_build_path(json_file, sizeof(json_file), base_path, ".json") != 0) {
        return -4;
    }

    const int sample_status = io_write_samples(bin_file, signal);
    if (sample_status != 0) {
        return sample_status;
    }
    return io_write_metadata(json_file, signal, metadata);
}

