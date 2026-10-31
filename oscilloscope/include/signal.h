/* include/signal.h */

#ifndef SIGNAL_H
#define SIGNAL_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>

/* Mono sample buffer. The storage is owned by the caller and is never
 * allocated or freed by library code. */
typedef struct {
    float *samples;
    size_t count;
    double sample_rate_hz;
} SignalBuffer;

static inline int signal_buffer_init(SignalBuffer *buffer, float *storage, size_t count,
                                     double sample_rate_hz) {
    if (buffer == NULL || storage == NULL || count == 0) {
        return -1;
    }
    if (!isfinite(sample_rate_hz) || sample_rate_hz <= 0.0) {
        return -2;
    }
    buffer->samples = storage;
    buffer->count = count;
    buffer->sample_rate_hz = sample_rate_hz;
    return 0;
}

static inline bool signal_buffer_is_valid(const SignalBuffer *buffer) {
    if (buffer == NULL || buffer->samples == NULL || buffer->count == 0) {
        return false;
    }
    return isfinite(buffer->sample_rate_hz) && buffer->sample_rate_hz > 0.0;
}

static inline double signal_buffer_duration_s(const SignalBuffer *buffer) {
    if (!signal_buffer_is_valid(buffer)) {
        return 0.0;
    }
    return (double)buffer->count / buffer->sample_rate_hz;
}

#endif /* SIGNAL_H */
