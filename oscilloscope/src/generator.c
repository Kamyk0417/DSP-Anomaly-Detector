/* src/generator.c */

#include "generator.h"

#include <math.h>

#define GENERATOR_TWO_PI 6.283185307179586476925286766559

static const double generator_deg_to_rad = 0.01745329251994329576923690768489;

static uint64_t generator_rng_next(GeneratorRng *rng) {
    rng->state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = rng->state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static int generator_prepare(const GeneratorConfig *config, const SignalBuffer *out) {
    const int config_status = generator_config_validate(config);
    if (config_status != 0) {
        return config_status;
    }
    if (!signal_buffer_is_valid(out)) {
        return -20;
    }
    return 0;
}

/* Position inside the current cycle in [0, 1). */
static double generator_cycle_fraction(double phase_rad) {
    double phase = fmod(phase_rad, GENERATOR_TWO_PI);
    if (phase < 0.0) {
        phase += GENERATOR_TWO_PI;
    }
    return phase / GENERATOR_TWO_PI;
}

int generator_config_default(GeneratorConfig *config) {
    if (config == NULL) {
        return -1;
    }
    config->type = GENERATOR_SINE;
    config->amplitude_v = 2.5;
    config->frequency_hz = 1000.0;
    config->phase_deg = 0.0;
    config->offset_v = 0.0;
    config->duty_cycle = 0.5;
    config->snr_db = 0.0;
    config->component_count = 0;
    for (size_t i = 0; i < GENERATOR_MAX_COMPONENTS; ++i) {
        config->components[i].freq_hz = 0.0;
        config->components[i].amp_v = 0.0;
        config->components[i].phase_deg = 0.0;
    }
    return 0;
}

int generator_config_validate(const GeneratorConfig *config) {
    if (config == NULL) {
        return -1;
    }
    if (!isfinite(config->amplitude_v) || !(config->amplitude_v > 0.0)) {
        return -2;
    }
    if (!isfinite(config->frequency_hz) || config->frequency_hz < 0.0) {
        return -3;
    }
    if (!isfinite(config->phase_deg)) {
        return -4;
    }
    if (!isfinite(config->offset_v)) {
        return -5;
    }
    if (!isfinite(config->duty_cycle)) {
        return -6;
    }
    if (!isfinite(config->snr_db)) {
        return -7;
    }
    if (config->component_count > GENERATOR_MAX_COMPONENTS) {
        return -8;
    }

    switch (config->type) {
    case GENERATOR_SINE:
    case GENERATOR_TRIANGLE:
    case GENERATOR_SAWTOOTH:
        if (!(config->frequency_hz > 0.0)) {
            return -9;
        }
        break;
    case GENERATOR_SQUARE:
        if (!(config->frequency_hz > 0.0)) {
            return -9;
        }
        if (!(config->duty_cycle > 0.0) || !(config->duty_cycle < 1.0)) {
            return -10;
        }
        break;
    case GENERATOR_NOISE:
        break;
    case GENERATOR_MULTI_TONE:
        if (config->component_count == 0) {
            return -11;
        }
        for (size_t i = 0; i < config->component_count; ++i) {
            const ToneComponent *component = &config->components[i];
            if (!isfinite(component->freq_hz) || !(component->freq_hz > 0.0)) {
                return -12;
            }
            if (!isfinite(component->amp_v)) {
                return -13;
            }
            if (!isfinite(component->phase_deg)) {
                return -14;
            }
        }
        break;
    default:
        return -15;
    }
    return 0;
}

void generator_rng_seed(GeneratorRng *rng, uint64_t seed) {
    if (rng == NULL) {
        return;
    }
    rng->state = seed;
    rng->spare = 0.0;
    rng->has_spare = false;
}

double generator_rng_uniform(GeneratorRng *rng) {
    if (rng == NULL) {
        return 0.0;
    }
    return (double)(generator_rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

double generator_rng_normal(GeneratorRng *rng) {
    if (rng == NULL) {
        return 0.0;
    }
    if (rng->has_spare) {
        rng->has_spare = false;
        return rng->spare;
    }

    const double u1 = generator_rng_uniform(rng);
    const double u2 = generator_rng_uniform(rng);
    const double radius = sqrt(-2.0 * log(1.0 - u1));
    const double theta = GENERATOR_TWO_PI * u2;
    rng->spare = radius * sin(theta);
    rng->has_spare = true;
    return radius * cos(theta);
}

int generator_generate_sine(const GeneratorConfig *config, SignalBuffer *out) {
    const int status = generator_prepare(config, out);
    if (status != 0) {
        return status;
    }

    const double dt = 1.0 / out->sample_rate_hz;
    const double omega = GENERATOR_TWO_PI * config->frequency_hz;
    const double phase0 = config->phase_deg * generator_deg_to_rad;
    for (size_t i = 0; i < out->count; ++i) {
        const double t = (double)i * dt;
        out->samples[i] = (float)(config->offset_v + config->amplitude_v * sin(omega * t + phase0));
    }
    return 0;
}

int generator_generate_square(const GeneratorConfig *config, SignalBuffer *out) {
    const int status = generator_prepare(config, out);
    if (status != 0) {
        return status;
    }

    const double dt = 1.0 / out->sample_rate_hz;
    const double omega = GENERATOR_TWO_PI * config->frequency_hz;
    const double phase0 = config->phase_deg * generator_deg_to_rad;
    for (size_t i = 0; i < out->count; ++i) {
        const double t = (double)i * dt;
        const double fraction = generator_cycle_fraction(omega * t + phase0);
        const double shape = fraction < config->duty_cycle ? 1.0 : -1.0;
        out->samples[i] = (float)(config->offset_v + config->amplitude_v * shape);
    }
    return 0;
}

int generator_generate_triangle(const GeneratorConfig *config, SignalBuffer *out) {
    const int status = generator_prepare(config, out);
    if (status != 0) {
        return status;
    }

    const double dt = 1.0 / out->sample_rate_hz;
    const double omega = GENERATOR_TWO_PI * config->frequency_hz;
    const double phase0 = config->phase_deg * generator_deg_to_rad;
    for (size_t i = 0; i < out->count; ++i) {
        const double t = (double)i * dt;
        const double fraction = generator_cycle_fraction(omega * t + phase0);
        const double shape = 4.0 * fabs(fraction - 0.5) - 1.0;
        out->samples[i] = (float)(config->offset_v + config->amplitude_v * shape);
    }
    return 0;
}

int generator_generate_sawtooth(const GeneratorConfig *config, SignalBuffer *out) {
    const int status = generator_prepare(config, out);
    if (status != 0) {
        return status;
    }

    const double dt = 1.0 / out->sample_rate_hz;
    const double omega = GENERATOR_TWO_PI * config->frequency_hz;
    const double phase0 = config->phase_deg * generator_deg_to_rad;
    for (size_t i = 0; i < out->count; ++i) {
        const double t = (double)i * dt;
        const double fraction = generator_cycle_fraction(omega * t + phase0);
        const double shape = 2.0 * fraction - 1.0;
        out->samples[i] = (float)(config->offset_v + config->amplitude_v * shape);
    }
    return 0;
}

int generator_generate_noise(const GeneratorConfig *config, GeneratorRng *rng, SignalBuffer *out) {
    const int status = generator_prepare(config, out);
    if (status != 0) {
        return status;
    }
    if (rng == NULL) {
        return -21;
    }

    const double std_dev = config->amplitude_v;
    for (size_t i = 0; i < out->count; ++i) {
        out->samples[i] = (float)(config->offset_v + std_dev * generator_rng_normal(rng));
    }
    return 0;
}

int generator_generate_multi_tone(const GeneratorConfig *config, SignalBuffer *out) {
    const int status = generator_prepare(config, out);
    if (status != 0) {
        return status;
    }

    const double dt = 1.0 / out->sample_rate_hz;
    for (size_t i = 0; i < out->count; ++i) {
        const double t = (double)i * dt;
        double value = config->offset_v;
        for (size_t c = 0; c < config->component_count; ++c) {
            const ToneComponent *component = &config->components[c];
            const double omega = GENERATOR_TWO_PI * component->freq_hz;
            const double phase = component->phase_deg * generator_deg_to_rad;
            value += component->amp_v * sin(omega * t + phase);
        }
        out->samples[i] = (float)value;
    }
    return 0;
}

int generator_add_noise(SignalBuffer *buffer, double snr_db, GeneratorRng *rng) {
    if (!signal_buffer_is_valid(buffer) || rng == NULL) {
        return -1;
    }
    if (!isfinite(snr_db)) {
        return -2;
    }

    double sum_squares = 0.0;
    for (size_t i = 0; i < buffer->count; ++i) {
        const double value = (double)buffer->samples[i];
        sum_squares += value * value;
    }
    const double signal_rms = sqrt(sum_squares / (double)buffer->count);
    if (!(signal_rms > 0.0)) {
        return 0;
    }

    const double noise_std = signal_rms / pow(10.0, snr_db / 20.0);
    for (size_t i = 0; i < buffer->count; ++i) {
        const double noise = noise_std * generator_rng_normal(rng);
        buffer->samples[i] = (float)((double)buffer->samples[i] + noise);
    }
    return 0;
}

int generator_generate(const GeneratorConfig *config, GeneratorRng *rng, SignalBuffer *out) {
    int status = generator_config_validate(config);
    if (status != 0) {
        return status;
    }

    switch (config->type) {
    case GENERATOR_SINE:
        status = generator_generate_sine(config, out);
        break;
    case GENERATOR_SQUARE:
        status = generator_generate_square(config, out);
        break;
    case GENERATOR_TRIANGLE:
        status = generator_generate_triangle(config, out);
        break;
    case GENERATOR_SAWTOOTH:
        status = generator_generate_sawtooth(config, out);
        break;
    case GENERATOR_NOISE:
        return generator_generate_noise(config, rng, out);
    case GENERATOR_MULTI_TONE:
        status = generator_generate_multi_tone(config, out);
        break;
    default:
        return -16;
    }
    if (status != 0) {
        return status;
    }

    if (config->snr_db > 0.0) {
        status = generator_add_noise(out, config->snr_db, rng);
    }
    return status;
}
