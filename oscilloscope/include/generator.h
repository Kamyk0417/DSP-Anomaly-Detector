/* include/generator.h */

#ifndef GENERATOR_H
#define GENERATOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "signal.h"

#define GENERATOR_MAX_COMPONENTS 8U

typedef enum {
    GENERATOR_SINE,
    GENERATOR_SQUARE,
    GENERATOR_TRIANGLE,
    GENERATOR_SAWTOOTH,
    GENERATOR_NOISE,
    GENERATOR_MULTI_TONE,
    GENERATOR_TYPE_COUNT
} GeneratorType;

typedef struct {
    double freq_hz;
    double amp_v;
    double phase_deg;
} ToneComponent;

/* amplitude_v is the peak amplitude of the waveform. For GENERATOR_NOISE it is
 * the standard deviation of the white gaussian noise. snr_db > 0 enables
 * additive white gaussian noise at that signal-to-noise ratio; snr_db <= 0
 * disables it. */
typedef struct {
    GeneratorType type;
    double amplitude_v;
    double frequency_hz;
    double phase_deg;
    double offset_v;
    double duty_cycle;
    double snr_db;
    ToneComponent components[GENERATOR_MAX_COMPONENTS];
    size_t component_count;
} GeneratorConfig;

typedef struct {
    uint64_t state;
    double spare;
    bool has_spare;
} GeneratorRng;

int generator_config_default(GeneratorConfig *config);
int generator_config_validate(const GeneratorConfig *config);

void generator_rng_seed(GeneratorRng *rng, uint64_t seed);
double generator_rng_uniform(GeneratorRng *rng);
double generator_rng_normal(GeneratorRng *rng);

/* rng may be NULL unless the configuration needs it: type GENERATOR_NOISE or
 * snr_db > 0. */
int generator_generate(const GeneratorConfig *config, GeneratorRng *rng, SignalBuffer *out);

int generator_generate_sine(const GeneratorConfig *config, SignalBuffer *out);
int generator_generate_square(const GeneratorConfig *config, SignalBuffer *out);
int generator_generate_triangle(const GeneratorConfig *config, SignalBuffer *out);
int generator_generate_sawtooth(const GeneratorConfig *config, SignalBuffer *out);
int generator_generate_noise(const GeneratorConfig *config, GeneratorRng *rng, SignalBuffer *out);
int generator_generate_multi_tone(const GeneratorConfig *config, SignalBuffer *out);

/* Adds zero-mean gaussian noise so that the resulting SNR equals snr_db. */
int generator_add_noise(SignalBuffer *buffer, double snr_db, GeneratorRng *rng);

#endif /* GENERATOR_H */
