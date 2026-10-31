/* src/main.c */

#include <SDL.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "adc.h"
#include "features.h"
#include "fft.h"
#include "filter.h"
#include "generator.h"
#include "io.h"
#include "signal.h"
#include "visualize.h"
#include "window.h"

#define MAIN_TITLE "OSCILLOSCOPE"
/* 16384 samples = 163.84 ms at 100 kSa/s. Two consumers: the 16384-point
 * analysis FFT (6.1 Hz per bin) and the spectrogram, whose 1024-sample chunks
 * give 10.24 ms per row (time resolution) at the cost of 97.7 Hz per bin; the
 * buffer is large enough to keep a full 128-row waterfall. */
#define MAIN_SAMPLE_COUNT 16384U
#define MAIN_SAMPLE_RATE_HZ ADC_SAMPLE_RATE_HZ_DEFAULT
#define MAIN_AMPLITUDE_V 2.5
#define MAIN_CUTOFF_HZ 10000.0
#define MAIN_FIR_ORDER FILTER_FIR_ORDER_DEFAULT
#define MAIN_IIR_ORDER 4U
#define MAIN_CHEBYSHEV1_RIPPLE_DB 1.0
#define MAIN_CHEBYSHEV2_STOPBAND_DB 40.0
#define MAIN_CHUNK_SAMPLES 1024U
#define MAIN_CHUNK_HOP 64U
#define MAIN_SPECTROGRAM_FRAMES 128U
#define MAIN_SEED_DEFAULT 12345U
#define MAIN_GENERATOR_COUNT (sizeof(main_generator_types) / sizeof(main_generator_types[0]))
#define MAIN_FREQUENCY_COUNT (sizeof(main_frequency_presets) / sizeof(main_frequency_presets[0]))

#define MAIN_HEADLESS_SAMPLES_DEFAULT 100000U
#define MAIN_HEADLESS_OUTPUT_DEFAULT "sig_001"
#define MAIN_HEADLESS_LIMIT_SAMPLES 16000000U
#define MAIN_FEATURE_BANDS 4U

typedef enum {
    MAIN_FILTER_HAMMING = 0,
    MAIN_FILTER_CHEBYSHEV1,
    MAIN_FILTER_CHEBYSHEV2,
    MAIN_FILTER_BUTTERWORTH,
    MAIN_FILTER_KIND_COUNT
} MainFilterKind;

typedef struct {
    float analog[MAIN_SAMPLE_COUNT];
    float digitized[MAIN_SAMPLE_COUNT];
    float processed[MAIN_SAMPLE_COUNT];
    float analysis_re[MAIN_SAMPLE_COUNT];
    float analysis_im[MAIN_SAMPLE_COUNT];
    float magnitude[MAIN_SAMPLE_COUNT / 2U + 1U];
    float chunk_samples[MAIN_CHUNK_SAMPLES];
    float chunk_im[MAIN_CHUNK_SAMPLES];
    float chunk_magnitude[MAIN_CHUNK_SAMPLES / 2U + 1U];
    uint32_t codes[MAIN_SAMPLE_COUNT];
    double window[MAIN_SAMPLE_COUNT];
    double chunk_window[MAIN_CHUNK_SAMPLES];

    SignalBuffer analog_view;
    SignalBuffer processed_view;
    Visualizer *visualizer;

    GeneratorConfig generator;
    GeneratorRng rng;
    AdcConfig adc;
    FirCoeffs fir;
    FilterCoeffs iir;
    FftPlan plan;
    FftPlan chunk_plan;

    size_t generator_index;
    size_t frequency_index;
    uint64_t seed;
    MainFilterKind filter_kind;
    bool adc_enabled;
    bool filter_enabled;

    double rms;
    double peak_to_peak;
    double quantization_rms;
    double dominant_frequency_hz;
    char status[VISUALIZER_TEXT_CHARS];
} MainApp;

static const GeneratorType main_generator_types[] = {
    GENERATOR_SINE, GENERATOR_SQUARE,  GENERATOR_TRIANGLE,
    GENERATOR_SAWTOOTH, GENERATOR_MULTI_TONE, GENERATOR_NOISE,
};

static const double main_frequency_presets[] = {
    100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0,
};

static MainApp main_state;

static const char *main_generator_name(GeneratorType type) {
    switch (type) {
    case GENERATOR_SINE:
        return "SINE";
    case GENERATOR_SQUARE:
        return "SQUARE";
    case GENERATOR_TRIANGLE:
        return "TRIANGLE";
    case GENERATOR_SAWTOOTH:
        return "SAWTOOTH";
    case GENERATOR_MULTI_TONE:
        return "MULTI TONE";
    case GENERATOR_NOISE:
        return "NOISE";
    default:
        return "UNKNOWN";
    }
}

static const char *main_filter_name(MainFilterKind kind) {
    switch (kind) {
    case MAIN_FILTER_HAMMING:
        return "FIR HAMMING";
    case MAIN_FILTER_CHEBYSHEV1:
        return "CHEBYSHEV I";
    case MAIN_FILTER_CHEBYSHEV2:
        return "CHEBYSHEV II";
    case MAIN_FILTER_BUTTERWORTH:
        return "BUTTERWORTH";
    default:
        return "UNKNOWN";
    }
}

static void main_format_volts(char *text, size_t size, double volts) {
    if (fabs(volts) < 1.0) {
        snprintf(text, size, "%.1f MV", volts * 1000.0);
    } else {
        snprintf(text, size, "%.2f V", volts);
    }
}

static void main_format_frequency(char *text, size_t size, double hertz) {
    if (fabs(hertz) >= 1000.0) {
        snprintf(text, size, "%.2f KHZ", hertz / 1000.0);
    } else {
        snprintf(text, size, "%.0f HZ", hertz);
    }
}

static double main_fraction(double value) {
    if (!(value > 0.0)) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

static void main_configure_generator(MainApp *state) {
    GeneratorConfig *config = &state->generator;

    generator_config_default(config);
    config->type = main_generator_types[state->generator_index];
    config->amplitude_v = MAIN_AMPLITUDE_V;
    config->frequency_hz = main_frequency_presets[state->frequency_index];
    config->duty_cycle = 0.3;

    if (config->type == GENERATOR_MULTI_TONE) {
        config->component_count = 2U;
        config->components[0].freq_hz = config->frequency_hz;
        config->components[0].amp_v = 1.5;
        config->components[0].phase_deg = 0.0;
        config->components[1].freq_hz = config->frequency_hz * 3.0;
        config->components[1].amp_v = 0.5;
        config->components[1].phase_deg = 0.0;
    }
}

static int main_configure_filter(MainApp *state) {
    switch (state->filter_kind) {
    case MAIN_FILTER_HAMMING:
        return filter_design_fir_lowpass(&state->fir, MAIN_CUTOFF_HZ, MAIN_SAMPLE_RATE_HZ,
                                         MAIN_FIR_ORDER, WINDOW_HAMMING);
    case MAIN_FILTER_CHEBYSHEV1:
        return filter_design_iir_lowpass(&state->iir, FILTER_IIR_CHEBYSHEV1, MAIN_CUTOFF_HZ,
                                         MAIN_SAMPLE_RATE_HZ, MAIN_IIR_ORDER,
                                         MAIN_CHEBYSHEV1_RIPPLE_DB);
    case MAIN_FILTER_CHEBYSHEV2:
        return filter_design_iir_lowpass(&state->iir, FILTER_IIR_CHEBYSHEV2, MAIN_CUTOFF_HZ,
                                         MAIN_SAMPLE_RATE_HZ, MAIN_IIR_ORDER,
                                         MAIN_CHEBYSHEV2_STOPBAND_DB);
    case MAIN_FILTER_BUTTERWORTH:
        return filter_design_iir_lowpass(&state->iir, FILTER_IIR_BUTTERWORTH, MAIN_CUTOFF_HZ,
                                         MAIN_SAMPLE_RATE_HZ, MAIN_IIR_ORDER, 0.0);
    default:
        return -1;
    }
}

static void main_update_parameters(MainApp *state) {
    Visualizer *visualizer = state->visualizer;
    char value[VISUALIZER_TEXT_CHARS];
    const double nyquist_hz = MAIN_SAMPLE_RATE_HZ * 0.5;

    visualizer_clear_parameters(visualizer);

    visualizer_add_parameter(visualizer, "WAVE", main_generator_name(state->generator.type), 0.0);

    main_format_frequency(value, sizeof(value), state->generator.frequency_hz);
    visualizer_add_parameter(visualizer, "FREQUENCY", value,
                             main_fraction(state->generator.frequency_hz / nyquist_hz));

    main_format_volts(value, sizeof(value), state->generator.amplitude_v);
    visualizer_add_parameter(visualizer, "AMPLITUDE", value,
                             main_fraction(state->generator.amplitude_v / 5.0));

    snprintf(value, sizeof(value), "%.1f KHZ", MAIN_SAMPLE_RATE_HZ / 1000.0);
    visualizer_add_parameter(visualizer, "SAMPLE RATE", value, 1.0);

    if (state->adc_enabled) {
        snprintf(value, sizeof(value), "%u BIT", state->adc.bits);
        visualizer_add_parameter(visualizer, "ADC", value, (double)state->adc.bits / 16.0);
        main_format_volts(value, sizeof(value), state->quantization_rms);
        visualizer_add_parameter(visualizer, "ADC RMS ERR", value,
                                 main_fraction(state->quantization_rms / 0.01));
    } else {
        visualizer_add_parameter(visualizer, "ADC", "OFF", 0.0);
        visualizer_add_parameter(visualizer, "ADC RMS ERR", "-", 0.0);
    }

    if (state->filter_enabled) {
        visualizer_add_parameter(visualizer, "FILTER", main_filter_name(state->filter_kind), 1.0);
        main_format_frequency(value, sizeof(value), MAIN_CUTOFF_HZ);
        visualizer_add_parameter(visualizer, "CUTOFF", value,
                                 main_fraction(MAIN_CUTOFF_HZ / nyquist_hz));
    } else {
        visualizer_add_parameter(visualizer, "FILTER", "OFF", 0.0);
        visualizer_add_parameter(visualizer, "CUTOFF", "-", 0.0);
    }

    main_format_volts(value, sizeof(value), state->rms);
    visualizer_add_parameter(visualizer, "RMS", value, main_fraction(state->rms / 5.0));

    main_format_volts(value, sizeof(value), state->peak_to_peak);
    visualizer_add_parameter(visualizer, "PEAK TO PEAK", value,
                             main_fraction(state->peak_to_peak / 10.0));

    main_format_frequency(value, sizeof(value), state->dominant_frequency_hz);
    visualizer_add_parameter(visualizer, "DOMINANT", value,
                             main_fraction(state->dominant_frequency_hz / nyquist_hz));
}

static void main_update_status(MainApp *state) {
    char frequency[VISUALIZER_TEXT_CHARS];
    char adc[VISUALIZER_TEXT_CHARS];
    char filter[VISUALIZER_TEXT_CHARS];

    main_format_frequency(frequency, sizeof(frequency), state->generator.frequency_hz);

    if (state->adc_enabled) {
        snprintf(adc, sizeof(adc), "ADC %u BIT", state->adc.bits);
    } else {
        snprintf(adc, sizeof(adc), "ADC OFF");
    }

    if (state->filter_enabled) {
        snprintf(filter, sizeof(filter), "FILTER %s", main_filter_name(state->filter_kind));
    } else {
        snprintf(filter, sizeof(filter), "FILTER OFF");
    }

    snprintf(state->status, sizeof(state->status), "%s %s  %s  %s  RMS %.2f V",
             main_generator_name(state->generator.type), frequency, adc, filter, state->rms);
}

static int main_regenerate(MainApp *state) {
    main_configure_generator(state);
    generator_rng_seed(&state->rng, state->seed);

    if (generator_generate(&state->generator, &state->rng, &state->analog_view) != 0) {
        return -1;
    }

    /* ADC stage: quantize to codes, then dequantize back to volts. */
    state->quantization_rms = 0.0;
    if (state->adc_enabled) {
        if (adc_quantize_buffer(&state->adc, state->analog, state->codes, MAIN_SAMPLE_COUNT) != 0 ||
            adc_dequantize_buffer(&state->adc, state->codes, state->digitized,
                                  MAIN_SAMPLE_COUNT) != 0) {
            return -2;
        }
        double error_sum = 0.0;
        for (size_t i = 0; i < MAIN_SAMPLE_COUNT; ++i) {
            const double difference = (double)state->digitized[i] - (double)state->analog[i];
            error_sum += difference * difference;
        }
        state->quantization_rms = sqrt(error_sum / (double)MAIN_SAMPLE_COUNT);
    } else {
        memcpy(state->digitized, state->analog, sizeof(state->analog));
    }

    /* Optional filter stage. */
    if (state->filter_enabled) {
        const int status =
            (state->filter_kind == MAIN_FILTER_HAMMING)
                ? filter_apply_fir(&state->fir, state->digitized, state->processed,
                                   MAIN_SAMPLE_COUNT)
                : filter_apply_iir_sos(&state->iir, state->digitized, state->processed,
                                       MAIN_SAMPLE_COUNT);
        if (status != 0) {
            return -3;
        }
    } else {
        memcpy(state->processed, state->digitized, sizeof(state->processed));
    }

    double sum_squares = 0.0;
    float minimum = state->processed[0];
    float maximum = state->processed[0];
    for (size_t i = 0; i < MAIN_SAMPLE_COUNT; ++i) {
        const double value = (double)state->processed[i];
        sum_squares += value * value;
        if (state->processed[i] < minimum) {
            minimum = state->processed[i];
        }
        if (state->processed[i] > maximum) {
            maximum = state->processed[i];
        }
    }
    state->rms = sqrt(sum_squares / (double)MAIN_SAMPLE_COUNT);
    state->peak_to_peak = (double)maximum - (double)minimum;

    for (size_t i = 0; i < MAIN_SAMPLE_COUNT; ++i) {
        state->analysis_re[i] = (float)((double)state->processed[i] * state->window[i]);
    }
    if (fft_forward(&state->plan, state->analysis_re, state->analysis_re, state->analysis_im) !=
            0 ||
        fft_magnitude(&state->plan, state->analysis_re, state->analysis_im, state->magnitude) !=
            0) {
        return -4;
    }

    {
        const size_t bins = fft_positive_half_size(MAIN_SAMPLE_COUNT);
        double best = 0.0;
        size_t best_bin = 0;
        for (size_t bin = 1; bin < bins; ++bin) {
            if ((double)state->magnitude[bin] > best) {
                best = (double)state->magnitude[bin];
                best_bin = bin;
            }
        }
        state->dominant_frequency_hz =
            (double)best_bin * MAIN_SAMPLE_RATE_HZ / (double)MAIN_SAMPLE_COUNT;
    }

    if (visualizer_set_signal(state->visualizer, &state->processed_view) != 0 ||
        visualizer_set_spectrum(state->visualizer, state->magnitude,
                                fft_positive_half_size(MAIN_SAMPLE_COUNT),
                                MAIN_SAMPLE_RATE_HZ / (double)MAIN_SAMPLE_COUNT) != 0) {
        return -5;
    }

    /* Overlapping chunks feed the waterfall view. */
    for (size_t frame = 0; frame < MAIN_SPECTROGRAM_FRAMES; ++frame) {
        const size_t start = frame * MAIN_CHUNK_HOP;
        if (start + MAIN_CHUNK_SAMPLES > MAIN_SAMPLE_COUNT) {
            break;
        }
        for (size_t i = 0; i < MAIN_CHUNK_SAMPLES; ++i) {
            state->chunk_samples[i] =
                (float)((double)state->processed[start + i] * state->chunk_window[i]);
        }
        if (fft_forward(&state->chunk_plan, state->chunk_samples, state->chunk_samples,
                        state->chunk_im) != 0 ||
            fft_magnitude(&state->chunk_plan, state->chunk_samples, state->chunk_im,
                          state->chunk_magnitude) != 0) {
            return -6;
        }
        visualizer_push_spectrogram(state->visualizer, state->chunk_magnitude,
                                    fft_positive_half_size(MAIN_CHUNK_SAMPLES));
    }

    main_update_parameters(state);
    main_update_status(state);
    return 0;
}

static void main_handle_command(MainApp *state, VisualizerCommand command, bool *running) {
    switch (command) {
    case VISUALIZER_COMMAND_NONE:
        return;
    case VISUALIZER_COMMAND_QUIT:
        *running = false;
        return;
    case VISUALIZER_COMMAND_NEXT_SIGNAL:
        state->frequency_index = (state->frequency_index + 1U) % MAIN_FREQUENCY_COUNT;
        ++state->seed;
        break;
    case VISUALIZER_COMMAND_NEXT_GENERATOR:
        state->generator_index = (state->generator_index + 1U) % MAIN_GENERATOR_COUNT;
        ++state->seed;
        break;
    case VISUALIZER_COMMAND_TOGGLE_ADC:
        state->adc_enabled = !state->adc_enabled;
        break;
    case VISUALIZER_COMMAND_TOGGLE_FILTER:
        state->filter_enabled = !state->filter_enabled;
        break;
    case VISUALIZER_COMMAND_NEXT_FILTER:
        state->filter_kind =
            (MainFilterKind)((state->filter_kind + 1) % MAIN_FILTER_KIND_COUNT);
        if (main_configure_filter(state) != 0) {
            fprintf(stderr, "oscilloscope: filter design failed\n");
        }
        state->filter_enabled = true;
        break;
    default:
        return;
    }

    if (main_regenerate(state) != 0) {
        fprintf(stderr, "oscilloscope: signal processing failed\n");
    }
}

static int main_initialize(MainApp *state) {
    memset(state, 0, sizeof(*state));

    state->seed = MAIN_SEED_DEFAULT;
    state->adc_enabled = true;
    state->filter_enabled = false;
    state->filter_kind = MAIN_FILTER_CHEBYSHEV1;

    if (signal_buffer_init(&state->analog_view, state->analog, MAIN_SAMPLE_COUNT,
                           MAIN_SAMPLE_RATE_HZ) != 0 ||
        signal_buffer_init(&state->processed_view, state->processed, MAIN_SAMPLE_COUNT,
                           MAIN_SAMPLE_RATE_HZ) != 0) {
        return -1;
    }
    if (adc_config_default(&state->adc) != 0) {
        return -2;
    }
    if (window_generate(WINDOW_HANN, state->window, MAIN_SAMPLE_COUNT) != 0 ||
        window_generate(WINDOW_HANN, state->chunk_window, MAIN_CHUNK_SAMPLES) != 0) {
        return -3;
    }
    if (fft_plan_create(&state->plan, MAIN_SAMPLE_COUNT) != 0 ||
        fft_plan_create(&state->chunk_plan, MAIN_CHUNK_SAMPLES) != 0) {
        return -4;
    }
    if (main_configure_filter(state) != 0) {
        return -5;
    }

    state->visualizer =
        visualizer_create(MAIN_TITLE, VISUALIZER_WIDTH_DEFAULT, VISUALIZER_HEIGHT_DEFAULT);
    if (state->visualizer == NULL) {
        return -6;
    }

    return main_regenerate(state);
}

static void main_shutdown(MainApp *state) {
    visualizer_destroy(state->visualizer);
    state->visualizer = NULL;
    fft_plan_destroy(&state->plan);
    fft_plan_destroy(&state->chunk_plan);
}

/* ---------------------------------------------------------------- headless -- */

typedef struct {
    const char *output;
    GeneratorType type;
    double frequency_hz;
    double amplitude_v;
    size_t samples;
    double sample_rate_hz;
    uint64_t seed;
    bool adc_enabled;
} MainHeadlessOptions;

static const char *main_generator_key(GeneratorType type) {
    switch (type) {
    case GENERATOR_SINE:
        return "sine";
    case GENERATOR_SQUARE:
        return "square";
    case GENERATOR_TRIANGLE:
        return "triangle";
    case GENERATOR_SAWTOOTH:
        return "sawtooth";
    case GENERATOR_NOISE:
        return "noise";
    case GENERATOR_MULTI_TONE:
        return "multi_tone";
    default:
        return "sine";
    }
}

static int main_generator_from_key(const char *name, GeneratorType *type) {
    static const GeneratorType types[] = {GENERATOR_SINE,      GENERATOR_SQUARE,
                                          GENERATOR_TRIANGLE, GENERATOR_SAWTOOTH,
                                          GENERATOR_NOISE,    GENERATOR_MULTI_TONE};
    for (size_t i = 0; i < MAIN_GENERATOR_COUNT; ++i) {
        if (strcmp(name, main_generator_key(types[i])) == 0) {
            *type = types[i];
            return 0;
        }
    }
    return -1;
}

static void main_print_usage(const char *program) {
    printf("usage: %s [--headless] [options]\n", program);
    printf("\n");
    printf("  --headless             generate and export a signal without opening a window\n");
    printf("  --output <base>        base path for the exported .bin/.json (default %s)\n",
           MAIN_HEADLESS_OUTPUT_DEFAULT);
    printf("  --type <name>          sine, square, triangle, sawtooth, noise, multi_tone\n");
    printf("  --frequency <hz>       generator frequency (default %.0f)\n", 1000.0);
    printf("  --amplitude <v>        peak amplitude in volts (default %.1f)\n", MAIN_AMPLITUDE_V);
    printf("  --samples <count>      sample count (default %u)\n", MAIN_HEADLESS_SAMPLES_DEFAULT);
    printf("  --sample-rate <hz>     sample rate (default %.0f)\n", MAIN_SAMPLE_RATE_HZ);
    printf("  --seed <n>             generator seed (default %u)\n", MAIN_SEED_DEFAULT);
    printf("  --no-adc               skip ADC quantization\n");
    printf("  --help                 show this message\n");
    printf("\n");
    printf("Without --headless the interactive SDL2 oscilloscope starts.\n");
}

static int main_parse_arguments(int argc, char *argv[], bool *headless,
                                MainHeadlessOptions *options) {
    *headless = false;
    options->output = MAIN_HEADLESS_OUTPUT_DEFAULT;
    options->type = GENERATOR_SINE;
    options->frequency_hz = 1000.0;
    options->amplitude_v = MAIN_AMPLITUDE_V;
    options->samples = MAIN_HEADLESS_SAMPLES_DEFAULT;
    options->sample_rate_hz = MAIN_SAMPLE_RATE_HZ;
    options->seed = MAIN_SEED_DEFAULT;
    options->adc_enabled = true;

    for (int index = 1; index < argc; ++index) {
        const char *argument = argv[index];
        const bool has_value = (index + 1) < argc;

        if (strcmp(argument, "--headless") == 0) {
            *headless = true;
        } else if (strcmp(argument, "--no-adc") == 0) {
            options->adc_enabled = false;
        } else if (strcmp(argument, "--help") == 0) {
            main_print_usage(argv[0]);
            return 1;
        } else if (strcmp(argument, "--output") == 0 && has_value) {
            options->output = argv[++index];
        } else if (strcmp(argument, "--type") == 0 && has_value) {
            const char *name = argv[++index];
            if (main_generator_from_key(name, &options->type) != 0) {
                fprintf(stderr, "oscilloscope: unknown signal type '%s'\n", name);
                return -1;
            }
        } else if (strcmp(argument, "--frequency") == 0 && has_value) {
            options->frequency_hz = atof(argv[++index]);
        } else if (strcmp(argument, "--amplitude") == 0 && has_value) {
            options->amplitude_v = atof(argv[++index]);
        } else if (strcmp(argument, "--sample-rate") == 0 && has_value) {
            options->sample_rate_hz = atof(argv[++index]);
        } else if (strcmp(argument, "--samples") == 0 && has_value) {
            const unsigned long value = strtoul(argv[++index], NULL, 10);
            if (value == 0UL || value > (unsigned long)MAIN_HEADLESS_LIMIT_SAMPLES) {
                fprintf(stderr, "oscilloscope: --samples must be between 1 and %u\n",
                        MAIN_HEADLESS_LIMIT_SAMPLES);
                return -1;
            }
            options->samples = (size_t)value;
        } else if (strcmp(argument, "--seed") == 0 && has_value) {
            options->seed = (uint64_t)strtoul(argv[++index], NULL, 10);
        } else {
            fprintf(stderr, "oscilloscope: unknown or incomplete argument '%s'\n", argument);
            return -1;
        }
    }
    return 0;
}

static int main_run_headless(const MainHeadlessOptions *options) {
    float *storage = malloc(options->samples * sizeof(float));
    if (storage == NULL) {
        fprintf(stderr, "oscilloscope: cannot allocate %zu samples\n", options->samples);
        return 1;
    }

    SignalBuffer buffer;
    if (signal_buffer_init(&buffer, storage, options->samples, options->sample_rate_hz) != 0) {
        fprintf(stderr, "oscilloscope: invalid sample count or rate\n");
        free(storage);
        return 1;
    }

    GeneratorConfig config;
    generator_config_default(&config);
    config.type = options->type;
    config.amplitude_v = options->amplitude_v;
    config.frequency_hz = options->frequency_hz;
    if (config.type == GENERATOR_MULTI_TONE) {
        config.component_count = 2U;
        config.components[0].freq_hz = options->frequency_hz;
        config.components[0].amp_v = 0.6 * options->amplitude_v;
        config.components[0].phase_deg = 0.0;
        config.components[1].freq_hz = 3.0 * options->frequency_hz;
        config.components[1].amp_v = 0.2 * options->amplitude_v;
        config.components[1].phase_deg = 0.0;
    }

    GeneratorRng rng;
    generator_rng_seed(&rng, options->seed);
    if (generator_generate(&config, &rng, &buffer) != 0) {
        fprintf(stderr, "oscilloscope: cannot generate a '%s' signal at %g Hz\n",
                main_generator_key(options->type), options->frequency_hz);
        free(storage);
        return 1;
    }

    AdcConfig adc;
    if (adc_config_default(&adc) != 0) {
        free(storage);
        return 1;
    }

    if (options->adc_enabled) {
        uint32_t *codes = malloc(options->samples * sizeof(uint32_t));
        if (codes == NULL) {
            fprintf(stderr, "oscilloscope: cannot allocate ADC codes\n");
            free(storage);
            return 1;
        }
        const int quantize_status = adc_quantize_buffer(&adc, storage, codes, buffer.count);
        const int dequantize_status = (quantize_status == 0)
                                          ? adc_dequantize_buffer(&adc, codes, storage,
                                                                  buffer.count)
                                          : -1;
        free(codes);
        if (quantize_status != 0 || dequantize_status != 0) {
            fprintf(stderr, "oscilloscope: ADC conversion failed\n");
            free(storage);
            return 1;
        }
    }

    SignalMetadata metadata;
    io_metadata_init(&metadata);
    snprintf(metadata.generator_type, IO_NAME_LENGTH, "%s", main_generator_key(options->type));
    metadata.amplitude_v = config.amplitude_v;
    metadata.frequency_hz = config.frequency_hz;
    metadata.phase_deg = config.phase_deg;
    metadata.offset_v = config.offset_v;
    metadata.duty_cycle = config.duty_cycle;
    metadata.snr_db = config.snr_db;
    metadata.adc_bits = options->adc_enabled ? adc.bits : 0U;
    metadata.adc_v_min = adc.v_min;
    metadata.adc_v_max = adc.v_max;

    FeatureExtractor extractor;
    const size_t fft_size = fft_next_power_of_two(buffer.count);
    if (fft_size == 0U || features_extractor_create(&extractor, fft_size) != 0) {
        fprintf(stderr, "oscilloscope: cannot create the feature extractor\n");
        free(storage);
        return 1;
    }

    double band_edges[MAIN_FEATURE_BANDS + 1U];
    SignalFeatures features;
    memset(&features, 0, sizeof(features));
    const int feature_status =
        features_default_band_edges(band_edges, MAIN_FEATURE_BANDS,
                                    buffer.sample_rate_hz * 0.5) == 0
            ? features_extract(&extractor, &buffer, band_edges, MAIN_FEATURE_BANDS, &features)
            : -1;
    const size_t spectrum_bins = features.spectrum_bins;
    features_extractor_destroy(&extractor);
    if (feature_status != 0) {
        fprintf(stderr, "oscilloscope: feature extraction failed (%d)\n", feature_status);
        free(storage);
        return 1;
    }

    metadata.has_features = true;
    metadata.features.rms = features.rms;
    metadata.features.peak_to_peak = features.peak_to_peak;
    metadata.features.variance = features.variance;
    metadata.features.skewness = features.skewness;
    metadata.features.kurtosis = features.kurtosis;
    metadata.features.dominant_freq_hz = features.dominant_freq_hz;
    metadata.features.band_count = features.band_count;
    for (size_t band = 0; band < features.band_count; ++band) {
        metadata.features.band_energies[band] = features.band_energies[band];
    }

    const int write_status = io_write_signal(options->output, &buffer, &metadata);
    if (write_status != 0) {
        fprintf(stderr, "oscilloscope: cannot write '%s' (error %d)\n", options->output,
                write_status);
        free(storage);
        return 1;
    }

    printf("wrote %s.bin and %s.json\n", options->output, options->output);
    printf("  type        %s\n", main_generator_key(options->type));
    printf("  samples     %zu\n", buffer.count);
    printf("  sample rate %.0f Hz\n", buffer.sample_rate_hz);
    printf("  duration    %g s\n", signal_buffer_duration_s(&buffer));
    printf("  amplitude   %g V (peak)\n", config.amplitude_v);
    printf("  frequency   %g Hz\n", config.frequency_hz);
    printf("  adc         %s\n", options->adc_enabled ? "12-bit quantize/dequantize" : "disabled");
    printf("  rms         %.6f V\n", features.rms);
    printf("  peak-peak   %.6f V\n", features.peak_to_peak);
    printf("  variance    %.6f V^2\n", features.variance);
    printf("  skewness    %.6f\n", features.skewness);
    printf("  kurtosis    %.6f (excess)\n", features.kurtosis);
    printf("  dominant    %.3f Hz (spectrum %zu bins)\n", features.dominant_freq_hz,
           spectrum_bins);

    free(storage);
    return 0;
}

int main(int argc, char *argv[]) {
    bool headless = false;
    MainHeadlessOptions options;
    const int parse_status = main_parse_arguments(argc, argv, &headless, &options);
    if (parse_status > 0) {
        return 0;
    }
    if (parse_status < 0) {
        return 2;
    }
    if (headless) {
        return main_run_headless(&options);
    }

    MainApp *state = &main_state;

    if (main_initialize(state) != 0) {
        fprintf(stderr, "oscilloscope: initialization failed: %s\n", SDL_GetError());
        main_shutdown(state);
        return 1;
    }

    bool running = true;
    while (running) {
        VisualizerCommand command = VISUALIZER_COMMAND_NONE;
        if (visualizer_poll(state->visualizer, &command) != 0) {
            break;
        }
        main_handle_command(state, command, &running);
        if (visualizer_render(state->visualizer) != 0) {
            break;
        }
    }

    main_shutdown(state);
    return 0;
}

