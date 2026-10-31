/* src/adc.c */

#include "adc.h"

#include <math.h>
#include <stdbool.h>

static bool adc_bits_supported(unsigned bits) {
    return bits == 8U || bits == 10U || bits == 12U || bits == 16U;
}

int adc_config_default(AdcConfig *config) {
    if (config == NULL) {
        return -1;
    }
    config->bits = ADC_BITS_DEFAULT;
    config->v_min = ADC_V_MIN_DEFAULT;
    config->v_max = ADC_V_MAX_DEFAULT;
    return 0;
}

int adc_config_validate(const AdcConfig *config) {
    if (config == NULL) {
        return -1;
    }
    if (!adc_bits_supported(config->bits)) {
        return -2;
    }
    if (!isfinite(config->v_min) || !isfinite(config->v_max)) {
        return -3;
    }
    if (!(config->v_max > config->v_min)) {
        return -4;
    }
    return 0;
}

uint32_t adc_code_max(const AdcConfig *config) {
    if (adc_config_validate(config) != 0) {
        return 0U;
    }
    return ((uint32_t)1U << config->bits) - 1U;
}

uint32_t adc_quantize(const AdcConfig *config, float volts) {
    const uint32_t code_max = adc_code_max(config);
    if (code_max == 0U) {
        return 0U;
    }

    const double volts_d = (double)volts;
    if (isnan(volts_d)) {
        return 0U;
    }

    const double span = config->v_max - config->v_min;
    const double scaled = (volts_d - config->v_min) / span * (double)code_max;
    if (!(scaled > 0.0)) {
        return 0U;
    }
    if (scaled >= (double)code_max) {
        return code_max;
    }
    return (uint32_t)llround(scaled);
}

float adc_dequantize(const AdcConfig *config, uint32_t code) {
    const uint32_t code_max = adc_code_max(config);
    if (code_max == 0U) {
        return 0.0f;
    }

    const uint32_t clamped = code > code_max ? code_max : code;
    const double span = config->v_max - config->v_min;
    return (float)(config->v_min + (double)clamped / (double)code_max * span);
}

int adc_quantize_buffer(const AdcConfig *config, const float *in, uint32_t *out, size_t count) {
    if (in == NULL || out == NULL || count == 0 || adc_config_validate(config) != 0) {
        return -1;
    }
    for (size_t i = 0; i < count; ++i) {
        out[i] = adc_quantize(config, in[i]);
    }
    return 0;
}

int adc_dequantize_buffer(const AdcConfig *config, const uint32_t *in, float *out, size_t count) {
    if (in == NULL || out == NULL || count == 0 || adc_config_validate(config) != 0) {
        return -1;
    }
    for (size_t i = 0; i < count; ++i) {
        out[i] = adc_dequantize(config, in[i]);
    }
    return 0;
}
