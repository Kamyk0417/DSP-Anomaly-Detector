/* include/adc.h */

#ifndef ADC_H
#define ADC_H

#include <stddef.h>
#include <stdint.h>

#define ADC_BITS_DEFAULT 12U
#define ADC_V_MIN_DEFAULT (-5.0)
#define ADC_V_MAX_DEFAULT (5.0)
#define ADC_SAMPLE_RATE_HZ_DEFAULT 100000.0
#define ADC_SAMPLE_RATE_HZ_MIN 1000.0
#define ADC_SAMPLE_RATE_HZ_MAX 1000000.0

typedef struct {
    unsigned bits;
    double v_min;
    double v_max;
} AdcConfig;

int adc_config_default(AdcConfig *config);
int adc_config_validate(const AdcConfig *config);

/* Returns 2^bits - 1, or 0 when the configuration is invalid. */
uint32_t adc_code_max(const AdcConfig *config);

/* Quantization: code = round((v - v_min) / (v_max - v_min) * (2^bits - 1)),
 * clipped to [0, 2^bits - 1]. NaN maps to code 0; the result is 0 when the
 * configuration is invalid. */
uint32_t adc_quantize(const AdcConfig *config, float volts);
float adc_dequantize(const AdcConfig *config, uint32_t code);

int adc_quantize_buffer(const AdcConfig *config, const float *in, uint32_t *out, size_t count);
int adc_dequantize_buffer(const AdcConfig *config, const uint32_t *in, float *out, size_t count);

#endif /* ADC_H */
