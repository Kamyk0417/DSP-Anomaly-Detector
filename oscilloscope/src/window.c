/* src/window.c */

#include "window.h"

#include <math.h>

#define WINDOW_TWO_PI 6.283185307179586476925286766559

bool window_type_is_valid(WindowType type) {
    return (unsigned)type < (unsigned)WINDOW_TYPE_COUNT;
}

int window_generate(WindowType type, double *coefficients, size_t count) {
    if (!window_type_is_valid(type)) {
        return -1;
    }
    if (coefficients == NULL || count == 0) {
        return -2;
    }
    if (count == 1U) {
        coefficients[0] = 1.0;
        return 0;
    }

    const double denominator = (double)(count - 1U);
    for (size_t i = 0; i < count; ++i) {
        const double angle = WINDOW_TWO_PI * (double)i / denominator;
        double value = 1.0;
        switch (type) {
        case WINDOW_HAMMING:
            value = 0.54 - 0.46 * cos(angle);
            break;
        case WINDOW_HANN:
            value = 0.5 - 0.5 * cos(angle);
            break;
        case WINDOW_BLACKMAN:
            value = 0.42 - 0.5 * cos(angle) + 0.08 * cos(2.0 * angle);
            break;
        case WINDOW_RECTANGULAR:
        default:
            value = 1.0;
            break;
        }
        coefficients[i] = value;
    }
    return 0;
}
