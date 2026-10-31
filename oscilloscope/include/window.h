/* include/window.h */

#ifndef WINDOW_H
#define WINDOW_H

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    WINDOW_RECTANGULAR,
    WINDOW_HAMMING,
    WINDOW_HANN,
    WINDOW_BLACKMAN,
    WINDOW_TYPE_COUNT
} WindowType;

bool window_type_is_valid(WindowType type);

/* Fills `coefficients` with `count` symmetric window samples in [0, 1]. */
int window_generate(WindowType type, double *coefficients, size_t count);

#endif /* WINDOW_H */
