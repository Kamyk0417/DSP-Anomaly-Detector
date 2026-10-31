/* include/visualize.h */

#ifndef VISUALIZE_H
#define VISUALIZE_H

#include <stdbool.h>
#include <stddef.h>

#include "signal.h"

#define VISUALIZER_WIDTH_DEFAULT 1280
#define VISUALIZER_HEIGHT_DEFAULT 720
#define VISUALIZER_MAX_PARAMETERS 12
#define VISUALIZER_SPECTROGRAM_ROWS 192
#define VISUALIZER_TEXT_CHARS 96

typedef enum {
    VIEW_TIME_DOMAIN = 0,
    VIEW_SPECTRUM,
    VIEW_SPECTROGRAM,
    VIEW_PARAMETERS,
    VIEW_COUNT
} ViewMode;

/* Keyboard commands reported to the caller. View switching (keys 1-4, Tab,
 * Shift+Tab) is handled inside the visualizer itself. */
typedef enum {
    VISUALIZER_COMMAND_NONE = 0,
    VISUALIZER_COMMAND_QUIT,
    VISUALIZER_COMMAND_NEXT_SIGNAL,
    VISUALIZER_COMMAND_NEXT_GENERATOR,
    VISUALIZER_COMMAND_TOGGLE_ADC,
    VISUALIZER_COMMAND_TOGGLE_FILTER,
    VISUALIZER_COMMAND_NEXT_FILTER
} VisualizerCommand;

typedef struct Visualizer Visualizer;

/* Returns an owned visualizer, or NULL when SDL is unavailable. */
Visualizer *visualizer_create(const char *title, int width, int height);
void visualizer_destroy(Visualizer *visualizer);

bool visualizer_is_open(const Visualizer *visualizer);
ViewMode visualizer_view(const Visualizer *visualizer);

/* All data is copied; the caller keeps ownership of its buffers. */
int visualizer_set_signal(Visualizer *visualizer, const SignalBuffer *signal);
int visualizer_set_spectrum(Visualizer *visualizer, const float *magnitude, size_t count,
                            double bin_hz);
int visualizer_push_spectrogram(Visualizer *visualizer, const float *magnitude, size_t count);

int visualizer_add_parameter(Visualizer *visualizer, const char *label, const char *value,
                             double fraction);
void visualizer_clear_parameters(Visualizer *visualizer);
void visualizer_set_status(Visualizer *visualizer, const char *text);

int visualizer_poll(Visualizer *visualizer, VisualizerCommand *command);
int visualizer_render(Visualizer *visualizer);

#endif /* VISUALIZE_H */

