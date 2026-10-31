/* src/visualize.c */

#include "visualize.h"

#include <SDL.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VISUALIZER_HEADER_HEIGHT 34
#define VISUALIZER_FOOTER_HEIGHT 30
#define VISUALIZER_AXIS_BAND 14
#define VISUALIZER_MARGIN 16
#define VISUALIZER_GRID_COLUMNS 10
#define VISUALIZER_GRID_ROWS 6
#define VISUALIZER_SPECTROGRAM_WIDTH 512
#define VISUALIZER_SPECTROGRAM_DECAY 0.995
#define VISUALIZER_SPECTROGRAM_GAMMA 0.35
#define VISUALIZER_BAR_HEIGHT 6
#define VISUALIZER_FONT_FIRST 32
#define VISUALIZER_FONT_LAST 90
#define VISUALIZER_FONT_COUNT (VISUALIZER_FONT_LAST - VISUALIZER_FONT_FIRST + 1)
#define VISUALIZER_FONT_WIDTH 5
#define VISUALIZER_FONT_HEIGHT 7
#define VISUALIZER_FONT_ADVANCE 6

/* 5x7 bitmap font, ASCII 32..90, one byte per column, bit 0 is the top row. */
static const unsigned char visualizer_font[VISUALIZER_FONT_COUNT][5] = {
    {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x5F, 0x00, 0x00},
    {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x00, 0x00, 0x00},
    {0x00, 0x00, 0x00, 0x00, 0x00}, {0x63, 0x13, 0x08, 0x64, 0x63},
    {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x00, 0x00, 0x00},
    {0x00, 0x1C, 0x22, 0x41, 0x00}, {0x00, 0x41, 0x22, 0x1C, 0x00},
    {0x2A, 0x1C, 0x3E, 0x1C, 0x2A}, {0x08, 0x08, 0x3E, 0x08, 0x08},
    {0x00, 0x70, 0x30, 0x00, 0x00}, {0x08, 0x08, 0x08, 0x08, 0x08},
    {0x00, 0x60, 0x60, 0x00, 0x00}, {0x60, 0x10, 0x08, 0x04, 0x03},
    {0x3E, 0x51, 0x49, 0x45, 0x3E}, {0x00, 0x42, 0x7F, 0x40, 0x00},
    {0x42, 0x61, 0x51, 0x49, 0x46}, {0x41, 0x49, 0x49, 0x49, 0x36},
    {0x18, 0x14, 0x12, 0x7F, 0x10}, {0x27, 0x45, 0x45, 0x45, 0x39},
    {0x3E, 0x49, 0x49, 0x49, 0x30}, {0x01, 0x71, 0x09, 0x05, 0x03},
    {0x36, 0x49, 0x49, 0x49, 0x36}, {0x06, 0x49, 0x49, 0x49, 0x3E},
    {0x00, 0x36, 0x36, 0x00, 0x00}, {0x00, 0x00, 0x00, 0x00, 0x00},
    {0x08, 0x14, 0x22, 0x41, 0x00}, {0x14, 0x14, 0x14, 0x14, 0x14},
    {0x00, 0x41, 0x22, 0x14, 0x08}, {0x02, 0x01, 0x51, 0x09, 0x06},
    {0x00, 0x00, 0x00, 0x00, 0x00}, {0x7E, 0x09, 0x09, 0x09, 0x7E},
    {0x7F, 0x49, 0x49, 0x49, 0x36}, {0x3E, 0x41, 0x41, 0x41, 0x22},
    {0x7F, 0x41, 0x41, 0x41, 0x3E}, {0x7F, 0x49, 0x49, 0x49, 0x41},
    {0x7F, 0x09, 0x09, 0x09, 0x01}, {0x3E, 0x41, 0x49, 0x49, 0x3A},
    {0x7F, 0x08, 0x08, 0x08, 0x7F}, {0x00, 0x41, 0x7F, 0x41, 0x00},
    {0x20, 0x40, 0x41, 0x3F, 0x01}, {0x7F, 0x08, 0x14, 0x22, 0x41},
    {0x7F, 0x40, 0x40, 0x40, 0x40}, {0x7F, 0x02, 0x04, 0x02, 0x7F},
    {0x7F, 0x02, 0x04, 0x08, 0x7F}, {0x3E, 0x41, 0x41, 0x41, 0x3E},
    {0x7F, 0x09, 0x09, 0x09, 0x06}, {0x3E, 0x41, 0x51, 0x21, 0x5E},
    {0x7F, 0x09, 0x19, 0x29, 0x46}, {0x46, 0x49, 0x49, 0x49, 0x31},
    {0x01, 0x01, 0x7F, 0x01, 0x01}, {0x3F, 0x40, 0x40, 0x40, 0x3F},
    {0x1F, 0x20, 0x40, 0x20, 0x1F}, {0x7F, 0x20, 0x10, 0x20, 0x7F},
    {0x63, 0x14, 0x08, 0x14, 0x63}, {0x03, 0x04, 0x78, 0x04, 0x03},
    {0x61, 0x51, 0x49, 0x45, 0x43},
};

typedef struct {
    char label[VISUALIZER_TEXT_CHARS];
    char value[VISUALIZER_TEXT_CHARS];
    double fraction;
} ParameterRow;

struct Visualizer {
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture *spectrogram_texture;

    char title[VISUALIZER_TEXT_CHARS];
    char status[VISUALIZER_TEXT_CHARS];

    ViewMode view;
    int window_width;
    int window_height;

    float *signal;
    size_t signal_count;
    size_t signal_capacity;
    double signal_sample_rate_hz;

    float *spectrum;
    size_t spectrum_count;
    size_t spectrum_capacity;
    double spectrum_bin_hz;

    ParameterRow parameters[VISUALIZER_MAX_PARAMETERS];
    size_t parameter_count;

    uint32_t *spectrogram_pixels;
    double spectrogram_peak;
    bool spectrogram_dirty;
    size_t spectrogram_rows;
};

static const SDL_Color visualizer_color_background = {10, 12, 18, 255};
static const SDL_Color visualizer_color_panel = {18, 22, 31, 255};
static const SDL_Color visualizer_color_grid = {34, 41, 55, 255};
static const SDL_Color visualizer_color_axis = {58, 70, 92, 255};
static const SDL_Color visualizer_color_text = {226, 234, 244, 255};
static const SDL_Color visualizer_color_text_dim = {120, 134, 156, 255};
static const SDL_Color visualizer_color_trace = {64, 226, 163, 255};
static const SDL_Color visualizer_color_spectrum = {86, 170, 255, 255};
static const SDL_Color visualizer_color_accent = {255, 184, 107, 255};

static const char *const visualizer_view_names[VIEW_COUNT] = {"TIME DOMAIN", "SPECTRUM",
                                                              "SPECTROGRAM", "PARAMETERS"};

static void visualizer_set_color(SDL_Renderer *renderer, SDL_Color color) {
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
}

static void visualizer_copy_text(char *destination, size_t size, const char *source) {
    if (destination == NULL || size == 0U) {
        return;
    }
    if (source == NULL) {
        destination[0] = '\0';
        return;
    }

    size_t index = 0;
    while (index + 1U < size && source[index] != '\0') {
        destination[index] = source[index];
        ++index;
    }
    destination[index] = '\0';
}

static int visualizer_text_width(const char *text, int scale) {
    if (text == NULL || text[0] == '\0') {
        return 0;
    }
    return (int)strlen(text) * VISUALIZER_FONT_ADVANCE * scale - scale;
}

static void visualizer_draw_text(Visualizer *visualizer, int x, int y, int scale, SDL_Color color,
                                 const char *text) {
    if (visualizer == NULL || visualizer->renderer == NULL || text == NULL || scale < 1) {
        return;
    }

    visualizer_set_color(visualizer->renderer, color);

    int cursor = x;
    for (const char *character = text; *character != '\0'; ++character) {
        int code = (int)(unsigned char)*character;
        if (code >= 'a' && code <= 'z') {
            code -= 'a' - 'A';
        }
        if (code >= VISUALIZER_FONT_FIRST && code <= VISUALIZER_FONT_LAST) {
            const unsigned char *glyph = visualizer_font[code - VISUALIZER_FONT_FIRST];
            for (int column = 0; column < VISUALIZER_FONT_WIDTH; ++column) {
                const unsigned char bits = glyph[column];
                int row = 0;
                while (row < VISUALIZER_FONT_HEIGHT) {
                    if ((bits & (1U << row)) != 0U) {
                        int run = 1;
                        while (row + run < VISUALIZER_FONT_HEIGHT &&
                               (bits & (1U << (row + run))) != 0U) {
                            ++run;
                        }
                        const SDL_Rect pixel = {cursor + column * scale, y + row * scale, scale,
                                                run * scale};
                        SDL_RenderFillRect(visualizer->renderer, &pixel);
                        row += run;
                    } else {
                        ++row;
                    }
                }
            }
        }
        cursor += VISUALIZER_FONT_ADVANCE * scale;
    }
}

static void visualizer_format_frequency(char *text, size_t size, double hertz) {
    if (fabs(hertz) >= 1000.0) {
        snprintf(text, size, "%.2f KHZ", hertz / 1000.0);
    } else {
        snprintf(text, size, "%.1f HZ", hertz);
    }
}

static SDL_Rect visualizer_plot_rect(const Visualizer *visualizer) {
    SDL_Rect plot;
    plot.x = VISUALIZER_MARGIN;
    plot.y = VISUALIZER_HEADER_HEIGHT;
    plot.w = visualizer->window_width - 2 * VISUALIZER_MARGIN;
    plot.h = visualizer->window_height - VISUALIZER_HEADER_HEIGHT - VISUALIZER_FOOTER_HEIGHT -
             VISUALIZER_AXIS_BAND;
    if (plot.w < 64) {
        plot.w = 64;
    }
    if (plot.h < 64) {
        plot.h = 64;
    }
    return plot;
}

static void visualizer_draw_grid(Visualizer *visualizer, const SDL_Rect *plot) {
    visualizer_set_color(visualizer->renderer, visualizer_color_grid);
    for (int index = 0; index <= VISUALIZER_GRID_ROWS; ++index) {
        const int y = plot->y + (plot->h - 1) * index / VISUALIZER_GRID_ROWS;
        SDL_RenderDrawLine(visualizer->renderer, plot->x, y, plot->x + plot->w - 1, y);
    }
    for (int index = 0; index <= VISUALIZER_GRID_COLUMNS; ++index) {
        const int x = plot->x + (plot->w - 1) * index / VISUALIZER_GRID_COLUMNS;
        SDL_RenderDrawLine(visualizer->renderer, x, plot->y, x, plot->y + plot->h - 1);
    }
}

static int visualizer_value_to_y(float value, double low, double span, const SDL_Rect *plot) {
    if (!(span > 0.0)) {
        return plot->y + plot->h / 2;
    }
    const double normalized = ((double)value - low) / span;
    int y = plot->y + (int)((1.0 - normalized) * (double)(plot->h - 1));
    if (y < plot->y) {
        y = plot->y;
    }
    if (y > plot->y + plot->h - 1) {
        y = plot->y + plot->h - 1;
    }
    return y;
}

static void visualizer_render_time_domain(Visualizer *visualizer, const SDL_Rect *plot) {
    visualizer_draw_grid(visualizer, plot);

    if (visualizer->signal == NULL || visualizer->signal_count < 2U) {
        visualizer_draw_text(visualizer, plot->x + 8, plot->y + 10, 2, visualizer_color_text_dim,
                             "NO SIGNAL");
        return;
    }

    float minimum = visualizer->signal[0];
    float maximum = visualizer->signal[0];
    for (size_t i = 1; i < visualizer->signal_count; ++i) {
        if (visualizer->signal[i] < minimum) {
            minimum = visualizer->signal[i];
        }
        if (visualizer->signal[i] > maximum) {
            maximum = visualizer->signal[i];
        }
    }

    double low = (double)minimum;
    double high = (double)maximum;
    if (!(high - low > 1.0e-9)) {
        low -= 0.5;
        high += 0.5;
    } else {
        const double padding = 0.05 * (high - low);
        low -= padding;
        high += padding;
    }
    const double span = high - low;

    if (low < 0.0 && high > 0.0) {
        const int zero_y = visualizer_value_to_y(0.0f, low, span, plot);
        visualizer_set_color(visualizer->renderer, visualizer_color_axis);
        SDL_RenderDrawLine(visualizer->renderer, plot->x, zero_y, plot->x + plot->w - 1, zero_y);
    }

    visualizer_set_color(visualizer->renderer, visualizer_color_trace);
    const size_t sample_count = visualizer->signal_count;
    if ((size_t)plot->w >= sample_count) {
        int previous_x = plot->x;
        int previous_y = visualizer_value_to_y(visualizer->signal[0], low, span, plot);
        for (size_t i = 1; i < sample_count; ++i) {
            const int x = plot->x +
                          (int)((double)i * (double)(plot->w - 1) / (double)(sample_count - 1U));
            const int y = visualizer_value_to_y(visualizer->signal[i], low, span, plot);
            SDL_RenderDrawLine(visualizer->renderer, previous_x, previous_y, x, y);
            previous_x = x;
            previous_y = y;
        }
    } else {
        int previous_minimum_y = -1;
        int previous_maximum_y = -1;
        for (int column = 0; column < plot->w; ++column) {
            const size_t first =
                (size_t)((double)column * (double)sample_count / (double)plot->w);
            size_t last =
                (size_t)((double)(column + 1) * (double)sample_count / (double)plot->w);
            if (last <= first) {
                last = first + 1U;
            }
            if (last > sample_count) {
                last = sample_count;
            }
            if (first >= last) {
                continue;
            }

            float column_minimum = visualizer->signal[first];
            float column_maximum = visualizer->signal[first];
            for (size_t i = first + 1U; i < last; ++i) {
                if (visualizer->signal[i] < column_minimum) {
                    column_minimum = visualizer->signal[i];
                }
                if (visualizer->signal[i] > column_maximum) {
                    column_maximum = visualizer->signal[i];
                }
            }
            const int top = visualizer_value_to_y(column_maximum, low, span, plot);
            const int bottom = visualizer_value_to_y(column_minimum, low, span, plot);
            SDL_RenderDrawLine(visualizer->renderer, plot->x + column, top, plot->x + column,
                               bottom);
            if (previous_minimum_y >= 0) {
                SDL_RenderDrawLine(visualizer->renderer, plot->x + column - 1, previous_maximum_y,
                                   plot->x + column, top);
                SDL_RenderDrawLine(visualizer->renderer, plot->x + column - 1, previous_minimum_y,
                                   plot->x + column, bottom);
            }
            previous_minimum_y = bottom;
            previous_maximum_y = top;
        }
    }

    char label[VISUALIZER_TEXT_CHARS];
    snprintf(label, sizeof(label), "%.2f V", high);
    visualizer_draw_text(visualizer, plot->x + 4, plot->y + 2, 1, visualizer_color_text_dim,
                         label);
    snprintf(label, sizeof(label), "%.2f V", low);
    visualizer_draw_text(visualizer, plot->x + 4, plot->y + plot->h - 9, 1,
                         visualizer_color_text_dim, label);

    snprintf(label, sizeof(label), "0 MS");
    visualizer_draw_text(visualizer, plot->x, plot->y + plot->h + 4, 1, visualizer_color_text_dim,
                         label);
    snprintf(label, sizeof(label), "%.2f MS",
             (double)sample_count / visualizer->signal_sample_rate_hz * 1000.0);
    visualizer_draw_text(visualizer,
                         plot->x + plot->w - visualizer_text_width(label, 1),
                         plot->y + plot->h + 4, 1, visualizer_color_text_dim, label);
}

static void visualizer_render_spectrum(Visualizer *visualizer, const SDL_Rect *plot) {
    visualizer_draw_grid(visualizer, plot);

    if (visualizer->spectrum == NULL || visualizer->spectrum_count < 2U) {
        visualizer_draw_text(visualizer, plot->x + 8, plot->y + 10, 2, visualizer_color_text_dim,
                             "NO SPECTRUM");
        return;
    }

    float spectrum_peak = 0.0f;
    for (size_t i = 0; i < visualizer->spectrum_count; ++i) {
        if (visualizer->spectrum[i] > spectrum_peak) {
            spectrum_peak = visualizer->spectrum[i];
        }
    }
    if (!(spectrum_peak > 0.0f)) {
        spectrum_peak = 1.0f;
    }

    visualizer_set_color(visualizer->renderer, visualizer_color_spectrum);
    for (int column = 0; column < plot->w; ++column) {
        const size_t first =
            (size_t)((double)column * (double)visualizer->spectrum_count / (double)plot->w);
        size_t last = (size_t)((double)(column + 1) * (double)visualizer->spectrum_count /
                               (double)plot->w);
        if (last <= first) {
            last = first + 1U;
        }
        if (last > visualizer->spectrum_count) {
            last = visualizer->spectrum_count;
        }

        float column_peak = 0.0f;
        for (size_t i = first; i < last; ++i) {
            if (visualizer->spectrum[i] > column_peak) {
                column_peak = visualizer->spectrum[i];
            }
        }

        const double normalized = (double)column_peak / (double)spectrum_peak;
        const int height = (int)(normalized * (double)(plot->h - 1));
        if (height > 0) {
            const SDL_Rect bar = {plot->x + column, plot->y + plot->h - height, 1, height};
            SDL_RenderFillRect(visualizer->renderer, &bar);
        }
    }

    char label[VISUALIZER_TEXT_CHARS];
    visualizer_draw_text(visualizer, plot->x, plot->y + plot->h + 4, 1, visualizer_color_text_dim,
                         "0 HZ");
    visualizer_format_frequency(
        label, sizeof(label),
        visualizer->spectrum_bin_hz * (double)(visualizer->spectrum_count - 1U));
    visualizer_draw_text(visualizer,
                         plot->x + plot->w - visualizer_text_width(label, 1),
                         plot->y + plot->h + 4, 1, visualizer_color_text_dim, label);
}

static uint32_t visualizer_colormap(double value) {
    static const double stops[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
    static const unsigned char red[5] = {8, 24, 48, 250, 255};
    static const unsigned char green[5] = {10, 56, 190, 224, 255};
    static const unsigned char blue[5] = {42, 168, 232, 64, 255};

    if (!(value > 0.0)) {
        value = 0.0;
    }
    if (value > 1.0) {
        value = 1.0;
    }

    size_t index = 0;
    while (index < 4U && value > stops[index + 1U]) {
        ++index;
    }
    const double fraction = (value - stops[index]) / (stops[index + 1U] - stops[index]);
    const unsigned char r = (unsigned char)((double)red[index] +
                                            fraction * ((double)red[index + 1U] -
                                                        (double)red[index]));
    const unsigned char g = (unsigned char)((double)green[index] +
                                            fraction * ((double)green[index + 1U] -
                                                        (double)green[index]));
    const unsigned char b = (unsigned char)((double)blue[index] +
                                            fraction * ((double)blue[index + 1U] -
                                                        (double)blue[index]));
    return 0xFF000000U | ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
}

static void visualizer_render_spectrogram(Visualizer *visualizer, const SDL_Rect *plot) {
    if (visualizer->spectrogram_texture == NULL || visualizer->spectrogram_rows == 0U) {
        visualizer_draw_grid(visualizer, plot);
        visualizer_draw_text(visualizer, plot->x + 8, plot->y + 10, 2, visualizer_color_text_dim,
                             "NO SPECTROGRAM DATA");
        return;
    }

    if (visualizer->spectrogram_dirty) {
        SDL_UpdateTexture(visualizer->spectrogram_texture, NULL, visualizer->spectrogram_pixels,
                          VISUALIZER_SPECTROGRAM_WIDTH * (int)sizeof(uint32_t));
        visualizer->spectrogram_dirty = false;
    }

    size_t filled_rows = visualizer->spectrogram_rows;
    if (filled_rows > VISUALIZER_SPECTROGRAM_ROWS) {
        filled_rows = VISUALIZER_SPECTROGRAM_ROWS;
    }
    const SDL_Rect source = {0, (int)(VISUALIZER_SPECTROGRAM_ROWS - filled_rows),
                             VISUALIZER_SPECTROGRAM_WIDTH, (int)filled_rows};
    SDL_RenderCopy(visualizer->renderer, visualizer->spectrogram_texture, &source, plot);

    visualizer_draw_text(visualizer, plot->x + 4, plot->y + 2, 1, visualizer_color_text,
                         "NEWEST");
    visualizer_draw_text(visualizer, plot->x + 4, plot->y + plot->h - 9, 1,
                         visualizer_color_text_dim, "OLDEST");
}

static void visualizer_render_parameters(Visualizer *visualizer, const SDL_Rect *plot) {
    if (visualizer->parameter_count == 0U) {
        visualizer_draw_text(visualizer, plot->x, plot->y + 10, 2, visualizer_color_text_dim,
                             "NO PARAMETERS");
        return;
    }

    const int row_count = (int)visualizer->parameter_count;
    int row_height = plot->h / row_count;
    if (row_height > 46) {
        row_height = 46;
    }
    const int scale = row_height >= 30 ? 2 : 1;
    const int bar_offset = scale == 2 ? 18 : 11;

    for (int index = 0; index < row_count; ++index) {
        const ParameterRow *parameter = &visualizer->parameters[index];
        const int top = plot->y + index * row_height;

        visualizer_draw_text(visualizer, plot->x, top, scale, visualizer_color_text,
                             parameter->label);
        visualizer_draw_text(visualizer,
                             plot->x + plot->w - visualizer_text_width(parameter->value, scale),
                             top, scale, visualizer_color_accent, parameter->value);

        const int bar_y = top + bar_offset;
        if (bar_y + VISUALIZER_BAR_HEIGHT < plot->y + plot->h) {
            const SDL_Rect track = {plot->x, bar_y, plot->w, VISUALIZER_BAR_HEIGHT};
            visualizer_set_color(visualizer->renderer, visualizer_color_grid);
            SDL_RenderFillRect(visualizer->renderer, &track);

            const int filled = (int)(parameter->fraction * (double)plot->w);
            if (filled > 0) {
                const SDL_Rect bar = {plot->x, bar_y, filled, VISUALIZER_BAR_HEIGHT};
                visualizer_set_color(visualizer->renderer, visualizer_color_trace);
                SDL_RenderFillRect(visualizer->renderer, &bar);
            }
        }
    }
}

static void visualizer_render_header(Visualizer *visualizer) {
    const SDL_Rect panel = {0, 0, visualizer->window_width, VISUALIZER_HEADER_HEIGHT - 8};
    visualizer_set_color(visualizer->renderer, visualizer_color_panel);
    SDL_RenderFillRect(visualizer->renderer, &panel);

    visualizer_draw_text(visualizer, VISUALIZER_MARGIN, 9, 2, visualizer_color_text,
                         visualizer->title);

    const char *view_name = visualizer_view_names[visualizer->view];
    visualizer_draw_text(visualizer,
                         visualizer->window_width - VISUALIZER_MARGIN -
                             visualizer_text_width(view_name, 2),
                         9, 2, visualizer_color_accent, view_name);

    const SDL_Rect separator = {0, VISUALIZER_HEADER_HEIGHT - 8, visualizer->window_width, 1};
    visualizer_set_color(visualizer->renderer, visualizer_color_axis);
    SDL_RenderFillRect(visualizer->renderer, &separator);
}

static void visualizer_render_footer(Visualizer *visualizer) {
    const int top = visualizer->window_height - VISUALIZER_FOOTER_HEIGHT + 6;

    visualizer_draw_text(visualizer, VISUALIZER_MARGIN, top, 1, visualizer_color_text_dim,
                         "1-4 TAB VIEW  G SIGNAL  T WAVE  A ADC  F FILTER  Y IIR  ESC QUIT");

    if (visualizer->status[0] != '\0') {
        visualizer_draw_text(visualizer, VISUALIZER_MARGIN, top + 12, 1, visualizer_color_text,
                             visualizer->status);
    }
}

Visualizer *visualizer_create(const char *title, int width, int height) {
    if (width <= 0 || height <= 0) {
        return NULL;
    }
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        return NULL;
    }

    Visualizer *visualizer = calloc(1U, sizeof(*visualizer));
    if (visualizer == NULL) {
        SDL_Quit();
        return NULL;
    }

    visualizer->view = VIEW_TIME_DOMAIN;
    visualizer_copy_text(visualizer->title, sizeof(visualizer->title), title);
    visualizer->window_width = width;
    visualizer->window_height = height;

    visualizer->window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          width, height, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    if (visualizer->window == NULL) {
        visualizer_destroy(visualizer);
        return NULL;
    }

    visualizer->renderer = SDL_CreateRenderer(
        visualizer->window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (visualizer->renderer == NULL) {
        visualizer->renderer = SDL_CreateRenderer(visualizer->window, -1, SDL_RENDERER_SOFTWARE);
    }
    if (visualizer->renderer == NULL) {
        visualizer_destroy(visualizer);
        return NULL;
    }

    visualizer->spectrogram_pixels = calloc(
        (size_t)VISUALIZER_SPECTROGRAM_WIDTH * (size_t)VISUALIZER_SPECTROGRAM_ROWS,
        sizeof(uint32_t));
    if (visualizer->spectrogram_pixels == NULL) {
        visualizer_destroy(visualizer);
        return NULL;
    }

    visualizer->spectrogram_texture =
        SDL_CreateTexture(visualizer->renderer, SDL_PIXELFORMAT_ARGB8888,
                          SDL_TEXTUREACCESS_STREAMING, VISUALIZER_SPECTROGRAM_WIDTH,
                          VISUALIZER_SPECTROGRAM_ROWS);
    if (visualizer->spectrogram_texture == NULL) {
        visualizer_destroy(visualizer);
        return NULL;
    }

    return visualizer;
}

void visualizer_destroy(Visualizer *visualizer) {
    if (visualizer == NULL) {
        return;
    }

    if (visualizer->spectrogram_texture != NULL) {
        SDL_DestroyTexture(visualizer->spectrogram_texture);
    }
    if (visualizer->renderer != NULL) {
        SDL_DestroyRenderer(visualizer->renderer);
    }
    if (visualizer->window != NULL) {
        SDL_DestroyWindow(visualizer->window);
    }

    free(visualizer->signal);
    free(visualizer->spectrum);
    free(visualizer->spectrogram_pixels);
    free(visualizer);

    SDL_Quit();
}

bool visualizer_is_open(const Visualizer *visualizer) {
    return visualizer != NULL && visualizer->window != NULL && visualizer->renderer != NULL;
}

ViewMode visualizer_view(const Visualizer *visualizer) {
    if (visualizer == NULL) {
        return VIEW_TIME_DOMAIN;
    }
    return visualizer->view;
}

static int visualizer_reserve(float **buffer, size_t *capacity, size_t count) {
    if (count <= *capacity) {
        return 0;
    }
    float *grown = realloc(*buffer, count * sizeof(float));
    if (grown == NULL) {
        return -1;
    }
    *buffer = grown;
    *capacity = count;
    return 0;
}

int visualizer_set_signal(Visualizer *visualizer, const SignalBuffer *signal) {
    if (visualizer == NULL || signal == NULL || signal->samples == NULL || signal->count == 0U) {
        return -1;
    }
    if (visualizer_reserve(&visualizer->signal, &visualizer->signal_capacity, signal->count) !=
        0) {
        return -2;
    }

    memcpy(visualizer->signal, signal->samples, signal->count * sizeof(float));
    visualizer->signal_count = signal->count;
    visualizer->signal_sample_rate_hz = signal->sample_rate_hz;
    return 0;
}

int visualizer_set_spectrum(Visualizer *visualizer, const float *magnitude, size_t count,
                            double bin_hz) {
    if (visualizer == NULL || magnitude == NULL || count == 0U) {
        return -1;
    }
    if (visualizer_reserve(&visualizer->spectrum, &visualizer->spectrum_capacity, count) != 0) {
        return -2;
    }

    memcpy(visualizer->spectrum, magnitude, count * sizeof(float));
    visualizer->spectrum_count = count;
    visualizer->spectrum_bin_hz = bin_hz;
    return 0;
}

int visualizer_push_spectrogram(Visualizer *visualizer, const float *magnitude, size_t count) {
    if (visualizer == NULL || visualizer->spectrogram_pixels == NULL) {
        return -1;
    }
    if (magnitude == NULL || count == 0U) {
        return -2;
    }

    float row_peak = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        if (magnitude[i] > row_peak) {
            row_peak = magnitude[i];
        }
    }

    visualizer->spectrogram_peak *= VISUALIZER_SPECTROGRAM_DECAY;
    if ((double)row_peak > visualizer->spectrogram_peak) {
        visualizer->spectrogram_peak = (double)row_peak;
    }
    if (!(visualizer->spectrogram_peak > 0.0)) {
        visualizer->spectrogram_peak = 1.0;
    }

    const size_t row_pixels = VISUALIZER_SPECTROGRAM_WIDTH;
    const size_t row_bytes = row_pixels * sizeof(uint32_t);
    memmove(visualizer->spectrogram_pixels, visualizer->spectrogram_pixels + row_pixels,
            (VISUALIZER_SPECTROGRAM_ROWS - 1U) * row_bytes);

    uint32_t *row =
        visualizer->spectrogram_pixels + (VISUALIZER_SPECTROGRAM_ROWS - 1U) * row_pixels;
    for (size_t column = 0; column < row_pixels; ++column) {
        const size_t first = column * count / row_pixels;
        size_t last = (column + 1U) * count / row_pixels;
        if (last <= first) {
            last = first + 1U;
        }
        if (last > count) {
            last = count;
        }

        float column_peak = 0.0f;
        for (size_t i = first; i < last; ++i) {
            if (magnitude[i] > column_peak) {
                column_peak = magnitude[i];
            }
        }
        const double normalized = (double)column_peak / visualizer->spectrogram_peak;
        row[column] = visualizer_colormap(pow(normalized, VISUALIZER_SPECTROGRAM_GAMMA));
    }

    visualizer->spectrogram_dirty = true;
    ++visualizer->spectrogram_rows;
    return 0;
}

int visualizer_add_parameter(Visualizer *visualizer, const char *label, const char *value,
                             double fraction) {
    if (visualizer == NULL || label == NULL) {
        return -1;
    }
    if (visualizer->parameter_count >= VISUALIZER_MAX_PARAMETERS) {
        return -2;
    }

    ParameterRow *parameter = &visualizer->parameters[visualizer->parameter_count];
    visualizer_copy_text(parameter->label, sizeof(parameter->label), label);
    visualizer_copy_text(parameter->value, sizeof(parameter->value), value);
    if (!(fraction > 0.0)) {
        fraction = 0.0;
    }
    if (fraction > 1.0) {
        fraction = 1.0;
    }
    parameter->fraction = fraction;
    ++visualizer->parameter_count;
    return 0;
}

void visualizer_clear_parameters(Visualizer *visualizer) {
    if (visualizer == NULL) {
        return;
    }
    visualizer->parameter_count = 0;
}

void visualizer_set_status(Visualizer *visualizer, const char *text) {
    if (visualizer == NULL) {
        return;
    }
    visualizer_copy_text(visualizer->status, sizeof(visualizer->status), text);
}

int visualizer_poll(Visualizer *visualizer, VisualizerCommand *command) {
    if (visualizer == NULL || command == NULL || visualizer->window == NULL) {
        return -1;
    }
    *command = VISUALIZER_COMMAND_NONE;

    SDL_Event event;
    while (SDL_PollEvent(&event) != 0) {
        if (event.type == SDL_QUIT) {
            *command = VISUALIZER_COMMAND_QUIT;
        } else if (event.type == SDL_WINDOWEVENT) {
            if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                visualizer->window_width = event.window.data1;
                visualizer->window_height = event.window.data2;
            }
        } else if (event.type == SDL_KEYDOWN) {
            const SDL_Keycode key = event.key.keysym.sym;
            const bool shift = (event.key.keysym.mod & KMOD_SHIFT) != 0;
            switch (key) {
            case SDLK_ESCAPE:
            case SDLK_q:
                *command = VISUALIZER_COMMAND_QUIT;
                break;
            case SDLK_1:
                visualizer->view = VIEW_TIME_DOMAIN;
                break;
            case SDLK_2:
                visualizer->view = VIEW_SPECTRUM;
                break;
            case SDLK_3:
                visualizer->view = VIEW_SPECTROGRAM;
                break;
            case SDLK_4:
                visualizer->view = VIEW_PARAMETERS;
                break;
            case SDLK_TAB:
                visualizer->view = (ViewMode)((visualizer->view + (shift ? VIEW_COUNT - 1 : 1)) %
                                              VIEW_COUNT);
                break;
            case SDLK_g:
                *command = VISUALIZER_COMMAND_NEXT_SIGNAL;
                break;
            case SDLK_t:
                *command = VISUALIZER_COMMAND_NEXT_GENERATOR;
                break;
            case SDLK_a:
                *command = VISUALIZER_COMMAND_TOGGLE_ADC;
                break;
            case SDLK_f:
                *command = VISUALIZER_COMMAND_TOGGLE_FILTER;
                break;
            case SDLK_y:
                *command = VISUALIZER_COMMAND_NEXT_FILTER;
                break;
            default:
                break;
            }
        }
    }
    return 0;
}

int visualizer_render(Visualizer *visualizer) {
    if (visualizer == NULL || visualizer->renderer == NULL) {
        return -1;
    }

    int width = visualizer->window_width;
    int height = visualizer->window_height;
    if (SDL_GetRendererOutputSize(visualizer->renderer, &width, &height) == 0) {
        visualizer->window_width = width;
        visualizer->window_height = height;
    }

    visualizer_set_color(visualizer->renderer, visualizer_color_background);
    SDL_RenderClear(visualizer->renderer);

    const SDL_Rect plot = visualizer_plot_rect(visualizer);
    switch (visualizer->view) {
    case VIEW_TIME_DOMAIN:
        visualizer_render_time_domain(visualizer, &plot);
        break;
    case VIEW_SPECTRUM:
        visualizer_render_spectrum(visualizer, &plot);
        break;
    case VIEW_SPECTROGRAM:
        visualizer_render_spectrogram(visualizer, &plot);
        break;
    case VIEW_PARAMETERS:
        visualizer_render_parameters(visualizer, &plot);
        break;
    default:
        break;
    }

    visualizer_render_header(visualizer);
    visualizer_render_footer(visualizer);

    SDL_RenderPresent(visualizer->renderer);
    return 0;
}

