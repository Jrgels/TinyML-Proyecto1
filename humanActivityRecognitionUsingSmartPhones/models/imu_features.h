
#pragma once
#include <math.h>

namespace har {

constexpr int WINDOW_SIZE = 128;
constexpr int CHANNELS = 6;

inline void extractFeatures(
    const float samples[WINDOW_SIZE][CHANNELS],
    float features[30]
) {
    for (int j = 0; j < CHANNELS; j++) {

        float suma = 0.0f;
        float sumaCuadrados = 0.0f;

        float minimo = samples[0][j];
        float maximo = samples[0][j];

        // Primera pasada: media, mínimo, máximo y RMS
        for (int i = 0; i < WINDOW_SIZE; i++) {

            float x = samples[i][j];

            suma += x;
            sumaCuadrados += x * x;

            if (x < minimo) minimo = x;
            if (x > maximo) maximo = x;
        }

        float media = suma / WINDOW_SIZE;

        // Segunda pasada: varianza
        float sumaDesviaciones = 0.0f;

        for (int i = 0; i < WINDOW_SIZE; i++) {

            float diferencia = samples[i][j] - media;

            sumaDesviaciones += diferencia * diferencia;
        }

        float desviacion = sqrtf(
            sumaDesviaciones / WINDOW_SIZE
        );

        float rms = sqrtf(
            sumaCuadrados / WINDOW_SIZE
        );

        // Mantener el orden usado durante entrenamiento
        features[j]      = media;
        features[6 + j]  = desviacion;
        features[12 + j] = minimo;
        features[18 + j] = maximo;
        features[24 + j] = rms;
    }
}

} // namespace har
