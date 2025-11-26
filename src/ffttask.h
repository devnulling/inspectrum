/*
 *  Copyright (C) 2015, Mike Walters <mike@flomp.net>
 *
 *  This file is part of inspectrum.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <QRunnable>
#include <QCache>
#include <QMutex>
#include <array>
#include <memory>
#include <complex>
#include "fft.h"
#include "samplesource.h"

class TileCacheKey;
class SpectrogramPlot;

class FFTTileTask : public QRunnable
{
public:
    FFTTileTask(
        int fftSize,
        int zoomLevel,
        int nfftSkip,
        size_t tileSample,
        std::shared_ptr<SampleSource<std::complex<float>>> inputSource,
        const float *window,
        size_t tileSize,
        QCache<TileCacheKey, std::array<float, 65536>> *fftCache,
        QMutex *cacheMutex,
        QSet<TileCacheKey> *pendingTiles,
        QMutex *pendingMutex,
        SpectrogramPlot *plot
    );

    void run() override;

private:
    std::shared_ptr<SampleSource<std::complex<float>>> inputSource;
    std::unique_ptr<float[]> windowCopy;
    int fftSize;
    int zoomLevel;
    int nfftSkip;
    size_t tileSize;
    size_t tileSample;
    int stride;

    QCache<TileCacheKey, std::array<float, 65536>> *fftCache;
    QMutex *cacheMutex;
    QSet<TileCacheKey> *pendingTiles;
    QMutex *pendingMutex;
    SpectrogramPlot *plot;

    void computeLine(float *dest, size_t sample, FFT *fft);
};
