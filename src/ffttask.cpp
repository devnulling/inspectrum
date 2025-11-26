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

#include "ffttask.h"
#include "spectrogramplot.h"
#include "util.h"
#include <limits>
#include <cmath>
#include <QDebug>
#include <QMetaObject>

FFTTileTask::FFTTileTask(
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
) : inputSource(inputSource),
    fftSize(fftSize),
    zoomLevel(zoomLevel),
    nfftSkip(nfftSkip),
    tileSize(tileSize),
    tileSample(tileSample),
    fftCache(fftCache),
    cacheMutex(cacheMutex),
    pendingTiles(pendingTiles),
    pendingMutex(pendingMutex),
    plot(plot)
{
    // Copy window function for thread safety
    windowCopy.reset(new float[fftSize]);
    for (int i = 0; i < fftSize; i++) {
        windowCopy[i] = window[i];
    }

    // Calculate stride with protection against division by zero
    if (zoomLevel == 0 || fftSize == 0) {
        stride = 1;
    } else {
        stride = fftSize * nfftSkip / zoomLevel;
    }

    // Auto-delete this task when done
    setAutoDelete(true);
}

void FFTTileTask::run()
{
    // Validate parameters before processing
    if (fftSize < 2 || stride < 1) {
        // Invalid parameters, remove from pending and exit
        TileCacheKey cacheKey(fftSize, zoomLevel, nfftSkip, tileSample);
        QMutexLocker locker(pendingMutex);
        pendingTiles->remove(cacheKey);
        return;
    }

    // Construct cache key for this tile
    TileCacheKey cacheKey(fftSize, zoomLevel, nfftSkip, tileSample);

    // Check if tile is still needed (might have been invalidated)
    {
        QMutexLocker locker(pendingMutex);
        if (!pendingTiles->contains(cacheKey)) {
            // Tile no longer needed, exit early
            return;
        }
    }

    // Create thread-local FFT object
    // FFTW plan creation is NOT thread-safe, must be protected with mutex
    FFT *fft;
    {
        QMutexLocker locker(&SpectrogramPlot::fftCreationMutex);
        fft = new FFT(fftSize);
    }

    // Allocate storage for the tile (must match tileSize exactly)
    if (tileSize > 65536) {
        // Tile size exceeds our fixed buffer size
        delete fft;
        QMutexLocker locker(pendingMutex);
        pendingTiles->remove(cacheKey);
        return;
    }

    std::array<float, 65536>* destStorage = new std::array<float, 65536>;
    float *ptr = destStorage->data();
    size_t sample = tileSample;

    // Calculate how many FFT lines fit in a tile
    // This must match SpectrogramPlot::linesPerTile() = tileSize / fftSize
    int numLines = tileSize / fftSize;

    // Compute all FFT lines in this tile
    for (int line = 0; line < numLines && (ptr - destStorage->data() + fftSize) <= 65536; line++) {
        computeLine(ptr, sample, fft);
        sample += stride;
        ptr += fftSize;
    }

    // Clean up FFT object
    delete fft;

    // Double-check tile is still needed before caching
    // (could have been invalidated while we were computing)
    {
        QMutexLocker locker(pendingMutex);
        if (!pendingTiles->contains(cacheKey)) {
            // Tile was invalidated while computing, discard result
            delete destStorage;
            return;
        }
        pendingTiles->remove(cacheKey);
    }

    // Store result in cache
    {
        QMutexLocker locker(cacheMutex);
        fftCache->insert(cacheKey, destStorage);
    }

    // Notify that tile is ready using QMetaObject::invokeMethod
    // This queues a call to onTileReady() in the main thread's event loop
    // and works correctly even after this task object is deleted
    QMetaObject::invokeMethod(plot, "onTileReady", Qt::QueuedConnection);
}

void FFTTileTask::computeLine(float *dest, size_t sample, FFT *fft)
{
    if (inputSource && fft) {
        // Make sample be the midpoint of the FFT, unless this takes us
        // past the beginning of the inputSource
        const auto first_sample = std::max(static_cast<ssize_t>(sample) - fftSize / 2,
                        static_cast<ssize_t>(0));
        auto buffer = inputSource->getSamples(first_sample, fftSize);
        if (buffer == nullptr) {
            auto neg_infinity = -1 * std::numeric_limits<float>::infinity();
            for (int i = 0; i < fftSize; i++, dest++)
                *dest = neg_infinity;
            return;
        }

        // Apply window function
        for (int i = 0; i < fftSize; i++) {
            buffer[i] *= windowCopy[i];
        }

        // Perform FFT
        fft->process(buffer.get(), buffer.get());

        // Convert to log power
        const float invFFTSize = 1.0f / fftSize;
        const float logMultiplier = 10.0f / log2f(10.0f);
        for (int i = 0; i < fftSize; i++) {
            // Start from the middle of the FFTW array and wrap
            // to rearrange the data
            int k = i ^ (fftSize >> 1);
            auto s = buffer[k] * invFFTSize;
            float power = s.real() * s.real() + s.imag() * s.imag();
            float logPower = log2f(power) * logMultiplier;
            *dest = logPower;
            dest++;
        }
    }
}
