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

#include "spectrogramplot.h"

#include <QDebug>
#include <QElapsedTimer>
#include <QPainter>
#include <QPaintEvent>
#include <QPixmapCache>
#include <QRect>
#include <liquid/liquid.h>
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <limits>
#include "util.h"
#include "ffttask.h"

// Static mutex to protect FFTW plan creation (FFTW is not thread-safe during planning)
QMutex SpectrogramPlot::fftCreationMutex;


SpectrogramPlot::SpectrogramPlot(std::shared_ptr<SampleSource<std::complex<float>>> src) : Plot(src), inputSource(src), fftSize(512), tuner(fftSize, this)
{
    setFFTSize(fftSize);
    zoomLevel = 1;
    nfftSkip = 1;
    powerMax = 0.0f;
    powerMin = -50.0f;
    sampleRate = 0;
    frequencyScaleEnabled = false;
    sigmfAnnotationsEnabled = true;
    sigmfAnnotationLabels = true;
    sigmfAnnotationColors = true;
    rfFreqEnabled = false;

    for (int i = 0; i < 256; i++) {
        float p = (float)i / 256;
        colormap[i] = QColor::fromHsvF(p * 0.83f, 1.0, 1.0 - p).rgba();
    }

    // Set cache sizes large enough to handle high FFT sizes on large displays
    // At fftSize=8192, each tile is only 8 pixels wide (65536/8192)
    // A 4K display (3840px) would need ~480 tiles just for width
    // Default QCache maxCost is 100, which causes cache thrashing
    // Use 4000 to handle ultra-wide and multi-monitor setups at max FFT
    fftCache.setMaxCost(4000);
    pixmapCache.setMaxCost(4000);

    // Initialize thread pool for async FFT computation
    threadPool = QThreadPool::globalInstance();

    // Initialize repaint throttling timer (16ms = ~60 FPS)
    repaintTimer = new QTimer(this);
    repaintTimer->setInterval(16);
    repaintTimer->setSingleShot(true);
    repaintPending = false;
    connect(repaintTimer, &QTimer::timeout, this, &SpectrogramPlot::performRepaint);

    tunerTransform = std::make_shared<TunerTransform>(src);
    connect(&tuner, &Tuner::tunerMoved, this, &SpectrogramPlot::tunerMoved);
}

void SpectrogramPlot::invalidateEvent()
{
    // HACK: this makes sure we update the height for real signals (as InputSource is passed here before the file is opened)
    setFFTSize(fftSize);

    pixmapCache.clear();
    {
        QMutexLocker locker(&cacheMutex);
        fftCache.clear();
    }
    {
        QMutexLocker locker(&pendingMutex);
        pendingTiles.clear();
    }
    {
        QMutexLocker locker(&placeholderMutex);
        placeholderTiles.clear();
    }
    emit repaint();
}

void SpectrogramPlot::paintFront(QPainter &painter, QRect &rect, range_t<size_t> sampleRange)
{
    if (tunerEnabled())
        tuner.paintFront(painter, rect, sampleRange);

    if (frequencyScaleEnabled)
        paintFrequencyScale(painter, rect);

    if (sigmfAnnotationsEnabled)
        paintAnnotations(painter, rect, sampleRange);
}

void SpectrogramPlot::paintFrequencyScale(QPainter &painter, QRect &rect)
{
    if (sampleRate == 0) {
        return;
    }

    if (sampleRate / 2 > UINT64_MAX) {
        return;
    }

    // Get center frequency if RF freq display is enabled
    double centerFreq = 0.0;
    if (rfFreqEnabled) {
        centerFreq = inputSource->getFrequency();
    }

    // At which pixel is F_+sampleRate/2
    int y = rect.y();

    int plotHeight = rect.height();
    if (inputSource->realSignal())
        plotHeight *= 2;

    double bwPerPixel = (double)sampleRate / plotHeight;
    int tickHeight = 50;

    uint64_t bwPerTick = 10 * pow(10, floor(log(bwPerPixel * tickHeight) / log(10)));

    if (bwPerTick < 1) {
        return;
    }

    painter.save();

    QPen pen(Qt::white, 1, Qt::SolidLine);
    painter.setPen(pen);
    QFontMetrics fm(painter.font());

    // Helper lambda to format frequency value
    auto formatFreq = [](char *buf, size_t bufSize, double freq) {
        double absFreq = fabs(freq);
        const char *sign = freq < 0 ? "-" : " ";
        if (absFreq >= 1000000000.0) {
            snprintf(buf, bufSize, "%s%.3f GHz", sign, absFreq / 1000000000.0);
        } else if (absFreq >= 1000000.0) {
            snprintf(buf, bufSize, "%s%.3f MHz", sign, absFreq / 1000000.0);
        } else if (absFreq >= 1000.0) {
            snprintf(buf, bufSize, "%s%.3f kHz", sign, absFreq / 1000.0);
        } else {
            snprintf(buf, bufSize, "%s%.0f Hz", sign, absFreq);
        }
    };

    uint64_t tick = 0;

    while (tick <= sampleRate / 2) {

        int tickpy = plotHeight / 2 - tick / bwPerPixel + y;
        int tickny = plotHeight / 2 + tick / bwPerPixel + y;

        if (!inputSource->realSignal())
            painter.drawLine(0, tickny, 30, tickny);
        painter.drawLine(0, tickpy, 30, tickpy);

        char buf[128];

        if (rfFreqEnabled && centerFreq != 0.0) {
            // RF frequency mode - show actual RF frequencies
            double freqUpper = centerFreq + tick;
            double freqLower = centerFreq - tick;

            formatFreq(buf, sizeof(buf), freqUpper);
            painter.drawText(5, tickpy + 15, buf);

            if (!inputSource->realSignal() && tick != 0) {
                formatFreq(buf, sizeof(buf), freqLower);
                painter.drawText(5, tickny - 5, buf);
            }
        } else {
            // Baseband mode - show offset from center
            if (tick != 0) {
                if (bwPerTick % 1000000000 == 0) {
                    snprintf(buf, sizeof(buf), "-%lu GHz", tick / 1000000000);
                } else if (bwPerTick % 1000000 == 0) {
                    snprintf(buf, sizeof(buf), "-%lu MHz", tick / 1000000);
                } else if(bwPerTick % 1000 == 0) {
                    snprintf(buf, sizeof(buf), "-%lu kHz", tick / 1000);
                } else {
                    snprintf(buf, sizeof(buf), "-%lu Hz", tick);
                }

                if (!inputSource->realSignal())
                    painter.drawText(5, tickny - 5, buf);

                buf[0] = ' ';
                painter.drawText(5, tickpy + 15, buf);
            }
        }

        tick += bwPerTick;
    }

    // Draw small ticks
    bwPerTick /= 10;

    if (bwPerTick >= 1 ) {
        tick = 0;
        while (tick <= sampleRate / 2) {

            int tickpy = plotHeight / 2 - tick / bwPerPixel + y;
            int tickny = plotHeight / 2 + tick / bwPerPixel + y;

            if (!inputSource->realSignal())
                painter.drawLine(0, tickny, 3, tickny);
            painter.drawLine(0, tickpy, 3, tickpy);

            tick += bwPerTick;
        }
    }
    painter.restore();
}

void SpectrogramPlot::paintAnnotations(QPainter &painter, QRect &rect, range_t<size_t> sampleRange)
{
    // Pixel (from the top) at which 0 Hz sits
    int zero = rect.y() + rect.height() / 2;

    painter.save();
    QPen pen(Qt::white, 1, Qt::SolidLine);
    painter.setPen(pen);
    QFontMetrics fm(painter.font());

    visibleAnnotationLocations.clear();

    for (int i = 0; i < inputSource->annotationList.size(); i++) {
        Annotation a = inputSource->annotationList.at(i);

        size_t labelLength = fm.boundingRect(a.label).width() * getStride();

        // Check if:
        //  (1) End of annotation (might be maximum, or end of label text) is still visible in time
        //  (2) Part of the annotation is already visible in time
        //
        // Currently there is no check if the annotation is visible in frequency. This is a
        // possible performance improvement
        //
        size_t start = a.sampleRange.minimum;
        size_t end = std::max(a.sampleRange.minimum + labelLength, a.sampleRange.maximum);

        if(start <= sampleRange.maximum && end >= sampleRange.minimum) {

            double frequency = a.frequencyRange.maximum - inputSource->getFrequency();
            int x = (a.sampleRange.minimum - sampleRange.minimum) / getStride();
            int y = zero - frequency / sampleRate * rect.height();
            int height = (a.frequencyRange.maximum - a.frequencyRange.minimum) / sampleRate * rect.height();
            int width = (a.sampleRange.maximum - a.sampleRange.minimum) / getStride();

            if (sigmfAnnotationColors) {
                painter.setPen(a.boxColor);
            }
            if (sigmfAnnotationLabels) {
                // Draw the label 2 pixels above the box
                painter.drawText(x, y - 2, a.label);
            }
            painter.drawRect(x, y, width, height);

            visibleAnnotationLocations.emplace_back(a, x, y, width, height);
        }
    }

    painter.restore();
}

QString *SpectrogramPlot::mouseAnnotationComment(const QMouseEvent *event) {
    auto pos = event->pos();
    int mouse_x = pos.x();
    int mouse_y = pos.y();

    for (auto& a : visibleAnnotationLocations) {
        if (!a.annotation.comment.isEmpty() && a.isInside(mouse_x, mouse_y)) {
            return &a.annotation.comment;
        }
    }
    return nullptr;
}

void SpectrogramPlot::paintMid(QPainter &painter, QRect &rect, range_t<size_t> sampleRange)
{
    if (!inputSource || inputSource->count() == 0) {
        return;
    }

    size_t sampleOffset = sampleRange.minimum % (getStride() * linesPerTile());
    size_t tileID = sampleRange.minimum - sampleOffset;
    int xoffset = sampleOffset / getStride();

    // Paint first (possibly partial) tile
    painter.drawPixmap(QRect(rect.left(), rect.y(), linesPerTile() - xoffset, height()), *getPixmapTile(tileID), QRect(xoffset, 0, linesPerTile() - xoffset, height()));
    tileID += getStride() * linesPerTile();

    // Paint remaining tiles
    for (int x = linesPerTile() - xoffset; x < rect.right(); x += linesPerTile()) {
        // TODO: don't draw past rect.right()
        // TODO: handle partial final tile
        painter.drawPixmap(QRect(x, rect.y(), linesPerTile(), height()), *getPixmapTile(tileID), QRect(0, 0, linesPerTile(), height()));
        tileID += getStride() * linesPerTile();
    }
}

QPixmap* SpectrogramPlot::getPixmapTile(size_t tile)
{
    TileCacheKey key(fftSize, zoomLevel, nfftSkip, tile);

    // Try to get FFT data
    float *fftTile = getFFTTile(tile);

    if (fftTile != nullptr) {
        // FFT data is available
        bool isPlaceholder = false;
        {
            QMutexLocker locker(&placeholderMutex);
            isPlaceholder = placeholderTiles.contains(key);
        }

        // If we have a real (non-placeholder) cached pixmap, return it
        QPixmap *cachedPixmap = pixmapCache.object(key);
        if (cachedPixmap != nullptr && !isPlaceholder) {
            return cachedPixmap;
        }

        // Need to generate pixmap from FFT data
        QPixmap *obj = new QPixmap(linesPerTile(), fftSize);
        QImage image(linesPerTile(), fftSize, QImage::Format_RGB32);
        float powerRange = -1.0f / std::abs(int(powerMin - powerMax));
        for (int y = 0; y < fftSize; y++) {
            auto scanLine = (QRgb*)image.scanLine(fftSize - y - 1);
            for (int x = 0; x < linesPerTile(); x++) {
                float *fftLine = &fftTile[x * fftSize];
                float normPower = (fftLine[y] - powerMax) * powerRange;
                normPower = clamp(normPower, 0.0f, 1.0f);

                scanLine[x] = colormap[(uint8_t)(normPower * (256 - 1))];
            }
        }
        obj->convertFromImage(image);
        pixmapCache.insert(key, obj);

        // Remove from placeholder set if it was there
        {
            QMutexLocker locker(&placeholderMutex);
            placeholderTiles.remove(key);
        }

        return obj;
    }

    // FFT data not ready - use placeholder
    QPixmap *cachedPixmap = pixmapCache.object(key);
    if (cachedPixmap != nullptr) {
        // Already have a placeholder, return it
        return cachedPixmap;
    }

    // Create new placeholder
    QPixmap *placeholder = new QPixmap(linesPerTile(), fftSize);
    placeholder->fill(Qt::black);
    pixmapCache.insert(key, placeholder, 1);

    // Mark as placeholder
    {
        QMutexLocker locker(&placeholderMutex);
        placeholderTiles.insert(key);
    }

    return placeholder;
}

float* SpectrogramPlot::getFFTTile(size_t tile)
{
    TileCacheKey key(fftSize, zoomLevel, nfftSkip, tile);

    // Check cache first (thread-safe read)
    {
        QMutexLocker locker(&cacheMutex);
        std::array<float, tileSize>* obj = fftCache.object(key);
        if (obj != nullptr) {
            // Cache hit
            return obj->data();
        }
    }

    // Check if already computing this tile AND mark as pending atomically
    // This prevents race condition where multiple tasks could be started for same tile
    {
        QMutexLocker locker(&pendingMutex);
        if (pendingTiles.contains(key)) {
            // Tile is being computed, return nullptr for now
            return nullptr;
        }
        // Mark as pending while still holding the lock
        pendingTiles.insert(key);
    }

    // Start async computation (tile is already marked as pending)
    startAsyncTileComputation(key);
    return nullptr;
}

void SpectrogramPlot::getLine(float *dest, size_t sample)
{
    if (inputSource && fft) {
        // Make sample be the midpoint of the FFT, unless this takes us
        // past the beginning of the inputSource (if we remove the
        // std::max(·, 0), then an ugly red bar appears at the beginning
        // of the spectrogram with large zooms and FFT sizes).
        const auto first_sample = std::max(static_cast<ssize_t>(sample) - fftSize / 2,
                        static_cast<ssize_t>(0));
        auto buffer = inputSource->getSamples(first_sample, fftSize);
        if (buffer == nullptr) {
            auto neg_infinity = -1 * std::numeric_limits<float>::infinity();
            for (int i = 0; i < fftSize; i++, dest++)
                *dest = neg_infinity;
            return;
        }

        for (int i = 0; i < fftSize; i++) {
            buffer[i] *= window[i];
        }

        fft->process(buffer.get(), buffer.get());
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

void SpectrogramPlot::startAsyncTileComputation(const TileCacheKey &key)
{
    // Note: Tile is already marked as pending by getFFTTile() before calling this function
    // This ensures atomic check-and-mark to prevent duplicate tasks

    // Create and start async task
    // Pass 'this' pointer so task can notify us via QMetaObject::invokeMethod
    FFTTileTask *task = new FFTTileTask(
        key.fftSize,
        key.zoomLevel,
        key.nfftSkip,
        key.sample,
        inputSource,
        window.get(),
        tileSize,
        &fftCache,
        &cacheMutex,
        &pendingTiles,
        &pendingMutex,
        this
    );

    // Start task on thread pool
    // When task completes, it will call onTileReady() via QMetaObject::invokeMethod
    threadPool->start(task);
}

int SpectrogramPlot::getStride()
{
    if (zoomLevel == 0 || fftSize == 0) {
        return 1; // Prevent division by zero
    }
    return fftSize * nfftSkip / zoomLevel;
}

float SpectrogramPlot::getTunerPhaseInc()
{
    if (fftSize == 0) {
        return 0.0f; // Prevent division by zero
    }
    auto freq = 0.5f - tuner.centre() / (float)fftSize;
    return freq * Tau;
}

std::vector<float> SpectrogramPlot::getTunerTaps()
{
    if (fftSize == 0) {
        return std::vector<float>(1, 1.0f); // Return minimal valid filter
    }
    float cutoff = tuner.deviation() / (float)fftSize;
    float gain = pow(10.0f, powerMax / -10.0f);
    auto atten = 60.0f;
    auto len = estimate_req_filter_len(std::min(cutoff, 0.05f), atten);
    auto taps = std::vector<float>(len);
    liquid_firdes_kaiser(len, cutoff, atten, 0.0f, taps.data());
    std::transform(taps.begin(), taps.end(), taps.begin(),
                   std::bind(std::multiplies<float>(), std::placeholders::_1, gain));
    return taps;
}

int SpectrogramPlot::linesPerTile()
{
    if (fftSize == 0) {
        return 1; // Prevent division by zero
    }
    return tileSize / fftSize;
}

bool SpectrogramPlot::mouseEvent(QEvent::Type type, QMouseEvent event)
{
    if (tunerEnabled())
        return tuner.mouseEvent(type, event);

    return false;
}

std::shared_ptr<AbstractSampleSource> SpectrogramPlot::output()
{
    return tunerTransform;
}

void SpectrogramPlot::setFFTSize(int size)
{
    // Ensure valid FFT size
    if (size < 2) {
        size = 2;
    }

    qDebug() << "SpectrogramPlot::setFFTSize() - FFT size changed to:" << size;

    float sizeScale = float(size) / float(fftSize);
    fftSize = size;

    // Protect FFTW plan creation with mutex (not thread-safe)
    {
        QMutexLocker locker(&fftCreationMutex);
        fft.reset(new FFT(fftSize));
    }

    window.reset(new float[fftSize]);
    for (int i = 0; i < fftSize; i++) {
        window[i] = 0.5f * (1.0f - cos(Tau * i / (fftSize - 1)));
    }

    if (inputSource->realSignal()) {
        setHeight(fftSize/2);
    } else {
        setHeight(fftSize);
    }
    auto dev = tuner.deviation();
    auto centre = tuner.centre();
    tuner.setHeight(height());
    tuner.setDeviation( dev * sizeScale );
    tuner.setCentre( centre * sizeScale );

    // Clear caches since FFT size changed - old tiles are invalid
    pixmapCache.clear();
    {
        QMutexLocker locker(&cacheMutex);
        fftCache.clear();
    }
    {
        QMutexLocker locker(&pendingMutex);
        pendingTiles.clear();
    }
    {
        QMutexLocker locker(&placeholderMutex);
        placeholderTiles.clear();
    }
}

void SpectrogramPlot::setPowerMax(int power)
{
    qDebug() << "SpectrogramPlot::setPowerMax() - Power max changed to:" << power << "dB";
    powerMax = power;
    pixmapCache.clear();
    {
        QMutexLocker locker(&placeholderMutex);
        placeholderTiles.clear();
    }
    // No need to clear FFT cache, only pixmap rendering changes
    tunerMoved();
}

void SpectrogramPlot::setPowerMin(int power)
{
    qDebug() << "SpectrogramPlot::setPowerMin() - Power min changed to:" << power << "dB";
    powerMin = power;
    pixmapCache.clear();
    {
        QMutexLocker locker(&placeholderMutex);
        placeholderTiles.clear();
    }
    // No need to clear FFT cache, only pixmap rendering changes
    emit repaint();
}

void SpectrogramPlot::setZoomLevel(int zoom)
{
    qDebug() << "SpectrogramPlot::setZoomLevel() - Zoom level changed to:" << zoom;
    zoomLevel = zoom;

    // Clear caches since zoom affects tile stride calculations
    pixmapCache.clear();
    {
        QMutexLocker locker(&cacheMutex);
        fftCache.clear();
    }
    {
        QMutexLocker locker(&pendingMutex);
        pendingTiles.clear();
    }
    {
        QMutexLocker locker(&placeholderMutex);
        placeholderTiles.clear();
    }
}

void SpectrogramPlot::setSkip(int skip)
{
    qDebug() << "SpectrogramPlot::setSkip() - Skip changed to:" << skip;
    nfftSkip = skip;

    // Clear caches since skip affects tile stride calculations
    pixmapCache.clear();
    {
        QMutexLocker locker(&cacheMutex);
        fftCache.clear();
    }
    {
        QMutexLocker locker(&pendingMutex);
        pendingTiles.clear();
    }
    {
        QMutexLocker locker(&placeholderMutex);
        placeholderTiles.clear();
    }
}

void SpectrogramPlot::setSampleRate(double rate)
{
    sampleRate = rate;
}

void SpectrogramPlot::enableScales(bool enabled)
{
   frequencyScaleEnabled = enabled;
}

void SpectrogramPlot::enableAnnotations(bool enabled)
{
   sigmfAnnotationsEnabled = enabled;
}

bool SpectrogramPlot::isAnnotationsEnabled(void)
{
    return sigmfAnnotationsEnabled;
}

void SpectrogramPlot::enableAnnoLabels(bool enabled)
{
    sigmfAnnotationLabels = enabled;
}

void SpectrogramPlot::enableAnnoColors(bool enabled)
{
    sigmfAnnotationColors = enabled;
}

void SpectrogramPlot::enableRfFreq(bool enabled)
{
    rfFreqEnabled = enabled;
}

bool SpectrogramPlot::tunerEnabled()
{
    return (tunerTransform->subscriberCount() > 0);
}

void SpectrogramPlot::tunerMoved()
{
    tunerTransform->setFrequency(getTunerPhaseInc());
    tunerTransform->setTaps(getTunerTaps());
    tunerTransform->setRelativeBandwith(tuner.deviation() * 2.0 / height());

    // TODO: for invalidating traceplot cache, this shouldn't really go here
    QPixmapCache::clear();

    emit repaint();
}

void SpectrogramPlot::onTileReady()
{
    // Tile completed in background, schedule a throttled repaint
    // Don't trigger immediate repaint - use timer to batch multiple tile completions
    if (!repaintPending) {
        repaintPending = true;
        repaintTimer->start();
    }
}

void SpectrogramPlot::performRepaint()
{
    // Actually emit the repaint signal (throttled to max 60 FPS)
    repaintPending = false;
    emit repaint();
}

uint qHash(const TileCacheKey &key, uint seed)
{
    return key.fftSize ^ key.zoomLevel ^ key.nfftSkip ^ key.sample ^ seed;
}
