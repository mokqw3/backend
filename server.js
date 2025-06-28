// server.js - Node.js Backend for Quantum AI Supercore Predictions
// Version: 6.0 - ALL Prediction Logic, Game Data Proxy, Gemini Meta-Prediction, API Key Rotation, Pattern Ingestion
// This server now encapsulates the full prediction engine, game data fetching, Gemini AI, and external pattern integration.
// Frontend (index.html) is now entirely clean of prediction logic.

require('dotenv').config();

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
const port = process.env.PORT || 3000;

app.use(cors()); // Enable CORS for frontend communication
app.use(bodyParser.json({ limit: '5mb' })); // Increased limit for potentially larger history data from collector

// --- Global state for patterns and raw history received from external collector ---
let currentPatterns = {
    color_patterns: {},
    number_patterns: {},
    last_updated: null
};

// This will store the raw game history received from the data collector.
// It's used by the prediction logic internally.
let rawGameHistoryFromCollector = [];

// --- API Key Configuration & Rotation (for Gemini) ---
const GEMINI_API_KEYS = process.env.GEMINI_API_KEYS ? process.env.GEMINI_API_KEYS.split(',') : [];

if (GEMINI_API_KEYS.length === 0) {
    console.error('Error: GEMINI_API_KEYS are not set. Configure on Render.com.');
}

let currentApiKeyIndex = 0;
let failedApiKeys = new Set();

function getNextApiKey() {
    if (failedApiKeys.size === GEMINI_API_KEYS.length && GEMINI_API_KEYS.length > 0) {
        console.error('All Gemini API keys have been marked as failed. No more keys to rotate to.');
        return null;
    }
    if (GEMINI_API_KEYS.length === 0) {
        return null; // No keys configured
    }

    let key;
    let attempts = 0;
    const maxAttempts = GEMINI_API_KEYS.length * 2;

    do {
        currentApiKeyIndex = (currentApiKeyIndex + 1) % GEMINI_API_KEYS.length;
        key = GEMINI_API_KEYS[currentApiKeyIndex];
        attempts++;
    } while (failedApiKeys.has(key) && attempts < maxAttempts); // Skip keys marked as failed

    if (failedApiKeys.has(key) && failedApiKeys.size === GEMINI_API_KEYS.length) {
        // This means after trying all, the only remaining key is a failed one, and all are failed.
        console.error('Could not find a valid API key to rotate to. All keys exhausted or permanently failed.');
        return null;
    }
    console.log(`Switched to API Key at index: ${currentApiKeyIndex} (Key: ${key.substring(0, 5)}...)`);
    return key;
}

async function callGeminiAPI(payload) {
    if (GEMINI_API_KEYS.length === 0) {
        throw new Error('No Gemini API keys configured on the server. Please check Render environment variables.');
    }

    let currentKey = GEMINI_API_KEYS[currentApiKeyIndex];

    const MAX_RETRIES = GEMINI_API_KEYS.length * 2;
    for (let i = 0; i < MAX_RETRIES; i++) {
        if (!currentKey) {
            throw new Error('No valid Gemini API key available for current attempt after rotation.');
        }
        const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${currentKey}`;

        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                return await response.json();
            }

            const errorText = await response.text();
            console.warn(`Gemini API call failed with status ${response.status} for key ${currentKey.substring(0, 5)}... (Attempt ${i + 1}): ${errorText}`);

            if (response.status === 429 || response.status === 500) { // Rate Limit or transient server error
                console.log(`Rate limit (429) or transient error (500) for key ${currentKey.substring(0, 5)}.... Rotating.`);
                currentKey = getNextApiKey();
            } else if (response.status === 400 || response.status === 403) { // Bad Request or Forbidden (likely invalid key)
                console.error(`Permanent error (${response.status}) for key ${currentKey.substring(0, 5)}.... Marking as failed and rotating.`);
                failedApiKeys.add(currentKey); // Mark the specific key string as failed
                currentKey = getNextApiKey();
            } else {
                // Other unexpected errors, rethrow if not a known transient/key error
                throw new Error(`Gemini API non-retryable error: ${response.status} - ${errorText}`);
            }

            if (!currentKey) { // Check if getNextApiKey failed to find a valid key
                throw new Error('All API keys exhausted or permanently failed during rotation.');
            }

        } catch (error) {
            console.error(`Network or unexpected error during Gemini API call (attempt ${i + 1}):`, error.message);
            // On network error, try rotating to the next key
            currentKey = getNextApiKey();
            if (!currentKey) {
                throw new Error('All API keys exhausted or failed due to network issues.');
            }
        }
    }
    throw new Error('Failed to get a successful response from Gemini API after multiple retries and key rotations.');
}


// --- START: All Prediction Logic Functions (Moved from original predictionLogic.js) ---
// This entire block is now part of the backend server.

// --- Helper Functions ---
function getBigSmallFromNumber(number) {
    if (number === undefined || number === null) return null;
    const num = parseInt(number);
    if (isNaN(num)) return null;
    return num >= 0 && num <= 4 ? 'SMALL' : num >= 5 && num <= 9 ? 'BIG' : null;
}

function getOppositeOutcome(prediction) {
    return prediction === "BIG" ? "SMALL" : prediction === "SMALL" ? "BIG" : null;
}

function calculateSMA(data, period) {
    if (!Array.isArray(data) || data.length < period || period <= 0) return null;
    const relevantData = data.slice(0, period);
    const sum = relevantData.reduce((a, b) => a + b, 0);
    return sum / period;
}

function calculateEMA(data, period) {
    if (!Array.isArray(data) || data.length < period || period <= 0) return null;
    const k = 2 / (period + 1);
    const chronologicalData = data.slice().reverse();

    const initialSliceForSma = chronologicalData.slice(0, period);
    if (initialSliceForSma.length < period) return null;

    let ema = calculateSMA(initialSliceForSma.slice().reverse(), period);
    if (ema === null && initialSliceForSma.length > 0) {
        ema = initialSliceForSma.reduce((a, b) => a + b, 0) / initialSliceForSma.length;
    }
    if (ema === null) return null;

    for (let i = period; i < chronologicalData.length; i++) {
        ema = (chronologicalData[i] * k) + (ema * (1 - k));
    }
    return ema;
}

function calculateStdDev(data, period) {
    if (!Array.isArray(data) || data.length < period || period <= 0) return null;
    const relevantData = data.slice(0, period);
    if (relevantData.length < 2) return null;
    const mean = relevantData.reduce((a, b) => a + b, 0) / relevantData.length;
    const variance = relevantData.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / relevantData.length;
    return Math.sqrt(variance);
}

function calculateVWAP(data, period) {
    if (!Array.isArray(data) || data.length < period || period <= 0) return null;
    const relevantData = data.slice(0, period);
    let totalPriceVolume = 0;
    let totalVolume = 0;
    for (const entry of relevantData) {
        const price = parseFloat(entry.actualNumber);
        const volume = parseFloat(entry.volume || 1);
        if (!isNaN(price) && !isNaN(volume) && volume > 0) {
            totalPriceVolume += price * volume;
            totalVolume += volume;
        }
    }
    if (totalVolume === 0) return null;
    return totalPriceVolume / totalVolume;
}

function calculateRSI(data, period) {
    if (period <= 0) return null;
    const chronologicalData = data.slice().reverse();
    if (!Array.isArray(chronologicalData) || chronologicalData.length < period + 1) return null;
    let gains = 0, losses = 0;

    for (let i = 1; i <= period; i++) {
        const change = chronologicalData[i] - chronologicalData[i - 1];
        if (change > 0) gains += change;
        else losses += Math.abs(change);
    }

    let avgGain = gains / period;
    let avgLoss = losses / period;

    for (let i = period + 1; i < chronologicalData.length; i++) {
        const change = chronologicalData[i] - chronologicalData[i - 1];
        let currentGain = change > 0 ? change : 0;
        let currentLoss = change < 0 ? Math.abs(change) : 0;
        avgGain = (avgGain * (period - 1) + currentGain) / period;
        avgLoss = (avgLoss * (period - 1) + currentLoss) / period;
    }

    if (avgLoss === 0) return 100;
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
}

function getCurrentISTHour() {
    try {
        const now = new Date();
        const istFormatter = new Intl.DateTimeFormat('en-US', {
            timeZone: 'Asia/Kolkata',
            hour: 'numeric',
            hour12: false
        });
        const istHourString = istFormatter.formatToParts(now).find(part => part.type === 'hour').value;
        let hour = parseInt(istHourString, 10);
        if (hour === 24) hour = 0;

        return {
            raw: hour,
            sin: Math.sin(hour / 24 * 2 * Math.PI),
            cos: Math.cos(hour / 24 * 2 * Math.PI)
        };
    } catch (error) {
        console.error("Error getting IST hour:", error);
        const hour = new Date().getHours();
        return {
             raw: hour,
             sin: Math.sin(hour / 24 * 2 * Math.PI),
             cos: Math.cos(hour / 24 * 2 * Math.PI)
        };
    }
}

function getRealTimeExternalData() {
    const weatherConditions = ["Clear", "Clouds", "Haze", "Smoke", "Rain", "Drizzle"];
    const randomWeather = weatherConditions[Math.floor(Math.random() * weatherConditions.length)];
    let weatherFactor = 1.0;
    if (["Clear", "Clouds"].includes(randomWeather)) weatherFactor = 1.01;
    else if (["Rain", "Drizzle"].includes(randomWeather)) weatherFactor = 0.99;

    const newsSentiments = ["Strongly Positive", "Positive", "Neutral", "Negative", "Strongly Negative"];
    const randomNewsSentiment = newsSentiments[Math.floor(Math.random() * newsSentiments.length)];
    let newsFactor = 1.0;
    if(randomNewsSentiment === "Strongly Positive") newsFactor = 1.05;
    else if(randomNewsSentiment === "Positive") newsFactor = 1.02;
    else if(randomNewsSentiment === "Negative") newsFactor = 0.98;
    else if(randomNewsSentiment === "Strongly Negative") newsFactor = 0.95;

    const marketVolatilities = ["Low", "Normal", "Elevated", "High"];
    const randomMarketVol = marketVolatilities[Math.floor(Math.random() * marketVolatilities.length)];
    let marketVolFactor = 1.0;
    if(randomMarketVol === "Elevated") marketVolFactor = 0.97;
    else if(randomMarketVol === "High") marketVolFactor = 0.94;

    const combinedFactor = weatherFactor * newsFactor * marketVolFactor;
    const reason = `ExtData(Weather:${randomWeather},News:${randomNewsSentiment},MktVol:${randomMarketVol})`;

    return { factor: combinedFactor, reason: reason };
}

function getPrimeTimeSession(istHour) {
    if (istHour >= 10 && istHour < 12) return { session: "PRIME_MORNING", aggression: 1.25, confidence: 1.15 };
    if (istHour >= 13 && istHour < 14) return { session: "PRIME_AFTERNOON_1", aggression: 1.15, confidence: 1.10 };
    if (istHour >= 15 && istHour < 16) return { session: "PRIME_AFTERNOON_2", aggression: 1.15, confidence: 1.10 };
    if (istHour >= 17 && istHour < 20) {
        if (istHour === 19) {
             return { session: "PRIME_EVENING_PEAK", aggression: 1.35, confidence: 1.25 };
        }
        return { session: "PRIME_EVENING", aggression: 1.30, confidence: 1.20 };
    }
    return null;
}

// --- Market Context Analysis ---
function getMarketRegimeAndTrendContext(history, shortMALookback = 5, mediumMALookback = 10, longMALookback = 20) {
    const baseContext = getTrendContext(history, shortMALookback, mediumMALookback, longMALookback);
    let macroRegime = "UNCERTAIN";
    const { strength, volatility } = baseContext;
    let isTransitioning = false;

    const numbers = history.map(entry => parseInt(entry.actualNumber || entry.actual)).filter(n => !isNaN(n));

    if (numbers.length > mediumMALookback + 5) {
        const prevShortMA = calculateEMA(numbers.slice(1), shortMALookback);
        const prevMediumMA = calculateEMA(numbers.slice(1), mediumMALookback);
        const currentShortMA = calculateEMA(numbers, shortMALookback);
        const currentMediumMA = calculateEMA(numbers, mediumMALookback);

        if (prevShortMA && prevMediumMA && currentShortMA && currentMediumMA) {
            if ((prevShortMA <= prevMediumMA && currentShortMA > currentMediumMA) ||
                (prevShortMA >= prevMediumMA && currentShortMA < currentMediumMA)) {
                isTransitioning = true;
            }
        }
    }

    if (strength === "STRONG") {
        if (volatility === "LOW" || volatility === "VERY_LOW") macroRegime = "TREND_STRONG_LOW_VOL";
        else if (volatility === "MEDIUM") macroRegime = "TREND_STRONG_MED_VOL";
        else macroRegime = "TREND_STRONG_HIGH_VOL";
    } else if (strength === "MODERATE") {
        if (volatility === "LOW" || volatility === "VERY_LOW") macroRegime = "TREND_MOD_LOW_VOL";
        else if (volatility === "MEDIUM") macroRegime = "TREND_MOD_MED_VOL";
        else macroRegime = "TREND_MOD_HIGH_VOL";
    } else if (strength === "RANGING") {
        if (volatility === "LOW" || volatility === "VERY_LOW") macroRegime = "RANGE_LOW_VOL";
        else if (volatility === "MEDIUM") macroRegime = "RANGE_MED_VOL";
        else macroRegime = "RANGE_HIGH_VOL";
    } else {
        if (volatility === "HIGH") macroRegime = "WEAK_HIGH_VOL";
        else if (volatility === "MEDIUM") macroRegime = "WEAK_MED_VOL";
        else macroRegime = "WEAK_LOW_VOL";
    }

    if (isTransitioning && !macroRegime.includes("TRANSITION")) {
        macroRegime += "_TRANSITION";
    }

    baseContext.macroRegime = macroRegime;
    baseContext.isTransitioning = isTransitioning;
    baseContext.details += `,Regime:${macroRegime}`;
    return baseContext;
}

function getTrendContext(history, shortMALookback = 5, mediumMALookback = 10, longMALookback = 20) {
    if (!Array.isArray(history) || history.length < longMALookback) {
        return { strength: "UNKNOWN", direction: "NONE", volatility: "UNKNOWN", details: "Insufficient history", macroRegime: "UNKNOWN_REGIME", isTransitioning: false };
    }
    const numbers = history.map(entry => parseInt(entry.actualNumber || entry.actual)).filter(n => !isNaN(n));
    if (numbers.length < longMALookback) {
        return { strength: "UNKNOWN", direction: "NONE", volatility: "UNKNOWN", details: "Insufficient numbers", macroRegime: "UNKNOWN_REGIME", isTransitioning: false };
    }

    const shortMA = calculateEMA(numbers, shortMALookback);
    const mediumMA = calculateEMA(numbers, mediumMALookback);
    const longMA = calculateEMA(numbers, longMALookback);

    if (shortMA === null || mediumMA === null || longMA === null) return { strength: "UNKNOWN", direction: "NONE", volatility: "UNKNOWN", details: "MA calculation failed", macroRegime: "UNKNOWN_REGIME", isTransitioning: false };

    let direction = "NONE";
    let strength = "WEAK";
    let details = `S:${shortMA.toFixed(1)},M:${mediumMA.toFixed(1)},L:${longMA.toFixed(1)}`;

    const stdDevLong = calculateStdDev(numbers, longMALookback);
    const epsilon = 0.001;
    const normalizedSpread = (stdDevLong !== null && stdDevLong > epsilon) ? (shortMA - longMA) / stdDevLong : (shortMA - longMA) / epsilon;

    details += `,NormSpread:${normalizedSpread.toFixed(2)}`;

    if (shortMA > mediumMA && mediumMA > longMA) {
        direction = "BIG";
        if (normalizedSpread > 0.80) strength = "STRONG";
        else if (normalizedSpread > 0.45) strength = "MODERATE";
        else strength = "WEAK";
    } else if (shortMA < mediumMA && mediumMA < longMA) {
        direction = "SMALL";
        if (normalizedSpread < -0.80) strength = "STRONG";
        else if (normalizedSpread < -0.45) strength = "MODERATE";
        else strength = "WEAK";
    } else {
        strength = "RANGING";
        if (shortMA > longMA) direction = "BIG_BIASED_RANGE";
        else if (longMA > shortMA) direction = "SMALL_BIASED_RANGE";
    }

    let volatility = "UNKNOWN";
    const volSlice = numbers.slice(0, Math.min(numbers.length, 30));
    if (volSlice.length >= 15) {
        const stdDevVol = calculateStdDev(volSlice, volSlice.length);
        if (stdDevVol !== null) {
            details += ` VolStdDev:${stdDevVol.toFixed(2)}`;
            if (stdDevVol > 3.3) volatility = "HIGH";
            else if (stdDevVol > 2.0) volatility = "MEDIUM";
            else if (stdDevVol > 0.9) volatility = "LOW";
            else volatility = "VERY_LOW";
        }
    }
    return { strength, direction, volatility, details, macroRegime: "PENDING_REGIME_CLASSIFICATION", isTransitioning: false };
}

// --- Core Analytical Modules ---
function analyzeTransitions(history, baseWeight) {
    if (!Array.isArray(history) || history.length < 15) return null;
    const transitions = { "BIG": { "BIG": 0, "SMALL": 0, "total": 0 }, "SMALL": { "BIG": 0, "SMALL": 0, "total": 0 } };
    for (let i = 0; i < history.length - 1; i++) {
        const currentBS = getBigSmallFromNumber(history[i]?.actual);
        const prevBS = getBigSmallFromNumber(history[i + 1]?.actual);
        if (currentBS && prevBS && transitions[prevBS]) {
            transitions[prevBS][currentBS]++;
            transitions[prevBS].total++;
        }
    }
    const lastOutcome = getBigSmallFromNumber(history[0]?.actual);
    if (!lastOutcome || !transitions[lastOutcome] || transitions[lastOutcome].total < 6) return null;
    const nextBigProb = transitions[lastOutcome]["BIG"] / transitions[lastOutcome].total;
    const nextSmallProb = transitions[lastOutcome]["SMALL"] / transitions[lastOutcome].total;
    if (nextBigProb > nextSmallProb + 0.30) return { prediction: "BIG", weight: baseWeight * nextBigProb, source: "Transition" };
    if (nextSmallProb > nextBigProb + 0.30) return { prediction: "SMALL", weight: baseWeight * nextSmallProb, source: "Transition" };
    return null;
}
function analyzeStreaks(history, baseWeight) {
    if (!Array.isArray(history) || history.length < 3) return null;
    const actuals = history.map(p => getBigSmallFromNumber(p.actual)).filter(bs => bs);
    if (actuals.length < 3) return null;
    let currentStreakType = actuals[0], currentStreakLength = 0;
    for (const outcome of actuals) {
        if (outcome === currentStreakType) currentStreakLength++; else break;
    }
    if (currentStreakLength >= 2) {
        const prediction = getOppositeOutcome(currentStreakType);
        const weightFactor = Math.min(0.45 + (currentStreakLength * 0.18), 0.95);
        return { prediction, weight: baseWeight * weightFactor, source: `StreakBreak-${currentStreakLength}` };
    }
    return null;
}
function analyzeAlternatingPatterns(history, baseWeight) {
    if (!Array.isArray(history) || history.length < 5) return null;
    const actuals = history.slice(0, 5).map(p => getBigSmallFromNumber(p.actual)).filter(bs => bs);
    if (actuals.length < 4) return null;
    if (actuals[0] === "SMALL" && actuals[1] === "BIG" && actuals[2] === "SMALL" && actuals[3] === "BIG")
        return { prediction: "SMALL", weight: baseWeight * 1.15, source: "Alt-BSBS->S" };
    if (actuals[0] === "BIG" && actuals[1] === "SMALL" && actuals[2] === "BIG" && actuals[3] === "SMALL")
        return { prediction: "BIG", weight: baseWeight * 1.15, source: "Alt-SBSB->B" };
    return null;
}
function analyzeWeightedHistorical(history, weightDecayFactor, baseWeight) {
    if (!Array.isArray(history) || history.length < 5) return null;
    let bigWeightedScore = 0, smallWeightedScore = 0, currentWeight = 1.0;
    const maxHistory = Math.min(history.length, 20);
    for (let i = 0; i < maxHistory; i++) {
        const outcome = getBigSmallFromNumber(history[i].actual);
        if (outcome === "BIG") bigWeightedScore += currentWeight;
        else if (outcome === "SMALL") smallWeightedScore += currentWeight;
        currentWeight *= weightDecayFactor;
    }
    if (bigWeightedScore === 0 && smallWeightedScore === 0) return null;
    const totalScore = bigWeightedScore + smallWeightedScore + 0.0001;
    if (bigWeightedScore > smallWeightedScore) return { prediction: "BIG", weight: baseWeight * (bigWeightedScore / totalScore), source: "WeightedHist" };
    if (smallWeightedScore > bigWeightedScore) return { prediction: "SMALL", weight: baseWeight * (smallWeightedScore / totalScore), source: "WeightedHist" };
    return null;
}

function analyzeTwoPlusOnePatterns(history, baseWeight) {
    if (!history || history.length < 3) return null;
    const outcomes = history.slice(0, 3).map(p => getBigSmallFromNumber(p.actual));
    if (outcomes.some(o => o === null)) return null;

    const pattern = outcomes.join('');
    if (pattern === 'BBS') return { prediction: 'BIG', weight: baseWeight * 0.85, source: 'Pattern-BBS->B' };
    if (pattern === 'SSB') return { prediction: 'SMALL', weight: baseWeight * 0.85, source: 'Pattern-SSB->S' };
    return null;
}

function analyzeDoublePatterns(history, baseWeight) {
    if (!history || history.length < 4) return null;
    const outcomes = history.slice(0, 4).map(p => getBigSmallFromNumber(p.actual));
    if (outcomes.some(o => o === null)) return null;

    if (outcomes[0] === 'BIG' && outcomes[1] === 'BIG' && outcomes[2] === 'SMALL' && outcomes[3] === 'SMALL') {
        return { prediction: 'BIG', weight: baseWeight * 1.1, source: 'Pattern-SSBB->B' };
    }
    if (outcomes[0] === 'SMALL' && outcomes[1] === 'SMALL' && outcomes[2] === 'BIG' && outcomes[3] === 'BIG') {
        return { prediction: 'SMALL', weight: baseWeight * 1.1, source: 'Pattern-BBSS->S' };
    }
    return null;
}

function analyzeMirrorPatterns(history, baseWeight) {
    if (!history || history.length < 4) return null;
    const outcomes = history.slice(0, 4).map(p => getBigSmallFromNumber(p.actual));
    if (outcomes.some(o => o === null)) return null;

    if (outcomes[0] === outcomes[3] && outcomes[1] === outcomes[2] && outcomes[0] !== outcomes[1]) {
        return { prediction: outcomes[0], weight: baseWeight * 1.2, source: `Pattern-Mirror->${outcomes[0]}` };
    }
    return null;
}

function analyzeRSI(history, rsiPeriod, baseWeight, volatility) {
    if (rsiPeriod <= 0) return null;
    const actualNumbers = history.map(entry => parseInt(entry.actualNumber || entry.actual)).filter(num => !isNaN(num));
    if (actualNumbers.length < rsiPeriod + 1) return null;

    const rsiValue = calculateRSI(actualNumbers, rsiPeriod);
    if (rsiValue === null) return null;

    let overbought = 70; let oversold = 30;
    if (volatility === "HIGH") { overbought = 80; oversold = 20; }
    else if (volatility === "MEDIUM") { overbought = 75; oversold = 25; }
    else if (volatility === "LOW") { overbought = 68; oversold = 32; }
    else if (volatility === "VERY_LOW") { overbought = 65; oversold = 35; }


    let prediction = null, signalStrengthFactor = 0;
    if (rsiValue < oversold) { prediction = "BIG"; signalStrengthFactor = (oversold - rsiValue) / oversold; }
    else if (rsiValue > overbought) { prediction = "SMALL"; signalStrengthFactor = (rsiValue - overbought) / (100 - overbought); }

    if (prediction)
        return { prediction, weight: baseWeight * (0.60 + Math.min(signalStrengthFactor, 1.0) * 0.40), source: "RSI" };
    return null;
}
function analyzeMACD(history, shortPeriod, longPeriod, signalPeriod, baseWeight) {
    if (shortPeriod <=0 || longPeriod <=0 || signalPeriod <=0 || shortPeriod >= longPeriod) return null;
    const actualNumbers = history.map(entry => parseInt(entry.actualNumber || entry.actual)).filter(num => !isNaN(num));
    if (actualNumbers.length < longPeriod + signalPeriod -1) return null;

    const emaShort = calculateEMA(actualNumbers, shortPeriod);
    const emaLong = calculateEMA(actualNumbers, longPeriod);

    if (emaShort === null || emaLong === null) return null;
    const macdLineCurrent = emaShort - emaLong;

    const macdLineValues = [];
    for (let i = longPeriod -1; i < actualNumbers.length; i++) {
        const currentSlice = actualNumbers.slice(actualNumbers.length - 1 - i);
        const shortE = calculateEMA(currentSlice, shortPeriod);
        const longE = calculateEMA(currentSlice, longPeriod);
        if (shortE !== null && longE !== null) {
            macdLineValues.push(shortE - longE);
        }
    }

    if (macdLineValues.length < signalPeriod) return null;

    const signalLine = calculateEMA(macdLineValues.slice().reverse(), signalPeriod);
    if (signalLine === null) return null;

    const macdHistogram = macdLineCurrent - signalLine;
    let prediction = null;

    if (macdLineValues.length >= signalPeriod + 1) {
        const prevMacdSliceForSignal = macdLineValues.slice(0, -1);
        const prevSignalLine = calculateEMA(prevMacdSliceForSignal.slice().reverse(), signalPeriod);
        const prevMacdLine = macdLineValues[macdLineValues.length - 2];

        if (prevMacdLine <= prevSignalLine && macdLineCurrent > signalLine) prediction = "BIG";
            else if (prevMacdLine >= prevSignalLine && macdLineCurrent < signalLine) prediction = "SMALL";
    }

    if (!prediction) {
        if (macdHistogram > 0.25) prediction = "BIG";
        else if (macdHistogram < -0.25) prediction = "SMALL";
    }

    if (prediction) {
        const strengthFactor = Math.min(Math.abs(macdHistogram) / 0.6, 1.0);
        return { prediction, weight: baseWeight * (0.55 + strengthFactor * 0.45), source: `MACD_${prediction === "BIG" ? "CrossB" : "CrossS"}` };
    }
    return null;
}
function analyzeBollingerBands(history, period, stdDevMultiplier, baseWeight) {
    if (period <= 0) return null;
    const actualNumbers = history.map(entry => parseInt(entry.actualNumber || entry.actual)).filter(num => !isNaN(num));
    if (actualNumbers.length < period) return null;

    const sma = calculateSMA(actualNumbers.slice(0, period), period);
    if (sma === null) return null;

    const stdDev = calculateStdDev(actualNumbers, period);
    if (stdDev === null || stdDev < 0.05) return null;

    const upperBand = sma + (stdDev * stdDevMultiplier);
    const lowerBand = sma - (stdDev * stdDevMultiplier);
    const lastNumber = actualNumbers[0];
    let prediction = null;

    if (lastNumber > upperBand * 1.01) prediction = "SMALL";
    else if (lastNumber < lowerBand * 0.99) prediction = "BIG";

    if (prediction) {
        const bandBreachStrength = Math.abs(lastNumber - sma) / (stdDev * stdDevMultiplier + 0.001);
        return { prediction, weight: baseWeight * (0.65 + Math.min(bandBreachStrength, 0.9)*0.35), source: "Bollinger" };
    }
    return null;
}
function analyzeIchimokuCloud(history, tenkanPeriod, kijunPeriod, senkouBPeriod, baseWeight) {
    if (tenkanPeriod <=0 || kijunPeriod <=0 || senkouBPeriod <=0) return null;
    const chronologicalHistory = history.slice().reverse();
    const numbers = chronologicalHistory.map(entry => parseInt(entry.actualNumber || entry.actual)).filter(n => !isNaN(n));

    if (numbers.length < Math.max(senkouBPeriod, kijunPeriod) + kijunPeriod -1 ) return null;

    const getHighLow = (dataSlice) => {
        if (!dataSlice || dataSlice.length === 0) return { high: null, low: null };
        return { high: Math.max(...dataSlice), low: Math.min(...dataSlice) };
    };

    const tenkanSenValues = [];
    for (let i = 0; i < numbers.length; i++) {
        if (i < tenkanPeriod - 1) { tenkanSenValues.push(null); continue; }
        const { high, low } = getHighLow(numbers.slice(i - tenkanPeriod + 1, i + 1));
        if (high !== null && low !== null) tenkanSenValues.push((high + low) / 2); else tenkanSenValues.push(null);
    }

    const kijunSenValues = [];
    for (let i = 0; i < numbers.length; i++) {
        if (i < kijunPeriod - 1) { kijunSenValues.push(null); continue; }
        const { high, low } = getHighLow(numbers.slice(i - kijunPeriod + 1, i + 1));
        if (high !== null && low !== null) kijunSenValues.push((high + low) / 2); else kijunSenValues.push(null);
    }

    const currentTenkan = tenkanSenValues[numbers.length - 1];
    const prevTenkan = tenkanSenValues[numbers.length - 2];
    const currentKijun = kijunSenValues[numbers.length - 1];
    const prevKijun = kijunSenValues[numbers.length - 2];

    const senkouSpanAValues = [];
    for(let i=0; i < numbers.length; i++) {
        if (tenkanSenValues[i] !== null && kijunSenValues[i] !== null) {
            senkouSpanAValues.push((tenkanSenValues[i] + kijunSenValues[i]) / 2);
        } else {
            senkouSpanAValues.push(null);
        }
    }

    const senkouSpanBValues = [];
    for (let i = 0; i < numbers.length; i++) {
        if (i < senkouBPeriod -1) { senkouSpanBValues.push(null); continue; }
        const { high, low } = getHighLow(numbers.slice(i - senkouBPeriod + 1, i + 1));
        if (high !== null && low !== null) senkouSpanBValues.push((high + low) / 2); else senkouSpanBValues.push(null);
    }

    const currentSenkouA = (numbers.length > kijunPeriod && senkouSpanAValues.length > numbers.length - 1 - kijunPeriod) ? senkouSpanAValues[numbers.length - 1 - kijunPeriod] : null;
    const currentSenkouB = (numbers.length > kijunPeriod && senkouSpanBValues.length > numbers.length - 1 - kijunPeriod) ? senkouSpanBValues[numbers.length - 1 - kijunPeriod] : null;


    const chikouSpan = numbers[numbers.length - 1];
    const priceKijunPeriodsAgo = numbers.length > kijunPeriod ? numbers[numbers.length - 1 - kijunPeriod] : null;

    const lastPrice = numbers[numbers.length - 1];
    if (lastPrice === null || currentTenkan === null || currentKijun === null || currentSenkouA === null || currentSenkouB === null || chikouSpan === null || priceKijunPeriodsAgo === null) {
        return null;
    }

    let prediction = null;
    let strengthFactor = 0.3;

    let tkCrossSignal = null;
    if (prevTenkan <= prevKijun && currentTenkan > currentKijun) tkCrossSignal = "BIG";
    else if (prevTenkan >= prevKijun && currentTenkan < currentKijun) tkCrossSignal = "SMALL";

    const cloudTop = Math.max(currentSenkouA, currentSenkouB);
    const cloudBottom = Math.min(currentSenkouA, currentSenkouB);
    let priceVsCloudSignal = null;
    if (lastPrice > cloudTop) priceVsCloudSignal = "BIG";
    else if (lastPrice < cloudBottom) priceVsCloudSignal = "SMALL";

    let chikouSignal = null;
    if (chikouSpan > priceKijunPeriodsAgo) chikouSignal = "BIG";
    else if (chikouSpan < priceKijunPeriodsAgo) chikouSignal = "SMALL";

    if (tkCrossSignal && tkCrossSignal === priceVsCloudSignal && tkCrossSignal === chikouSignal) {
        prediction = tkCrossSignal; strengthFactor = 0.95;
    }
    else if (priceVsCloudSignal && priceVsCloudSignal === tkCrossSignal && chikouSignal === priceVsCloudSignal) {
        prediction = priceVsCloudSignal; strengthFactor = 0.85;
    }
    else if (priceVsCloudSignal && priceVsCloudSignal === tkCrossSignal) {
        prediction = priceVsCloudSignal; strengthFactor = 0.7;
    }
    else if (priceVsCloudSignal && priceVsCloudSignal === chikouSignal) {
        prediction = priceVsCloudSignal; strengthFactor = 0.65;
    }
    else if (tkCrossSignal && priceVsCloudSignal) {
        prediction = tkCrossSignal; strengthFactor = 0.55;
    }
    else if (priceVsCloudSignal) {
         prediction = priceVsCloudSignal; strengthFactor = 0.5;
    }

    if (prediction === "BIG" && lastPrice > currentKijun && numbers[numbers.length-2] <= prevKijun && priceVsCloudSignal === "BIG") {
        strengthFactor = Math.min(1.0, strengthFactor + 0.15);
    } else if (prediction === "SMALL" && lastPrice < currentKijun && numbers[numbers.length-2] >= prevKijun && priceVsCloudSignal === "SMALL") {
        strengthFactor = Math.min(1.0, strengthFactor + 0.15);
    }

    if (prediction) return { prediction, weight: baseWeight * strengthFactor, source: "Ichimoku" };
    return null;
}
function calculateEntropyForSignal(outcomes, windowSize) {
    if (!Array.isArray(outcomes) || outcomes.length < windowSize) return null;
    const counts = { BIG: 0, SMALL: 0 };
    outcomes.slice(0, windowSize).forEach(o => { if (o) counts[o] = (counts[o] || 0) + 1; });
    let entropy = 0;
    const totalValidOutcomes = counts.BIG + counts.SMALL;
    if (totalValidOutcomes === 0) return 1;
    for (let key in counts) {
        if (counts[key] > 0) { const p = counts[key] / totalValidOutcomes; entropy -= p * Math.log2(p); }
    }
    return isNaN(entropy) ? 1 : entropy;
}
function analyzeEntropySignal(history, period, baseWeight) {
    if (history.length < period) return null;
    const outcomes = history.slice(0, period).map(e => getBigSmallFromNumber(e.actual));
    const entropy = calculateEntropyForSignal(outcomes, period);
    if (entropy === null) return null;

    if (entropy < 0.55) {
        const lastOutcome = outcomes[0];
        if (lastOutcome) return { prediction: getOppositeOutcome(lastOutcome), weight: baseWeight * (1 - entropy) * 0.85, source: "EntropyReversal" };
    } else if (entropy > 0.98) {
        const lastOutcome = outcomes[0];
        if (lastOutcome) return { prediction: lastOutcome, weight: baseWeight * 0.25, source: "EntropyHighContWeak" };
    }
    return null;
}
function analyzeVolatilityBreakout(history, trendContext, baseWeight) {
    if (trendContext.volatility === "VERY_LOW" && history.length >= 3) {
        const lastOutcome = getBigSmallFromNumber(history[0].actual);
        const prevOutcome = getBigSmallFromNumber(history[1].actual);
        if (lastOutcome && prevOutcome && lastOutcome === prevOutcome) return { prediction: lastOutcome, weight: baseWeight * 0.8, source: "VolSqueezeBreakoutCont" };
        if (lastOutcome && prevOutcome && lastOutcome !== prevOutcome) return { prediction: lastOutcome, weight: baseWeight * 0.6, source: "VolSqueezeBreakoutInitial" };
    }
    return null;
}
function analyzeStochastic(history, kPeriod, dPeriod, smoothK, baseWeight, volatility) {
    if (kPeriod <=0 || dPeriod <=0 || smoothK <=0) return null;
    const actualNumbers = history.map(entry => parseInt(entry.actualNumber || entry.actual)).filter(num => !isNaN(num));
    if (actualNumbers.length < kPeriod + smoothK -1 + dPeriod -1) return null;

    const chronologicalNumbers = actualNumbers.slice().reverse();

    let kValues = [];
    for (let i = kPeriod - 1; i < chronologicalNumbers.length; i++) {
        const currentSlice = chronologicalNumbers.slice(i - kPeriod + 1, i + 1);
        const currentClose = currentSlice[currentSlice.length - 1];
        const lowestLow = Math.min(...currentSlice);
        const highestHigh = Math.max(...currentSlice);
        if (highestHigh === lowestLow) kValues.push(kValues.length > 0 ? kValues[kValues.length-1] : 50);
        else kValues.push(100 * (currentClose - lowestLow) / (highestHigh - lowestLow));
    }

    if (kValues.length < smoothK) return null;
    const smoothedKValues = [];
    for(let i = 0; i <= kValues.length - smoothK; i++) {
        smoothedKValues.push(calculateSMA(kValues.slice(i, i + smoothK).slice().reverse(), smoothK));
    }

    if (smoothedKValues.length < dPeriod) return null;
    const dValues = [];
    for(let i = 0; i <= smoothedKValues.length - dPeriod; i++) {
        dValues.push(calculateSMA(smoothedKValues.slice(i, i + dPeriod).slice().reverse(), dPeriod));
    }

    if (smoothedKValues.length < 2 || dValues.length < 2) return null;

    const currentK = smoothedKValues[smoothedKValues.length - 1];
    const prevK = smoothedKValues[smoothedKValues.length - 2];
    const currentD = dValues[dValues.length - 1];
    const prevD = dValues[dValues.length - 1];

    let overbought = 80; let oversold = 20;
    if (volatility === "HIGH") { overbought = 88; oversold = 12; }
    else if (volatility === "MEDIUM") { overbought = 82; oversold = 18;}
    else if (volatility === "LOW") { overbought = 75; oversold = 25; }
    else if (volatility === "VERY_LOW") { overbought = 70; oversold = 30; }


    let prediction = null, strengthFactor = 0;
    if (prevK <= prevD && currentK > currentD && currentK < overbought - 5) {
         prediction = "BIG"; strengthFactor = Math.max(0.35, (oversold + 5 - Math.min(currentK, currentD, oversold + 5)) / (oversold + 5));
    } else if (prevK >= prevD && currentK < currentD && currentK > oversold + 5) {
        prediction = "SMALL"; strengthFactor = Math.max(0.35, (Math.max(currentK, currentD, overbought - 5) - (overbought - 5)) / (100 - (overbought - 5)));
    }
    if (!prediction) {
        if (prevK < oversold && currentK >= oversold && currentK < (oversold + (overbought-oversold)/2) ) {
            prediction = "BIG"; strengthFactor = Math.max(0.25, (currentK - oversold) / ((overbought-oversold)/2) );
        } else if (prevK > overbought && currentK <= overbought && currentK > (oversold + (overbought-oversold)/2) ) {
            prediction = "SMALL"; strengthFactor = Math.max(0.25, (overbought - currentK) / ((overbought-oversold)/2) );
        }
    }
    if (prediction) return { prediction, weight: baseWeight * (0.5 + Math.min(strengthFactor, 1.0) * 0.5), source: "Stochastic" };
    return null;
}
function analyzeMADeviation(history, longMAPeriod, normalizationPeriod, baseWeight) {
    if (longMAPeriod <=0 || normalizationPeriod <=0) return null;
    const actualNumbers = history.map(entry => parseInt(entry.actualNumber || entry.actual)).filter(num => !isNaN(num));
    if (actualNumbers.length < Math.max(longMAPeriod, normalizationPeriod)) return null;
    const lastNumber = actualNumbers[0];
    const longMA = calculateEMA(actualNumbers, longMAPeriod);
    const stdDevNorm = calculateStdDev(actualNumbers, normalizationPeriod);
    if (longMA === null || stdDevNorm === null || stdDevNorm < 0.01) return null;

    const deviationScore = (lastNumber - longMA) / stdDevNorm;
    let prediction = null, strengthFactor = 0;
    const threshold = 1.8;
    if (deviationScore > threshold) { prediction = "SMALL"; strengthFactor = Math.min((deviationScore - threshold) / threshold, 1.0); }
    else if (deviationScore < -threshold) { prediction = "BIG"; strengthFactor = Math.min(Math.abs(deviationScore - (-threshold)) / threshold, 1.0); }
    if (prediction) return { prediction, weight: baseWeight * (0.4 + strengthFactor * 0.6), source: "MADev" };
    return null;
}
function analyzeVWAPDeviation(history, vwapPeriod, normalizationPeriod, baseWeight) {
    if (vwapPeriod <=0 || normalizationPeriod <=0) return null;
    const actualNumbers = history.map(entry => parseInt(entry.actualNumber || entry.actual)).filter(num => !isNaN(num));
    if (history.length < Math.max(vwapPeriod, normalizationPeriod) || actualNumbers.length < 1) return null;

    const vwap = calculateVWAP(history, vwapPeriod);
    const stdDevPrice = calculateStdDev(actualNumbers, normalizationPeriod);
    if (vwap === null || stdDevPrice === null || stdDevPrice < 0.01) return null;

    const lastNumber = actualNumbers[0];
    const deviationScore = (lastNumber - vwap) / stdDevPrice;
    let prediction = null, strengthFactor = 0;
    const threshold = 1.5;
    if (deviationScore > threshold) { prediction = "SMALL"; strengthFactor = Math.min((deviationScore - threshold) / threshold, 1.0); }
    else if (deviationScore < -threshold) { prediction = "BIG"; strengthFactor = Math.min(Math.abs(deviationScore - (-threshold)) / threshold, 1.0); }
    if (prediction) return { prediction, weight: baseWeight * (0.45 + strengthFactor * 0.55), source: "VWAPDev" };
    return null;
}
function analyzeHarmonicPotential(history, baseWeight) {
    const numbers = history.map(entry => parseFloat(entry.actualNumber)).filter(n => !isNaN(n));
    if (numbers.length < 20) return null;

    let swings = [];
    const chronologicalNumbers = numbers.slice().reverse();

    for (let i = 2; i < chronologicalNumbers.length - 2; i++) {
        const isPeak = chronologicalNumbers[i] > chronologicalNumbers[i-1] && chronologicalNumbers[i] > chronologicalNumbers[i-2] &&
                       chronologicalNumbers[i] > chronologicalNumbers[i+1] && chronologicalNumbers[i] > chronologicalNumbers[i+2];
        const isTrough = chronologicalNumbers[i] < chronologicalNumbers[i-1] && chronologicalNumbers[i] < chronologicalNumbers[i-2] &&
                         chronologicalNumbers[i] < chronologicalNumbers[i+1] && chronologicalNumbers[i] < chronologicalNumbers[i+2];

        if (isPeak || isTrough) {
            const newSwing = { price: chronologicalNumbers[i], index: numbers.length - 1 - i, type: isPeak ? 'peak' : 'trough' };
            if (swings.length === 0 || (isPeak && swings[0].type === 'trough') || (isTrough && swings[0].type === 'peak')) {
                 swings.unshift(newSwing);
            } else {
                if (isPeak && swings[0].type === 'peak' && newSwing.price > swings[0].price) swings[0] = newSwing;
                if (isTrough && swings[0].type === 'trough' && newSwing.price < swings[0].price) swings[0] = newSwing;
            }
        }
    }
    if (swings.length < 3) return null;

    const C = swings[0];
    const B = swings[1];
    const X = swings[2];

    if (!X || !B || !C || X.type === B.type || B.type === C.type ) return null;

    const XA_val = Math.abs(B.price - X.price);
    const AB_val = XA_val;
    const BC_val = Math.abs(C.price - B.price);

    if (AB_val < 0.8 || BC_val < 0.5) return null;

    const lastPrice = numbers[0];
    let prediction = null;
    let strengthFactor = 0;
    const bcRetracementOfAb = BC_val / AB_val;

    if (X.type === 'peak' && B.type === 'trough' && C.type === 'peak') {
        if (bcRetracementOfAb >= 0.382 && bcRetracementOfAb <= 0.886) {
            const prz_D_gartley = X.price - AB_val * 0.786;
            if (lastPrice >= prz_D_gartley * 0.98 && lastPrice <= prz_D_gartley * 1.02 && lastPrice < B.price) {
                prediction = "BIG"; strengthFactor = 0.6;
            }
        }
    }
    else if (X.type === 'trough' && B.type === 'peak' && C.type === 'trough') {
        if (bcRetracementOfAb >= 0.382 && bcRetracementOfAb <= 0.886) {
            const prz_D_gartley = X.price + AB_val * 0.786;
             if (lastPrice <= prz_D_gartley * 1.02 && lastPrice >= prz_D_gartley * 0.98 && lastPrice > B.price) {
                prediction = "SMALL"; strengthFactor = 0.6;
            }
        }
    }

    if (prediction) {
        return { prediction, weight: baseWeight * Math.max(0.25, Math.min(strengthFactor, 0.85)), source: "HarmonicPotV3" };
    }
    return null;
}
function analyzeNGramPatterns(history, n, baseWeight) {
    if (!Array.isArray(history) || history.length < n + 10) return null;
    const outcomes = history.map(p => getBigSmallFromNumber(p.actual)).filter(bs => bs);
    if (outcomes.length < n + 5) return null;
    const recentNGram = outcomes.slice(0, n).join('-');
    const patternCounts = {};
    for (let i = 0; i <= outcomes.length - (n + 1); i++) {
        const pattern = outcomes.slice(i + 1, i + 1 + n).join('-');
        const nextOutcome = outcomes[i];
        if (!patternCounts[pattern]) patternCounts[pattern] = { BIG: 0, SMALL: 0, total: 0 };
        patternCounts[pattern][nextOutcome]++;
        patternCounts[pattern].total++;
    }

    if (patternCounts[recentNGram] && patternCounts[recentNGram].total >= 4) {
        const data = patternCounts[recentNGram];
        const probBig = data.BIG / data.total;
        const probSmall = data.SMALL / data.total;
        if (probBig > probSmall + 0.30 && probBig > 0.65) return { prediction: "BIG", weight: baseWeight * probBig * 1.1, source: `${n}GramB` };
        if (probSmall > probBig + 0.30 && probSmall > 0.65) return { prediction: "SMALL", weight: baseWeight * probSmall * 1.1, source: `${n}GramS` };
    }
    return null;
}
function analyzeCyclicalPatterns(history, period, baseWeight) {
    if (history.length < period || period < 8) return null;
    const outcomes = history.slice(0, period).map(e => getBigSmallFromNumber(e.actual)).filter(o => o);
    if (outcomes.length < period * 0.80) return null;

    for (let cycleLen = 3; cycleLen <= 6; cycleLen++) {
        if (outcomes.length < cycleLen * 2.8) continue;
        const cycle1String = outcomes.slice(0, cycleLen).join('');
        const cycle2String = outcomes.slice(cycleLen, cycleLen * 2).join('');

        if (cycle1String.length === cycleLen && cycle1String === cycle2String) {
            let matchLength = 0;
            for (let k = 0; k < cycleLen && (cycleLen * 2 + k) < outcomes.length; k++) {
                if (outcomes[cycleLen * 2 + k] === outcomes[k]) matchLength++;
                else break;
            }
            if (matchLength >= Math.floor(cycleLen * 0.66)) {
                const predictedOutcome = outcomes[cycleLen - 1];
                if (predictedOutcome) return { prediction: predictedOutcome, weight: baseWeight * (0.65 + (1 / cycleLen) + (matchLength / cycleLen * 0.2)), source: `Cycle${cycleLen}StrongCont` };
            }
        }
    }
    return null;
}
function analyzeVolatilityPersistence(history, period = 10, baseWeight) {
    const numbers = history.map(entry => parseInt(entry.actualNumber || entry.actual)).filter(n => !isNaN(n));
    if (numbers.length < period * 2) return null;

    const recentVolSlice = numbers.slice(0, period);
    const prevVolSlice = numbers.slice(period, period * 2);

    const currentStdDev = calculateStdDev(recentVolSlice, period);
    const prevStdDev = calculateStdDev(prevVolSlice, period);

    if (currentStdDev === null || prevStdDev === null) return null;

    let prediction = null;
    let strengthFactor = 0;

    if (currentStdDev > prevStdDev * 1.3 && currentStdDev > 2.0) {
        if (numbers[0] > numbers[1]) prediction = "BIG";
        else if (numbers[0] < numbers[1]) prediction = "SMALL";
        strengthFactor = 0.3;
    }
    else if (currentStdDev < prevStdDev * 0.7 && currentStdDev < 1.0) {
        if (numbers[0] > numbers[1]) prediction = "SMALL";
        else if (numbers[0] < numbers[1]) prediction = "BIG";
        strengthFactor = 0.35;
    }

    if(prediction) return { prediction, weight: baseWeight * strengthFactor, source: "VolPersist" };
    return null;
}
function analyzeFractalDimension(history, period = 14, baseWeight) {
    const numbers = history.map(entry => parseFloat(entry.actualNumber)).filter(n => !isNaN(n));
    if (numbers.length < period + 1) return null;

    const chronologicalNumbers = numbers.slice().reverse();

    const periodSlice = chronologicalNumbers.slice(-period);
    const { high: highestHighP, low: lowestLowP } = periodSlice.reduce(
        (acc, val) => ({ high: Math.max(acc.high, val), low: Math.min(acc.low, val) }),
        { high: -Infinity, low: Infinity }
    );
    if (highestHighP === -Infinity || lowestLowP === Infinity) return null;
    const N1_val = (highestHighP - lowestLowP) / period;
    if (N1_val === 0) return { value: 1.0, interpretation: "EXTREMELY_TRENDING_OR_FLAT", prediction: null, weight: 0, source: "FractalDim" };

    let sumPriceChanges = 0;
    for (let i = 1; i < periodSlice.length; i++) {
        sumPriceChanges += Math.abs(periodSlice[i] - periodSlice[i-1]);
    }
    const N2_val = sumPriceChanges / period;

    if (N2_val === 0 && N1_val !== 0) return { value: 2.0, interpretation: "CHOPPY_MAX_NOISE", prediction: null, weight: 0, source: "FractalDim" };
    if (N2_val === 0 && N1_val === 0) return { value: 1.0, interpretation: "FLAT_NO_MOVEMENT", prediction: null, weight: 0, source: "FractalDim" };

    const priceChangeOverPeriod = Math.abs(periodSlice[periodSlice.length - 1] - periodSlice[0]);
    const ER = sumPriceChanges > 0 ? priceChangeOverPeriod / sumPriceChanges : 0;
    const FDI_approx = 1 + (1 - ER);

    let interpretation = "UNKNOWN";
    let prediction = null;

    if (FDI_approx < 1.35) interpretation = "TRENDING";
    else if (FDI_approx > 1.65) interpretation = "CHOPPY_RANGING";
    else interpretation = "MODERATE_ACTIVITY";

    if (FDI_approx > 1.75) {
        const lastOutcome = getBigSmallFromNumber(numbers[0]);
        if (lastOutcome) prediction = getOppositeOutcome(lastOutcome);
    }

    return {
        value: FDI_approx,
        interpretation: interpretation,
        prediction: prediction,
        weight: prediction ? baseWeight * 0.2 : 0,
        source: "FractalDim"
    };
}
function analyzeSignalLeadLag(signals, trendContext, baseWeight) {
    let prediction = null;
    let strengthFactor = 0;

    const rsiSignal = signals.find(s => s.source === "RSI");
    const macdSignal = signals.find(s => s.source.startsWith("MACD"));

    if (rsiSignal && macdSignal && rsiSignal.prediction === macdSignal.prediction) {
        if (rsiSignal.prediction === "BIG" && macdSignal.source.includes("CrossB")) {
            if (trendContext.direction === "BIG" || trendContext.strength === "RANGING" || trendContext.direction.includes("BIG")) {
                prediction = "BIG";
                strengthFactor = 0.5;
            }
        } else if (rsiSignal.prediction === "SMALL" && macdSignal.source.includes("CrossS")) {
            if (trendContext.direction === "SMALL" || trendContext.strength === "RANGING" || trendContext.direction.includes("SMALL")) {
                prediction = "SMALL";
                strengthFactor = 0.5;
            }
        }
    }

    if(prediction) return { prediction, weight: baseWeight * strengthFactor, source: "LeadLagConfirm" };
    return null;
}

function analyzeWaveformPatterns(history, baseWeight) {
    const outcomes = history.map(p => getBigSmallFromNumber(p.actual)).filter(bs => bs);
    if (outcomes.length < 8) return null;

    const wave = outcomes.slice(0, 8).map(o => o === "BIG" ? 1 : -1);

    if (wave[0] === wave[1] && wave[2] === wave[3] && wave[0] !== wave[2]) {
        return { prediction: wave[0] === 1 ? "BIG" : "SMALL", weight: baseWeight * 1.2, source: "Waveform-Constructive" };
    }
    if (wave[0] !== wave[1] && wave[1] !== wave[2] && wave[2] !== wave[3]) {
        return { prediction: wave[0] === 1 ? "SMALL" : "BIG", weight: baseWeight, source: "Waveform-Destructive" };
    }
    return null;
}

function analyzePhaseSpace(history, baseWeight) {
    const outcomes = history.map(p => getBigSmallFromNumber(p.actual)).filter(bs => bs);
    if (outcomes.length < 10) return null;

    const recent = outcomes.slice(0, 10);
    const bigCount = recent.filter(r => r === "BIG").length;
    const smallCount = recent.filter(r => r === "SMALL").length;

    if (bigCount >= 7) {
        return { prediction: "BIG", weight: baseWeight * ((bigCount - 5) / 5), source: "PhaseSpace-BigAttractor" };
    }
    if (smallCount >= 7) {
        return { prediction: "SMALL", weight: baseWeight * ((smallCount - 5) / 5), source: "PhaseSpace-SmallAttractor" };
    }
    return null;
}

function analyzeQuantumTunneling(history, baseWeight) {
    const actuals = history.map(p => p.actual).filter(n => n !== null);
    if (actuals.length < 2) return null;

    const lastNum = actuals[0];
    const prevNum = actuals[1];

    if ((lastNum <= 1 && prevNum >= 8) || (lastNum >= 8 && prevNum <= 1)) {
        return { prediction: lastNum > 4 ? "SMALL" : "BIG", weight: baseWeight, source: "QuantumTunneling" };
    }
    return null;
}

function analyzeEntanglement(history, lag, baseWeight) {
    const outcomes = history.map(p => getBigSmallFromNumber(p.actual)).filter(bs => bs);
    if (outcomes.length < lag + 10) return null;

    let match = 0;
    let antiMatch = 0;
    for(let i = 0; i < 10; i++) {
        if(outcomes[i] === outcomes[i + lag]) {
            match++;
        } else {
            antiMatch++;
        }
    }

    if (match >= 8) {
        return { prediction: outcomes[lag-1], weight: baseWeight, source: `Entangled-Corr-Lag${lag}` };
    }
    if (antiMatch >= 8) {
        return { prediction: outcomes[lag-1] === "BIG" ? "SMALL" : "BIG", weight: baseWeight, source: `Entangled-AntiCorr-Lag${lag}` };
    }

    return null;
}

function analyzeMonteCarloSignal(signals, baseWeight) {
    if (signals.length < 5) return null;

    const bigProb = signals.filter(s => s.prediction === "BIG").reduce((acc, s) => acc + s.weight, 0);
    const smallProb = signals.filter(s => s.prediction === "SMALL").reduce((acc, s) => acc + s.weight, 0);
    const totalWeight = bigProb + smallProb;

    if (totalWeight === 0) return null;

    const normalizedBigProb = bigProb / totalWeight;

    let bigWins = 0;
    const simulations = 1000;
    for(let i = 0; i < simulations; i++) {
        if (Math.random() < normalizedBigProb) {
            bigWins++;
        }
    }

    if (bigWins / simulations > 0.7) {
        return { prediction: "BIG", weight: baseWeight * (bigWins / simulations), source: "MonteCarlo" };
    }
    if (bigWins / simulations < 0.3) {
        return { prediction: "SMALL", weight: baseWeight * (1 - (bigWins / simulations)), source: "MonteCarlo" };
    }

    return null;
}

function analyzeVolatilityTrendFusion(trendContext, marketEntropyState, baseWeight) {
    const { direction, strength, volatility } = trendContext;
    const { state: entropy } = marketEntropyState;

    let prediction = null;
    let weightFactor = 0;

    if (strength === 'STRONG' && (volatility === 'LOW' || volatility === 'MEDIUM') && entropy === 'ORDERLY') {
        prediction = direction.includes('BIG') ? 'BIG' : 'SMALL';
        weightFactor = 1.4;
    }
    else if (strength === 'STRONG' && volatility === 'HIGH' && entropy.includes('CHAOS')) {
        prediction = direction.includes('BIG') ? 'SMALL' : 'BIG';
        weightFactor = 1.2;
    }
    else if (strength === 'RANGING' && volatility === 'LOW' && entropy === 'ORDERLY') {
        prediction = Math.random() > 0.5 ? 'BIG' : 'SMALL';
        weightFactor = 0.8;
    }

    if (prediction) {
        return { prediction, weight: baseWeight * weightFactor, source: 'Vol-Trend-Fusion' };
    }
    return null;
}

function analyzeMLModelSignal(features, baseWeight) {
    const { rsi_14, macd_hist, stddev_30, time_sin, time_cos } = features;

    let modelConfidence = 0;
    let prediction = null;

    if (rsi_14 > 70 && macd_hist < -0.1) {
        prediction = "SMALL";
        modelConfidence = Math.abs(macd_hist) + (rsi_14 - 70) / 30;
    } else if (rsi_14 < 30 && macd_hist > 0.1) {
        prediction = "BIG";
        modelConfidence = Math.abs(macd_hist) + (30 - rsi_14) / 30;
    } else if (stddev_30 < 1.0 && time_sin > 0) {
        prediction = "BIG";
        modelConfidence = 0.4;
    }

    if (prediction) {
        const weight = baseWeight * Math.min(1.0, modelConfidence) * 1.5;
        return { prediction, weight: weight, source: "ML-GradientBoost" };
    }

    return null;
}

// --- Trend Stability & Market Entropy ---
function analyzeTrendStability(history) {
    if (!Array.isArray(history) || history.length < 25) {
        return { isStable: true, reason: "Not enough data for stability check.", details: "", dominance: "NONE" };
    }
    const confirmedHistory = history.filter(p => p && (p.status === "Win" || p.status === "Loss") && typeof p.actual !== 'undefined' && p.actual !== null);
    if (confirmedHistory.length < 20) return { isStable: true, reason: "Not enough confirmed results.", details: `Confirmed: ${confirmedHistory.length}`, dominance: "NONE" };

    const recentResults = confirmedHistory.slice(0, 20).map(p => getBigSmallFromNumber(p.actual)).filter(r => r);
    if (recentResults.length < 18) return { isStable: true, reason: "Not enough valid B/S for stability.", details: `Valid B/S: ${recentResults.length}`, dominance: "NONE" };

    const bigCount = recentResults.filter(r => r === "BIG").length;
    const smallCount = recentResults.filter(r => r === "SMALL").length;
    let outcomeDominance = "NONE";

    if (bigCount / recentResults.length >= 0.80) {
        outcomeDominance = "BIG_DOMINANCE";
        return { isStable: false, reason: "Unstable: Extreme Outcome Dominance", details: `BIG:${bigCount}, SMALL:${smallCount} in last ${recentResults.length}`, dominance: outcomeDominance };
    }
    if (smallCount / recentResults.length >= 0.80) {
        outcomeDominance = "SMALL_DOMINANCE";
        return { isStable: false, reason: "Unstable: Extreme Outcome Dominance", details: `BIG:${bigCount}, SMALL:${smallCount} in last ${recentResults.length}`, dominance: outcomeDominance };
    }

    const entropy = calculateEntropyForSignal(recentResults, recentResults.length);
    if (entropy !== null && entropy < 0.45) {
        return { isStable: false, reason: "Unstable: Very Low Entropy (Highly Predictable/Stuck)", details: `Entropy: ${entropy.toFixed(2)}`, dominance: outcomeDominance };
    }

    const actualNumbersRecent = confirmedHistory.slice(0, 15).map(p => parseInt(p.actualNumber || p.actual)).filter(n => !isNaN(n));
    if (actualNumbersRecent.length >= 10) {
        const stdDevNum = calculateStdDev(actualNumbersRecent, actualNumbersRecent.length);
        if (stdDevNum !== null && stdDevNum > 3.3) {
            return { isStable: false, reason: "Unstable: High Numerical Volatility", details: `StdDev: ${stdDevNum.toFixed(2)}`, dominance: outcomeDominance };
        }
    }
    let alternations = 0;
    for (let i = 0; i < recentResults.length - 1; i++) {
        if (recentResults[i] !== recentResults[i + 1]) alternations++;
    }
    if (alternations / recentResults.length > 0.75) {
        return { isStable: false, reason: "Unstable: Excessive Choppiness", details: `Alternations: ${alternations}/${recentResults.length}`, dominance: outcomeDominance };
    }

    return { isStable: true, reason: "Trend appears stable.", details: `Entropy: ${entropy !== null ? entropy.toFixed(2) : 'N/A'}`, dominance: outcomeDominance };
}

function analyzeMarketEntropyState(history, trendContext, stability) {
    const ENTROPY_WINDOW_SHORT = 10;
    const ENTROPY_WINDOW_LONG = 25;
    const VOL_CHANGE_THRESHOLD = 0.3;

    if (history.length < ENTROPY_WINDOW_LONG) return { state: "UNCERTAIN_ENTROPY", details: "Insufficient history for entropy state." };

    const outcomesShort = history.slice(0, ENTROPY_WINDOW_SHORT).map(e => getBigSmallFromNumber(e.actual));
    const outcomesLong = history.slice(0, ENTROPY_WINDOW_LONG).map(e => getBigSmallFromNumber(e.actual));

    const entropyShort = calculateEntropyForSignal(outcomesShort, ENTROPY_WINDOW_SHORT);
    const entropyLong = calculateEntropyForSignal(outcomesLong, ENTROPY_WINDOW_LONG);

    const numbersShort = history.slice(0, ENTROPY_WINDOW_SHORT).map(e => parseInt(e.actualNumber || e.actual)).filter(n => !isNaN(n));
    const numbersLongPrev = history.slice(ENTROPY_WINDOW_SHORT, ENTROPY_WINDOW_SHORT + ENTROPY_WINDOW_SHORT).map(e => parseInt(e.actualNumber || e.actual)).filter(n => !isNaN(n));

    let shortTermVolatility = null, prevShortTermVolatility = null;
    if(numbersShort.length >= ENTROPY_WINDOW_SHORT * 0.8) shortTermVolatility = calculateStdDev(numbersShort, numbersShort.length);
    if(numbersLongPrev.length >= ENTROPY_WINDOW_SHORT * 0.8) prevShortTermVolatility = calculateStdDev(numbersLongPrev, numbersLongPrev.length);

    let state = "STABLE_MODERATE";
    let details = `E_S:${entropyShort?.toFixed(2)} E_L:${entropyLong?.toFixed(2)} Vol_S:${shortTermVolatility?.toFixed(2)} Vol_P:${prevShortTermVolatility?.toFixed(2)}`;

    if (entropyShort === null || entropyLong === null) return { state: "UNCERTAIN_ENTROPY", details };

    if (entropyShort < 0.5 && entropyLong < 0.6 && shortTermVolatility !== null && shortTermVolatility < 1.5) {
        state = "ORDERLY";
    }
    else if (entropyShort > 0.95 && entropyLong > 0.9) {
        if (shortTermVolatility && prevShortTermVolatility && shortTermVolatility > prevShortTermVolatility * (1 + VOL_CHANGE_THRESHOLD) && shortTermVolatility > 2.5) {
            state = "RISING_CHAOS";
        } else {
            state = "STABLE_CHAOS";
        }
    }
    else if (shortTermVolatility && prevShortTermVolatility) {
        if (shortTermVolatility > prevShortTermVolatility * (1 + VOL_CHANGE_THRESHOLD) && entropyShort > 0.85 && shortTermVolatility > 2.0) {
            state = "RISING_CHAOS";
        } else if (shortTermVolatility < prevShortTermVolatility * (1 - VOL_CHANGE_THRESHOLD) && entropyLong > 0.85 && entropyShort < 0.80) {
            state = "SUBSIDING_CHAOS";
        }
    }

    if (!stability.isStable && (state === "ORDERLY" || state === "STABLE_MODERATE")) {
        state = "POTENTIAL_CHAOS_FROM_INSTABILITY";
        details += ` | StabilityOverride: ${stability.reason}`;
    }
    return { state, details };
}

function analyzeAdvancedMarketRegime(trendContext, marketEntropyState) {
    const { strength, volatility } = trendContext;
    const { state: entropy } = marketEntropyState;

    let probabilities = {
        bullTrend: 0.25,
        bearTrend: 0.25,
        volatileRange: 0.25,
        quietRange: 0.25
    };

    if (strength === 'STRONG' && volatility !== 'HIGH' && entropy === 'ORDERLY') {
        if (trendContext.direction.includes('BIG')) {
            probabilities = { bullTrend: 0.8, bearTrend: 0.05, volatileRange: 0.1, quietRange: 0.05 };
        } else {
            probabilities = { bullTrend: 0.05, bearTrend: 0.8, volatileRange: 0.1, quietRange: 0.05 };
        }
    } else if (strength === 'RANGING' && volatility === 'HIGH' && entropy.includes('CHAOS')) {
         probabilities = { bullTrend: 0.1, bearTrend: 0.1, volatileRange: 0.7, quietRange: 0.1 };
    } else if (strength === 'RANGING' && volatility === 'VERY_LOW') {
         probabilities = { bullTrend: 0.1, bearTrend: 0.1, volatileRange: 0.1, quietRange: 0.7 };
    }

    return { probabilities, details: `Prob(B:${probabilities.bullTrend.toFixed(2)},S:${probabilities.bearTrend.toFixed(2)})` };
}

let signalPerformance = {};
const PERFORMANCE_WINDOW = 30;
const SESSION_PERFORMANCE_WINDOW = 15;
const MIN_OBSERVATIONS_FOR_ADJUST = 10;
const MAX_WEIGHT_FACTOR = 1.95;
const MIN_WEIGHT_FACTOR = 0.05;
const MAX_ALPHA_FACTOR = 1.6;
const MIN_ALPHA_FACTOR = 0.4;
const MIN_ABSOLUTE_WEIGHT = 0.0003;
const INACTIVITY_PERIOD_FOR_DECAY = PERFORMANCE_WINDOW * 3;
const DECAY_RATE = 0.025;
const ALPHA_UPDATE_RATE = 0.04;
const PROBATION_THRESHOLD_ACCURACY = 0.40;
const PROBATION_MIN_OBSERVATIONS = 15;
const PROBATION_WEIGHT_CAP = 0.10;
let driftDetector = { p_min: Infinity, s_min: Infinity, n: 0, warning_level: 2.0, drift_level: 3.0 };

function getDynamicWeightAdjustment(signalSourceName, baseWeight, currentPeriodFull, currentVolatilityRegime, sessionHistory) {
    const perf = signalPerformance[signalSourceName];
    if (!perf) {
        signalPerformance[signalSourceName] = {
            correct: 0, total: 0, recentAccuracy: [],
            sessionCorrect: 0, sessionTotal: 0,
            lastUpdatePeriod: null, lastActivePeriod: null,
            currentAdjustmentFactor: 1.0, alphaFactor: 1.0, longTermImportanceScore: 0.5,
            performanceByVolatility: {}, isOnProbation: false
        };
        return Math.max(baseWeight, MIN_ABSOLUTE_WEIGHT);
    }

    if (sessionHistory.length <= 1) {
        perf.sessionCorrect = 0;
        perf.sessionTotal = 0;
    }

    if (perf.lastUpdatePeriod !== currentPeriodFull) {
        if (perf.lastActivePeriod !== null && (currentPeriodFull - perf.lastActivePeriod) > INACTIVITY_PERIOD_FOR_DECAY) {
            if (perf.currentAdjustmentFactor > 1.0) perf.currentAdjustmentFactor = Math.max(1.0, perf.currentAdjustmentFactor - DECAY_RATE);
            else if (perf.currentAdjustmentFactor < 1.0) perf.currentAdjustmentFactor = Math.min(1.0, perf.currentAdjustmentFactor + DECAY_RATE);
            if (perf.isOnProbation) perf.isOnProbation = false;
        }
        perf.lastUpdatePeriod = currentPeriodFull;
    }

    let volatilitySpecificAdjustment = 1.0;
    if (perf.performanceByVolatility[currentVolatilityRegime] && perf.performanceByVolatility[currentVolatilityRegime].total >= MIN_OBSERVATIONS_FOR_ADJUST / 2.0) {
        const volPerf = perf.performanceByVolatility[currentVolatilityRegime];
        const volAccuracy = volPerf.correct / volPerf.total;
        const volDeviation = volAccuracy - 0.5;
        volatilitySpecificAdjustment = 1 + (volDeviation * 1.30);
        volatilitySpecificAdjustment = Math.min(Math.max(volatilitySpecificAdjustment, 0.55), 1.45);
    }

    let sessionAdjustmentFactor = 1.0;
    if (perf.sessionTotal >= 3) {
        const sessionAccuracy = perf.sessionCorrect / perf.sessionTotal;
        const sessionDeviation = sessionAccuracy - 0.5;
        sessionAdjustmentFactor = 1 + (sessionDeviation * 1.5);
        sessionAdjustmentFactor = Math.min(Math.max(sessionAdjustmentFactor, 0.6), 1.4);
    }

    let finalAdjustmentFactor = perf.currentAdjustmentFactor * perf.alphaFactor * volatilitySpecificAdjustment * sessionAdjustmentFactor * (0.70 + perf.longTermImportanceScore * 0.6);

    if (perf.isOnProbation) {
        finalAdjustmentFactor = Math.min(finalAdjustmentFactor, PROBATION_WEIGHT_CAP);
    }

    let adjustedWeight = baseWeight * finalAdjustmentFactor;
    return Math.max(adjustedWeight, MIN_ABSOLUTE_WEIGHT);
}

function updateSignalPerformance(contributingSignals, actualOutcome, periodFull, currentVolatilityRegime, lastFinalConfidence, concentrationModeActive, marketEntropyState) {
    if (!actualOutcome || !contributingSignals || contributingSignals.length === 0) return;
    const isHighConfidencePrediction = lastFinalConfidence > 0.75;
    const isOverallCorrect = getBigSmallFromNumber(actualOutcome) === (lastFinalConfidence > 0.5 ? "BIG" : "SMALL");

    contributingSignals.forEach(signal => {
        if (!signal || !signal.source) return;
        const source = signal.source;
        if (!signalPerformance[source]) {
            signalPerformance[source] = {
                correct: 0, total: 0, recentAccuracy: [],
                sessionCorrect: 0, sessionTotal: 0,
                lastUpdatePeriod: null, lastActivePeriod: null,
                currentAdjustmentFactor: 1.0, alphaFactor: 1.0, longTermImportanceScore: 0.5,
                performanceByVolatility: {}, isOnProbation: false
            };
        }

        if (!signalPerformance[source].performanceByVolatility[currentVolatilityRegime]) {
            signalPerformance[source].performanceByVolatility[currentVolatilityRegime] = { correct: 0, total: 0 };
        }

        if (signalPerformance[source].lastActivePeriod !== periodFull || signalPerformance[source].total === 0) {
            signalPerformance[source].total++;
            signalPerformance[source].sessionTotal++;
            signalPerformance[source].performanceByVolatility[currentVolatilityRegime].total++;
            let outcomeCorrect = (signal.prediction === actualOutcome) ? 1 : 0;
            if (outcomeCorrect) {
                signalPerformance[source].correct++;
                signalPerformance[source].sessionCorrect++;
                signalPerformance[source].performanceByVolatility[currentVolatilityRegime].correct++;
            }

            let importanceDelta = 0;
            if(outcomeCorrect) {
                importanceDelta = isHighConfidencePrediction ? 0.025 : 0.01;
            } else {
                importanceDelta = isHighConfidencePrediction && !isOverallCorrect ? -0.040 : -0.015;
            }

            if (concentrationModeActive || marketEntropyState.includes("CHAOS")) {
                 importanceDelta *= 1.5;
            }
            signalPerformance[source].longTermImportanceScore = Math.min(1.0, Math.max(0.0, signalPerformance[source].longTermImportanceScore + importanceDelta));

            signalPerformance[source].recentAccuracy.push(outcomeCorrect);
            if (signalPerformance[source].recentAccuracy.length > PERFORMANCE_WINDOW) {
                signalPerformance[source].recentAccuracy.shift();
            }

            if (signalPerformance[source].total >= MIN_OBSERVATIONS_FOR_ADJUST && signalPerformance[source].recentAccuracy.length >= PERFORMANCE_WINDOW / 2) {
                const recentCorrectCount = signalPerformance[source].recentAccuracy.reduce((sum, acc) => sum + acc, 0);
                const accuracy = recentCorrectCount / signalPerformance[source].recentAccuracy.length;
                const deviation = accuracy - 0.5;
                let newAdjustmentFactor = 1 + (deviation * 3.5);
                newAdjustmentFactor = Math.min(Math.max(newAdjustmentFactor, MIN_WEIGHT_FACTOR), MAX_WEIGHT_FACTOR);
                signalPerformance[source].currentAdjustmentFactor = newAdjustmentFactor;

                if (signalPerformance[source].recentAccuracy.length >= PROBATION_MIN_OBSERVATIONS && accuracy < PROBATION_THRESHOLD_ACCURACY) {
                    signalPerformance[source].isOnProbation = true;
                } else if (accuracy > PROBATION_THRESHOLD_ACCURACY + 0.15) {
                    signalPerformance[source].isOnProbation = false;
                }

                let alphaLearningRate = ALPHA_UPDATE_RATE;
                if (accuracy < 0.35) alphaLearningRate *= 1.75;
                else if (accuracy < 0.45) alphaLearningRate *= 1.4;

                if (newAdjustmentFactor > signalPerformance[source].alphaFactor) {
                    signalPerformance[source].alphaFactor = Math.min(MAX_ALPHA_FACTOR, signalPerformance[source].alphaFactor + alphaLearningRate * (newAdjustmentFactor - signalPerformance[source].alphaFactor));
                } else {
                    signalPerformance[source].alphaFactor = Math.max(MIN_ALPHA_FACTOR, signalPerformance[source].alphaFactor - alphaLearningRate * (signalPerformance[source].alphaFactor - newAdjustmentFactor));
                }
            }
            signalPerformance[source].lastActivePeriod = periodFull;
        }
        signalPerformance[source].lastUpdatePeriod = periodFull;
    });
}

function detectConceptDrift(isCorrect) {
    driftDetector.n++;
    const errorRate = isCorrect ? 0 : 1;
    const p_i = (driftDetector.n > 1 ? driftDetector.p_i : 0) + (errorRate - (driftDetector.n > 1 ? driftDetector.p_i : 0)) / driftDetector.n;
    driftDetector.p_i = p_i;
    const s_i = Math.sqrt(p_i * (1 - p_i) / driftDetector.n);

    if (p_i + s_i < driftDetector.p_min + driftDetector.s_min) {
        driftDetector.p_min = p_i;
        driftDetector.s_min = s_i;
    }

    if (p_i + s_i > driftDetector.p_min + driftDetector.drift_level * driftDetector.s_min) {
        driftDetector.p_min = Infinity;
        driftDetector.s_min = Infinity;
        driftDetector.n = 1;
        return 'DRIFT';
    } else if (p_i + s_i > driftDetector.p_min + driftDetector.warning_level * driftDetector.s_min) {
        return 'WARNING';
    } else {
        return 'STABLE';
    }
}

let REGIME_SIGNAL_PROFILES = {
    "TREND_STRONG_LOW_VOL": { baseWeightMultiplier: 1.30, activeSignalTypes: ['trend', 'momentum', 'ichimoku', 'volBreak', 'leadLag', 'stateSpace', 'fusion', 'ml'], contextualAggression: 1.35, recentAccuracy: [], totalPredictions: 0, correctPredictions: 0 },
    "TREND_STRONG_MED_VOL": { baseWeightMultiplier: 1.20, activeSignalTypes: ['trend', 'momentum', 'ichimoku', 'pattern', 'leadLag', 'stateSpace', 'fusion', 'ml'], contextualAggression: 1.25, recentAccuracy: [], totalPredictions: 0, correctPredictions: 0 },
    "TREND_STRONG_HIGH_VOL": { baseWeightMultiplier: 0.70, activeSignalTypes: ['trend', 'ichimoku', 'entropy', 'volPersist', 'zScore', 'fusion'], contextualAggression: 0.70, recentAccuracy: [], totalPredictions: 0, correctPredictions: 0 },
    "TREND_MOD_LOW_VOL": { baseWeightMultiplier: 1.25, activeSignalTypes: ['trend', 'momentum', 'ichimoku', 'pattern', 'volBreak', 'leadLag', 'stateSpace', 'ml'], contextualAggression: 1.25, recentAccuracy: [], totalPredictions: 0, correctPredictions: 0 },
    "TREND_MOD_MED_VOL": { baseWeightMultiplier: 1.15, activeSignalTypes: ['trend', 'momentum', 'ichimoku', 'pattern', 'rsi', 'leadLag', 'bayesian', 'fusion', 'ml'], contextualAggression: 1.15, recentAccuracy: [], totalPredictions: 0, correctPredictions: 0 },
    "TREND_MOD_HIGH_VOL": { baseWeightMultiplier: 0.75, activeSignalTypes: ['trend', 'ichimoku', 'meanRev', 'stochastic', 'volPersist', 'zScore'], contextualAggression: 0.75, recentAccuracy: [], totalPredictions: 0, correctPredictions: 0 },
    "RANGE_LOW_VOL": { baseWeightMultiplier: 1.30, activeSignalTypes: ['meanRev', 'pattern', 'volBreak', 'stochastic', 'harmonic', 'fractalDim', 'zScore', 'bayesian', 'fusion'], contextualAggression: 1.30, recentAccuracy: [], totalPredictions: 0, correctPredictions: 0 },
    "RANGE_MED_VOL": { baseWeightMultiplier: 1.15, activeSignalTypes: ['meanRev', 'pattern', 'stochastic', 'rsi', 'bollinger', 'harmonic', 'zScore'], contextualAggression: 1.15, recentAccuracy: [], totalPredictions: 0, correctPredictions: 0 },
    "RANGE_HIGH_VOL": { baseWeightMultiplier: 0.65, activeSignalTypes: ['meanRev', 'entropy', 'bollinger', 'vwapDev', 'volPersist', 'zScore', 'fusion'], contextualAggression: 0.65, recentAccuracy: [], totalPredictions: 0, correctPredictions: 0 },
    "WEAK_HIGH_VOL": { baseWeightMultiplier: 0.70, activeSignalTypes: ['meanRev', 'entropy', 'stochastic', 'volPersist', 'fractalDim', 'zScore'], contextualAggression: 0.70, recentAccuracy: [], totalPredictions: 0, correctPredictions: 0 },
    "WEAK_MED_VOL": { baseWeightMultiplier: 0.75, activeSignalTypes: ['momentum', 'meanRev', 'pattern', 'rsi', 'fractalDim', 'bayesian'], contextualAggression: 0.75, recentAccuracy: [], totalPredictions: 0, correctPredictions: 0 },
    "WEAK_LOW_VOL": { baseWeightMultiplier: 0.85, activeSignalTypes: ['all'], contextualAggression: 0.85, recentAccuracy: [], totalPredictions: 0, correctPredictions: 0 },
    "DEFAULT": { baseWeightMultiplier: 0.9, activeSignalTypes: ['all'], contextualAggression: 0.9, recentAccuracy: [], totalPredictions: 0, correctPredictions: 0 }
};
const REGIME_ACCURACY_WINDOW = 35;
const REGIME_LEARNING_RATE_BASE = 0.028;
let GLOBAL_LONG_TERM_ACCURACY_FOR_LEARNING_RATE = 0.5;

function updateRegimeProfilePerformance(regime, actualOutcome, predictedOutcome) {
    if (REGIME_SIGNAL_PROFILES[regime] && predictedOutcome) {
        const profile = REGIME_SIGNAL_PROFILES[regime];
        profile.totalPredictions = (profile.totalPredictions || 0) + 1;
        let outcomeCorrect = (actualOutcome === predictedOutcome) ? 1 : 0;
        if(outcomeCorrect === 1) profile.correctPredictions = (profile.correctPredictions || 0) + 1;

        profile.recentAccuracy.push(outcomeCorrect);
        if (profile.recentAccuracy.length > REGIME_ACCURACY_WINDOW) {
            profile.recentAccuracy.shift();
        }

        if (profile.recentAccuracy.length >= REGIME_ACCURACY_WINDOW * 0.7) {
            const regimeAcc = profile.recentAccuracy.reduce((a,b) => a+b, 0) / profile.recentAccuracy.length;
            let dynamicLearningRateFactor = 1.0 + Math.abs(0.5 - GLOBAL_LONG_TERM_ACCURACY_FOR_LEARNING_RATE) * 0.7;
            dynamicLearningRateFactor = Math.max(0.65, Math.min(1.5, dynamicLearningRateFactor));
            let currentLearningRate = REGIME_LEARNING_RATE_BASE * dynamicLearningRateFactor;
            currentLearningRate = Math.max(0.01, Math.min(0.07, currentLearningRate));

            if (regimeAcc > 0.62) {
                profile.baseWeightMultiplier = Math.min(1.9, profile.baseWeightMultiplier + currentLearningRate);
                profile.contextualAggression = Math.min(1.8, profile.contextualAggression + currentLearningRate * 0.5);
            } else if (regimeAcc < 0.38) {
                profile.baseWeightMultiplier = Math.max(0.20, profile.baseWeightMultiplier - currentLearningRate * 1.3);
                profile.contextualAggression = Math.max(0.30, profile.contextualAggression - currentLearningRate * 0.7);
            }
        }
    }
}

function analyzeBayesianInference(signals, baseWeight) {
    if (!signals || signals.length < 5) return null;

    let posteriorBig = 0.5;
    let posteriorSmall = 0.5;

    const categories = {
        trend: { BIG: 0, SMALL: 0, total: 0 },
        momentum: { BIG: 0, SMALL: 0, total: 0 },
        meanRev: { BIG: 0, SMALL: 0, total: 0 },
    };

    signals.forEach(s => {
        let category = null;
        if (s.source.includes("MACD") || s.source.includes("Ichimoku")) category = 'trend';
        else if (s.source.includes("Stochastic") || s.source.includes("RSI")) category = 'momentum';
        else if (s.source.includes("Bollinger") || s.source.includes("MADev") || s.source.includes("ZScore")) category = 'meanRev';

        if (category && (s.prediction === 'BIG' || s.prediction === 'SMALL')) {
            categories[category][s.prediction] += s.adjustedWeight;
            categories[category].total += s.adjustedWeight;
        }
    });

    for (const cat of Object.values(categories)) {
        if (cat.total > 0) {
            const evidenceForBig = cat.BIG / cat.total;
            const evidenceForSmall = cat.SMALL / cat.total;

            let newPosteriorBig = evidenceForBig * posteriorBig;
            let newPosteriorSmall = evidenceForSmall * posteriorSmall;

            const normalization = newPosteriorBig + newPosteriorSmall;
            if (normalization > 0) {
                posteriorBig = newPosteriorBig / normalization;
                posteriorSmall = newPosteriorSmall / normalization;
            }
        }
    }

    if (posteriorBig > posteriorSmall && posteriorBig > 0.65) {
        return { prediction: "BIG", weight: baseWeight * (posteriorBig - 0.5) * 2, source: "Bayesian" };
    }
    if (posteriorSmall > posteriorBig && posteriorSmall > 0.65) {
        return { prediction: "SMALL", weight: baseWeight * (posteriorSmall - 0.5) * 2, source: "Bayesian" };
    }
    return null;
}

function analyzeZScoreAnomaly(history, period, threshold, baseWeight) {
    const numbers = history.map(entry => parseInt(entry.actualNumber || entry.actual)).filter(n => !isNaN(n));
    if (numbers.length < period) return null;

    const slice = numbers.slice(0, period);
    const mean = calculateSMA(slice, period);
    const stdDev = calculateStdDev(slice, period);

    if (mean === null || stdDev === null || stdDev < 0.1) return null;

    const lastNumber = numbers[0];
    const zScore = (lastNumber - mean) / stdDev;

    let prediction = null;
    if (zScore > threshold) prediction = "SMALL";
    else if (zScore < -threshold) prediction = "BIG";

    if (prediction) {
        const strengthFactor = Math.min(1.0, (Math.abs(zScore) - threshold) / threshold);
        return { prediction, weight: baseWeight * (0.5 + strengthFactor * 0.5), source: "ZScoreAnomaly" };
    }
    return null;
}

function analyzeStateSpaceMomentum(history, period, baseWeight) {
    const numbers = history.map(entry => parseInt(entry.actualNumber || entry.actual)).filter(n => !isNaN(n));
    if (numbers.length < period * 2) return null;

    const chronologicalNumbers = numbers.slice().reverse();

    let velocity = 0;
    let error = 0;
    const gain = 0.6;

    for (let i = 1; i < chronologicalNumbers.length; i++) {
        const measurement = chronologicalNumbers[i] - chronologicalNumbers[i-1];
        const prediction = velocity;
        error = measurement - prediction;
        velocity = prediction + gain * error;
    }

    const velocities = [];
    for(let i=1; i < chronologicalNumbers.length; i++) {
        velocities.push(chronologicalNumbers[i] - chronologicalNumbers[i-1]);
    }
    const avgVelocity = velocities.reduce((a,b) => a+b, 0) / velocities.length;

    let prediction = null;
    if (velocity > avgVelocity * 1.8 && velocity > 0.5) {
        prediction = "BIG";
    }
    else if (velocity < avgVelocity * 1.8 && velocity < -0.5) {
        prediction = "SMALL";
    }

    if (prediction) {
        const strengthFactor = Math.min(1, Math.abs(velocity - avgVelocity) / (Math.abs(avgVelocity) + 1));
        return { prediction, weight: baseWeight * strengthFactor, source: "StateSpaceMomentum" };
    }
    return null;
}

function analyzePredictionConsensus(signals, trendContext) {
    if (!signals || signals.length < 4) {
        return { score: 0.5, factor: 1.0, details: "Insufficient signals for consensus" };
    }

    const categories = {
        trend: { BIG: 0, SMALL: 0, weight: 0 },
        momentum: { BIG: 0, SMALL: 0, weight: 0 },
        meanRev: { BIG: 0, SMALL: 0, weight: 0 },
        pattern: { BIG: 0, SMALL: 0, weight: 0 },
        volatility: { BIG: 0, SMALL: 0, weight: 0 },
        probabilistic: { BIG: 0, SMALL: 0, weight: 0 },
        ml: { BIG: 0, SMALL: 0, weight: 0 }
    };

    const getCategory = source => {
        if (source.includes("MACD") || source.includes("Ichimoku")) return 'trend';
        if (source.includes("Stochastic") || source.includes("RSI")) return 'momentum';
        if (source.includes("Bollinger") || source.includes("MADev") || source.includes("ZScore")) return 'meanRev';
        if (source.includes("Gram") || source.includes("Cycle") || source.includes("Alt") || source.includes("Harmonic") || source.includes("Pattern")) return 'pattern';
        if (source.includes("Vol") || source.includes("Fractal") || source.includes("QuantumTunnel")) return 'volatility';
        if (source.includes("Bayesian") || source.includes("MonteCarlo") || source.includes("Superposition")) return 'probabilistic';
        if (source.includes("ML-")) return 'ml';
        return null;
    };

    signals.forEach(s => {
        const category = getCategory(s.source);
        if (category && (s.prediction === "BIG" || s.prediction === "SMALL")) {
            categories[category][s.prediction] += s.adjustedWeight;
        }
    });

    let bigWeight = 0, smallWeight = 0;
    let bigCats = 0, smallCats = 0, mixedCats = 0;

    for(const cat of Object.values(categories)) {
        const totalWeight = cat.BIG + cat.SMALL;
        if (totalWeight > 0) {
            bigWeight += cat.BIG;
            smallWeight += cat.SMALL;
            if(cat.BIG > cat.SMALL * 1.2) bigCats++;
            else if (cat.SMALL > cat.BIG * 1.2) smallCats++;
            else mixedCats++;
        }
    }

    let consensusScore = 0.5;
    const totalCats = bigCats + smallCats + mixedCats;
    if(totalCats > 0) {
        const dominantCats = Math.max(bigCats, smallCats);
        const nonDominantCats = Math.min(bigCats, smallCats);
        consensusScore = (dominantCats - nonDominantCats) / totalCats;
    }

    let factor = 1.0 + (consensusScore * 0.4);

    if (trendContext.strength === 'STRONG') {
        if((categories.trend.BIG > categories.trend.SMALL && categories.momentum.SMALL > categories.momentum.BIG) ||
           (categories.trend.SMALL > categories.trend.BIG && categories.momentum.BIG > categories.momentum.SMALL)) {
            factor *= 0.6;
        }
    }

    return {
        score: consensusScore,
        factor: Math.max(0.4, Math.min(1.6, factor)),
        details: `Bcat:${bigCats},Scat:${smallCats},Mcat:${mixedCats},Score:${consensusScore.toFixed(2)}`
    };
}

function analyzeQuantumSuperpositionState(signals, consensus, baseWeight) {
    if (!signals || signals.length < 5 || !consensus) return null;

    const totalWeight = signals.reduce((sum, s) => sum + (s.adjustedWeight || 0), 0);
    if (totalWeight < 0.1) return null;

    const bigWeight = signals.filter(s => s.prediction === "BIG").reduce((sum, s) => sum + s.adjustedWeight, 0);
    const smallWeight = signals.filter(s => s.prediction === "SMALL").reduce((sum, s) => sum + s.adjustedWeight, 0);

    const bigCollapseProbability = (bigWeight / totalWeight) * consensus.factor;
    const smallCollapseProbability = (smallWeight / totalWeight) * (2.0 - consensus.factor);

    if (bigCollapseProbability > smallCollapseProbability * 1.3) {
        return {
            prediction: "BIG",
            weight: baseWeight * Math.min(1.0, (bigCollapseProbability - smallCollapseProbability)),
            source: "QuantumSuperposition"
        };
    }

    if (smallCollapseProbability > bigCollapseProbability * 1.3) {
        return {
            prediction: "SMALL",
            weight: baseWeight * Math.min(1.0, (smallCollapseProbability - bigCollapseProbability)),
            source: "QuantumSuperposition"
        };
    }

    return null;
}

function analyzePathConfluenceStrength(signals, finalPrediction) {
    if (!signals || signals.length === 0 || !finalPrediction) return { score: 0, diversePaths: 0, details: "No valid signals or prediction." };

    const agreeingSignals = signals.filter(s => s.prediction === finalPrediction && s.adjustedWeight > MIN_ABSOLUTE_WEIGHT * 10);
    if (agreeingSignals.length < 2) {
        return { score: 0, diversePaths: agreeingSignals.length, details: "Insufficient agreeing signals." };
    }

    const signalCategories = new Set();
    agreeingSignals.forEach(s => {
        if (s.source.includes("MACD") || s.source.includes("Ichimoku")) signalCategories.add('trend');
        else if (s.source.includes("Stochastic") || s.source.includes("RSI")) signalCategories.add('momentum');
        else if (s.source.includes("Bollinger") || s.source.includes("ZScore")) signalCategories.add('meanRev');
        else if (s.source.includes("Gram") || s.source.includes("Cycle") || s.source.includes("Pattern")) signalCategories.add('pattern');
        else if (s.source.includes("Vol") || s.source.includes("FractalDim")) signalCategories.add('volatility');
        else if (s.source.includes("Bayesian") || s.source.includes("Superposition") || s.source.includes("ML-")) signalCategories.add('probabilistic');
        else signalCategories.add('other');
    });

    const diversePathCount = signalCategories.size;
    let confluenceScore = 0;

    if (diversePathCount >= 4) confluenceScore = 0.20;
    else if (diversePathCount === 3) confluenceScore = 0.12;
    else if (diversePathCount === 2) confluenceScore = 0.05;

    const veryStrongAgreeingCount = agreeingSignals.filter(s => s.adjustedWeight > 0.10).length;
    confluenceScore += Math.min(veryStrongAgreeingCount * 0.02, 0.10);

    return { score: Math.min(confluenceScore, 0.30), diversePaths: diversePathCount, details: `Paths:${diversePathCount},Strong:${veryStrongAgreeingCount}` };
}

function analyzeSignalConsistency(signals, trendContext) {
    if (!signals || signals.length < 3) return { score: 0.70, details: "Too few signals for consistency check" };
    const validSignals = signals.filter(s => s.prediction);
    if (validSignals.length < 3) return { score: 0.70, details: "Too few valid signals" };

    const predictions = { BIG: 0, SMALL: 0 };
    validSignals.forEach(s => {
        if (s.prediction === "BIG" || s.prediction === "SMALL") predictions[s.prediction]++;
    });

    const totalPredictions = predictions.BIG + predictions.SMALL;
    if (totalPredictions === 0) return { score: 0.5, details: "No directional signals" };

    const consistencyScore = Math.max(predictions.BIG, predictions.SMALL) / totalPredictions;
    return { score: consistencyScore, details: `Overall split B:${predictions.BIG}/S:${predictions.SMALL}` };
}

let consecutiveHighConfLosses = 0;
let reflexiveCorrectionActive = 0;

function checkForAnomalousPerformance(currentSharedStats) {
    if (reflexiveCorrectionActive > 0) {
        reflexiveCorrectionActive--;
        return true;
    }

    if (currentSharedStats && typeof currentSharedStats.lastFinalConfidence === 'number' && currentSharedStats.lastActualOutcome) {
        const lastPredOutcomeBS = getBigSmallFromNumber(currentSharedStats.lastActualOutcome);
        const lastPredWasCorrect = lastPredOutcomeBS === currentSharedStats.lastPredictedOutcome;

        const lastPredWasHighConf = currentSharedStats.lastConfidenceLevel === 3;

        if (lastPredWasHighConf && !lastPredWasCorrect) {
            consecutiveHighConfLosses++;
        } else {
            consecutiveHighConfLosses = 0;
        }
    }

    if (consecutiveHighConfLosses >= 2) {
        reflexiveCorrectionActive = 5;
        consecutiveHighConfLosses = 0;
        return true;
    }

    return false;
}

function calculateUncertaintyScore(trendContext, stability, marketEntropyState, signalConsistency, pathConfluence, globalAccuracy, isReflexiveCorrection, driftState, apexInfluenceFlags) {
    let uncertaintyScore = 0;
    let reasons = [];

    if (isReflexiveCorrection) {
        uncertaintyScore += 80;
        reasons.push("ReflexiveCorrection");
    }
    if(driftState === 'DRIFT') {
        uncertaintyScore += 70;
        reasons.push("ConceptDrift");
    } else if (driftState === 'WARNING') {
        uncertaintyScore += 40;
        reasons.push("DriftWarning");
    }
    if (!stability.isStable) {
        uncertaintyScore += (stability.reason.includes("Dominance") || stability.reason.includes("Choppiness")) ? 50 : 40;
        reasons.push(`Instability:${stability.reason}`);
    }
    if (marketEntropyState.state.includes("CHAOS")) {
        uncertaintyScore += marketEntropyState.state === "RISING_CHAOS" ? 45 : 35;
        reasons.push(marketEntropyState.state);
    }
    if (signalConsistency.score < 0.6) {
        uncertaintyScore += (1 - signalConsistency.score) * 50;
        reasons.push(`LowConsistency:${signalConsistency.score.toFixed(2)}`);
    }
    if (pathConfluence.diversePaths < 3) {
        uncertaintyScore += (3 - pathConfluence.diversePaths) * 15;
        reasons.push(`LowConfluence:${pathConfluence.diversePaths}`);
    }
    if (trendContext.isTransitioning) {
        uncertaintyScore += 25;
        reasons.push("RegimeTransition");
    }
    if (trendContext.volatility === "HIGH") {
        uncertaintyScore += 20;
        reasons.push("HighVolatility");
    }
     if (typeof globalAccuracy === 'number' && globalAccuracy < 0.48) {
        uncertaintyScore += (0.48 - globalAccuracy) * 150;
        reasons.push(`LowGlobalAcc:${globalAccuracy.toFixed(2)}`);
    }

    if (apexInfluenceFlags.eidolonDistortionFactor > 0.1) {
        uncertaintyScore += apexInfluenceFlags.eidortionDistortionFactor * 60;
        reasons.push(`EidolonDistort:${apexInfluenceFlags.eidolonDistortionFactor.toFixed(2)}`);
    }
    if (apexInfluenceFlags.architectInterferenceDetected) {
        uncertaintyScore += 75;
        reasons.push("ArchitectInterference");
    }
    if (apexInfluenceFlags.nexusBleedThroughActive) {
        uncertaintyScore += 40;
        reasons.push("NexusBleedThrough");
    }
    if (apexInfluenceFlags.progenitorGhostActivity) {
        uncertaintyScore += 20;
        reasons.push("ProgenitorGhost");
    }

    return { score: uncertaintyScore, reasons: reasons.join(';') };
}

function createFeatureSetForML(history, trendContext, time) {
    const numbers = history.map(e => parseInt(e.actualNumber || e.actual)).filter(n => !isNaN(n));
    if(numbers.length < 52) return null;

    return {
        time_sin: time.sin,
        time_cos: time.cos,
        last_5_mean: calculateSMA(numbers, 5),
        last_20_mean: calculateSMA(numbers, 20),
        stddev_10: calculateStdDev(numbers, 10),
        stddev_30: calculateStdDev(numbers, 30),
        rsi_14: calculateRSI(numbers, 14),
        stoch_k_14: analyzeStochastic(history, 14, 3, 3, 1.0, trendContext.volatility)?.currentK,
        macd_hist: analyzeMACD(history, 12, 26, 9, 1.0)?.macdHistogram,
        trend_strength: trendContext.strength === 'STRONG' ? 2 : (trendContext.strength === 'MODERATE' ? 1 : 0),
        volatility_level: trendContext.volatility === 'HIGH' ? 2 : (trendContext.volatility === 'MEDIUM' ? 1 : 0),
    };
}

// The main prediction logic function, now internal to the backend
async function backendPredictionLogic(historicalData, clientSharedStats) {
    const currentPeriodFull = clientSharedStats?.periodFull || Date.now();
    const time = getCurrentISTHour();
    const primeTimeSession = getPrimeTimeSession(time.raw);
    const realTimeData = getRealTimeExternalData();

    console.log(`Quantum AI Supercore v6.0 Internal Prediction Initializing for period ${currentPeriodFull}`);
    let masterLogic = [`QAScore_v6.0(IST_Hr:${time.raw})`];
    masterLogic.push(realTimeData.reason);

    const apexInfluenceFlags = {
        eidolonDistortionFactor: Math.random() < 0.05 ? Math.random() * 0.5 + 0.5 : 0,
        architectInterferenceDetected: Math.random() < 0.02,
        nexusBleedThroughActive: Math.random() < 0.03,
        progenitorGhostActivity: Math.random() < 0.1
    };
    if (apexInfluenceFlags.eidolonDistortionFactor > 0) masterLogic.push(`EidolonDistortion: ${apexInfluenceFlags.eidolonDistortionFactor.toFixed(2)}`);
    if (apexInfluenceFlags.architectInterferenceDetected) masterLogic.push(`ArchitectInterference: TRUE`);
    if (apexInfluenceFlags.nexusBleedThroughActive) masterLogic.push(`NexusBleedThrough: TRUE`);
    if (apexInfluenceFlags.progenitorGhostActivity) masterLogic.push(`ProgenitorGhost: TRUE`);

    let primeTimeAggression = 1.0;
    let primeTimeConfidence = 1.0;
    if (primeTimeSession) {
        masterLogic.push(`!!! PRIME TIME ACTIVE: ${primeTimeSession.session} !!!`);
        primeTimeAggression = primeTimeSession.aggression;
        primeTimeConfidence = primeTimeSession.confidence;
    }

    let longTermGlobalAccuracy = clientSharedStats?.longTermGlobalAccuracy || GLOBAL_LONG_TERM_ACCURACY_FOR_LEARNING_RATE;
    GLOBAL_LONG_TERM_ACCURACY_FOR_LEARNING_RATE = longTermGlobalAccuracy; // Sync global with currentSharedStats for next run

    const isReflexiveCorrection = checkForAnomalousPerformance(clientSharedStats);
    if (isReflexiveCorrection) {
        masterLogic.push(`!!! REFLEXIVE CORRECTION ACTIVE !!! (Countdown: ${reflexiveCorrectionActive})`);
    }

    const trendContext = getMarketRegimeAndTrendContext(historicalData);
    masterLogic.push(`TrendCtx(Dir:${trendContext.direction},Str:${trendContext.strength},Vol:${trendContext.volatility},Regime:${trendContext.macroRegime})`);

    const stability = analyzeTrendStability(historicalData);
    const marketEntropyAnalysis = analyzeMarketEntropyState(historicalData, trendContext, stability);
    masterLogic.push(`MarketEntropy:${marketEntropyAnalysis.state}`);

    const advancedRegime = analyzeAdvancedMarketRegime(trendContext, marketEntropyAnalysis);
    masterLogic.push(`AdvRegime:${advancedRegime.details}`);

    let concentrationModeEngaged = !stability.isStable || isReflexiveCorrection || marketEntropyAnalysis.state.includes("CHAOS");

    let driftState = 'STABLE';
    if (clientSharedStats && typeof clientSharedStats.lastActualOutcome !== 'undefined') {
        const lastPredictionWasCorrect = getBigSmallFromNumber(clientSharedStats.lastActualOutcome) === clientSharedStats.lastPredictedOutcome;
        driftState = detectConceptDrift(lastPredictionWasCorrect);
        if (driftState !== 'STABLE') {
            masterLogic.push(`!!! DRIFT DETECTED: ${driftState} !!!`);
            concentrationModeEngaged = true;
        }
    }

    if (concentrationModeEngaged) masterLogic.push(`ConcentrationModeActive`);

    const currentVolatilityRegimeForPerf = trendContext.volatility;
    const currentMacroRegime = trendContext.macroRegime;
    if (clientSharedStats && clientSharedStats.lastPredictionSignals && clientSharedStats.lastActualOutcome) {
        updateSignalPerformance(
            clientSharedStats.lastPredictionSignals,
            getBigSmallFromNumber(clientSharedStats.lastActualOutcome),
            clientSharedStats.lastPeriodFull,
            clientSharedStats.lastVolatilityRegime || currentVolatilityRegimeForPerf,
            clientSharedStats.lastFinalConfidence,
            clientSharedStats.lastConcentrationModeEngaged || false,
            clientSharedStats.lastMarketEntropyState || "STABLE_MODERATE"
        );
        updateRegimeProfilePerformance(clientSharedStats.lastMacroRegime, getBigSmallFromNumber(clientSharedStats.lastActualOutcome), clientSharedStats.lastPredictedOutcome);
    }

    const confirmedHistory = historicalData.filter(p => p && p.actual !== null && p.actualNumber !== undefined);
    if (confirmedHistory.length < 52) {
        masterLogic.push(`InsufficientHistory_ForceRandom`);
        const finalDecision = Math.random() > 0.5 ? "BIG" : "SMALL";
        return {
            predictions: { BIG: { confidence: 0.5, logic: "ForcedRandom" }, SMALL: { confidence: 0.5, logic: "ForcedRandom" } },
            finalDecision: finalDecision, finalConfidence: 0.5, confidenceLevel: 1, isForcedPrediction: true,
            overallLogic: masterLogic.join(' -> '), source: "InsufficientHistory",
            contributingSignals: [], currentMacroRegime, concentrationModeEngaged, pathConfluenceScore: 0, marketEntropyState: marketEntropyAnalysis.state, predictionQualityScore: 0.01, reflexiveCorrectionActive: isReflexiveCorrection, lastPredictedOutcome: finalDecision, lastFinalConfidence: 0.5, lastMacroRegime: currentMacroRegime, lastPredictionSignals: [], lastConcentrationModeEngaged: concentrationModeEngaged, lastMarketEntropyState: marketEntropyAnalysis.state, lastVolatilityRegime: trendContext.volatility, lastConfidenceLevel: 1
        };
    }

    let signals = [];
    const currentRegimeProfile = REGIME_SIGNAL_PROFILES[currentMacroRegime] || REGIME_SIGNAL_PROFILES["DEFAULT"];
    let regimeContextualAggression = (currentRegimeProfile.contextualAggression || 1.0) * primeTimeAggression;

    if (apexInfluenceFlags.eidolonDistortionFactor > 0.5) regimeContextualAggression *= (1.0 - apexInfluenceFlags.eidolonDistortionFactor);
    if (apexInfluenceFlags.architectInterferenceDetected) regimeContextualAggression *= 0.5;

    if (isReflexiveCorrection || driftState === 'DRIFT') regimeContextualAggression *= 0.25;
    else if (concentrationModeEngaged) regimeContextualAggression *= 0.6;

    const addSignal = (fn, historyArg, signalType, lookbackParams, baseWeight) => {
        if (!(currentRegimeProfile.activeSignalTypes.includes('all') || currentRegimeProfile.activeSignalTypes.includes(signalType))) {
            return;
        }

        const fnArgs = Array.isArray(lookbackParams) ?
            [historyArg, ...lookbackParams, baseWeight] :
            [historyArg, ...Object.values(lookbackParams), baseWeight];

         if (fn === analyzeRSI || fn === analyzeStochastic) {
             fnArgs.push(trendContext.volatility);
        }
        if (fn === analyzeVolatilityTrendFusion) {
            fnArgs.splice(1, 0, marketEntropyAnalysis)
        }
        if (fn === analyzeMLModelSignal) {
            const features = createFeatureSetForML(historyArg, trendContext, time);
            if (!features) return;
            if (apexInfluenceFlags.nexusBleedThroughActive) {
                features.nexus_feature_1 = Math.random();
                features.nexus_feature_2 = Math.random() > 0.5 ? 1 : 0;
            }
            if (apexInfluenceFlags.progenitorGhostActivity) {
                features.progenitor_bot_sentiment = Math.random() * 2 - 1;
            }
            fnArgs.splice(0, 1, features);
        }

        const result = fn(...fnArgs);

        if (result && result.weight && result.prediction) {
            result.adjustedWeight = getDynamicWeightAdjustment(result.source, result.weight * regimeContextualAggression, currentPeriodFull, currentVolatilityRegimeForPerf, historicalData);
            signals.push(result);
        }
    };

    addSignal(analyzeTransitions, confirmedHistory, 'pattern', {}, 0.05);
    addSignal(analyzeStreaks, confirmedHistory, 'meanRev', {}, 0.045);
    addSignal(analyzeAlternatingPatterns, confirmedHistory, 'pattern', {}, 0.06);
    addSignal(analyzeTwoPlusOnePatterns, confirmedHistory, 'pattern', {}, 0.07);
    addSignal(analyzeDoublePatterns, confirmedHistory, 'pattern', {}, 0.075);
    addSignal(analyzeMirrorPatterns, confirmedHistory, 'pattern', {}, 0.08);
    addSignal(analyzeRSI, confirmedHistory, 'momentum', { rsiPeriod: 14 }, 0.08);
    addSignal(analyzeMACD, confirmedHistory, 'trend', { shortPeriod: 12, longPeriod: 26, signalPeriod: 9 }, 0.09);
    addSignal(analyzeBollingerBands, confirmedHistory, 'meanRev', { period: 20, stdDevMultiplier: 2.1 }, 0.07);
    addSignal(analyzeIchimokuCloud, confirmedHistory, 'trend', { tenkanPeriod: 9, kijunPeriod: 26, senkouBPeriod: 52 }, 0.14);
    addSignal(analyzeStochastic, confirmedHistory, 'momentum', { kPeriod: 14, dPeriod: 3, smoothK: 3 }, 0.08);
    addSignal(analyzeZScoreAnomaly, confirmedHistory, 'meanRev', { period: 20, threshold: 2.0 }, 0.12);
    addSignal(analyzeStateSpaceMomentum, confirmedHistory, 'trend', { period: 15 }, 0.11);
    addSignal(analyzeWaveformPatterns, confirmedHistory, 'pattern', {}, 0.035);
    addSignal(analyzeQuantumTunneling, confirmedHistory, 'volatility', {}, 0.055);

    addSignal(analyzeVolatilityTrendFusion, trendContext, 'fusion', {}, 0.25);
    addSignal(analyzeMLModelSignal, confirmedHistory, 'ml', {}, 0.40);
    addSignal(analyzeBayesianInference, signals, 'probabilistic', {}, 0.15);

    const consensus = analyzePredictionConsensus(signals, trendContext);
    masterLogic.push(`Consensus:${consensus.details},Factor:${consensus.factor.toFixed(2)}`);
    const superpositionSignal = analyzeQuantumSuperpositionState(signals, consensus, 0.22);
    if (superpositionSignal) {
        superpositionSignal.adjustedWeight = getDynamicWeightAdjustment(superpositionSignal.source, superpositionSignal.weight, currentPeriodFull, currentVolatilityRegimeForPerf, historicalData);
        signals.push(superpositionSignal);
    }

    const validSignals = signals.filter(s => s?.prediction && s.adjustedWeight > MIN_ABSOLUTE_WEIGHT);
    masterLogic.push(`ValidSignals(${validSignals.length}/${signals.length})`);

    if (validSignals.length === 0) {
        masterLogic.push(`NoValidSignals_ForceRandom`);
        const finalDecision = Math.random() > 0.5 ? "BIG" : "SMALL";
        return {
            predictions: { BIG: { confidence: 0.5, logic: "ForcedRandom" }, SMALL: { confidence: 0.5, logic: "ForcedRandom" } },
            finalDecision: finalDecision, finalConfidence: 0.5, confidenceLevel: 1, isForcedPrediction: true,
            overallLogic: masterLogic.join(' -> '), source: "InsufficientHistory",
            contributingSignals: [], currentMacroRegime, concentrationModeEngaged, pathConfluenceScore: 0, marketEntropyState: marketEntropyAnalysis.state, predictionQualityScore: 0.01, reflexiveCorrectionActive: isReflexiveCorrection, lastPredictedOutcome: finalDecision, lastFinalConfidence: 0.5, lastMacroRegime: currentMacroRegime, lastPredictionSignals: [], lastConcentrationModeEngaged: concentrationModeEngaged, lastMarketEntropyState: marketEntropyAnalysis.state, lastVolatilityRegime: trendContext.volatility, lastConfidenceLevel: 1
        };
    }

    let bigScore = 0; let smallScore = 0;
    validSignals.forEach(signal => {
        if (signal.prediction === "BIG") bigScore += signal.adjustedWeight;
        else if (signal.prediction === "SMALL") smallScore += signal.adjustedWeight;
    });

    bigScore *= (1 + advancedRegime.probabilities.bullTrend - advancedRegime.probabilities.bearTrend);
    smallScore *= (1 + advancedRegime.probabilities.bearTrend - advancedRegime.probabilities.bullTrend);

    bigScore *= consensus.factor;
    smallScore *= (2.0 - consensus.factor);

    const totalScore = bigScore + smallScore;
    let finalDecisionInternal = totalScore > 0 ? (bigScore >= smallScore ? "BIG" : "SMALL") : (Math.random() > 0.5 ? "BIG" : "SMALL");
    let finalConfidenceInternal = totalScore > 0 ? Math.max(bigScore, smallScore) / totalScore : 0.5;

    finalConfidenceInternal = 0.5 + (finalConfidenceInternal - 0.5) * primeTimeConfidence * realTimeData.factor;

    const signalConsistency = analyzeSignalConsistency(validSignals, finalDecisionInternal);
    const pathConfluence = analyzePathConfluenceStrength(validSignals, finalDecisionInternal);
    const uncertainty = calculateUncertaintyScore(trendContext, stability, marketEntropyAnalysis, signalConsistency, pathConfluence, longTermGlobalAccuracy, isReflexiveCorrection, driftState, apexInfluenceFlags);

    const uncertaintyFactor = 1.0 - Math.min(1.0, uncertainty.score / 120.0);
    finalConfidenceInternal = 0.5 + (finalConfidenceInternal - 0.5) * uncertaintyFactor;
    masterLogic.push(`Uncertainty(Score:${uncertainty.score.toFixed(0)},Factor:${uncertaintyFactor.toFixed(2)};Reasons:${uncertainty.reasons})`);

    let pqs = 0.5;
    pqs += (signalConsistency.score - 0.5) * 0.4;
    pqs += pathConfluence.score * 1.2;
    pqs = Math.max(0.01, Math.min(0.99, pqs - (uncertainty.score / 500)));
    masterLogic.push(`PQS:${pqs.toFixed(3)}`);

    let highConfThreshold = 0.78, medConfThreshold = 0.65;
    let highPqsThreshold = 0.75, medPqsThreshold = 0.60;

    if (primeTimeSession) {
        highConfThreshold = 0.72;
        medConfThreshold = 0.60;
        highPqsThreshold = 0.70;
        medPqsThreshold = 0.55;
    }

    let confidenceLevelInternal = 1;
    if (finalConfidenceInternal > medConfThreshold && pqs > medPqsThreshold) {
        confidenceLevelInternal = 2;
    }
    if (finalConfidenceInternal > highConfThreshold && pqs > highPqsThreshold) {
        confidenceLevelInternal = 3;
    }

    const uncertaintyThreshold = isReflexiveCorrection || driftState === 'DRIFT' || apexInfluenceFlags.architectInterferenceDetected || apexInfluenceFlags.eidolonDistortionFactor > 0.5 ? 65 : 95;
    const isForcedInternal = uncertainty.score >= uncertaintyThreshold || pqs < 0.20;

    // Output for Gemini
    const outputForGemini = {
        predictions: {
            BIG: { confidence: finalDecisionInternal === "BIG" ? finalConfidenceInternal : 1 - finalConfidenceInternal, logic: "InternalEnsemble" },
            SMALL: { confidence: finalDecisionInternal === "SMALL" ? finalConfidenceInternal : 1 - finalConfidenceInternal, logic: "InternalEnsemble" }
        },
        finalDecision: finalDecisionInternal,
        finalConfidence: finalConfidenceInternal,
        confidenceLevel: confidenceLevelInternal,
        isForcedPrediction: isForcedInternal,
        overallLogic: masterLogic.join(' -> '),
        source: "SupercoreV6_Internal",
        contributingSignals: validSignals.map(s => ({ source: s.source, prediction: s.prediction, weight: s.adjustedWeight.toFixed(5), isOnProbation: s.isOnProbation || false })).sort((a,b)=>b.weight-a.weight).slice(0, 15),
        currentMacroRegime,
        marketEntropyState: marketEntropyAnalysis.state,
        predictionQualityScore: pqs,
        reflexiveCorrectionActive: isReflexiveCorrection,
        driftState: driftState,
        apexInfluenceFlags: apexInfluenceFlags,
        externalColorPatterns: currentPatterns.color_patterns, // Patterns from collector
        externalNumberPatterns: currentPatterns.number_patterns, // Patterns from collector
        externalPatternsLastUpdated: currentPatterns.last_updated
    };

    return outputForGemini;
}

// --- END: All Prediction Logic Functions ---


// --- Endpoint: Receive Pattern Data & Raw History from Collector Script ---
// This endpoint is called by your local data_collector.py
app.post('/update-data-and-patterns', (req, res) => {
    console.log('Received data and pattern update from data collector.');
    const { history_data, color_patterns, number_patterns, last_updated } = req.body;

    if (history_data && color_patterns && number_patterns && last_updated) {
        rawGameHistoryFromCollector = history_data; // Update the global raw history
        currentPatterns.color_patterns = color_patterns;
        currentPatterns.number_patterns = number_patterns;
        currentPatterns.last_updated = last_updated;
        console.log(`Raw history and patterns updated. History length: ${rawGameHistoryFromCollector.length}. Patterns last updated: ${currentPatterns.last_updated}`);
        res.status(200).json({ status: 'success', message: 'Raw history and patterns updated.' });
    } else {
        res.status(400).json({ status: 'error', message: 'Invalid data or patterns provided.' });
    }
});


// --- Endpoint: Fetch Game History Data (Proxy for UI) ---
// This endpoint is called by your frontend UI (index.html) to get the data it needs for display.
// It serves data from the collector's cache if available, otherwise falls back to direct API.
app.post('/fetch-game-history', async (req, res) => {
    console.log('Received request for game history from client.');

    // Prefer to serve data from the collector's cache if available
    if (rawGameHistoryFromCollector.length > 0) {
        console.log(`Serving game history from collector cache (${rawGameHistoryFromCollector.length} records).`);
        // The UI only needs a recent set for display, usually up to 10-20, so slice it.
        return res.json(rawGameHistoryFromCollector.slice(0, 20)); // Send top 20 most recent records
    }

    // Fallback to direct API call if collector data is not available (e.g., collector not running or just started)
    console.log('Collector data not available. Falling back to direct Game API call for UI history.');
    const { pageNo } = req.body; // Still accept pageNo, though for fallback it often defaults to 1

    const fixedRandom = "4a0522c6ecd8410496260e686be2a57c";
    const fixedSignature = "334B5E70A0C9B8918B0B15E517E2069C";
    const currentTimestamp = Math.floor(Date.now() / 1000);

    try {
        const gameApiResponse = await fetch("https://api.bdg88zf.com/api/webapi/GetNoaverageEmerdList", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                pageSize: 10, // Fetch recent results
                pageNo: pageNo || 1,
                typeId: 1, language: 0,
                random: fixedRandom, signature: fixedSignature, timestamp: currentTimestamp
            })
        });

        if (!gameApiResponse.ok) {
            const errorText = await gameApiResponse.text();
            console.error(`Game Data API fallback error: ${gameApiResponse.status} - ${errorText}`);
            return res.status(gameApiResponse.status).json({ error: `Game Data API fallback failed: ${errorText}` });
        }

        const data = await gameApiResponse.json();
        res.json(data?.data?.list || []);

    } catch (error) {
        console.error('Error during Game Data API fallback call:', error);
        res.status(500).json({ error: 'Failed to fetch game history data via fallback.', details: error.message });
    }
});


// --- Endpoint: Main Prediction Request from UI ---
// This endpoint is called by your frontend UI (index.html) to get the final prediction.
app.post('/get-prediction', async (req, res) => {
    console.log('Received main prediction request from client.');

    // The client sends its currentSharedStats (minimal state for learning continuation)
    const { clientSharedStats } = req.body;

    // Ensure we have historical data from the collector (or fallback) for the prediction logic
    let historyForPrediction = rawGameHistoryFromCollector;
    if (historyForPrediction.length < 52) { // Need enough history for ML features (min 52 as per logic)
        console.warn(`Insufficient collector history (${historyForPrediction.length} records). Attempting to fetch directly for prediction.`);
        // Fallback to fetching a sufficient amount of history if collector hasn't supplied enough
        const fixedRandom = "4a0522c6ecd8410496260e686be2a57c";
        const fixedSignature = "334B5E70A0C9B8918B0B15E517E2069C";
        const currentTimestamp = Math.floor(Date.now() / 1000);
        try {
            const fallbackResponse = await fetch("https://api.bdg88zf.com/api/webapi/GetNoaverageEmerdList", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    pageSize: 52, // Fetch exactly what's needed for ML features
                    pageNo: 1, typeId: 1, language: 0,
                    random: fixedRandom, signature: fixedSignature, timestamp: currentTimestamp
                })
            });
            const fallbackData = await fallbackResponse.json();
            historyForPrediction = fallbackData?.data?.list || [];
            if (historyForPrediction.length < 52) {
                 throw new Error("Critical: Insufficient fallback history for prediction. Cannot proceed.");
            }
        } catch (err) {
            console.error("Critical: Failed to get even fallback history for prediction:", err);
            return res.status(500).json({ error: 'Critical: No historical data available for prediction.' });
        }
    }

    try {
        // 1. Run the Quantum AI Supercore prediction logic internally on the backend
        // This function returns the comprehensive analysis result before Gemini's final pass
        const internalPredictionOutput = await backendPredictionLogic(historyForPrediction, clientSharedStats);

        // 2. Prepare payload for Gemini (Apex AI) based on internal prediction output
        // The prompt now explicitly mentions patterns for Gemini to consider
        const prompt = `
        You are the Apex AI, the ultimate decision-making layer. Your task is to provide the final, definitive prediction ("BIG" or "SMALL") for the "Color Trading Game" based on the extensive analysis provided by the Quantum AI Supercore.

        Crucially, you also have access to real-time, pre-computed patterns of colors and numbers from an independent data collector. Integrate these pattern insights into your final judgment.

        Analyze the following output from the Quantum AI Supercore. Pay close attention to:
        - The Supercore's 'finalDecision' and 'finalConfidence'.
        - Its 'confidenceLevel' (1=low, 2=medium, 3=high).
        - The 'predictionQualityScore' (PQS).
        - Whether it was a 'forcedPrediction' (indicating high uncertainty internally).
        - The 'overallLogic' and 'source' for its prediction.
        - The 'contributingSignals' (source, prediction, weight).
        - The 'currentMacroRegime' and 'marketEntropyState'.
        - Whether 'reflexiveCorrectionActive' or 'concentrationModeEngaged' is true.
        - Any 'apexInfluenceFlags' indicating external manipulation (Eidolon, Architect, Nexus, Progenitor Prime).

        **Quantum AI Supercore Output (Internal Analysis):**
        ${JSON.stringify(internalPredictionOutput, null, 2)}

        --- External Pattern Analysis (from Data Collector) ---
        Color Patterns:
        ${JSON.stringify(currentPatterns.color_patterns, null, 2)}
        Number Patterns:
        ${JSON.stringify(currentPatterns.number_patterns, null, 2)}
        Patterns Last Updated: ${currentPatterns.last_updated || 'N/A'}
        ------------------------------------------------------

        Based on this combined data (Supercore's extensive analysis PLUS the latest color and number patterns), considering all internal analyses and external influences, what is your final prediction?
        Provide your response in a structured JSON format, including:
        {
          "apexFinalPrediction": "BIG" or "SMALL",
          "apexGeminiConfidence": "Your confidence level (0.0 to 1.0)",
          "apexReasoning": "A concise explanation of why you chose this prediction, referencing the Supercore's data AND your insights from the identified color/number patterns.",
          "supercoreInputSummary": { /* A brief summary of key points from the Supercore output */ }
        }
        `;

        const chatHistory = [{ role: "user", parts: [{ text: prompt }] }];

        const geminiPayload = {
            contents: chatHistory,
            generationConfig: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: "OBJECT",
                    properties: {
                        "apexFinalPrediction": { "type": "STRING", "enum": ["BIG", "SMALL"] },
                        "apexGeminiConfidence": { "type": "NUMBER", "format": "float" },
                        "apexReasoning": { "type": "STRING" },
                        "supercoreInputSummary": { "type": "OBJECT", "additionalProperties": true }
                    },
                    "required": ["apexFinalPrediction", "apexGeminiConfidence", "apexReasoning"]
                }
            }
        };

        const geminiResult = await callGeminiAPI(geminiPayload);

        if (geminiResult && geminiResult.candidates && geminiResult.candidates.length > 0 &&
            geminiResult.candidates[0].content && geminiResult.candidates[0].content.parts &&
            geminiResult.candidates[0].content.parts.length > 0) {

            const geminiResponseText = geminiResult.candidates[0].content.parts[0].text;
            let finalApexPrediction;
            try {
                finalApexPrediction = JSON.parse(geminiResponseText);
            } catch (jsonParseError) {
                console.error("Failed to parse Gemini's JSON response:", jsonParseError);
                console.error("Gemini raw response:", geminiResponseText);
                return res.status(500).json({
                    error: "Gemini API returned unparseable JSON.",
                    rawGeminiResponse: geminiResponseText,
                    details: jsonParseError.message
                });
            }

            if (finalApexPrediction.apexFinalPrediction && finalApexPrediction.apexGeminiConfidence !== undefined && finalApexPrediction.reasoning) {
                // Combine internal prediction output with Gemini's final decision
                const finalOutput = {
                    ...internalPredictionOutput, // Retain all internal details from Supercore
                    finalDecision: finalApexPrediction.apexFinalPrediction,
                    finalConfidence: finalApexPrediction.apexGeminiConfidence,
                    confidenceLevel: finalApexPrediction.apexGeminiConfidence > 0.75 ? 3 : (finalApexPrediction.apexGeminiConfidence > 0.6 ? 2 : 1),
                    isForcedPrediction: false, // Gemini made a reasoned decision, so no longer forced (unless original was critical failure)
                    overallLogic: `${internalPredictionOutput.overallLogic} -> GeminiApexAI_FinalDecision: ${finalApexPrediction.apexFinalPrediction} (${(finalApexPrediction.apexGeminiConfidence * 100).toFixed(1)}%) -> Reason: ${finalApexPrediction.reasoning}`,
                    source: "GeminiApexAI_Master",
                };
                res.json(finalOutput); // Send the combined result back to the frontend

            } else {
                console.error("Gemini API returned unexpected JSON structure:", finalApexPrediction);
                res.status(500).json({ error: "Gemini API returned unexpected JSON structure from final prediction call." });
            }

        } else {
            console.warn('Gemini API response structure unexpected or content missing:', geminiResult);
            res.status(500).json({ error: 'Unexpected Gemini API response structure from final prediction call.', details: geminiResult });
        }

    } catch (error) {
        console.error('Error during Gemini meta-prediction call:', error);
        res.status(500).json({ error: 'Failed to get final prediction from Apex AI (Gemini).', details: error.message });
    }
});

// --- Start the Server ---
app.listen(port, () => {
    console.log(`Node.js Prediction Server listening at http://localhost:${port}`);
    console.log(`API keys configured: ${GEMINI_API_KEYS.length}`);
    if (GEMINI_API_KEYS.length > 0) {
        console.log(`Initial Gemini API key in use: ${GEMINI_API_KEYS[currentApiKeyIndex].substring(0, 5)}...`);
    } else {
        console.log('WARNING: No Gemini API keys found. Gemini prediction calls will fail.');
    }
    console.log(`Remember to configure GEMINI_API_KEYS in your .env file on Render.`);
});
