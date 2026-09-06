const Weather = require("../models/Weather");
const { predictAnomaly } = require("../services/ml_service");
const Alert = require("../models/Alert");

// Add Weather Data
exports.addWeather = async (req, res) => {
    try {
        const {
            temperature,
            pressure,
            humidity,
            sensorHealth,
            location,
            stationId,
            timestamp,
            windSpeed,
            rainfall
        } = req.body;

        // 1. Input Validation: Check presence, non-empty, and finite numeric validity
        const isInvalidNumber = (val) => {
            if (val === undefined || val === null) return true;
            if (typeof val === "string" && val.trim() === "") return true;
            const num = Number(val);
            return isNaN(num) || !isFinite(num);
        };

        if (isInvalidNumber(temperature) || isInvalidNumber(pressure) || isInvalidNumber(humidity)) {
            return res.status(400).json({
                success: false,
                message: "Valid numeric temperature, pressure, and humidity are required"
            });
        }

        const numTemp = Number(temperature);
        const numPress = Number(pressure);
        const numHum = Number(humidity);
        const activeStationId = stationId || req.body.station_id || "AWS-24567";
        const readingTimestamp = timestamp ? new Date(timestamp) : new Date();

        // 2. ML Prediction with Graceful Offline Fallback
        let mlResult;
        let mlServiceOnline = true;

        try {
            mlResult = await predictAnomaly({
                temperature: numTemp,
                pressure: numPress,
                humidity: numHum,
                stationId: activeStationId,
                timestamp: readingTimestamp.toISOString()
            });
        } catch (mlError) {
            console.warn("⚠️ ML Service offline or unreachable; using local Tier 1 fallback:", mlError.message);
            mlServiceOnline = false;

            // Local deterministic fallback
            const isOob = (numTemp < -40.0 || numTemp > 55.0 || numPress < 800.0 || numPress > 1100.0 || numHum < 0.0 || numHum > 100.0);
            const oobSensor = (numTemp < -40.0 || numTemp > 55.0) ? "temperature" : (
                (numPress < 800.0 || numPress > 1100.0) ? "pressure" : (
                    (numHum < 0.0 || numHum > 100.0) ? "humidity" : "none"
                )
            );

            mlResult = {
                status: isOob ? "Anomaly" : "Normal",
                anomaly_score: isOob ? 1.0 : 0.0,
                severity: isOob ? "High" : "Normal",
                anomaly_type: isOob ? "OutOfBounds" : "Normal",
                affected_sensor: oobSensor,
                tier: isOob ? "Tier 1 (Local Fallback)" : "None (ML Offline)",
                sensor_health: {
                    temperature: (numTemp < -40.0 || numTemp > 55.0) ? 50 : 100,
                    pressure: (numPress < 800.0 || numPress > 1100.0) ? 50 : 100,
                    humidity: (numHum < 0.0 || numHum > 100.0) ? 50 : 100
                },
                data: {
                    temperature: numTemp,
                    pressure: numPress,
                    humidity: numHum,
                    prediction: isOob ? -1 : 1,
                    anomaly: isOob,
                    label: isOob ? "Anomaly" : "Normal",
                    severity: isOob ? "High" : "Normal",
                    anomalyScore: isOob ? 1.0 : 0.0,
                    threshold: 0.615997,
                    model: "Local Fallback Guardrail"
                }
            };
        }

        const isAnomaly = mlResult.status === "Anomaly" || Boolean(mlResult.data?.anomaly);
        const anomalyType = mlResult.anomaly_type || mlResult.data?.anomalyType || "Normal";
        const affectedSensor = mlResult.affected_sensor || mlResult.data?.affectedSensor || "none";
        const severity = mlResult.severity || mlResult.data?.severity || "Normal";
        const anomalyScore = mlResult.anomaly_score ?? mlResult.data?.anomalyScore ?? 0.0;
        const tier = mlResult.tier || mlResult.data?.tier || "None";

        // 3. Persist Weather reading with full Phase 2 & 3 metadata
        const weather = await Weather.create({
            stationId: activeStationId,
            location: location || "Pune, Maharashtra",
            temperature: numTemp,
            pressure: numPress,
            humidity: numHum,
            windSpeed: windSpeed !== undefined ? Number(windSpeed) : 0,
            rainfall: rainfall !== undefined ? Number(rainfall) : 0,
            sensorHealth: sensorHealth || mlResult.sensor_health || mlResult.data?.sensorHealth || {
                temperature: 100,
                humidity: 100,
                pressure: 100
            },
            anomalyScore: Number(anomalyScore),
            status: severity,
            isAnomaly: isAnomaly,
            anomalyType: anomalyType,
            affectedSensor: affectedSensor,
            tier: tier,
            timestamp: readingTimestamp
        });

        // 4. Create Alert if anomalous
        let createdAlert = null;
        if (isAnomaly) {
            const validSeverities = ["Low", "Medium", "High", "Critical"];
            const alertSeverity = validSeverities.includes(severity) ? severity : "Medium";
            createdAlert = await Alert.create({
                stationId: activeStationId,
                type: `${anomalyType} Anomaly`,
                message: `${severity} severity ${anomalyType} detected on ${affectedSensor}`,
                severity: alertSeverity,
                status: "Active",
                anomalyType: anomalyType,
                affectedSensor: affectedSensor,
                anomalyScore: Number(anomalyScore)
            });
        }

        res.status(201).json({
            success: true,
            message: "Weather data added successfully",
            mlServiceOnline,
            data: {
                weather,
                mlPrediction: mlResult,
                alert: createdAlert
            }
        });

    } catch (error) {
        console.error("Weather Controller Error:", error.message);

        res.status(500).json({
            success: false,
            message: error.message
        });
    }
};

// Get Latest Weather
exports.getLatestWeather = async (req, res) => {
    try {
        const query = req.query.stationId ? { stationId: req.query.stationId } : {};
        const weather = await Weather.findOne(query).sort({ timestamp: -1, createdAt: -1 });

        res.status(200).json({
            success: true,
            data: weather
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            message: error.message
        });
    }
};

// Get Weather History
exports.getWeatherHistory = async (req, res) => {
    try {
        const query = req.query.stationId ? { stationId: req.query.stationId } : {};
        const limit = req.query.limit ? parseInt(req.query.limit) : 100;
        const weather = await Weather.find(query).sort({ timestamp: -1, createdAt: -1 }).limit(limit);

        res.status(200).json({
            success: true,
            count: weather.length,
            data: weather
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            message: error.message
        });
    }
};