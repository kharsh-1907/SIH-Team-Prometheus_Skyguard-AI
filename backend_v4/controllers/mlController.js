const { predictAnomaly } = require("../services/ml_service");

exports.predict = async (req, res) => {
    try {
        const { temperature, pressure, humidity, stationId, timestamp } = req.body;

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

        const result = await predictAnomaly({
            temperature: Number(temperature),
            pressure: Number(pressure),
            humidity: Number(humidity),
            stationId: stationId || req.body.station_id || "AWS-24567",
            timestamp: timestamp || new Date().toISOString()
        });

        res.status(200).json(result);

    } catch (error) {
        console.error("ML Controller Error:", error.message);

        res.status(503).json({
            success: false,
            message: "ML microservice is offline or unreachable",
            error: error.message
        });
    }
};