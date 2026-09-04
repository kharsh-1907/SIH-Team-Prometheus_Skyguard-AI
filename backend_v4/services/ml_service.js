const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://127.0.0.1:8000";

const predictAnomaly = async (weatherData) => {
    try {
        const response = await fetch(`${ML_SERVICE_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                temperature: Number(weatherData.temperature),
                pressure: Number(weatherData.pressure),
                humidity: Number(weatherData.humidity),
                stationId: weatherData.stationId || weatherData.station_id || "AWS-24567",
                timestamp: weatherData.timestamp || new Date().toISOString()
            }),
            signal: AbortSignal.timeout(4000)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || "ML prediction failed");
        }

        return data;

    } catch (error) {
        console.error("ML Service Error:", error.message);
        throw error;
    }
};

const checkMLHealth = async () => {
    try {
        const response = await fetch(`${ML_SERVICE_URL}/health`, {
            signal: AbortSignal.timeout(4000)
        });

        if (!response.ok) {
            return false;
        }

        return true;

    } catch (error) {
        return false;
    }
};

module.exports = {
    predictAnomaly,
    checkMLHealth
};