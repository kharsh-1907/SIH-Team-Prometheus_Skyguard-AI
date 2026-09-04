const Weather = require("../models/Weather");
const Alert = require("../models/Alert");
const Report = require("../models/Report");

const getDashboard = async (req, res) => {
  try {
    const stationFilter = req.query.stationId ? { stationId: req.query.stationId } : {};
    const totalReadings = await Weather.countDocuments(stationFilter);
    const normalReadings = await Weather.countDocuments({ ...stationFilter, status: "Normal" });
    const anomalyReadings = await Weather.countDocuments({ ...stationFilter, status: { $ne: "Normal" } });
    
    const alertFilter = req.query.stationId ? { stationId: req.query.stationId } : {};
    const activeAlerts = await Alert.countDocuments({ ...alertFilter, status: "Active" });
    const resolvedAlerts = await Alert.countDocuments({ ...alertFilter, status: "Resolved" });
    const totalReports = await Report.countDocuments();

    const latestWeather = await Weather.findOne(stationFilter).sort({ timestamp: -1, createdAt: -1 });
    const recentWeather = await Weather.find(stationFilter)
      .sort({ timestamp: -1, createdAt: -1 })
      .limit(100)
      .select("temperature humidity pressure timestamp anomalyScore status anomalyType affectedSensor isAnomaly stationId sensorHealth");
    
    const anomalyRate = totalReadings > 0 ? ((anomalyReadings / totalReadings) * 100).toFixed(2) : "0.00";
    const stations = await Weather.distinct("stationId");

    res.status(200).json({
      totalReadings,
      normalReadings,
      anomalyReadings,
      anomalyRate: Number(anomalyRate),
      activeAlerts,
      resolvedAlerts,
      totalReports,
      latestWeather,
      recentWeather,
      stations: stations.length > 0 ? stations : ["AWS-24567"]
    });
  } catch (error) {
    res.status(500).json({
      message: "Failed to fetch dashboard data",
      error: error.message
    });
  }
};

module.exports = {
  getDashboard
};