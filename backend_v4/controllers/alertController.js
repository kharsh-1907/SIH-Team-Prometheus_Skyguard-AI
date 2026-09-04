const Alert = require("../models/Alert");

// Get all alerts
const getAlerts = async (req, res) => {
  try {
    const query = req.query.stationId ? { stationId: req.query.stationId } : {};
    const alerts = await Alert.find(query).sort({ createdAt: -1 });

    res.status(200).json(alerts);
  } catch (error) {
    res.status(500).json({
      message: "Failed to fetch alerts",
      error: error.message,
    });
  }
};

// Create a new alert
const createAlert = async (req, res) => {
  try {
    const { type, message, severity, status, stationId, anomalyType, affectedSensor, anomalyScore } = req.body;

    const alert = await Alert.create({
      type,
      message,
      severity,
      status: status || "Active",
      stationId: stationId || "AWS-24567",
      anomalyType: anomalyType || "Anomaly",
      affectedSensor: affectedSensor || "none",
      anomalyScore: anomalyScore !== undefined ? Number(anomalyScore) : 0,
    });

    res.status(201).json(alert);
  } catch (error) {
    res.status(500).json({
      message: "Failed to create alert",
      error: error.message,
    });
  }
};

// Update alert status
const updateAlert = async (req, res) => {
  try {
    const alert = await Alert.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true }
    );

    if (!alert) {
      return res.status(404).json({
        message: "Alert not found",
      });
    }

    res.status(200).json(alert);
  } catch (error) {
    res.status(500).json({
      message: "Failed to update alert",
      error: error.message,
    });
  }
};

module.exports = {
  getAlerts,
  createAlert,
  updateAlert,
};