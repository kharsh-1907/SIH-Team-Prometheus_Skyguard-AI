const mongoose = require("mongoose");

const alertSchema = new mongoose.Schema(
  {
    stationId: {
      type: String,
      default: "AWS-24567",
    },

    anomalyType: {
      type: String,
      default: "Anomaly",
    },

    affectedSensor: {
      type: String,
      default: "none",
    },

    anomalyScore: {
      type: Number,
      default: 0,
    },

    type: {
      type: String,
      required: true,
    },

    message: {
      type: String,
      required: true,
    },

    severity: {
      type: String,
      enum: ["Low", "Medium", "High", "Critical"],
      required: true,
    },

    status: {
      type: String,
      enum: ["Active", "Resolved"],
      default: "Active",
    },
  },
  {
    timestamps: true,
  }
);

module.exports = mongoose.model("Alert", alertSchema);