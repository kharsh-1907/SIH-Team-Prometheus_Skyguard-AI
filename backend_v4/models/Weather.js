const mongoose = require("mongoose");

const weatherSchema = new mongoose.Schema(
  {
    stationId: {
      type: String,
      default: "AWS-24567",
    },

    location: {
      type: String,
      default: "Pune, Maharashtra",
    },

    temperature: {
      type: Number,
      required: true,
    },

    humidity: {
      type: Number,
      required: true,
    },

    pressure: {
      type: Number,
      required: true,
    },

    windSpeed: {
      type: Number,
      default: 0,
    },

    rainfall: {
      type: Number,
      default: 0,
    },

    anomalyScore: {
      type: Number,
      default: 0,
    },

    status: {
      type: String,
      enum: ["Normal", "Low", "Medium", "High", "Unknown"],
      default: "Normal",
    },

    isAnomaly: {
      type: Boolean,
      default: false,
    },

    anomalyType: {
      type: String,
      default: "Normal",
    },

    affectedSensor: {
      type: String,
      default: "none",
    },

    tier: {
      type: String,
      default: "None",
    },
    sensorHealth: {
      temperature: {
        type: Number,
        default: 100,
      },
      humidity: {
        type: Number,
        default: 100,
      },
      pressure: {
        type: Number,
        default: 100,
      },
    },

    timestamp: {
      type: Date,
      default: Date.now,
    }
  },
  {
    timestamps: true,
  }
);

module.exports = mongoose.model("Weather", weatherSchema);