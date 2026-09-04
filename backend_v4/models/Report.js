const mongoose = require("mongoose");

const reportSchema = new mongoose.Schema(
  {
    title: {
      type: String,
      required: true,
    },

    description: {
      type: String,
      required: true,
    },

    totalReadings: {
      type: Number,
      default: 0,
    },

    normalReadings: {
      type: Number,
      default: 0,
    },

    anomalyReadings: {
      type: Number,
      default: 0,
    },

    anomalyRate: {
      type: Number,
      default: 0,
    },

    generatedAt: {
      type: Date,
      default: Date.now,
    },
  },
  {
    timestamps: true,
  }
);

module.exports = mongoose.model("Report", reportSchema);