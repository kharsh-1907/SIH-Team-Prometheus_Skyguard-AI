const Report = require("../models/Report");

// Get all reports
const getReports = async (req, res) => {
  try {
    const reports = await Report.find().sort({ createdAt: -1 });

    res.status(200).json(reports);
  } catch (error) {
    res.status(500).json({
      message: "Failed to fetch reports",
      error: error.message,
    });
  }
};

// Create a new report
const createReport = async (req, res) => {
  try {
    const {
      title,
      description,
      totalReadings,
      normalReadings,
      anomalyReadings,
      anomalyRate,
    } = req.body;

    const report = await Report.create({
      title,
      description,
      totalReadings,
      normalReadings,
      anomalyReadings,
      anomalyRate,
    });

    res.status(201).json(report);
  } catch (error) {
    res.status(500).json({
      message: "Failed to create report",
      error: error.message,
    });
  }
};

// Delete a report
const deleteReport = async (req, res) => {
  try {
    const report = await Report.findByIdAndDelete(req.params.id);

    if (!report) {
      return res.status(404).json({
        message: "Report not found",
      });
    }

    res.status(200).json({
      message: "Report deleted successfully",
      report,
    });
  } catch (error) {
    res.status(500).json({
      message: "Failed to delete report",
      error: error.message,
    });
  }
};

module.exports = {
  getReports,
  createReport,
  deleteReport,
};