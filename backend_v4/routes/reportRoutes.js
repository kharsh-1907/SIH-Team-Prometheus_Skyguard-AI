const express = require("express");
const router = express.Router();

const {
  getReports,
  createReport,
  deleteReport,
} = require("../controllers/reportController");

// Get all reports
router.get("/", getReports);

// Create a new report
router.post("/", createReport);

// Delete a report
router.delete("/:id", deleteReport);

module.exports = router;