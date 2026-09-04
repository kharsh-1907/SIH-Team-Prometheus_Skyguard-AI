const express = require("express");
const router = express.Router();

const {
  getAlerts,
  createAlert,
  updateAlert,
} = require("../controllers/alertController");

// GET all alerts
router.get("/", getAlerts);

// POST a new alert
router.post("/", createAlert);

// UPDATE an alert
router.put("/:id", updateAlert);

module.exports = router;