const express = require("express");

const router = express.Router();

const {
    addWeather,
    getLatestWeather,
    getWeatherHistory
} = require("../controllers/weatherController");

router.post("/", addWeather);

router.get("/latest", getLatestWeather);

router.get("/history", getWeatherHistory);

module.exports = router;