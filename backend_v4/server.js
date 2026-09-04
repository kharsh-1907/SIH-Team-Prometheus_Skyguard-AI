const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");
const weatherRoutes = require("./routes/weatherRoutes");
const alertRoutes = require("./routes/alertRoutes");
const reportRoutes = require("./routes/reportRoutes");
const dashboardRoutes = require("./routes/dashboardRoutes");
const mlRoutes = require("./routes/mlRoutes");


dotenv.config();

const connectDB = require("./config/db");

const app = express();

// Connect Database
connectDB();

// Middleware
app.use(cors());
app.use(express.json());

//weather api calling
app.use("/api/weather", weatherRoutes);

//Alert api calling
app.use("/api/alerts", alertRoutes);

// Report API calling
app.use("/api/reports", reportRoutes);

// Dashboard API calling
app.use("/api/dashboard", dashboardRoutes);

// Ml Route calling
app.use("/api/ml", mlRoutes);

// Test Route
app.get("/", (req, res) => {
    res.send("🚀 SkyGuard AI Backend is Running...");
});

// Server
const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
    console.log(`✅ Server running on http://localhost:${PORT}`);
});