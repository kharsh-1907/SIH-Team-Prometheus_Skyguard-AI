import React, { useState, useEffect } from "react";
import { Activity, AlertTriangle, BarChart3, Bell, CloudSun, Droplets, Gauge, Menu, Radio, Settings, ShieldCheck, Thermometer, X } from "lucide-react";

const nav = [["Dashboard", Activity], ["Live Monitor", Radio], ["Alerts", Bell], ["Analytics", BarChart3], ["Sensor Health", ShieldCheck], ["Settings", Settings]];
function Metric({ Icon, title, value, unit, status, subtitle, tone }) {
  const isWarn = status === "Warning" || status === "High" || status === "Medium" || status === "Critical";
  return <div className="metric"><div className={"metricIcon " + tone}><Icon size={19} /></div><div>
    <p>{title}</p><h3>{value}<small>{unit}</small></h3>
    <span className={"status " + (isWarn ? "warning" : "")}><i />{status}</span>
    {subtitle && <small style={{ display: "block", fontSize: "0.72rem", opacity: 0.85, marginTop: "3px" }}>{subtitle}</small>}
  </div></div>
}
function Health({ name, value, tone }) { return <div className="healthItem"><div><span>{name}</span><b>{value}%</b></div><section><i className={tone} style={{ width: value + "%" }} /></section></div> }
function getSensorStatus(weather, sensor) {
  if (!weather) return "---";

  const affectedSensor = weather.affectedSensor?.toLowerCase();

  if (
    weather.status &&
    weather.status !== "Normal" &&
    affectedSensor === sensor
  ) {
    return weather.status;
  }

  return "Normal";
}


function Chart({ readings }) {
  const data = [...(readings || [])].reverse();

  const [hoverIndex, setHoverIndex] = useState(null);
  const [lockedIndex, setLockedIndex] = useState(null);

  const activeIndex = lockedIndex ?? hoverIndex;

  const getPoints = (key) => {
    if (!data.length) return "";

    const min = Math.min(...data.map(d => d[key]));
    const max = Math.max(...data.map(d => d[key]));
    const range = max - min || 1;

    return data.map((d, i) => {
      const x = data.length === 1
        ? 475
        : 35 + i * (880 / (data.length - 1));

      const y = 220 - (((d[key] - min) / range) * 180);

      return `${x},${y}`;
    }).join(" ");
  };

  const getX = (index) => {
    if (data.length === 1) return 475;
    return 35 + index * (880 / (data.length - 1));
  };

  const getY = (key, value) => {
    const min = Math.min(...data.map(d => d[key]));
    const max = Math.max(...data.map(d => d[key]));
    const range = max - min || 1;

    return 220 - (((value - min) / range) * 180);
  };

  const handleMouseMove = (e) => {
    if (lockedIndex !== null || !data.length) return;

    const svg = e.currentTarget;
    const rect = svg.getBoundingClientRect();

    const mouseX = ((e.clientX - rect.left) / rect.width) * 950;

    const index = Math.round(
      ((mouseX - 35) / 880) * (data.length - 1)
    );

    setHoverIndex(
      Math.max(0, Math.min(data.length - 1, index))
    );
  };

  const handleClick = () => {
    if (hoverIndex !== null) {
      setLockedIndex(
        lockedIndex === null ? hoverIndex : null
      );
    }
  };

  // Show only a few timestamp labels while keeping ALL chart data points.
  const labelCount = Math.min(7, data.length);

  const labelIndices = Array.from(
    { length: labelCount },
    (_, i) =>
      labelCount === 1
        ? 0
        : Math.round((i * (data.length - 1)) / (labelCount - 1))
  );

  return (
    <div className="chart">

      <svg
        viewBox="0 0 950 245"
        preserveAspectRatio="none"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => {
          if (lockedIndex === null) {
            setHoverIndex(null);
          }
        }}
        onClick={handleClick}
      >

        {[20, 65, 110, 155, 200].map((y) => (
          <line
            key={y}
            x1="35"
            x2="915"
            y1={y}
            y2={y}
            className="grid"
          />
        ))}

        <polyline
          points={getPoints("temperature")}
          className="line temp"
        />

        <polyline
          points={getPoints("humidity")}
          className="line hum"
        />

        <polyline
          points={getPoints("pressure")}
          className="line press"
        />

        {activeIndex !== null && data[activeIndex] && (
          <>
            <line
              x1={getX(activeIndex)}
              x2={getX(activeIndex)}
              y1="20"
              y2="220"
              className="crosshair"
            />

            <circle
              cx={getX(activeIndex)}
              cy={getY("temperature", data[activeIndex].temperature)}
              r="5"
              className="crossPoint"
            />

            <circle
              cx={getX(activeIndex)}
              cy={getY("humidity", data[activeIndex].humidity)}
              r="5"
              className="crossPoint"
            />

            <circle
              cx={getX(activeIndex)}
              cy={getY("pressure", data[activeIndex].pressure)}
              r="5"
              className="crossPoint"
            />

            <foreignObject
              x={Math.min(getX(activeIndex) + 12, 730)}
              y="25"
              width="205"
              height="115"
            >
              <div className="chartTooltip">

                <strong>
                  {new Date(data[activeIndex].timestamp).toLocaleString("en-IN", {
                    day: "2-digit",
                    month: "short",
                    year: "numeric",
                    hour: "2-digit",
                    minute: "2-digit",
                    hour12: false
                  })}
                </strong>

                <span>
                  <i className="green" />
                  Temperature: {data[activeIndex].temperature} °C
                </span>

                <span>
                  <i className="blue" />
                  Humidity: {data[activeIndex].humidity}%
                </span>

                <span>
                  <i className="orange" />
                  Pressure: {data[activeIndex].pressure} hPa
                </span>

              </div>
            </foreignObject>
          </>
        )}

      </svg>

      <div className="times">
        {labelIndices.map((index) => {
          const d = data[index];

          return (
            <span key={d.timestamp}>
              {new Date(d.timestamp).toLocaleTimeString("en-IN", {
                hour: "2-digit",
                minute: "2-digit",
                hour12: false
              })}
            </span>
          );
        })}
      </div>

    </div>
  );
}
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:5000";

export default function App() {
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState("Dashboard");
  const [dashboard, setDashboard] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [chartRange, setChartRange] = useState("24h");
  const [selectedStation, setSelectedStation] = useState("");
  const [connected, setConnected] = useState(true);

  // Fetching Backend Data
  useEffect(() => {
    const fetchDashboardData = () => {
      const dashUrl = selectedStation
        ? `${API_BASE}/api/dashboard?stationId=${encodeURIComponent(selectedStation)}`
        : `${API_BASE}/api/dashboard`;

      const alertsUrl = selectedStation
        ? `${API_BASE}/api/alerts?stationId=${encodeURIComponent(selectedStation)}`
        : `${API_BASE}/api/alerts`;

      Promise.all([
        fetch(dashUrl).then(r => r.json()),
        fetch(alertsUrl).then(r => r.json())
      ]).then(([dashData, alertsData]) => {
        setDashboard(dashData);
        setAlerts(alertsData);
        setConnected(true);
      }).catch((error) => {
        console.warn("Backend connectivity interrupted:", error.message);
        setConnected(false);
      });
    };

    fetchDashboardData();
    const refresh = setInterval(fetchDashboardData, 5000);
    return () => clearInterval(refresh);
  }, [selectedStation]);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return <div className="app">
    <aside className={open ? "sidebar open" : "sidebar"}>
      <div className="brand"><div className="brandIcon"><CloudSun size={21} /></div><div><h2>SkyGuard <span>AI</span></h2><p>Weather Monitoring</p></div><button className="close" onClick={() => setOpen(false)}><X size={18} /></button></div>
      <nav>{nav.map(([name, Icon]) => <button className={page === name ? "active" : ""} key={name} onClick={() => { setPage(name); setOpen(false) }}><Icon size={16} />{name}</button>)}</nav>
      <div className="sidebarLive" style={{ color: connected ? "#27a65c" : "#d85d5d" }}>
        <i style={{ background: connected ? "#2ab768" : "#d85d5d" }} /> {connected ? "System Live" : "Reconnecting..."}
      </div>
    </aside>
    <main>
      <header>
        <button className="menu" onClick={() => setOpen(true)}><Menu size={19} /></button>
        <div><h1>{page}</h1><p>Weather station monitoring & anomaly detection</p></div>
        <div className="right">
          <span className="live" style={{ color: connected ? "#249b57" : "#d85d5d", background: connected ? "#f3fbf6" : "#fff0f0", borderColor: connected ? "#cde8d7" : "#fbd1d1" }}>
            <i style={{ background: connected ? "#2ab768" : "#d85d5d" }} /> {connected ? "LIVE" : "OFFLINE"}
          </span>
          <span>{currentTime.toLocaleString("en-IN", { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: true })}</span>
          {dashboard?.stations && dashboard.stations.length > 0 && (
            <select
              value={selectedStation}
              onChange={(e) => setSelectedStation(e.target.value)}
              style={{
                border: "1px solid #e0e6ea",
                background: "#fff",
                color: "#1677c3",
                fontWeight: 600,
                borderRadius: "6px",
                padding: "4px 8px",
                fontSize: "9px",
                cursor: "pointer"
              }}
            >
              <option value="">All Stations ({dashboard.stations.length})</option>
              {dashboard.stations.map((st) => (
                <option key={st} value={st}>{st}</option>
              ))}
            </select>
          )}
          <span>{selectedStation || dashboard?.latestWeather?.stationId || "AWS-24567"} · {dashboard?.latestWeather?.location ?? "Pune"}</span>
        </div>
      </header>
      <div className="content">
        {page === "Dashboard" && (
          <>
            <div className="title">
              <div>
                <label>STATION {dashboard?.latestWeather?.stationId ? `· ${dashboard.latestWeather.stationId}` : ""}</label>
                <h2>Current readings</h2>
              </div>
              <span>
                {dashboard?.latestWeather?.timestamp
                  ? new Date(dashboard.latestWeather.timestamp).toLocaleString("en-IN", {
                    day: "2-digit",
                    month: "short",
                    year: "numeric",
                    hour: "2-digit",
                    minute: "2-digit",
                    hour12: true
                  })
                  : "---"}
              </span>
            </div>

            <section className="metrics">
              <Metric Icon={Thermometer} title="Temperature" value={dashboard?.latestWeather?.temperature ?? "---"} unit=" °C" status={getSensorStatus(dashboard?.latestWeather, "temperature")} tone="green" />

              <Metric Icon={Droplets} title="Humidity" value={dashboard?.latestWeather?.humidity ?? "---"} unit=" %" status={getSensorStatus(dashboard?.latestWeather, "humidity")} tone="blue" />

              <Metric Icon={Gauge} title="Pressure" value={dashboard?.latestWeather?.pressure ?? "---"} unit=" hPa" status={getSensorStatus(dashboard?.latestWeather, "pressure")} tone="orange" />

              <Metric
                Icon={AlertTriangle}
                title="Anomaly Score"
                value={dashboard?.latestWeather?.anomalyScore !== undefined ? dashboard.latestWeather.anomalyScore.toFixed(3) : "---"}
                unit=""
                status={dashboard?.latestWeather?.status ?? "---"}
                subtitle={dashboard?.latestWeather?.anomalyType && dashboard.latestWeather.anomalyType !== "Normal" ? `${dashboard.latestWeather.anomalyType} · ${dashboard.latestWeather.affectedSensor}` : "All sensors nominal"}
                tone="purple"
              />
            </section>

            <section className="mainGrid">
              <div className="card chartCard">
                <div className="head">
                  <div>
                    <h3>Sensor readings</h3>
                    <p>Temperature, pressure and humidity</p>
                  </div>

                  <select value={chartRange} onChange={(e) => setChartRange(e.target.value)}>
                    <option value="24h">Last 24 hours</option>
                    <option value="7d">Last 7 days</option>
                  </select>
                </div>

                <div className="legend">
                  <span><i className="green" />Temperature</span>
                  <span><i className="blue" />Humidity</span>
                  <span><i className="orange" />Pressure</span>
                </div>

                <Chart readings={chartRange === "24h" ? (dashboard?.recentWeather || []).filter(r => Date.now() - new Date(r.timestamp).getTime() <= 24 * 60 * 60 * 1000) : (dashboard?.recentWeather || []).filter(r => Date.now() - new Date(r.timestamp).getTime() <= 7 * 24 * 60 * 60 * 1000)} />
              </div>

              <div className="card alerts">
                <div className="head">
                  <div>
                    <h3>Active alerts</h3>
                    <p>Current anomaly detections</p>
                  </div>

                  <b>{alerts.filter((alert) => alert.status === "Active").length}</b>
                </div>

                {alerts.filter((alert) => alert.status === "Active").slice(0, 3).map((alert) => (
                  <div className={`alert ${alert.severity === "High" || alert.severity === "Critical" ? "high" : "medium"}`} key={alert._id}>
                    <div>
                      <AlertTriangle size={17} />
                    </div>

                    <section>
                      <strong>{alert.type}</strong>
                      <p>{alert.message}</p>
                      <small>
                        {alert.severity} severity
                        {alert.stationId ? ` · ${alert.stationId}` : ""}
                        {alert.affectedSensor && alert.affectedSensor !== "none" ? ` · ${alert.affectedSensor}` : ""}
                        {alert.anomalyScore !== undefined ? ` · Score: ${Number(alert.anomalyScore).toFixed(3)}` : ""}
                        {alert.createdAt ? ` · ${new Date(alert.createdAt).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })}` : ""}
                      </small>
                    </section>
                  </div>
                ))}
              </div>
            </section>

            <section className="bottomGrid">
              <div className="card">
                <div className="head">
                  <div>
                    <h3>Sensor health</h3>
                    <p>Current sensor reliability</p>
                  </div>

                  <ShieldCheck size={17} />
                </div>

                <div className="health">
                  <Health name="Temperature" value={dashboard?.latestWeather?.sensorHealth?.temperature ?? 100} tone="g" />
                  <Health name="Humidity" value={dashboard?.latestWeather?.sensorHealth?.humidity ?? 100} tone="b" />
                  <Health name="Pressure" value={dashboard?.latestWeather?.sensorHealth?.pressure ?? 100} tone="o" />
                </div>
              </div>

              <div className="card">
                <div className="head">
                  <div>
                    <h3>Anomaly summary</h3>
                    <p>ML detection will appear here</p>
                  </div>

                  <BarChart3 size={17} />
                </div>

                <div className="summary">
                  <div>
                    <strong>{dashboard?.totalReadings ?? 0}</strong>
                    <span>Total readings</span>
                  </div>

                  <div onClick={() => setPage("Alerts")} style={{ cursor: "pointer" }}>
                    <strong className="red">{dashboard?.anomalyReadings ?? 0}</strong>
                    <span>Anomalies</span>
                  </div>

                  <div>
                    <strong className="greenText">
                      {dashboard?.totalReadings
                        ? ((dashboard.normalReadings / dashboard.totalReadings) * 100).toFixed(2)
                        : "0.00"}%
                    </strong>
                    <span>Normal</span>
                  </div>

                  <div>
                    <strong className="orangeText">{dashboard?.anomalyRate ?? 0}%</strong>
                    <span>Anomaly rate</span>
                  </div>
                </div>
              </div>
            </section>
          </>
        )}

        {page === "Alerts" && (
          <div className="card">
            <div className="head">
              <div>
                <h3>Alerts</h3>
                <p>Weather anomaly detections</p>
              </div>

              <b>{alerts.filter((alert) => alert.status === "Active").length}</b>
            </div>

            {alerts.length === 0 ? (
              <p style={{ padding: "16px", color: "#89969e" }}>No alerts found.</p>
            ) : (
              alerts.map((alert) => (
                <div className={`alert ${alert.severity === "High" || alert.severity === "Critical" ? "high" : "medium"}`} key={alert._id}>
                  <div>
                    <AlertTriangle size={17} />
                  </div>

                  <section>
                    <strong>{alert.type}</strong>
                    <p>{alert.message}</p>
                    <small>
                      {alert.severity} severity · {alert.status}
                      {alert.stationId ? ` · ${alert.stationId}` : ""}
                      {alert.affectedSensor && alert.affectedSensor !== "none" ? ` · ${alert.affectedSensor}` : ""}
                      {alert.anomalyScore !== undefined ? ` · Score: ${Number(alert.anomalyScore).toFixed(3)}` : ""}
                      {alert.createdAt ? ` · ${new Date(alert.createdAt).toLocaleString("en-IN", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" })}` : ""}
                    </small>
                  </section>
                </div>
              ))
            )}
          </div>
        )}

        {page === "Live Monitor" && (
          <div className="card">
            <div className="head">
              <div>
                <h3>Live Telemetry Stream</h3>
                <p>Recent sensor readings & ML status</p>
              </div>
              <b>{dashboard?.recentWeather?.length ?? 0} Readings</b>
            </div>
            <div style={{ padding: "0 16px 16px", overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "10px", textAlign: "left" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid #edf0f2", color: "#89969e" }}>
                    <th style={{ padding: "8px 6px" }}>Timestamp</th>
                    <th style={{ padding: "8px 6px" }}>Station</th>
                    <th style={{ padding: "8px 6px" }}>Temp (°C)</th>
                    <th style={{ padding: "8px 6px" }}>Hum (%)</th>
                    <th style={{ padding: "8px 6px" }}>Press (hPa)</th>
                    <th style={{ padding: "8px 6px" }}>Anomaly Score</th>
                    <th style={{ padding: "8px 6px" }}>Type</th>
                    <th style={{ padding: "8px 6px" }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(dashboard?.recentWeather || []).slice(0, 20).map((r, idx) => (
                    <tr key={idx} style={{ borderBottom: "1px solid #f4f6f8", color: "#334155" }}>
                      <td style={{ padding: "7px 6px" }}>{new Date(r.timestamp).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</td>
                      <td style={{ padding: "7px 6px", fontWeight: 600 }}>{r.stationId || "AWS-24567"}</td>
                      <td style={{ padding: "7px 6px" }}>{r.temperature?.toFixed(1)}</td>
                      <td style={{ padding: "7px 6px" }}>{r.humidity?.toFixed(1)}</td>
                      <td style={{ padding: "7px 6px" }}>{r.pressure?.toFixed(1)}</td>
                      <td style={{ padding: "7px 6px", color: r.anomalyScore > 0.616 ? "#db5e5e" : "#28a75e" }}>
                        {r.anomalyScore !== undefined ? r.anomalyScore.toFixed(4) : "---"}
                      </td>
                      <td style={{ padding: "7px 6px" }}>{r.anomalyType || "Normal"}</td>
                      <td style={{ padding: "7px 6px" }}>
                        <span className={"status " + (r.status !== "Normal" ? "warning" : "")}>
                          <i />{r.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {page === "Sensor Health" && (
          <div className="card">
            <div className="head">
              <div>
                <h3>Sensor Health & Reliability Matrix</h3>
                <p>Dynamic recovery & fault diagnostics</p>
              </div>
              <ShieldCheck size={17} />
            </div>
            <div className="health" style={{ padding: "16px 20px" }}>
              <Health name="Temperature Sensor" value={dashboard?.latestWeather?.sensorHealth?.temperature ?? 100} tone="g" />
              <p style={{ fontSize: "8px", color: "#89969e", margin: "-8px 0 14px 0" }}>Operating range: -40.0°C to 55.0°C · Max rate of change: 6.0°C/10min</p>
              <Health name="Humidity Sensor" value={dashboard?.latestWeather?.sensorHealth?.humidity ?? 100} tone="b" />
              <p style={{ fontSize: "8px", color: "#89969e", margin: "-8px 0 14px 0" }}>Operating range: 0% to 100% · Thermodynamic VPD verification</p>
              <Health name="Pressure Sensor" value={dashboard?.latestWeather?.sensorHealth?.pressure ?? 100} tone="o" />
              <p style={{ fontSize: "8px", color: "#89969e", margin: "-8px 0 14px 0" }}>Operating range: 800 to 1100 hPa · Barometric stability tracking</p>
            </div>
          </div>
        )}

        {page === "Analytics" && (
          <div className="card">
            <div className="head">
              <div>
                <h3>System Analytics</h3>
                <p>Ingestion and anomaly statistics</p>
              </div>
              <BarChart3 size={17} />
            </div>
            <div className="summary" style={{ padding: "16px" }}>
              <div><strong>{dashboard?.totalReadings ?? 0}</strong><span>Total readings</span></div>
              <div><strong className="greenText">{dashboard?.normalReadings ?? 0}</strong><span>Normal</span></div>
              <div><strong className="red">{dashboard?.anomalyReadings ?? 0}</strong><span>Anomalies</span></div>
              <div><strong className="orangeText">{dashboard?.anomalyRate ?? 0}%</strong><span>Anomaly rate</span></div>
            </div>
          </div>
        )}

        {page === "Settings" && (
          <div className="card">
            <div className="head">
              <div>
                <h3>Engine Configuration & System Status</h3>
                <p>Two-Tier Anomaly Engine & Environment</p>
              </div>
              <Settings size={17} />
            </div>
            <div style={{ padding: "16px 20px", fontSize: "10px", color: "#53636d", lineHeight: "1.8" }}>
              <p><strong>ML Microservice:</strong> Isolation Forest (22 Engineered Features, Threshold = 0.616)</p>
              <p><strong>Tier 1 Guardrails:</strong> Range Bounds, Rate-of-Change, Thermodynamics, Stuck Sensor Flatlines</p>
              <p><strong>Active Station:</strong> {selectedStation || dashboard?.latestWeather?.stationId || "All Stations"}</p>
              <p><strong>Available Stations:</strong> {(dashboard?.stations || ["AWS-24567"]).join(", ")}</p>
              <p><strong>Backend API:</strong> Node.js Express (:5000) · <strong>Database:</strong> MongoDB (:27017)</p>
            </div>
          </div>
        )}
      </div>
      <footer className="siteFooter">
        <span>© 2026 SkyGuard</span>
        <span>AI/ML-Based Smart Monitoring & Anomaly Detection</span>
        <span className="footerPrototype">Prototype</span>
      </footer>
    </main>
  </div>
} 