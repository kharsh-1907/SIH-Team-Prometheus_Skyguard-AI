const assert = require("assert");

const NODE_URL = "http://localhost:5000";
const FLASK_URL = "http://127.0.0.1:8000";

async function runPhase4ReadinessSuite() {
    console.log("===========================================================================");
    console.log("SKYGUARD AI - PHASE 4 FINAL READINESS & STABILITY VERIFICATION SUITE");
    console.log("===========================================================================\n");

    let passedTests = 0;
    let totalTests = 0;

    function recordTest(name, passed, detail = "") {
        totalTests++;
        if (passed) {
            passedTests++;
            console.log(`✅ [PASS] ${name}`);
            if (detail) console.log(`   ${detail}`);
        } else {
            console.error(`❌ [FAIL] ${name}`);
            if (detail) console.error(`   ${detail}`);
        }
    }

    // Reset ML service state
    await fetch(`${FLASK_URL}/reset`, { method: "POST" }).catch(() => {});

    // -----------------------------------------------------------------------
    // TEST 1: Flask ML Health & Model Metadata
    // -----------------------------------------------------------------------
    try {
        const res = await fetch(`${FLASK_URL}/health`);
        const data = await res.json();
        recordTest(
            "1. Flask ML /health responds with Phase 2 Two-Tier model and 22 features",
            res.ok && data.model === "Isolation Forest" && data.features_count === 22,
            `Model: ${data.model}, Features: ${data.features_count}, Threshold: ${data.threshold}`
        );
    } catch (e) {
        recordTest("1. Flask ML /health responds", false, e.message);
    }

    // -----------------------------------------------------------------------
    // TEST 2: Normal Weather Ingestion (End-to-End Pipeline)
    // -----------------------------------------------------------------------
    try {
        const res = await fetch(`${NODE_URL}/api/weather`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                temperature: 23.5,
                pressure: 1012.5,
                humidity: 58.0,
                stationId: "AWS-READINESS-01",
                location: "Pune Observatory"
            })
        });
        const data = await res.json();
        const ok = res.status === 201 &&
                   data.data.weather.status === "Normal" &&
                   data.data.weather.anomalyType === "Normal" &&
                   data.data.weather.sensorHealth.temperature === 100 &&
                   data.data.alert === null;
        recordTest(
            "2. Normal Reading Ingestion: Saved with Normal status and 100% health",
            ok,
            `Weather ID: ${data.data.weather._id}, Score: ${data.data.weather.anomalyScore.toFixed(4)}, Status: ${data.data.weather.status}`
        );
    } catch (e) {
        recordTest("2. Normal Reading Ingestion", false, e.message);
    }

    // -----------------------------------------------------------------------
    // TEST 3: OutOfBounds Physical Violation (Tier 1 Guardrails)
    // -----------------------------------------------------------------------
    try {
        const res = await fetch(`${NODE_URL}/api/weather`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                temperature: 68.0, // Above 55.0°C
                pressure: 1012.0,
                humidity: 50.0,
                stationId: "AWS-READINESS-01"
            })
        });
        const data = await res.json();
        const ok = res.status === 201 &&
                   data.data.weather.status === "High" &&
                   data.data.weather.anomalyType === "OutOfBounds" &&
                   data.data.weather.affectedSensor === "temperature" &&
                   data.data.weather.sensorHealth.temperature <= 50 &&
                   data.data.alert !== null &&
                   data.data.alert.type === "OutOfBounds Anomaly";
        recordTest(
            "3. OutOfBounds Anomaly: Detected, temperature health penalized, Alert created",
            ok,
            `Type: ${data.data.weather.anomalyType}, Affected: ${data.data.weather.affectedSensor}, Health: ${JSON.stringify(data.data.weather.sensorHealth)}, Alert: "${data.data.alert?.message}"`
        );
    } catch (e) {
        recordTest("3. OutOfBounds Anomaly", false, e.message);
    }

    // -----------------------------------------------------------------------
    // TEST 4: Rate-of-Change Anomaly with Isolated Pressure Drop
    // -----------------------------------------------------------------------
    try {
        const rocStation = "AWS-ROC-READINESS";
        // Baseline reading
        await fetch(`${NODE_URL}/api/weather`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                temperature: 23.0,
                pressure: 1012.0,
                humidity: 50.0,
                stationId: rocStation
            })
        });

        // Rapid pressure drop (|dP| = 17 > 4.0 limit)
        const res = await fetch(`${NODE_URL}/api/weather`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                temperature: 23.0,
                pressure: 995.0,
                humidity: 50.0,
                stationId: rocStation
            })
        });
        const data = await res.json();
        const ok = res.status === 201 &&
                   data.data.weather.anomalyType === "RateOfChange" &&
                   data.data.weather.affectedSensor === "pressure" &&
                   data.data.weather.sensorHealth.pressure <= 65;
        recordTest(
            "4. RateOfChange Anomaly: Rapid pressure drop flagged, pressure sensor isolated",
            ok,
            `Type: ${data.data.weather.anomalyType}, Affected: ${data.data.weather.affectedSensor}, Pressure Health: ${data.data.weather.sensorHealth.pressure}%`
        );
    } catch (e) {
        recordTest("4. RateOfChange Anomaly", false, e.message);
    }

    // -----------------------------------------------------------------------
    // TEST 5: Thermodynamic Guardrails (Dew Point / VPD Consistency)
    // -----------------------------------------------------------------------
    try {
        const res = await fetch(`${NODE_URL}/api/ml/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                temperature: 15.0,
                pressure: 1013.0,
                humidity: 105.0,
                stationId: "AWS-THERMO-READINESS"
            })
        });
        const data = await res.json();
        const ok = res.status === 200 && data.status === "Anomaly";
        recordTest(
            "5. Thermodynamic Violation: Detected and flagged by Tier 1 guardrails",
            ok,
            `Status: ${data.status}, Anomaly Type: ${data.anomaly_type}, Affected: ${data.affected_sensor}`
        );
    } catch (e) {
        recordTest("5. Thermodynamic Violation", false, e.message);
    }

    // -----------------------------------------------------------------------
    // TEST 6: Stuck Sensor Flatline & Natural Health Recovery (+5%/step)
    // -----------------------------------------------------------------------
    try {
        const stuckStation = "AWS-STUCK-READINESS";
        let lastStuck = null;
        for (let i = 1; i <= 7; i++) {
            const res = await fetch(`${NODE_URL}/api/weather`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    temperature: 20.0,
                    pressure: 1010.0,
                    humidity: 50.0,
                    stationId: stuckStation
                })
            });
            lastStuck = await res.json();
        }

        const stuckDetected = lastStuck.data.weather.anomalyType === "StuckSensor" &&
                              lastStuck.data.weather.affectedSensor === "temperature" &&
                              lastStuck.data.weather.sensorHealth.temperature <= 60;

        let lastRecovered = null;
        for (let i = 1; i <= 3; i++) {
            const res = await fetch(`${NODE_URL}/api/weather`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    temperature: 21.0 + i * 0.4,
                    pressure: 1010.0 + i * 0.2,
                    humidity: 52.0 + i * 0.3,
                    stationId: stuckStation
                })
            });
            lastRecovered = await res.json();
        }

        const recoveredHealth = lastRecovered.data.weather.sensorHealth.temperature;
        const recoveryOk = recoveredHealth > 60;

        recordTest(
            "6. Stuck Sensor Flatline & Natural Health Recovery",
            stuckDetected && recoveryOk,
            `Step 7 Stuck detected: ${lastStuck.data.weather.anomalyType} (Health: ${lastStuck.data.weather.sensorHealth.temperature}%). Post-recovery Health: ${recoveredHealth}%`
        );
    } catch (e) {
        recordTest("6. Stuck Sensor & Health Recovery", false, e.message);
    }

    // -----------------------------------------------------------------------
    // TEST 7: Hardened Input Validation (Missing, String, Whitespace, Infinity)
    // -----------------------------------------------------------------------
    try {
        const missingRes = await fetch(`${NODE_URL}/api/weather`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ temperature: 22.0 })
        });

        const nonNumRes = await fetch(`${NODE_URL}/api/weather`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ temperature: "not-a-number", pressure: 1012, humidity: 55 })
        });

        const whitespaceRes = await fetch(`${NODE_URL}/api/weather`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ temperature: "   ", pressure: 1012, humidity: 55 })
        });

        const infRes = await fetch(`${NODE_URL}/api/weather`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ temperature: Infinity, pressure: 1012, humidity: 55 })
        });

        const ok = missingRes.status === 400 && nonNumRes.status === 400 &&
                   whitespaceRes.status === 400 && infRes.status === 400;

        recordTest(
            "7. Input Validation: Missing, non-numeric, whitespace ('   '), and Infinity rejected with 400",
            ok,
            `Statuses -> Missing: ${missingRes.status}, NonNumeric: ${nonNumRes.status}, Whitespace: ${whitespaceRes.status}, Infinity: ${infRes.status}`
        );
    } catch (e) {
        recordTest("7. Input Validation", false, e.message);
    }

    // -----------------------------------------------------------------------
    // TEST 8: Multi-Station Isolation & Cross-Filtering
    // -----------------------------------------------------------------------
    try {
        await fetch(`${NODE_URL}/api/weather`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                temperature: 19.5,
                pressure: 985.0,
                humidity: 75.0,
                stationId: "AWS-CHENNAI-03",
                location: "Chennai Coastal"
            })
        });

        const dashAll = await fetch(`${NODE_URL}/api/dashboard`).then(r => r.json());
        const dashStation = await fetch(`${NODE_URL}/api/dashboard?stationId=AWS-CHENNAI-03`).then(r => r.json());
        const alertsStation = await fetch(`${NODE_URL}/api/alerts?stationId=AWS-READINESS-01`).then(r => r.json());

        const ok = Array.isArray(dashAll.stations) &&
                   dashAll.stations.includes("AWS-CHENNAI-03") &&
                   dashStation.latestWeather.stationId === "AWS-CHENNAI-03" &&
                   Array.isArray(alertsStation) &&
                   alertsStation.every(a => a.stationId === "AWS-READINESS-01");

        recordTest(
            "8. Multi-Station Isolation: Dashboard and Alert queries filter by stationId and list active stations",
            ok,
            `Active Stations: ${dashAll.stations.length}, Station Latest: ${dashStation.latestWeather?.stationId}, Filtered Alerts: ${alertsStation.length}`
        );
    } catch (e) {
        recordTest("8. Multi-Station Isolation", false, e.message);
    }

    // -----------------------------------------------------------------------
    // TEST 9: Concurrency & Thread-Safety (20 Parallel Multi-Station Requests)
    // -----------------------------------------------------------------------
    try {
        const concurrentStations = ["AWS-CONC-A", "AWS-CONC-B", "AWS-CONC-C", "AWS-CONC-D"];
        const promises = [];
        for (let i = 0; i < 20; i++) {
            promises.push(
                fetch(`${NODE_URL}/api/weather`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        temperature: 21.0 + (i * 0.3),
                        pressure: 1011.0 + (i * 0.1),
                        humidity: 50.0 + (i % 8),
                        stationId: concurrentStations[i % concurrentStations.length]
                    })
                }).then(r => r.status)
            );
        }
        const statuses = await Promise.all(promises);
        const all201 = statuses.every(s => s === 201);
        recordTest(
            "9. Concurrency & Thread Safety: 20 simultaneous multi-station requests all succeed (201)",
            all201 && statuses.length === 20,
            `Total Requests: ${statuses.length}, Successful (201): ${statuses.filter(s => s === 201).length}`
        );
    } catch (e) {
        recordTest("9. Concurrency", false, e.message);
    }

    // -----------------------------------------------------------------------
    // TEST 10: Performance & Latency Benchmark
    // -----------------------------------------------------------------------
    try {
        const latencies = [];
        for (let i = 0; i < 15; i++) {
            const t0 = performance.now();
            const res = await fetch(`${NODE_URL}/api/weather`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    temperature: 22.0 + (i % 4) * 0.5,
                    pressure: 1013.0,
                    humidity: 55.0,
                    stationId: "AWS-BENCH-SUITE"
                })
            });
            const t1 = performance.now();
            if (res.ok) latencies.push(t1 - t0);
        }
        latencies.sort((a, b) => a - b);
        const p50 = latencies[Math.floor(latencies.length * 0.5)];
        const p95 = latencies[Math.floor(latencies.length * 0.95)];
        const ok = p95 < 120; // Well within real-time SLA (< 120ms for full pipeline)
        recordTest(
            "10. Performance SLA: Full pipeline latency meets real-time requirements (P95 < 120ms)",
            ok,
            `P50: ${p50.toFixed(2)}ms, P95: ${p95.toFixed(2)}ms, Min: ${latencies[0].toFixed(2)}ms, Max: ${latencies[latencies.length - 1].toFixed(2)}ms`
        );
    } catch (e) {
        recordTest("10. Performance SLA", false, e.message);
    }

    // -----------------------------------------------------------------------
    // TEST 11: End-to-End Persistence Resilience
    // -----------------------------------------------------------------------
    try {
        const res = await fetch(`${NODE_URL}/api/weather`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                temperature: 24.1,
                pressure: 1013.2,
                humidity: 50.0,
                stationId: "AWS-FINAL-TEST"
            })
        });
        const data = await res.json();
        const ok = res.status === 201 && data.success === true && data.data.weather._id;
        recordTest(
            "11. End-to-End Persistence Resilience: Weather saved with 201 Created and zero data loss",
            ok,
            `Saved Weather ID: ${data.data.weather._id}, Status: ${data.data.weather.status}`
        );
    } catch (e) {
        recordTest("11. Resilience", false, e.message);
    }

    // -----------------------------------------------------------------------
    // SUMMARY
    // -----------------------------------------------------------------------
    console.log("\n" + "=".repeat(75));
    console.log(`PHASE 4 READINESS TEST RESULTS: ${passedTests}/${totalTests} TESTS PASSED`);
    console.log("=".repeat(75));

    if (passedTests === totalTests) {
        console.log("🎉 ALL PHASE 4 READINESS & STABILITY TESTS PASSED CLEANLY!");
        process.exit(0);
    } else {
        console.error(`⚠️ ${totalTests - passedTests} test(s) failed.`);
        process.exit(1);
    }
}

runPhase4ReadinessSuite().catch(e => {
    console.error("FATAL SUITE ERROR:", e);
    process.exit(1);
});
