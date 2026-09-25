# Dataset description

Operational data from a single utility-scale solar PV plant, together with the day-ahead
weather forecasts issued for it.

## Files

| File | Rows | Coverage |
|---|---|---|
| `data/train.csv` | 12,978 | 2024-01-01 to 2025-06-30, hourly |
| `data/test.csv` | 1,488 | 2025-07-01 to 2025-08-31, hourly |

Each row describes one clock hour. `timestamp` is the **hour beginning**, in Indian Standard
Time (UTC+05:30); India does not observe daylight saving. Every other value in the row is the
mean over that hour.

## Schema

### Weather forecast — present in both files

These are the day-ahead forecast values issued for the plant the previous afternoon. They are
forecasts, not observations, and they carry forecast error.

| Column | Unit | Description |
|---|---|---|
| `timestamp` | IST | Hour beginning |
| `forecast_cloud_cover` | fraction, 0–1 | Forecast total cloud cover |
| `forecast_temp_c` | °C | Forecast air temperature |
| `forecast_wind_ms` | m/s | Forecast wind speed |
| `forecast_humidity_pct` | % | Forecast relative humidity |

### Plant records — present in `data/train.csv` only

These come off the plant after the fact, from the on-site sensors and the SCADA system. They
do not exist for the test period.

| Column | Unit | Description |
|---|---|---|
| `measured_poa_wm2` | W/m² | Plane-of-array irradiance, on-site pyranometer |
| `measured_module_temp_c` | °C | Module back-sheet temperature |
| `measured_ambient_temp_c` | °C | Air temperature, on-site sensor |
| `rainfall_mm` | mm | Rainfall accumulated during the hour |
| `status` | text | Plant operating status reported by SCADA |
| `ac_power_kw` | kW | **Target.** Mean AC power exported to the grid during the hour |

## The plant

| | |
|---|---|
| Asset | Ground-mount solar PV block |
| Location | Tumakuru district, Karnataka, India |
| Latitude / longitude | 14.10° N, 77.28° E |
| Altitude | 680 m |
| Timezone | IST (UTC+05:30) |
| Array | Fixed tilt, 15°, due south (azimuth 180°) |
| Ground albedo | 0.20 (assumed) |
| Module technology | Monocrystalline PERC |
| Power temperature coefficient | −0.35 % per °C, relative to 25 °C cell temperature |
| DC capacity | 12,500 kWp |
| AC capacity | 10,000 kW (inverter rating; DC/AC ratio 1.25) |
| Commissioning | 2021-06-15 |

## Provenance

`data/train.csv` is an unprocessed export from the plant historian, joined to the forecast
archive on timestamp. No cleaning or validation has been applied to it and no quality flags
have been attached. It is supplied exactly as it came out of the system.

`data/test.csv` contains the forecast columns only.
