import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import {
  Clock,
  Train,
  AlertTriangle,
  ThermometerSun,
  MapPin,
  TrendingUp,
  Activity,
} from "lucide-react";

const TTCDelayDashboard = () => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [examples, setExamples] = useState([]);

  // Form state
  const [formData, setFormData] = useState({
    hour: new Date().getHours(),
    station: "UNION",
    subway_line: "YU",
    temperature: 15,
    precipitation: 0,
    is_weekend: new Date().getDay() >= 5,
  });

  const stations = {
    YU: [
      "FINCH",
      "NORTH YORK CENTRE",
      "SHEPPARD-YONGE",
      "EGLINTON",
      "ST CLAIR",
      "ROSEDALE",
      "BLOOR-YONGE",
      "COLLEGE",
      "DUNDAS",
      "QUEEN",
      "KING",
      "UNION",
    ],
    BD: [
      "KIPLING",
      "ISLINGTON",
      "ROYAL YORK",
      "JANE",
      "RUNNYMEDE",
      "HIGH PARK",
      "KEELE",
      "DUNDAS WEST",
      "LANSDOWNE",
      "DUFFERIN",
      "CHRISTIE",
      "BATHURST",
      "SPADINA",
      "ST GEORGE",
      "BAY",
      "BLOOR-YONGE",
      "SHERBOURNE",
      "CASTLE FRANK",
    ],
    SHP: ["SHEPPARD-YONGE", "BAYVIEW", "BESSARION", "LESLIE", "DON MILLS"],
    SRT: [
      "KENNEDY",
      "LAWRENCE EAST",
      "ELLESMERE",
      "MIDLAND",
      "SCARBOROUGH CENTRE",
    ],
  };

  const lineColors = {
    YU: "#FFD320", // Yellow
    BD: "#00B04F", // Green
    SHP: "#753BBD", // Purple
    SRT: "#009639", // Light Blue
  };

  // Load model info and examples on component mount
  useEffect(() => {
    fetchModelInfo();
    fetchExamples();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await fetch("/api/models");
      const data = await response.json();
      setModelInfo(data);
    } catch (error) {
      console.error("Failed to fetch model info:", error);
    }
  };

  const fetchExamples = async () => {
    try {
      const response = await fetch("/api/examples");
      const data = await response.json();
      setExamples(data.examples || []);
    } catch (error) {
      console.error("Failed to fetch examples:", error);
    }
  };

  const makePrediction = async () => {
    setLoading(true);
    try {
      const currentDate = new Date();

      const features = {
        hour: parseInt(formData.hour),
        month: currentDate.getMonth() + 1,
        day_of_month: currentDate.getDate(),
        week_of_year: Math.ceil(currentDate.getDate() / 7),
        is_weekend: formData.is_weekend,
        is_rush_hour: [7, 8, 17, 18, 19].includes(parseInt(formData.hour)),
        is_morning_rush: [7, 8].includes(parseInt(formData.hour)),
        is_evening_rush: [17, 18, 19].includes(parseInt(formData.hour)),
        temperature: parseFloat(formData.temperature),
        precipitation: parseFloat(formData.precipitation),
        is_extreme_cold: formData.temperature < -10,
        is_extreme_hot: formData.temperature > 30,
        has_precipitation: formData.precipitation > 0,
        heavy_precipitation: formData.precipitation > 5,
        is_major_station: [
          "UNION",
          "BLOOR-YONGE",
          "ST GEORGE",
          "EGLINTON",
          "FINCH",
        ].includes(formData.station),
        delays_last_7days: 3.5,
        delay_count_last_7days: 5,
        subway_line: formData.subway_line,
        station: formData.station,
        incident_code: "MUSAN",
        direction: "N",
        weather_condition: formData.precipitation > 0 ? "Rain" : "Clear",
        season: "Spring",
        line_capacity: formData.subway_line === "SHP" ? "Low" : "High",
      };

      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(features),
      });

      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      console.error("Prediction failed:", error);
      setPrediction({ error: "Failed to get prediction. Is the API running?" });
    }
    setLoading(false);
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case "low":
        return "text-green-600 bg-green-100";
      case "medium":
        return "text-yellow-600 bg-yellow-100";
      case "high":
        return "text-orange-600 bg-orange-100";
      case "critical":
        return "text-red-600 bg-red-100";
      default:
        return "text-gray-600 bg-gray-100";
    }
  };

  const sampleData = [
    { time: "6:00", delays: 1.2 },
    { time: "7:00", delays: 4.8 },
    { time: "8:00", delays: 6.2 },
    { time: "9:00", delays: 3.1 },
    { time: "17:00", delays: 5.9 },
    { time: "18:00", delays: 7.3 },
    { time: "19:00", delays: 4.7 },
    { time: "20:00", delays: 2.1 },
  ];

  const lineData = [
    { line: "Yonge-University", delays: 3036, color: lineColors.YU },
    { line: "Bloor-Danforth", delays: 2052, color: lineColors.BD },
    { line: "Sheppard", delays: 623, color: lineColors.SHP },
    { line: "Scarborough RT", delays: 316, color: lineColors.SRT },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2 flex items-center justify-center gap-3">
            <Train className="text-blue-600" size={40} />
            TTC Delay Prediction System
          </h1>
          <p className="text-lg text-gray-600">
            AI-Powered Transit Delay Forecasting for Toronto
          </p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">
                  Model Accuracy
                </p>
                <p className="text-2xl font-bold text-blue-600">±2.15 min</p>
              </div>
              <Activity className="text-blue-600" size={24} />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">
                  Stations Covered
                </p>
                <p className="text-2xl font-bold text-green-600">38</p>
              </div>
              <MapPin className="text-green-600" size={24} />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">
                  Training Records
                </p>
                <p className="text-2xl font-bold text-purple-600">6,027</p>
              </div>
              <TrendingUp className="text-purple-600" size={24} />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">R² Score</p>
                <p className="text-2xl font-bold text-orange-600">0.285</p>
              </div>
              <AlertTriangle className="text-orange-600" size={24} />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Prediction Form */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Clock className="text-blue-600" size={20} />
                Predict Delay
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Hour
                  </label>
                  <select
                    value={formData.hour}
                    onChange={(e) =>
                      setFormData({ ...formData, hour: e.target.value })
                    }
                    className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    {Array.from({ length: 24 }, (_, i) => (
                      <option key={i} value={i}>
                        {String(i).padStart(2, "0")}:00
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Subway Line
                  </label>
                  <select
                    value={formData.subway_line}
                    onChange={(e) => {
                      setFormData({
                        ...formData,
                        subway_line: e.target.value,
                        station: stations[e.target.value][0],
                      });
                    }}
                    className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="YU">Yonge-University (Yellow)</option>
                    <option value="BD">Bloor-Danforth (Green)</option>
                    <option value="SHP">Sheppard (Purple)</option>
                    <option value="SRT">Scarborough RT (Blue)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Station
                  </label>
                  <select
                    value={formData.station}
                    onChange={(e) =>
                      setFormData({ ...formData, station: e.target.value })
                    }
                    className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    {stations[formData.subway_line].map((station) => (
                      <option key={station} value={station}>
                        {station}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Temperature (°C)
                  </label>
                  <input
                    type="number"
                    value={formData.temperature}
                    onChange={(e) =>
                      setFormData({ ...formData, temperature: e.target.value })
                    }
                    className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    min="-30"
                    max="40"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Precipitation (mm)
                  </label>
                  <input
                    type="number"
                    value={formData.precipitation}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        precipitation: e.target.value,
                      })
                    }
                    className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    min="0"
                    max="50"
                  />
                </div>

                <div className="flex items-center">
                  <input
                    type="checkbox"
                    checked={formData.is_weekend}
                    onChange={(e) =>
                      setFormData({ ...formData, is_weekend: e.target.checked })
                    }
                    className="mr-2"
                  />
                  <label className="text-sm font-medium text-gray-700">
                    Weekend
                  </label>
                </div>

                <button
                  onClick={makePrediction}
                  disabled={loading}
                  className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? "Predicting..." : "Predict Delay"}
                </button>
              </div>

              {/* Prediction Result */}
              {prediction && (
                <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                  {prediction.error ? (
                    <div className="text-red-600">
                      <strong>Error:</strong> {prediction.error}
                    </div>
                  ) : (
                    <div>
                      <div className="text-center mb-3">
                        <div className="text-3xl font-bold text-blue-600">
                          {prediction.prediction} min
                        </div>
                        <div
                          className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${getSeverityColor(prediction.severity)}`}
                        >
                          {prediction.category}
                        </div>
                      </div>
                      <div className="text-xs text-gray-500 text-center">
                        Confidence: {prediction.confidence} | Model:{" "}
                        {prediction.model_used}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Charts and Analytics */}
          <div className="lg:col-span-2 space-y-6">
            {/* Rush Hour Pattern */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-bold mb-4">
                Rush Hour Delay Patterns
              </h3>
              <LineChart width={600} height={300} data={sampleData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="delays"
                  stroke="#2563eb"
                  strokeWidth={3}
                />
              </LineChart>
            </div>

            {/* Line Performance */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-bold mb-4">Delays by Subway Line</h3>
              <BarChart width={600} height={300} data={lineData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="line" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="delays" fill="#8884d8" />
              </BarChart>
            </div>
          </div>
        </div>

        {/* Example Predictions */}
        {examples.length > 0 && (
          <div className="mt-8 bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-bold mb-4">Example Scenarios</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {examples.map((example, index) => (
                <div
                  key={index}
                  className="border border-gray-200 rounded-lg p-4"
                >
                  <h4 className="font-semibold text-gray-900 mb-2">
                    {example.scenario}
                  </h4>
                  {example.prediction && !example.prediction.error && (
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {example.prediction.prediction} min
                      </div>
                      <div
                        className={`inline-block px-2 py-1 rounded text-xs ${getSeverityColor(example.prediction.severity)}`}
                      >
                        {example.prediction.category}
                      </div>
                    </div>
                  )}
                  <div className="mt-2 text-xs text-gray-500">
                    {example.input.station} Station • {example.input.hour}:00
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Model Performance */}
        {modelInfo && (
          <div className="mt-8 bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-bold mb-4">
              Model Performance Metrics
            </h3>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Model
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      MAE (min)
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      R² Score
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  <tr>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      Optimized Random Forest
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      2.15
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      0.285
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800">
                        Best Performer
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      Random Forest
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      2.20
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      0.266
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-blue-100 text-blue-800">
                        Production Ready
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      Gradient Boosting
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      2.21
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      0.245
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-yellow-100 text-yellow-800">
                        Alternative
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="mt-8 text-center text-gray-500 text-sm">
          <p>
            Built with React, Python Flask, and scikit-learn • Data from Toronto
            Open Data Portal
          </p>
          <p>
            Predicting delays with ±2.15 minute accuracy across 38 TTC stations
          </p>
        </div>
      </div>
    </div>
  );
};

export default TTCDelayDashboard;
