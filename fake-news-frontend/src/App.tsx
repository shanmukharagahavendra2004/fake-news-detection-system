import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const apiUrl = process.env.REACT_APP_API_URL;
const App: React.FC = () => {
  const [text, setText] = useState("");
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [fakeProb, setFakeProb] = useState<number | null>(null);
  const [realProb, setRealProb] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const res = await axios.post(`${apiUrl}/predict`, { text });

      setPrediction(res.data.prediction);
      setConfidence(res.data.confidence);
      setFakeProb(res.data.probabilities.fake);
      setRealProb(res.data.probabilities.real);
    } catch (err) {
      setError("Failed to get prediction. Is Flask server running?");
    } finally {
      setLoading(false);
    }
  };

 return (
  <div className="min-h-screen bg-gradient-to-tr from-indigo-600 via-purple-600 to-pink-600 flex items-center justify-center p-6">
    <div className="backdrop-blur-xl bg-white/90 shadow-2xl rounded-3xl p-10 max-w-2xl w-full animate-fadeIn">
      
      {/* Header */}
      <h1 className="text-4xl font-extrabold text-center text-purple-700 mb-2">
        Fake News Detector 📰
      </h1>
      <p className="text-center text-gray-500 mb-6">
        AI-powered text classification using Machine Learning
      </p>

      {/* Textarea */}
      <div className="relative">
        <textarea
          className="w-full h-44 p-4 border-2 border-purple-300 rounded-2xl focus:outline-none focus:ring-4 focus:ring-purple-400 resize-none text-gray-800 transition"
          placeholder="Paste or type a news article here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <span className="absolute bottom-2 right-4 text-xs text-gray-400">
          {text.length} characters
        </span>
      </div>

      {/* Button */}
      <button
        onClick={handleSubmit}
        disabled={loading}
        className="w-full mt-5 bg-gradient-to-r from-purple-600 to-pink-600 text-white py-3 rounded-2xl font-semibold text-lg shadow-lg hover:scale-[1.02] transition-all disabled:opacity-50"
      >
        {loading ? (
          <span className="flex justify-center items-center gap-2">
            <span className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
            Analyzing...
          </span>
        ) : (
          "Check News"
        )}
      </button>

      {/* Error */}
      {error && (
        <p className="text-red-500 text-center mt-4 font-semibold">
          {error}
        </p>
      )}

      {/* Results */}
      {prediction && confidence !== null && (
        <div className="mt-8 space-y-6">

          {/* Prediction Card */}
          <div
            className={`p-6 rounded-2xl text-center shadow-md ${
              prediction === "REAL"
                ? "bg-green-100 text-green-800"
                : "bg-red-100 text-red-800"
            }`}
          >
            <h2 className="text-2xl font-bold flex justify-center items-center gap-2">
              {prediction === "REAL" ? "✅ REAL NEWS" : "❌ FAKE NEWS"}
            </h2>
            <p className="mt-2 text-lg">
              Confidence: <span className="font-bold">{confidence}%</span>
            </p>
          </div>

          {/* Probability Bars */}
          {fakeProb !== null && realProb !== null && (
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm font-semibold text-red-600">
                  <span>Fake Probability</span>
                  <span>{fakeProb}%</span>
                </div>
                <div className="w-full bg-red-200 rounded-full h-4 overflow-hidden">
                  <div
                    className="bg-red-500 h-4 rounded-full transition-all duration-700"
                    style={{ width: `${fakeProb}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm font-semibold text-green-600">
                  <span>Real Probability</span>
                  <span>{realProb}%</span>
                </div>
                <div className="w-full bg-green-200 rounded-full h-4 overflow-hidden">
                  <div
                    className="bg-green-500 h-4 rounded-full transition-all duration-700"
                    style={{ width: `${realProb}%` }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Footer */}
      <p className="text-gray-400 text-center mt-8 text-sm">
        ⚙️ React + TypeScript + Tailwind + Flask ML Backend
      </p>
    </div>
  </div>
);
};

export default App;
