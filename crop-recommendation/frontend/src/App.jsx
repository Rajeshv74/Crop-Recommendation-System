import { useState, useEffect } from 'react'

// Field definitions: label, unit, sensible min/max/step/default based on the
// standard crop-recommendation dataset ranges. Adjust if your data differs.
const FIELDS = [
  { key: 'N', name: 'Nitrogen', unit: 'kg/ha', min: 0, max: 140, step: 1, default: 70 },
  { key: 'P', name: 'Phosphorous', unit: 'kg/ha', min: 5, max: 145, step: 1, default: 60 },
  { key: 'K', name: 'Potassium', unit: 'kg/ha', min: 5, max: 205, step: 1, default: 45 },
  { key: 'temperature', name: 'Temperature', unit: '°C', min: 8, max: 44, step: 0.1, default: 25 },
  { key: 'humidity', name: 'Humidity', unit: '%', min: 14, max: 100, step: 0.1, default: 70 },
  { key: 'ph', name: 'Soil pH', unit: 'pH', min: 3.5, max: 9.9, step: 0.1, default: 6.5 },
  { key: 'rainfall', name: 'Rainfall', unit: 'mm', min: 20, max: 300, step: 1, default: 120 },
]

// Dataset labels -> proper Wikipedia article titles (dataset labels are
// lowercase/concatenated, e.g. "kidneybeans", "mothbeans", "pigeonpeas").
const CROP_WIKI_TITLE = {
  rice: 'Rice',
  maize: 'Maize',
  chickpea: 'Chickpea',
  kidneybeans: 'Kidney bean',
  pigeonpeas: 'Pigeon pea',
  mothbeans: 'Moth bean',
  mungbean: 'Mung bean',
  blackgram: 'Black gram',
  lentil: 'Lentil',
  pomegranate: 'Pomegranate',
  banana: 'Banana',
  mango: 'Mango',
  grapes: 'Grape',
  watermelon: 'Watermelon',
  muskmelon: 'Muskmelon',
  apple: 'Apple',
  orange: 'Orange',
  papaya: 'Papaya',
  coconut: 'Coconut',
  cotton: 'Cotton',
  jute: 'Jute',
  coffee: 'Coffee',
}

// Fetches a representative photo for the recommended crop from Wikipedia's
// public REST API (CORS-enabled, no API key needed). Returns null if the
// article has no image so the UI can fall back gracefully.
async function fetchCropImage(cropLabel) {
  const title = CROP_WIKI_TITLE[cropLabel.toLowerCase()] || cropLabel
  const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`
  const res = await fetch(url)
  if (!res.ok) throw new Error('No image found')
  const data = await res.json()
  const source = data.originalimage?.source || data.thumbnail?.source
  if (!source) throw new Error('No image found')
  return { url: source, attributionUrl: data.content_urls?.desktop?.page }
}

const API_BASE = '/api'

export default function App() {
  const [values, setValues] = useState(
    Object.fromEntries(FIELDS.map((f) => [f.key, f.default]))
  )
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [image, setImage] = useState(null)       // { url, attributionUrl }
  const [imageLoading, setImageLoading] = useState(false)

  useEffect(() => {
    if (!result) {
      setImage(null)
      return
    }
    let cancelled = false
    setImageLoading(true)
    setImage(null)
    fetchCropImage(result.recommended_crop)
      .then((img) => { if (!cancelled) setImage(img) })
      .catch(() => { if (!cancelled) setImage(null) })
      .finally(() => { if (!cancelled) setImageLoading(false) })
    return () => { cancelled = true }
  }, [result])

  const handleChange = (key, raw) => {
    const num = raw === '' ? '' : Number(raw)
    setValues((prev) => ({ ...prev, [key]: num }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setResult(null)

    const missing = FIELDS.filter((f) => values[f.key] === '' || Number.isNaN(values[f.key]))
    if (missing.length) {
      setError(`Enter a value for: ${missing.map((f) => f.name).join(', ')}`)
      return
    }

    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Prediction failed')
      setResult(data)
    } catch (err) {
      setError(err.message || 'Could not reach the prediction server. Is the Flask API running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <div className="horizon-strip">
        <span className="o" /><span className="a" /><span className="b" /><span className="c" /><span className="r" />
      </div>

      <header className="masthead">
        <div>
          <p className="eyebrow">Field Report No. 01 — Soil &amp; Climate Assay</p>
          <h1>Crop Recommendation</h1>
        </div>
        <p>
          Enter your plot's soil chemistry and local climate readings.
          A Random Forest model trained on 2,200+ field samples returns
          the best-matched crop.
        </p>
      </header>

      <div className="grid">
        <form className="panel" onSubmit={handleSubmit}>
          <div className="panel-title">
            <span>Sample Readings</span>
            <span className="count">7 parameters</span>
          </div>

          {FIELDS.map((f) => (
            <div className="field" key={f.key}>
              <div className="field-label">
                <span className="name">{f.name}</span>
                <span className="unit">{f.unit}</span>
              </div>
              <div className="field-input-row">
                <input
                  type="range"
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  value={values[f.key] === '' ? f.min : values[f.key]}
                  onChange={(e) => handleChange(f.key, e.target.value)}
                />
                <input
                  className="field-value"
                  type="number"
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  value={values[f.key]}
                  onChange={(e) => handleChange(f.key, e.target.value)}
                />
              </div>
              <div className="field-range-hint">typical range {f.min}–{f.max}</div>
            </div>
          ))}

          <button className="submit-btn" type="submit" disabled={loading}>
            {loading ? 'Analyzing sample…' : 'Recommend Crop'}
          </button>

          {error && <div className="error-note">{error}</div>}
        </form>

        <div className="ticket">
          {loading && (
            <div className="ticket-loading">
              <div className="spinner" />
              <span>Running Random Forest inference…</span>
            </div>
          )}

          {!loading && !result && (
            <div className="ticket-empty">
              <div className="glyph">◈</div>
              <p>Submit a sample to generate a field report with the recommended crop and confidence breakdown.</p>
            </div>
          )}

          {!loading && result && (
            <div>
              <div className="crop-photo">
                {imageLoading && <div className="crop-photo-skeleton" />}
                {!imageLoading && image && (
                  <img src={image.url} alt={result.recommended_crop} loading="lazy" />
                )}
                {!imageLoading && !image && (
                  <div className="crop-photo-fallback">
                    <span>{result.recommended_crop.slice(0, 1).toUpperCase()}</span>
                  </div>
                )}
              </div>

              <p className="result-eyebrow">Recommended Crop</p>
              <h2 className="result-crop">{result.recommended_crop}</h2>
              <p className="result-confidence">
                Model confidence <b>{result.confidence}%</b>
              </p>
              {image?.attributionUrl && (
                <a className="photo-credit" href={image.attributionUrl} target="_blank" rel="noreferrer">
                  Photo via Wikipedia ↗
                </a>
              )}

              <hr className="perforation" />

              <p className="top3-title">Top Matches</p>
              {result.top_3.map((c) => (
                <div className="top3-row" key={c.crop}>
                  <span className="top3-name">{c.crop}</span>
                  <div className="top3-bar-track">
                    <div className="top3-bar-fill" style={{ width: `${c.confidence}%` }} />
                  </div>
                  <span className="top3-pct">{c.confidence}%</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="footer-note">
        <span>Model: Random Forest Classifier</span>
        <span>Backend: Flask · /api/predict</span>
      </div>
    </div>
  )
}
