# Crop Recommendation System
Random Forest (scikit-learn) + Flask API + React (Vite) frontend.

```
crop-recommendation/
├── backend/
│   ├── app.py              # Flask API
│   ├── train_model.py      # trains the Random Forest on your CSV
│   ├── requirements.txt
│   └── model/               # crop_model.pkl / label_encoder.pkl saved here after training
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── main.jsx
    │   └── index.css
    ├── index.html
    ├── package.json
    └── vite.config.js
```

Your dataset needs exactly these 8 columns (case-sensitive):
`N, P, K, temperature, humidity, ph, rainfall, label`

---

## Step 1 — Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2 — Train the model on YOUR dataset

Put your CSV anywhere (e.g. `backend/data/Crop_recommendation.csv`) and run:

```bash
python train_model.py --data data/Crop_recommendation.csv
```

This prints accuracy, a classification report, and feature importances, then
saves `crop_model.pkl` and `label_encoder.pkl` into `backend/model/`.

## Step 3 — Run the Flask API

```bash
python app.py
```

Runs on `http://localhost:5000`. Test it:

```bash
curl http://localhost:5000/api/health

curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":20.8,"humidity":82,"ph":6.5,"rainfall":202}'
```

## Step 4 — Frontend setup

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The Vite dev server proxies `/api/*` calls to
Flask on port 5000 (configured in `vite.config.js`), so no CORS setup is
needed in dev — `flask-cors` is included for when you deploy them separately.

## Step 5 — Use it

Fill in (or drag the sliders for) the 7 soil/climate readings and click
**Recommend Crop**. You'll get the predicted crop, a confidence score, and
the top-3 closest matches.

## Step 6 — Production build (optional)

```bash
cd frontend
npm run build       # outputs static files to frontend/dist
```

Serve `frontend/dist` with any static host (Nginx, Vercel, Netlify) and point
it at your deployed Flask API — just update `API_BASE` in `src/App.jsx` or
add an environment-based config if you deploy backend and frontend on
different domains.

---

### Notes
- `train_model.py` does an 80/20 train/test split plus 5-fold cross-validation
  so you get a realistic accuracy estimate, not just training accuracy.
- `/api/predict` also returns `top_3`, useful if a farmer wants alternatives.
- To retrain after adding more data, just rerun Step 2 — the API auto-loads
  whatever's in `backend/model/` on startup.
