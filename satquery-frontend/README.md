# SatQuery AI — Frontend Web-GIS Console
**Owner:** Vinayak (Frontend & Web-GIS Lead)

## 🗺️ Overview
The SatQuery AI frontend is an interactive Web-GIS operational console built with **React 19**, **Vite**, and **Tailwind CSS**.
The active, verified frontend codebase is located in [`frontend/`](../frontend/).

## 🚀 Key Features
1. **Interactive Map Canvas (`src/components/map/MapCanvas.tsx`):**
   - High-performance raster tile rendering via GPU WebGL shaders.
   - Dynamic GeoJSON polygon overlays with area metrics tooltips (hectares / km²).
2. **Dual-Pane Synchronized Split Swipe Slider (`src/components/swipe/SwipeSlider.tsx`):**
   - Side-by-side interactive swipe divider comparing:
     - $T_1$ vs $T_2$ bi-temporal changes (urban expansion / flood recession).
     - Optical RGB vs SAR cloud penetration.
   - Synchronized pan/zoom viewport listeners.
3. **Conversational AI Drawer (`src/components/chat/ChatDrawer.tsx`):**
   - Natural language query input with preset operational prompts (*"Detect flood inundation"*, *"Highlight runway"*, *"Bi-temporal change"*).
4. **Auditable Trace Inspector (`src/components/trace/TraceInspector.tsx`):**
   - Real-time side modal displaying live JSON execution telemetry logs for jury inspection.

## 📦 Setup & Development
```bash
# Install dependencies
npm install

# Start local dev server
npm run dev
```
Open `http://localhost:5173` to view the console.
