import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { UploadCloud, FileImage, X, Loader2, Send, Database, Info, Code2, FileDown, CheckCircle2, Layers, Sparkles, Globe, Eye, ShieldCheck } from 'lucide-react';
import { API_BASE } from '../config';

export default function Analysis() {
  const navigate = useNavigate();
  const [files, setFiles] = useState([]);
  const [query, setQuery] = useState('');
  const [includeXai, setIncludeXai] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(65);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingPreset, setIsLoadingPreset] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  const fileInputRef = useRef(null);

  const loadPreset = async (presetType) => {
    setIsLoadingPreset(true);
    setError(null);
    try {
      if (presetType === 'water') {
        const res = await axios.get(`${API_BASE}/api/samples/optical.tif`, { responseType: 'blob' });
        const file = new File([res.data], 'optical.tif', { type: 'image/tiff' });
        setFiles([file]);
        setQuery('Highlight and segment the water body in this image');
      } else if (presetType === 'change') {
        const res1 = await axios.get(`${API_BASE}/api/samples/bitemporal_t1.tif`, { responseType: 'blob' });
        const res2 = await axios.get(`${API_BASE}/api/samples/bitemporal_t2.tif`, { responseType: 'blob' });
        const file1 = new File([res1.data], 'bitemporal_t1.tif', { type: 'image/tiff' });
        const file2 = new File([res2.data], 'bitemporal_t2.tif', { type: 'image/tiff' });
        setFiles([file1, file2]);
        setQuery('What changed between these two temporal acquisitions?');
      } else if (presetType === 'fusion') {
        const res1 = await axios.get(`${API_BASE}/api/samples/optical.tif`, { responseType: 'blob' });
        const res2 = await axios.get(`${API_BASE}/api/samples/sar.tif`, { responseType: 'blob' });
        const file1 = new File([res1.data], 'optical.tif', { type: 'image/tiff' });
        const file2 = new File([res2.data], 'sar.tif', { type: 'image/tiff' });
        setFiles([file1, file2]);
        setQuery('Use optical and SAR together to detect built-up and water covered regions');
      }
    } catch (err) {
      console.error(err);
      setError('Failed to load sample dataset.');
    } finally {
      setIsLoadingPreset(false);
    }
  };

  const handleDownloadDossier = async (traceId) => {
    setIsDownloading(true);
    try {
      const response = await axios.post(`${API_BASE}/api/export-report/${traceId}`, null, {
        responseType: 'blob'
      });
      const blob = new Blob([response.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${traceId}_report.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to download report', err);
    } finally {
      setIsDownloading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.length) {
      setFiles((prev) => [...prev, ...Array.from(e.dataTransfer.files)]);
    }
  };

  const removeFile = (index) => setFiles((prev) => prev.filter((_, i) => i !== index));

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!files.length) return setError('Please upload imagery.');
    if (!query.trim()) return setError('Please enter a prompt.');

    setError(null);
    setIsLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append('query', query);
    formData.append('include_xai', includeXai);
    files.forEach((file) => formData.append('files', file));

    try {
      const { data } = await axios.post(`${API_BASE}/api/query`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Connection failed.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
      {/* Left Panel: Inputs */}
      <section className="panel p-6 lg:col-span-5 flex flex-col gap-6">
        <h2 className="text-lg font-semibold flex items-center gap-2 text-textMain border-b border-border pb-4">
          <Database className="w-5 h-5 text-primary" /> Input Configuration
        </h2>

        {/* 1-Click Instant Demo Presets */}
        <div className="bg-slate-100 p-3.5 rounded-xl border border-slate-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-amber-500" /> Instant Demo Datasets
            </span>
            {isLoadingPreset && <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" />}
          </div>
          <div className="flex flex-col sm:flex-row flex-wrap gap-2">
            <button
              type="button"
              onClick={() => loadPreset('water')}
              disabled={isLoadingPreset}
              className="text-xs font-medium px-2.5 py-1.5 rounded-lg bg-white border border-slate-200 hover:border-primary hover:text-primary transition-colors text-slate-700 shadow-2xs"
            >
              🌊 Water Body (Optical)
            </button>
            <button
              type="button"
              onClick={() => loadPreset('change')}
              disabled={isLoadingPreset}
              className="text-xs font-medium px-2.5 py-1.5 rounded-lg bg-white border border-slate-200 hover:border-primary hover:text-primary transition-colors text-slate-700 shadow-2xs"
            >
              🏗️ Bi-Temporal (T1 vs T2)
            </button>
            <button
              type="button"
              onClick={() => loadPreset('fusion')}
              disabled={isLoadingPreset}
              className="text-xs font-medium px-2.5 py-1.5 rounded-lg bg-white border border-slate-200 hover:border-primary hover:text-primary transition-colors text-slate-700 shadow-2xs"
            >
              🛰️ Optical + SAR Fusion
            </button>
          </div>
          <p className="text-[11px] text-slate-500 mt-2">
            Click any button to auto-load sample rasters & query, or drag from <code className="bg-white px-1 py-0.5 rounded border text-slate-700">data/samples/</code>.
          </p>
        </div>

        <div
          className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center transition-all cursor-pointer bg-slate-50 ${
            isDragging ? 'border-primary bg-blue-50' : 'border-slate-300 hover:border-slate-400 hover:bg-slate-100'
          }`}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input type="file" multiple className="hidden" ref={fileInputRef} onChange={(e) => setFiles((p) => [...p, ...Array.from(e.target.files)])} accept=".tif,.tiff,.jpg,.jpeg,.png" />
          <div className="bg-white border border-slate-200 p-3 rounded-full mb-3 shadow-sm text-primary">
            <UploadCloud className="w-6 h-6" />
          </div>
          <p className="font-medium text-textMain">Drop GeoTIFF imagery here</p>
          <p className="text-textMuted text-xs mt-1">Supports multi-band TIF, PNG, JPG</p>
        </div>

        {files.length > 0 && (
          <div className="space-y-3">
            <span className="text-xs font-semibold text-textMuted uppercase tracking-wider block">
              Loaded Satellite Rasters ({files.length})
            </span>
            <div className="space-y-2">
              {files.map((file, i) => {
                const previewUrl = {
                  'optical.tif': `${API_BASE}/api/samples/preview/optical_preview.png`,
                  'sar.tif': `${API_BASE}/api/samples/preview/sar_preview.png`,
                  'bitemporal_t1.tif': `${API_BASE}/api/samples/preview/bitemporal_t1_preview.png`,
                  'bitemporal_t2.tif': `${API_BASE}/api/samples/preview/bitemporal_t2_preview.png`,
                }[file.name];

                return (
                  <div key={i} className="flex items-center justify-between bg-white p-3 rounded-lg border border-border shadow-sm group/item">
                    <div className="flex items-center gap-3 overflow-hidden">
                      {previewUrl ? (
                        <img 
                          src={previewUrl} 
                          alt={file.name} 
                          className="w-10 h-10 object-cover rounded border border-slate-200 bg-slate-900 shrink-0" 
                        />
                      ) : (
                        <div className="w-10 h-10 rounded bg-blue-50 border border-blue-100 flex items-center justify-center shrink-0">
                          <FileImage className="w-5 h-5 text-primary" />
                        </div>
                      )}
                      <div className="overflow-hidden">
                        <span className="truncate text-sm text-textMain font-medium block">{file.name}</span>
                        <span className="text-[10px] text-slate-500 font-mono">
                          {file.name.includes('sar') ? 'Sentinel-1 C-Band SAR' : 'Sentinel-2 Multispectral 10m'}
                        </span>
                      </div>
                    </div>
                    <button onClick={() => removeFile(i)} className="text-slate-400 hover:text-red-500 transition-colors p-1">
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        <div className="flex flex-col gap-2">
          <label className="text-xs font-semibold text-textMuted uppercase tracking-wider">Agent Prompt</label>
          <textarea
            rows="4"
            className="w-full bg-white border border-border rounded-lg p-4 text-textMain focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all resize-none shadow-inner-soft text-sm"
            placeholder="e.g., Identify new constructions between these two scenes."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>

        {/* RS-XAI Explainability Toggle */}
        <div className="bg-slate-100/80 border border-slate-200 p-3.5 rounded-xl flex items-center justify-between">
          <div className="flex flex-col">
            <span className="text-xs font-semibold text-slate-800 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-indigo-600" /> Explainable AI (RS-XAI)
            </span>
            <span className="text-[11px] text-slate-500">
              Compute exact Shapley attribution, physics backscatter & saliency
            </span>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={includeXai}
              onChange={(e) => setIncludeXai(e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-9 h-5 bg-slate-300 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-indigo-600"></div>
          </label>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 p-3 rounded-lg flex items-center gap-2">
            <Info className="w-4 h-4 shrink-0" />
            <p className="text-sm font-medium">{error}</p>
          </div>
        )}

        <button
          onClick={handleSubmit}
          disabled={isLoading}
          className="w-full bg-primary hover:bg-primaryHover text-white font-medium py-3 rounded-lg transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" /> Routing Inference...
            </>
          ) : (
            <>
              <Send className="w-4 h-4" /> Execute Pipeline
            </>
          )}
        </button>
      </section>

      {/* Right Panel: Results */}
      <section className="panel p-6 lg:col-span-7 flex flex-col h-full min-h-[500px] bg-slate-50">
        <h2 className="text-lg font-semibold flex items-center gap-2 text-textMain border-b border-border pb-4 mb-6">
          <Code2 className="w-5 h-5 text-textMuted" /> Execution Trace
        </h2>

        {!result && !isLoading && (
          <div className="flex-1 flex flex-col items-center justify-center text-textMuted">
            <Database className="w-12 h-12 mb-3 opacity-20" />
            <p className="font-medium text-textMain">Awaiting Telemetry</p>
            <p className="text-sm">Configure task and execute to view trace.</p>
          </div>
        )}

        {isLoading && (
          <div className="flex-1 flex flex-col items-center justify-center text-primary">
            <Loader2 className="w-10 h-10 animate-spin mb-4" />
            <p className="font-medium text-sm">Processing Payload...</p>
          </div>
        )}

        {result && !isLoading && (
          <div className="flex flex-col gap-5 animate-in fade-in duration-300">
            <div className="bg-white border border-border p-5 rounded-xl shadow-sm">
              <h3 className="text-xs uppercase tracking-wider text-textMuted mb-2 font-semibold">User Objective</h3>
              <p className="text-sm font-medium text-textMain">"{result.query}"</p>
            </div>

            {/* AI Textual Intelligence Response */}
            {result.text_response && (
              <div className="bg-white border border-border p-5 rounded-xl shadow-sm">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs uppercase tracking-wider text-primary font-semibold flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" /> Intelligence Synthesis
                  </h3>
                  {result.confidence_score && (
                    <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                      {(result.confidence_score * 100).toFixed(1)}% Confidence
                    </span>
                  )}
                </div>
                <p className="text-textMain text-sm leading-relaxed whitespace-pre-line">
                  {result.text_response}
                </p>
              </div>
            )}

            {/* Vector Layers / Detected Features */}
            {result.vector_layers?.length > 0 && (
              <div className="bg-white border border-border p-5 rounded-xl shadow-sm">
                <h3 className="text-xs uppercase tracking-wider text-textMuted mb-3 font-semibold flex items-center gap-1.5">
                  <Layers className="w-4 h-4 text-blue-600" /> Geospatial Detections & Metrics
                </h3>
                <div className="space-y-2">
                  {result.vector_layers.map((vl, idx) => (
                    <div key={idx} className="bg-slate-50 p-3 rounded-lg border border-slate-200 flex items-center justify-between text-xs">
                      <span className="font-semibold text-slate-800">{vl.layer_name}</span>
                      <div className="flex items-center gap-3">
                        <span className="text-slate-500">{vl.feature_count} features</span>
                        {vl.metrics?.area_hectares !== undefined && (
                          <span className="font-mono bg-blue-100 text-blue-800 px-2 py-0.5 rounded">
                            {vl.metrics.area_hectares.toFixed(2)} ha
                          </span>
                        )}
                        {vl.metrics?.change_percentage !== undefined && (
                          <span className="font-mono bg-amber-100 text-amber-800 px-2 py-0.5 rounded">
                            {vl.metrics.change_percentage.toFixed(1)}% Δ
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                <button
                  onClick={() => navigate('/globe', { state: { vector_layers: result.vector_layers } })}
                  className="mt-3 w-full bg-blue-50 hover:bg-blue-100 text-blue-700 border border-blue-200 font-medium py-2 rounded-lg transition-colors flex items-center justify-center gap-2 text-xs cursor-pointer shadow-2xs"
                >
                  <Globe className="w-3.5 h-3.5 text-blue-600" />
                  View Detected Features in 3D Cesium Globe &rarr;
                </button>
              </div>
            )}

            {/* RS-XAI Explainability Card */}
            {result.xai_explanation && (
              <div className="bg-white border border-indigo-100 p-5 rounded-xl shadow-sm border-l-4 border-l-indigo-600 space-y-4">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-indigo-600" />
                    <h3 className="text-xs uppercase tracking-wider text-indigo-900 font-bold">
                      Visual Evidence & Explainability (RS-XAI)
                    </h3>
                  </div>
                  <div className="flex items-center gap-2">
                    {result.xai_explanation.runtime_ms !== undefined && (
                      <span className="text-[10px] font-mono bg-indigo-50 text-indigo-700 border border-indigo-200 px-2 py-0.5 rounded">
                        {result.xai_explanation.runtime_ms.toFixed(1)} ms
                      </span>
                    )}
                    <span className="text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200 px-2 py-0.5 rounded flex items-center gap-1">
                      <ShieldCheck className="w-3 h-3" /> Zero-VRAM Saliency
                    </span>
                  </div>
                </div>

                <div className="bg-slate-50 border border-slate-200 p-3 rounded-lg text-xs text-slate-700 leading-relaxed font-medium">
                  {result.xai_explanation.summary}
                </div>

                {/* Modality Attribution (Shapley) */}
                {result.xai_explanation.modality_attribution && (
                  <div className="space-y-2 bg-slate-50 p-3 rounded-lg border border-slate-200">
                    <div className="flex justify-between items-center text-xs font-semibold text-slate-700">
                      <span>Modality Reliance (2-Player Shapley)</span>
                      <span className="font-mono text-[11px] text-slate-500">Exact Closed-Form</span>
                    </div>
                    {/* Attribution Bar */}
                    <div className="h-5 w-full bg-slate-200 rounded-full overflow-hidden flex text-[10px] font-bold text-white leading-5 text-center">
                      {result.xai_explanation.modality_attribution.optical !== undefined && (
                        <div
                          style={{ width: `${(result.xai_explanation.modality_attribution.optical * 100).toFixed(1)}%` }}
                          className="bg-blue-600 flex items-center justify-center transition-all duration-500 truncate px-1"
                          title={`Optical: ${(result.xai_explanation.modality_attribution.optical * 100).toFixed(1)}%`}
                        >
                          Optical {(result.xai_explanation.modality_attribution.optical * 100).toFixed(0)}%
                        </div>
                      )}
                      {result.xai_explanation.modality_attribution.sar !== undefined && (
                        <div
                          style={{ width: `${(result.xai_explanation.modality_attribution.sar * 100).toFixed(1)}%` }}
                          className="bg-amber-600 flex items-center justify-center transition-all duration-500 truncate px-1"
                          title={`SAR C-Band: ${(result.xai_explanation.modality_attribution.sar * 100).toFixed(1)}%`}
                        >
                          SAR {(result.xai_explanation.modality_attribution.sar * 100).toFixed(0)}%
                        </div>
                      )}
                    </div>
                    <p className="text-[10px] text-slate-500 italic">
                      Model-internal feature attribution across combinatorial coalitions; not direct causal proof.
                    </p>
                  </div>
                )}

                {/* Physics Microwave Scattering */}
                {result.xai_explanation.physics_rationale?.scattering_distribution_pct && (
                  <div className="space-y-1.5">
                    <span className="text-[11px] uppercase tracking-wider text-slate-600 font-semibold block">
                      Microwave Backscatter Regimes (C-Band σ⁰)
                    </span>
                    <div className="grid grid-cols-3 gap-2 text-center">
                      <div className="bg-sky-50 border border-sky-200 p-2.5 rounded-lg">
                        <span className="text-[10px] text-sky-700 font-medium block">Surface / Specular</span>
                        <span className="text-sm font-bold text-sky-900 font-mono">
                          {result.xai_explanation.physics_rationale.scattering_distribution_pct.surface_specular_pct}%
                        </span>
                        <span className="text-[9px] text-sky-600 block">&lt; -16 dB (Water)</span>
                      </div>
                      <div className="bg-emerald-50 border border-emerald-200 p-2.5 rounded-lg">
                        <span className="text-[10px] text-emerald-700 font-medium block">Volume Canopy</span>
                        <span className="text-sm font-bold text-emerald-900 font-mono">
                          {result.xai_explanation.physics_rationale.scattering_distribution_pct.volume_canopy_pct}%
                        </span>
                        <span className="text-[9px] text-emerald-600 block">[-16, -6] dB</span>
                      </div>
                      <div className="bg-purple-50 border border-purple-200 p-2.5 rounded-lg">
                        <span className="text-[10px] text-purple-700 font-medium block">Double-Bounce</span>
                        <span className="text-sm font-bold text-purple-900 font-mono">
                          {result.xai_explanation.physics_rationale.scattering_distribution_pct.double_bounce_urban_pct}%
                        </span>
                        <span className="text-[9px] text-purple-600 block">&gt; -6 dB (Urban)</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Spectral Sensitivities */}
                {result.xai_explanation.spectral_sensitivity && (
                  <div className="space-y-1.5">
                    <span className="text-[11px] uppercase tracking-wider text-slate-600 font-semibold block">
                      Analytical Spectral Sensitivities (∂Index/∂Band)
                    </span>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(result.xai_explanation.spectral_sensitivity).map(([band, val], idx) => (
                        <div key={idx} className="bg-slate-100 border border-slate-200 px-2.5 py-1 rounded text-xs flex items-center gap-1.5 font-mono text-slate-800">
                          <span className="text-slate-500 font-sans capitalize">{band.replace('_', ' ')}:</span>
                          <span className="font-bold text-indigo-700">{(val * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Interactive Saliency Overlay & Opacity Slider */}
                {result.xai_explanation.heatmap_overlay_base64 && (
                  <div className="space-y-2 bg-slate-900 p-3.5 rounded-xl border border-slate-800 text-white">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold flex items-center gap-1.5 text-slate-200">
                        <Eye className="w-3.5 h-3.5 text-indigo-400" /> Saliency Heatmap Overlay
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="text-[11px] text-slate-400">Opacity: {overlayOpacity}%</span>
                        <input
                          type="range"
                          min="0"
                          max="100"
                          value={overlayOpacity}
                          onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                          className="w-24 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                        />
                      </div>
                    </div>
                    <div className="relative rounded-lg overflow-hidden border border-slate-700 aspect-video max-h-56 flex items-center justify-center bg-black">
                      <img
                        src={result.xai_explanation.heatmap_overlay_base64}
                        alt="Saliency Heatmap"
                        style={{ opacity: overlayOpacity / 100 }}
                        className="w-full h-full object-contain transition-opacity duration-150"
                      />
                    </div>
                  </div>
                )}

                {/* Limitations Caveat */}
                {result.xai_explanation.limitations?.length > 0 && (
                  <div className="text-[10px] text-slate-500 italic bg-amber-50/60 p-2.5 rounded border border-amber-200/50">
                    <b>Scientific Caveats:</b> {result.xai_explanation.limitations.join(' • ')}
                  </div>
                )}
              </div>
            )}

            {/* Execution Trace */}
            {result.execution_trace && (
              <div className="flex flex-col gap-4">
                <div className="bg-blue-50 border border-blue-100 p-5 rounded-xl border-l-4 border-l-primary">
                  <h3 className="text-xs uppercase tracking-wider text-primary mb-1 font-semibold">Selected Tool</h3>
                  <p className="text-base font-mono text-blue-900 font-semibold">{result.execution_trace.selected_tool || result.task_category}</p>
                </div>

                <div className="bg-white border border-border p-5 rounded-xl shadow-sm">
                  <h3 className="text-xs uppercase tracking-wider text-textMuted mb-2 font-semibold">Agent Reasoning</h3>
                  <p className="text-textMain text-sm leading-relaxed">
                    {result.execution_trace.reasoning}
                  </p>
                </div>

                <div className="bg-white border border-border p-5 rounded-xl shadow-sm">
                  <h3 className="text-xs uppercase tracking-wider text-textMuted mb-3 font-semibold">Execution Artifacts</h3>
                  <div className="flex flex-wrap gap-2">
                    {result.execution_trace.inputs?.length > 0 ? (
                      result.execution_trace.inputs.map((input, idx) => (
                        <div key={idx} className="bg-slate-100 text-xs px-3 py-1.5 rounded flex items-center gap-2 font-mono text-slate-700 border border-slate-200">
                          <FileImage className="w-3 h-3 text-slate-400" />
                          {input}
                        </div>
                      ))
                    ) : (
                      <span className="text-sm text-textMuted italic">No artifacts extracted.</span>
                    )}
                  </div>
                </div>

                {/* 1-Click Intelligence Dossier PDF Export */}
                {result.trace_id && (
                  <button
                    onClick={() => handleDownloadDossier(result.trace_id)}
                    disabled={isDownloading}
                    className="w-full bg-slate-900 hover:bg-slate-800 text-white font-medium py-3 rounded-lg transition-colors shadow-sm flex items-center justify-center gap-2 text-sm"
                  >
                    {isDownloading ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" /> Compiling ReportLab PDF...
                      </>
                    ) : (
                      <>
                        <FileDown className="w-4 h-4 text-emerald-400" /> Download Verified Intelligence Dossier (PDF)
                      </>
                    )}
                  </button>
                )}
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
