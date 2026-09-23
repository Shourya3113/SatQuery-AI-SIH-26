import React, { useState } from 'react';
import { 
  Sparkles, 
  ShieldCheck, 
  Eye, 
  HelpCircle, 
  AlertTriangle, 
  Info, 
  Target, 
  Layers, 
  Sliders 
} from 'lucide-react';

export default function ExplainabilityHUD({ 
  explanation, 
  overlayOpacity = 65, 
  setOverlayOpacity,
  className = "" 
}) {
  const [showTooltip, setShowTooltip] = useState(false);
  const [showLimitations, setShowLimitations] = useState(false);

  if (!explanation) return null;

  const isPartialOrFallback = explanation.status === 'partial' || explanation.status === 'fallback';

  return (
    <div className={`bg-white border border-indigo-100 p-5 rounded-xl shadow-sm border-l-4 border-l-indigo-600 space-y-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-indigo-600" />
          <h3 className="text-xs uppercase tracking-wider text-indigo-900 font-bold">
            Visual Evidence & Explainability (RS-XAI)
          </h3>
          {explanation.status && (
            <span className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded font-semibold ${
              explanation.status === 'full' 
                ? 'bg-emerald-100 text-emerald-800' 
                : 'bg-amber-100 text-amber-800'
            }`}>
              {explanation.status}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {explanation.runtime_ms !== undefined && (
            <span className="text-[10px] font-mono bg-indigo-50 text-indigo-700 border border-indigo-200 px-2 py-0.5 rounded">
              {explanation.runtime_ms.toFixed(1)} ms
            </span>
          )}
          <span className="text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200 px-2 py-0.5 rounded flex items-center gap-1">
            <ShieldCheck className="w-3 h-3" /> Zero-VRAM Saliency
          </span>
        </div>
      </div>

      {/* Partial / Fallback Alert if applicable */}
      {isPartialOrFallback && (
        <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg flex items-start gap-2 text-xs text-amber-900">
          <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-bold">Constrained / Fallback Mode Active</span>
            <p className="text-amber-800">{explanation.fallback_reason || "Some high-level attribution routines were skipped."}</p>
            {explanation.unavailable_methods?.length > 0 && (
              <span className="text-[11px] text-amber-700 block">
                Unavailable: {explanation.unavailable_methods.join(", ")}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Natural Language Summary */}
      <div className="bg-slate-50 border border-slate-200 p-3 rounded-lg text-xs text-slate-700 leading-relaxed font-medium">
        {explanation.summary}
      </div>

      {/* Modality Attribution (2-Player Shapley) */}
      {explanation.modality_attribution && (
        <div className="space-y-2 bg-slate-50 p-3 rounded-lg border border-slate-200">
          <div className="flex justify-between items-center text-xs font-semibold text-slate-700">
            <div className="flex items-center gap-1.5">
              <span>Modality Reliance (2-Player Shapley)</span>
              <button 
                type="button"
                onClick={() => setShowTooltip(!showTooltip)}
                className="text-slate-400 hover:text-slate-600 focus:outline-none"
                title="Click for scientific definition"
              >
                <HelpCircle className="w-3.5 h-3.5" />
              </button>
            </div>
            <span className="font-mono text-[11px] text-slate-500">Exact Closed-Form (n=2)</span>
          </div>

          {showTooltip && (
            <div className="p-2.5 bg-indigo-50 border border-indigo-200 rounded text-[11px] text-indigo-900 leading-relaxed">
              <strong>Shapley Attribution Formula:</strong> Calculated via exact 2-player cooperative game theory across all 4 coalition subsets. Quantifies model internal feature reliance; not direct physical causality.
            </div>
          )}

          {/* Attribution Bar */}
          <div className="h-5 w-full bg-slate-200 rounded-full overflow-hidden flex text-[10px] font-bold text-white leading-5 text-center">
            {explanation.modality_attribution.optical !== undefined && (
              <div
                style={{ width: `${(explanation.modality_attribution.optical * 100).toFixed(1)}%` }}
                className="bg-blue-600 flex items-center justify-center transition-all duration-500 truncate px-1"
                title={`Optical: ${(explanation.modality_attribution.optical * 100).toFixed(1)}%`}
              >
                Optical {(explanation.modality_attribution.optical * 100).toFixed(0)}%
              </div>
            )}
            {explanation.modality_attribution.sar !== undefined && (
              <div
                style={{ width: `${(explanation.modality_attribution.sar * 100).toFixed(1)}%` }}
                className="bg-amber-600 flex items-center justify-center transition-all duration-500 truncate px-1"
                title={`SAR C-Band: ${(explanation.modality_attribution.sar * 100).toFixed(1)}%`}
              >
                SAR {(explanation.modality_attribution.sar * 100).toFixed(0)}%
              </div>
            )}
            {explanation.modality_attribution.temporal_t1 !== undefined && (
              <div
                style={{ width: `${(explanation.modality_attribution.temporal_t1 * 100).toFixed(1)}%` }}
                className="bg-indigo-600 flex items-center justify-center transition-all duration-500 truncate px-1"
              >
                T1 {(explanation.modality_attribution.temporal_t1 * 100).toFixed(0)}%
              </div>
            )}
            {explanation.modality_attribution.temporal_t2 !== undefined && (
              <div
                style={{ width: `${(explanation.modality_attribution.temporal_t2 * 100).toFixed(1)}%` }}
                className="bg-rose-600 flex items-center justify-center transition-all duration-500 truncate px-1"
              >
                T2 {(explanation.modality_attribution.temporal_t2 * 100).toFixed(0)}%
              </div>
            )}
          </div>
          <p className="text-[10px] text-slate-500 italic">
            Model-internal feature attribution across combinatorial coalitions; not direct causal proof.
          </p>
        </div>
      )}

      {/* Physics Microwave Scattering Regimes */}
      {explanation.physics_rationale?.scattering_distribution_pct && (
        <div className="space-y-1.5">
          <span className="text-[11px] uppercase tracking-wider text-slate-600 font-semibold block">
            Microwave Backscatter Regimes (Sentinel-1 C-Band σ⁰)
          </span>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="bg-sky-50 border border-sky-200 p-2.5 rounded-lg">
              <span className="text-[10px] text-sky-700 font-medium block">Surface / Specular</span>
              <span className="text-sm font-bold text-sky-900 font-mono">
                {explanation.physics_rationale.scattering_distribution_pct.surface_specular_pct}%
              </span>
              <span className="text-[9px] text-sky-600 block">&lt; -16 dB (Water)</span>
            </div>
            <div className="bg-emerald-50 border border-emerald-200 p-2.5 rounded-lg">
              <span className="text-[10px] text-emerald-700 font-medium block">Volume Canopy</span>
              <span className="text-sm font-bold text-emerald-900 font-mono">
                {explanation.physics_rationale.scattering_distribution_pct.volume_canopy_pct}%
              </span>
              <span className="text-[9px] text-emerald-600 block">[-16, -6] dB</span>
            </div>
            <div className="bg-purple-50 border border-purple-200 p-2.5 rounded-lg">
              <span className="text-[10px] text-purple-700 font-medium block">Double-Bounce</span>
              <span className="text-sm font-bold text-purple-900 font-mono">
                {explanation.physics_rationale.scattering_distribution_pct.double_bounce_urban_pct}%
              </span>
              <span className="text-[9px] text-purple-600 block">&gt; -6 dB (Urban)</span>
            </div>
          </div>
        </div>
      )}

      {/* Spatial Localization / Pointing Game */}
      {explanation.localization_metrics && (
        <div className="space-y-1.5 bg-slate-50 p-3 rounded-lg border border-slate-200">
          <div className="flex justify-between items-center text-xs font-semibold text-slate-700">
            <span className="flex items-center gap-1.5">
              <Target className="w-3.5 h-3.5 text-indigo-600" /> Saliency Localization Metrics
            </span>
            <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-100 text-emerald-800">
              POINTING HIT: {explanation.localization_metrics.pointing_game_hit ? "YES" : "NO"}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs pt-1">
            <div className="bg-white p-2 rounded border border-slate-200">
              <span className="text-[10px] text-slate-500 block">Energy in Mask Ratio (EIMR)</span>
              <span className="font-bold text-slate-800 font-mono">
                {(explanation.localization_metrics.energy_in_mask_ratio * 100).toFixed(1)}%
              </span>
            </div>
            <div className="bg-white p-2 rounded border border-slate-200">
              <span className="text-[10px] text-slate-500 block">Top-20% Saliency IoU</span>
              <span className="font-bold text-slate-800 font-mono">
                {(explanation.localization_metrics.saliency_mask_iou * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Analytical Spectral Sensitivities */}
      {explanation.spectral_sensitivity && (
        <div className="space-y-1.5">
          <span className="text-[11px] uppercase tracking-wider text-slate-600 font-semibold block">
            Analytical Spectral Sensitivities (∂Index/∂Band)
          </span>
          <div className="flex flex-wrap gap-2">
            {Object.entries(explanation.spectral_sensitivity).map(([band, val], idx) => (
              <div key={idx} className="bg-slate-100 border border-slate-200 px-2.5 py-1 rounded text-xs flex items-center gap-1.5 font-mono text-slate-800">
                <span className="text-slate-500 font-sans capitalize">{band.replace('_', ' ')}:</span>
                <span className="font-bold text-indigo-700">{(val * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Interactive Saliency Overlay & Opacity Slider */}
      {explanation.heatmap_overlay_base64 && (
        <div className="space-y-2 bg-slate-900 p-3.5 rounded-xl border border-slate-800 text-white">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold flex items-center gap-1.5 text-slate-200">
              <Eye className="w-3.5 h-3.5 text-indigo-400" /> Saliency Heatmap Overlay
            </span>
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-slate-400">Opacity: {overlayOpacity}%</span>
              {setOverlayOpacity && (
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={overlayOpacity}
                  onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                  className="w-24 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
              )}
            </div>
          </div>

          <div className="relative rounded-lg overflow-hidden border border-slate-800 bg-slate-950 flex items-center justify-center p-2">
            <img
              src={explanation.heatmap_overlay_base64}
              alt="RS-XAI Saliency Overlay"
              style={{ opacity: overlayOpacity / 100.0 }}
              className="max-h-60 w-auto object-contain rounded transition-opacity duration-200"
            />
          </div>

          {/* Colormap Color Bar Legend */}
          <div className="flex items-center justify-between text-[10px] text-slate-400 pt-1">
            <span>Low Relevance</span>
            <div className="h-2 w-44 rounded bg-gradient-to-r from-blue-600 via-cyan-400 via-green-400 via-yellow-400 to-red-600"></div>
            <span>High Salience</span>
          </div>
        </div>
      )}

      {/* Limitations / Assumptions Toggle */}
      {explanation.limitations && explanation.limitations.length > 0 && (
        <div className="pt-1">
          <button
            type="button"
            onClick={() => setShowLimitations(!showLimitations)}
            className="text-[11px] text-slate-500 hover:text-slate-700 flex items-center gap-1 font-medium"
          >
            <Info className="w-3 h-3" />
            {showLimitations ? "Hide Assumptions & Limitations" : "View Scientific Assumptions & Boundaries"}
          </button>
          {showLimitations && (
            <ul className="mt-2 space-y-1 pl-4 list-disc text-[11px] text-slate-600 bg-slate-50 p-2.5 rounded border border-slate-200">
              {explanation.limitations.map((lim, idx) => (
                <li key={idx}>{lim}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
