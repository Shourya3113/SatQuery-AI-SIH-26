import { useState, useEffect } from 'react';
import axios from 'axios';
import { KeyRound, CheckCircle, Loader2, AlertCircle, Globe } from 'lucide-react';
import { API_BASE } from '../config';

export default function Settings() {
  const [apiKey, setApiKey] = useState('');
  const [currentMaskedKey, setCurrentMaskedKey] = useState('');
  const [hasKey, setHasKey] = useState(false);

  const [cesiumToken, setCesiumToken] = useState('');
  const [currentMaskedCesiumToken, setCurrentMaskedCesiumToken] = useState('');
  const [hasCesiumKey, setHasCesiumKey] = useState(false);
  
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [isSavingCesium, setIsSavingCesium] = useState(false);
  
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);

  const [cesiumMessage, setCesiumMessage] = useState(null);
  const [cesiumError, setCesiumError] = useState(null);

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      const { data } = await axios.get(`${API_BASE}/api/settings`);
      setCurrentMaskedKey(data.google_api_key_masked);
      setHasKey(data.has_key);
      setCurrentMaskedCesiumToken(data.cesium_ion_token_masked);
      setHasCesiumKey(data.has_cesium_key);
    } catch (err) {
      console.error(err);
      setError('Failed to fetch settings');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSave = async (e) => {
    e.preventDefault();
    if (!apiKey.trim()) return setError('Please enter a valid API key.');
    
    setIsSaving(true);
    setError(null);
    setMessage(null);
    
    try {
      const { data } = await axios.post(`${API_BASE}/api/settings`, {
        google_api_key: apiKey
      });
      setMessage(data.message);
      setApiKey('');
      await fetchSettings();
    } catch (err) {
      console.error(err);
      setError('Failed to update API key.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleSaveCesium = async (e) => {
    e.preventDefault();
    if (!cesiumToken.trim()) return setCesiumError('Please enter a valid Cesium Ion token.');
    
    setIsSavingCesium(true);
    setCesiumError(null);
    setCesiumMessage(null);
    
    try {
      const { data } = await axios.post(`${API_BASE}/api/settings`, {
        cesium_ion_token: cesiumToken
      });
      setCesiumMessage(data.message);
      setCesiumToken('');
      await fetchSettings();
    } catch (err) {
      console.error(err);
      setCesiumError('Failed to update Cesium Ion token.');
    } finally {
      setIsSavingCesium(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-12 text-primary">
        <Loader2 className="w-8 h-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto flex flex-col gap-8 animate-in fade-in duration-300">
      <div>
        <h1 className="text-2xl font-bold text-textMain">System Configuration</h1>
        <p className="text-textMuted mt-1">Manage your platform integrations, API keys, and 3D geospatial environment.</p>
      </div>

      {/* Google Gemini API Key Panel */}
      <div className="panel p-8">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-blue-50 rounded-lg text-primary">
            <KeyRound className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-textMain">Google Gemini API Key</h2>
            <p className="text-xs text-textMuted">Used for natural language query decomposition and vision-language agent routing.</p>
          </div>
        </div>

        <div className="mb-6 p-4 rounded-lg border border-border bg-slate-50 flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-textMain">Current Status</p>
            <p className="text-sm text-textMuted mt-1">
              {hasKey 
                ? `Key configured (${currentMaskedKey})` 
                : 'No API key configured.'}
            </p>
          </div>
          {hasKey ? (
            <span className="flex items-center gap-1 text-sm font-medium text-green-600 bg-green-50 px-3 py-1 rounded-full border border-green-200">
              <CheckCircle className="w-4 h-4" /> Active
            </span>
          ) : (
            <span className="flex items-center gap-1 text-sm font-medium text-orange-600 bg-orange-50 px-3 py-1 rounded-full border border-orange-200">
              <AlertCircle className="w-4 h-4" /> Missing
            </span>
          )}
        </div>

        <form onSubmit={handleSave} className="flex flex-col gap-4">
          <div>
            <label htmlFor="apiKey" className="block text-sm font-medium text-textMain mb-2">Update API Key</label>
            <input
              type="password"
              id="apiKey"
              placeholder="Paste your new GOOGLE_API_KEY here"
              className="w-full bg-white border border-border rounded-lg p-3 text-textMain focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all text-sm shadow-inner-soft"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
            <p className="text-xs text-textMuted mt-2">Get your free key from <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noreferrer" className="text-primary hover:underline">Google AI Studio</a>. Updates local environment.</p>
          </div>

          {error && (
            <p className="text-sm text-red-600 font-medium">{error}</p>
          )}
          {message && (
            <p className="text-sm text-green-600 font-medium">{message}</p>
          )}

          <div className="flex justify-end mt-2">
            <button
              type="submit"
              disabled={isSaving || !apiKey.trim()}
              className="bg-primary hover:bg-primaryHover text-white px-6 py-2.5 rounded-lg font-medium transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 text-sm cursor-pointer"
            >
              {isSaving && <Loader2 className="w-4 h-4 animate-spin" />}
              Save API Key
            </button>
          </div>
        </form>
      </div>

      {/* Cesium Ion 3D Environment Panel */}
      <div className="panel p-8">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-indigo-50 rounded-lg text-indigo-600">
            <Globe className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-textMain">Cesium Ion 3D Digital Globe Integration</h2>
            <p className="text-xs text-textMuted">Enables high-resolution Cesium World Terrain, Bing aerial imagery, and 3D photogrammetry.</p>
          </div>
        </div>

        <div className="mb-6 p-4 rounded-lg border border-border bg-slate-50 flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-textMain">Cesium Ion Status</p>
            <p className="text-sm text-textMuted mt-1">
              {hasCesiumKey 
                ? `Custom Ion Token Active (${currentMaskedCesiumToken})` 
                : 'Using Open Basemap Fallback (ESRI / OpenStreetMap / WGS84 Ellipsoid)'}
            </p>
          </div>
          {hasCesiumKey ? (
            <span className="flex items-center gap-1 text-sm font-medium text-green-600 bg-green-50 px-3 py-1 rounded-full border border-green-200">
              <CheckCircle className="w-4 h-4" /> Ion Active
            </span>
          ) : (
            <span className="flex items-center gap-1 text-sm font-medium text-blue-600 bg-blue-50 px-3 py-1 rounded-full border border-blue-200">
              <Globe className="w-4 h-4" /> Open Fallback Ready
            </span>
          )}
        </div>

        <form onSubmit={handleSaveCesium} className="flex flex-col gap-4">
          <div>
            <label htmlFor="cesiumToken" className="block text-sm font-medium text-textMain mb-2">Cesium Ion Access Token</label>
            <input
              type="password"
              id="cesiumToken"
              placeholder="Paste your CESIUM_ION_ACCESS_TOKEN here"
              className="w-full bg-white border border-border rounded-lg p-3 text-textMain focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all text-sm shadow-inner-soft"
              value={cesiumToken}
              onChange={(e) => setCesiumToken(e.target.value)}
            />
            <p className="text-xs text-textMuted mt-2">
              Create a free token at <a href="https://ion.cesium.com/tokens" target="_blank" rel="noreferrer" className="text-primary hover:underline">cesium.com/ion</a>. Even without a token, the 3D globe works automatically using open GIS imagery.
            </p>
          </div>

          {cesiumError && (
            <p className="text-sm text-red-600 font-medium">{cesiumError}</p>
          )}
          {cesiumMessage && (
            <p className="text-sm text-green-600 font-medium">{cesiumMessage}</p>
          )}

          <div className="flex justify-end mt-2">
            <button
              type="submit"
              disabled={isSavingCesium || !cesiumToken.trim()}
              className="bg-indigo-600 hover:bg-indigo-500 text-white px-6 py-2.5 rounded-lg font-medium transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 text-sm cursor-pointer"
            >
              {isSavingCesium && <Loader2 className="w-4 h-4 animate-spin" />}
              Save Cesium Token
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
