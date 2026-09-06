import { useState, useEffect } from 'react';
import axios from 'axios';
import { KeyRound, CheckCircle, Loader2, AlertCircle } from 'lucide-react';

export default function Settings() {
  const [apiKey, setApiKey] = useState('');
  const [currentMaskedKey, setCurrentMaskedKey] = useState('');
  const [hasKey, setHasKey] = useState(false);
  
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      const { data } = await axios.get('http://localhost:8000/api/settings');
      setCurrentMaskedKey(data.google_api_key_masked);
      setHasKey(data.has_key);
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
      const { data } = await axios.post('http://localhost:8000/api/settings', {
        google_api_key: apiKey
      });
      setMessage(data.message);
      setApiKey(''); // Clear input after save
      await fetchSettings(); // Refresh status
    } catch (err) {
      console.error(err);
      setError('Failed to update API key.');
    } finally {
      setIsSaving(false);
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
    <div className="max-w-3xl mx-auto flex flex-col gap-8">
      <div>
        <h1 className="text-2xl font-bold text-textMain">System Configuration</h1>
        <p className="text-textMuted mt-1">Manage your platform integrations and API keys.</p>
      </div>

      <div className="panel p-8">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-blue-50 rounded-lg text-primary">
            <KeyRound className="w-5 h-5" />
          </div>
          <h2 className="text-lg font-semibold text-textMain">Google Gemini API Key</h2>
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
            <p className="text-xs text-textMuted mt-2">Get your free key from <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noreferrer" className="text-primary hover:underline">Google AI Studio</a>. This will update your local .env file.</p>
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
              className="bg-primary hover:bg-primaryHover text-white px-6 py-2.5 rounded-lg font-medium transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 text-sm"
            >
              {isSaving && <Loader2 className="w-4 h-4 animate-spin" />}
              Save Configuration
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
