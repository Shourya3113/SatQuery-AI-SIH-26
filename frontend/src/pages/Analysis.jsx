import { useState, useRef } from 'react';
import axios from 'axios';
import { UploadCloud, FileImage, X, Loader2, Send, Database, Info, Code2 } from 'lucide-react';

export default function Analysis() {
  const [files, setFiles] = useState([]);
  const [query, setQuery] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  const fileInputRef = useRef(null);

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
    files.forEach((file) => formData.append('files', file));

    try {
      const { data } = await axios.post('http://localhost:8000/api/query', formData, {
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
          <div className="space-y-2">
            {files.map((file, i) => (
              <div key={i} className="flex items-center justify-between bg-white p-3 rounded-lg border border-border shadow-sm group/item">
                <div className="flex items-center gap-3 overflow-hidden">
                  <FileImage className="w-4 h-4 text-primary shrink-0" />
                  <span className="truncate text-sm text-textMain font-medium">{file.name}</span>
                </div>
                <button onClick={() => removeFile(i)} className="text-slate-400 hover:text-red-500 transition-colors">
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
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

            {result.execution_trace && (
              <div className="flex flex-col gap-4">
                <div className="bg-blue-50 border border-blue-100 p-5 rounded-xl border-l-4 border-l-primary">
                  <h3 className="text-xs uppercase tracking-wider text-primary mb-1 font-semibold">Selected Tool</h3>
                  <p className="text-base font-mono text-blue-900 font-semibold">{result.execution_trace.selected_tool}</p>
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
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
