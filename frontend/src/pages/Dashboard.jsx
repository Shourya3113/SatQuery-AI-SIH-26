import { useState, useEffect } from 'react';
import axios from 'axios';
import { ArrowRight, Image as ImageIcon, CheckCircle, Clock, Loader2 } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Dashboard() {
  const [statsData, setStatsData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const { data } = await axios.get('http://localhost:8000/api/stats');
      setStatsData(data);
    } catch (err) {
      console.error('Failed to fetch stats', err);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-12 text-primary">
        <Loader2 className="w-8 h-8 animate-spin" />
      </div>
    );
  }

  const stats = [
    { label: 'Total Images Processed', value: statsData?.total_processed || 0, icon: ImageIcon, color: 'text-blue-600', bg: 'bg-blue-50' },
    { label: 'Success Rate', value: statsData?.success_rate || '0%', icon: CheckCircle, color: 'text-green-600', bg: 'bg-green-50' },
    { label: 'Avg Inference Time', value: statsData?.avg_time || '0s', icon: Clock, color: 'text-orange-600', bg: 'bg-orange-50' },
  ];

  return (
    <div className="flex flex-col gap-8 animate-in fade-in duration-300">
      <div>
        <h1 className="text-2xl font-bold text-textMain">Welcome back, Analyst</h1>
        <p className="text-textMuted mt-1">Here's an overview of your remote sensing operations.</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {stats.map((stat, idx) => {
          const Icon = stat.icon;
          return (
            <div key={idx} className="panel p-6 flex items-center gap-4 hover:shadow-md transition-shadow">
              <div className={`p-4 rounded-lg ${stat.bg}`}>
                <Icon className={`w-6 h-6 ${stat.color}`} />
              </div>
              <div>
                <p className="text-sm font-medium text-textMuted">{stat.label}</p>
                <p className="text-2xl font-bold text-textMain mt-1">{stat.value}</p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Quick Action */}
      <div className="panel p-8 bg-gradient-to-r from-slate-50 to-white flex flex-col md:flex-row items-center justify-between gap-6 border-l-4 border-l-primary">
        <div>
          <h2 className="text-lg font-semibold text-textMain">Start a new analysis</h2>
          <p className="text-textMuted mt-1">Upload GeoTIFF imagery and use natural language to query the agent.</p>
        </div>
        <Link
          to="/analysis"
          className="flex items-center gap-2 bg-primary hover:bg-primaryHover text-white px-6 py-3 rounded-lg font-medium transition-colors whitespace-nowrap"
        >
          Go to Workspace <ArrowRight className="w-4 h-4" />
        </Link>
      </div>

      {/* Recent Activity */}
      <div className="panel p-6">
        <h3 className="font-semibold text-textMain mb-4">Recent Queries</h3>
        {statsData?.recent_queries?.length > 0 ? (
          <div className="divide-y divide-border">
            {statsData.recent_queries.map((q, idx) => (
              <div key={idx} className="py-4 flex items-center justify-between">
                <span className="text-sm text-textMain font-medium">"{q}"</span>
                <span className="text-xs text-textMuted bg-slate-100 px-2 py-1 rounded border border-slate-200">Processed</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="py-8 text-center border-2 border-dashed border-border rounded-lg bg-slate-50">
            <p className="text-sm text-textMuted font-medium">No recent queries.</p>
            <p className="text-xs text-slate-400 mt-1">Navigate to the workspace to run your first analysis.</p>
          </div>
        )}
      </div>
    </div>
  );
}
