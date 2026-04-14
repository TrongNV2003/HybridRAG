import React, { useState } from 'react';
import { Search, Loader2, Database, AlertCircle, CheckCircle2 } from 'lucide-react';
import { indexingApi, IndexingResponse } from '../../services/api';
import './IndexingPanel.css';

interface IndexingPanelProps {
  onIndexingComplete: (stats: IndexingResponse) => void;
}

const IndexingPanel: React.FC<IndexingPanelProps> = ({ onIndexingComplete }) => {
  const [query, setQuery] = useState('Elizabeth I');
  const [maxDocs, setMaxDocs] = useState(10);
  const [clearOld, setClearOld] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [result, setResult] = useState<IndexingResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleIndex = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isIndexing) return;

    setIsIndexing(true);
    setError(null);
    setResult(null);

    try {
      const response = await indexingApi.wikipedia(query, maxDocs, clearOld);
      setResult(response.data);
      onIndexingComplete(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Đã có lỗi xảy ra trong quá trình indexing.');
    } finally {
      setIsIndexing(false);
    }
  };

  return (
    <div className="indexing-container">
      <div className="indexing-header">
        <Database size={24} className="accent-icon" />
        <h2>Knowledge Base Ingestion</h2>
        <p>Thêm dữ liệu mới vào HybridRAG từ Wikipedia.</p>
      </div>

      <form className="indexing-form glass" onSubmit={handleIndex}>
        <div className="form-group">
          <label>Wikipedia Search Query</label>
          <div className="input-with-icon">
            <Search size={18} />
            <input 
              type="text" 
              value={query} 
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., Vietnam history, Quantum computing..."
              disabled={isIndexing}
            />
          </div>
        </div>

        <div className="form-group">
          <label>Max Documents to Load</label>
          <input 
            type="number" 
            min="1" 
            max="50" 
            value={maxDocs} 
            onChange={(e) => setMaxDocs(parseInt(e.target.value) || 1)}
            disabled={isIndexing}
            className="indexing-number-input"
          />
        </div>

        <div className="form-group checkbox-group">
          <label className="checkbox-container">
            <input 
              type="checkbox" 
              checked={clearOld} 
              onChange={(e) => setClearOld(e.target.checked)}
              disabled={isIndexing}
            />
            <span className="checkbox-label">Clear existing graph data before indexing</span>
          </label>
        </div>

        <button type="submit" className="index-btn" disabled={isIndexing}>
          {isIndexing ? (
            <>
              <Loader2 size={18} className="spin" />
              <span>Indexing...</span>
            </>
          ) : (
            <span>Start Indexing</span>
          )}
        </button>
      </form>

      {error && (
        <div className="message-box error">
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {result && (
        <div className="message-box success">
          <CheckCircle2 size={20} />
          <div>
            <h4>Indexing Complete!</h4>
            <div className="result-stats">
              <div className="res-stat"><span>Entities:</span> {result.entities_count}</div>
              <div className="res-stat"><span>Relationships:</span> {result.relationships_count}</div>
              <div className="res-stat"><span>Chunks:</span> {result.chunks_count}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default IndexingPanel;
