import React from 'react';
import { Share2, RefreshCcw } from 'lucide-react';
import { graphApi } from '../../services/api';
import './GraphView.css';

interface GraphViewProps {
  limit?: number;
}

const GraphView: React.FC<GraphViewProps> = ({ limit: initialLimit = 100 }) => {
  const [limit, setLimit] = React.useState<number | string>(initialLimit);
  const [searchLimit, setSearchLimit] = React.useState(initialLimit);
  const [refreshKey, setRefreshKey] = React.useState(0);
  
  const visualizationUrl = `${graphApi.getVisualizationUrl(searchLimit)}&t=${refreshKey}`;

  const handleSearch = () => {
    const finalLimit = typeof limit === 'string' ? parseInt(limit) || 10 : limit;
    setSearchLimit(finalLimit);
    setRefreshKey(prev => prev + 1);
  };

  const handleRefresh = () => {
    setRefreshKey(prev => prev + 1);
  };

  return (
    <div className="graph-view-container">
      <div className="graph-view-header glass">
        <div className="header-left">
          <Share2 size={20} className="accent-icon" />
          <h3>Knowledge Graph Explorer</h3>
        </div>
        <div className="header-actions">
          <div className="limit-selector">
            <span>Limit:</span>
            <input 
              type="number" 
              min="1" 
              max="1000" 
              value={limit} 
              onChange={(e) => setLimit(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              className="limit-input"
              placeholder="100"
            />
            <button className="search-btn" onClick={handleSearch}>
              Search
            </button>
          </div>
          <button className="action-btn" onClick={handleRefresh}>
            <RefreshCcw size={16} />
            <span>Refresh</span>
          </button>
        </div>
      </div>
      
      <div className="graph-iframe-wrapper glass">
        <iframe 
          key={refreshKey + searchLimit}
          title="Neo4j Graph Visualization"
          src={visualizationUrl}
          width="100%"
          height="100%"
          style={{ border: 'none' }}
        />
      </div>
    </div>
  );
};

export default GraphView;
