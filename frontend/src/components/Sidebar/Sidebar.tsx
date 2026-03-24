import React from 'react';
import { 
  MessageSquare, 
  Database, 
  Share2, 
  TrendingUp, 
  Settings as SettingsIcon,
  HelpCircle,
  Plus
} from 'lucide-react';
import './Sidebar.css';

export type NavItem = 'chat' | 'indexing' | 'graph' | 'stats' | 'settings';

interface SidebarProps {
  activeItem: NavItem;
  onNavigate: (item: NavItem) => void;
  onNewSession?: () => void;
  stats?: {
    entities: number;
    rels: number;
    chunks: number;
  };
}

const Sidebar: React.FC<SidebarProps> = ({ activeItem, onNavigate, onNewSession, stats }) => {
  return (
    <div className="sidebar-inner">
      <div className="sidebar-header">
        <div className="logo">
          <div className="logo-icon">H</div>
          <span className="logo-text gradient-text">HybridRAG</span>
        </div>
        <button className="new-chat-btn" onClick={onNewSession}>
          <Plus size={18} />
          <span>New Session</span>
        </button>
      </div>

      <nav className="sidebar-nav">
        <button 
          className={`nav-item ${activeItem === 'chat' ? 'active' : ''}`}
          onClick={() => onNavigate('chat')}
        >
          <MessageSquare size={20} />
          <span>Chat Console</span>
        </button>
        <button 
          className={`nav-item ${activeItem === 'indexing' ? 'active' : ''}`}
          onClick={() => onNavigate('indexing')}
        >
          <Database size={20} />
          <span>Knowledge Base</span>
        </button>
        <button 
          className={`nav-item ${activeItem === 'graph' ? 'active' : ''}`}
          onClick={() => onNavigate('graph')}
        >
          <Share2 size={20} />
          <span>Graph Explorer</span>
        </button>
        <button 
          className={`nav-item ${activeItem === 'stats' ? 'active' : ''}`}
          onClick={() => onNavigate('stats')}
        >
          <TrendingUp size={20} />
          <span>System Insights</span>
        </button>
        <button 
          className={`nav-item ${activeItem === 'settings' ? 'active' : ''}`}
          onClick={() => onNavigate('settings')}
        >
          <SettingsIcon size={20} />
          <span>Config RAG</span>
        </button>
      </nav>

      <div className="sidebar-footer">
        {stats && (
          <div className="stats-box">
            <div className="stat-row">
              <span className="stat-label">Entities</span>
              <span className="stat-value">{stats.entities}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Relationships</span>
              <span className="stat-value">{stats.rels}</span>
            </div>
          </div>
        )}
        <div className="footer-actions">
          <button className="footer-btn" onClick={() => onNavigate('settings')} title="Settings">
            <SettingsIcon size={18} />
          </button>
          <button className="footer-btn" title="Help"><HelpCircle size={18} /></button>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
