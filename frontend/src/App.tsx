import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Layout from './components/Layout/Layout';
import Sidebar, { NavItem } from './components/Sidebar/Sidebar';
import Chat, { Message } from './components/Chat/Chat';
import IndexingPanel from './components/Indexing/IndexingPanel';
import GraphView from './components/Graph/GraphView';
import SparqlPanel from './components/Sparql/SparqlPanel';
import SourcePanel from './components/Chat/SourcePanel';
import { queryApi, graphApi, backupApi } from './services/api';
import { Download, Upload, Loader2 } from 'lucide-react';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState<NavItem>('chat');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedMessage, setSelectedMessage] = useState<Message | null>(null);
  const [stats, setStats] = useState({ entities: 0, rels: 0, chunks: 0 });
  const [notification, setNotification] = useState<{ message: string, type: 'success' | 'error' } | null>(null);
  const [querySettings, setQuerySettings] = useState({
    top_k: 5,
    threshold: 0.5,
    graph_limit: 10,
    strategy: 'full_hybrid' as 'full_hybrid' | 'semantic_hybrid' | 'dense' | 'graph'
  });

  const fetchStats = async () => {
    try {
      const response = await graphApi.getStats();
      setStats({
        entities: response.data.entities_count,
        rels: response.data.relationships_count,
        chunks: response.data.chunks_count
      });
    } catch (err) {
      console.error('Failed to fetch stats', err);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      let response;
      const { strategy, top_k, threshold, graph_limit } = querySettings;

      switch (strategy) {
        case 'semantic_hybrid':
          response = await queryApi.semanticHybrid(content, top_k, threshold);
          break;
        case 'dense':
          response = await queryApi.dense(content, top_k, threshold);
          break;
        case 'graph':
          response = await queryApi.graph(content, graph_limit);
          break;
        case 'full_hybrid':
        default:
          response = await queryApi.fullHybrid(content, top_k, threshold, graph_limit);
          break;
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.data.answer,
        timestamp: new Date(),
        sources: {
          graph: response.data.graph_context,
          chunks: response.data.chunk_context
        }
      };
      setMessages(prev => [...prev, assistantMessage]);
      setSelectedMessage(assistantMessage);
    } catch (err) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Xin lỗi, tôi đã gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng kiểm tra kết nối server.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleIndexingComplete = () => {
    fetchStats();
  };

  const handleNewSession = () => {
    setMessages([]);
    setActiveTab('chat');
  };

  return (
    <Layout
      sidebar={
        <Sidebar
          activeItem={activeTab}
          onNavigate={setActiveTab}
          onNewSession={handleNewSession}
          stats={stats}
        />
      }
    >
      <div className="main-viewport">
        {activeTab === 'chat' && (
          <div className="chat-layout-wrapper">
            <Chat
              messages={messages}
              isLoading={isLoading}
              onSendMessage={handleSendMessage}
              strategy={querySettings.strategy}
              onStrategyChange={(s) => setQuerySettings({ ...querySettings, strategy: s })}
              onSelectMessage={setSelectedMessage}
              selectedMessage={selectedMessage}
            />
            <AnimatePresence>
              {selectedMessage && (
                <SourcePanel
                  message={selectedMessage}
                  onClose={() => setSelectedMessage(null)}
                  strategy={querySettings.strategy}
                />
              )}
            </AnimatePresence>
          </div>
        )}
        {activeTab === 'indexing' && (
          <IndexingPanel onIndexingComplete={handleIndexingComplete} />
        )}
        {activeTab === 'graph' && (
          <GraphView />
        )}
        {activeTab === 'sparql' && (
          <SparqlPanel />
        )}
        {activeTab === 'stats' && (
          <div className="stats-dashboard">
            <h2>System Insights</h2>
            <div className="stats-grid">
              <div className="stat-card glass">
                <span className="sc-label">Total Entities</span>
                <span className="sc-value gradient-text">{stats.entities}</span>
              </div>
              <div className="stat-card glass">
                <span className="sc-label">Social Relationships</span>
                <span className="sc-value gradient-text">{stats.rels}</span>
              </div>
              <div className="stat-card glass">
                <span className="sc-label">Indexed Chunks</span>
                <span className="sc-value gradient-text">{stats.chunks}</span>
              </div>
            </div>
          </div>
        )}
        {activeTab === 'settings' && (
          <SettingsTab
            currentSettings={querySettings}
            onSave={(newSettings) => {
              setQuerySettings(newSettings);
              setNotification({ message: "Cấu hình đã được lưu thành công!", type: 'success' });
              setTimeout(() => setNotification(null), 3000);
            }}
            onRefreshStats={fetchStats}
          />
        )}
      </div>

      <AnimatePresence>
        {notification && (
          <motion.div 
            className={`notification ${notification.type} glass`}
            initial={{ x: 300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 300, opacity: 0 }}
            transition={{ type: 'spring', damping: 20, stiffness: 200 }}
          >
            {notification.message}
          </motion.div>
        )}
      </AnimatePresence>
    </Layout>
  );
}

// Separate component for Settings Tab to manage its own local state
const SettingsTab: React.FC<{
  currentSettings: any,
  onSave: (settings: any) => void,
  onRefreshStats: () => void
}> = ({ currentSettings, onSave, onRefreshStats }) => {
  const [localSettings, setLocalSettings] = useState(currentSettings);
  const [isRestoring, setIsRestoring] = useState(false);
  const [restoreFile, setRestoreFile] = useState<File | null>(null);

  // Sync if current settings change externally (rare)
  useEffect(() => {
    setLocalSettings(currentSettings);
  }, [currentSettings.strategy]);

  const handleBackup = () => {
    try {
      backupApi.downloadBackup();
    } catch (err) {
      console.error('Backup failed', err);
    }
  };

  const handleRestore = async () => {
    if (!restoreFile) return;
    
    setIsRestoring(true);
    try {
      const response = await backupApi.restoreGraph(restoreFile);
      if (response.data.status === 'success') {
        alert(`Restore thành công! Đã khôi phục ${response.data.nodes_restored} nodes.`);
        onRefreshStats();
        setRestoreFile(null);
      }
    } catch (err) {
      console.error('Restore failed', err);
      alert('Restore thất bại. Vui lòng kiểm tra lại file backup.');
    } finally {
      setIsRestoring(false);
    }
  };

  return (
    <div className="settings-page">
      <div className="settings-container glass">
        <div className="settings-header-box">
          <h2>RAG Configuration</h2>
          <p className="settings-desc">Tùy chỉnh các tham số truy vấn cho hệ thống HybridRAG.</p>
        </div>

        <div className="settings-group">
          {(localSettings.strategy !== 'graph') && (
            <>
              <div className="setting-item">
                <div className="setting-info">
                  <label>Top K Chunks ({localSettings.top_k})</label>
                  <span>Số lượng đoạn văn bản lấy ra từ Vector DB.</span>
                </div>
                <input
                  type="number" min="1" max="50" step="1"
                  value={localSettings.top_k}
                  onChange={(e) => setLocalSettings({ ...localSettings, top_k: e.target.value })}
                  className="settings-input"
                />
              </div>

              <div className="setting-item">
                <div className="setting-info">
                  <label>Similarity Threshold ({localSettings.threshold})</label>
                  <span>Ngưỡng độ tương đồng tối thiểu để lấy kết quả.</span>
                </div>
                <input
                  type="number" min="0" max="1" step="0.05"
                  value={localSettings.threshold}
                  onChange={(e) => setLocalSettings({ ...localSettings, threshold: e.target.value })}
                  className="settings-input"
                />
              </div>
            </>
          )}

          {(localSettings.strategy === 'full_hybrid' || localSettings.strategy === 'graph') && (
            <div className="setting-item">
              <div className="setting-info">
                <label>Graph Limit ({localSettings.graph_limit})</label>
                <span>Số lượng quan hệ tối đa lấy từ Knowledge Graph.</span>
              </div>
              <input
                type="number" min="1" max="50" step="1"
                value={localSettings.graph_limit}
                onChange={(e) => setLocalSettings({ ...localSettings, graph_limit: e.target.value })}
                className="settings-input"
              />
            </div>
          )}
        </div>

        <button
          className="save-settings-btn"
          onClick={() => {
            onSave({
              ...localSettings,
              top_k: parseInt(localSettings.top_k as string) || 1,
              threshold: parseFloat(localSettings.threshold as string) || 0,
              graph_limit: parseInt(localSettings.graph_limit as string) || 1
            });
          }}
        >
          Save config
        </button>

        <hr className="settings-divider" />

        <div className="settings-header-box" style={{ marginTop: '2rem' }}>
          <h2>Graph Data Management</h2>
          <p className="settings-desc">Sao lưu và khôi phục toàn bộ cơ sở dữ liệu đồ thị.</p>
        </div>

        <div className="management-grid" style={{ 
          display: 'grid', 
          gridTemplateColumns: '1fr 1fr', 
          gap: '1.5rem',
          marginTop: '1.5rem'
        }}>
          <div className="mgmt-card glass-sm" style={{ padding: '1.5rem', borderRadius: '12px' }}>
            <div className="mgmt-icon" style={{ marginBottom: '1rem', color: 'var(--accent)' }}>
              <Download size={32} />
            </div>
            <h3 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>Backup Graph</h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
              Tải xuống toàn bộ dữ liệu (Nodes, Relationships) dưới dạng file ZIP.
            </p>
            <button 
              className="mgmt-btn export" 
              onClick={handleBackup}
              style={{
                width: '100%',
                padding: '0.75rem',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid var(--border-soft)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem'
              }}
            >
              <Download size={16} /> Export ZIP
            </button>
          </div>

          <div className="mgmt-card glass-sm" style={{ padding: '1.5rem', borderRadius: '12px' }}>
            <div className="mgmt-icon" style={{ marginBottom: '1rem', color: 'var(--accent)' }}>
              <Upload size={32} />
            </div>
            <h3 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>Restore Graph</h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
              Khôi phục dữ liệu từ file backup. Lưu ý: Sẽ xóa dữ liệu hiện tại.
            </p>
            
            <div className="file-upload-wrapper" style={{ marginBottom: '1rem' }}>
              <input 
                type="file" 
                accept=".zip" 
                id="restore-file"
                onChange={(e) => setRestoreFile(e.target.files?.[0] || null)}
                style={{ display: 'none' }}
              />
              <label 
                htmlFor="restore-file" 
                style={{
                  display: 'block',
                  padding: '0.75rem',
                  backgroundColor: 'rgba(255, 255, 255, 0.02)',
                  border: '1px dashed var(--border-soft)',
                  borderRadius: '8px',
                  textAlign: 'center',
                  fontSize: '0.85rem',
                  cursor: 'pointer',
                  color: restoreFile ? 'var(--accent)' : 'var(--text-muted)'
                }}
              >
                {restoreFile ? restoreFile.name : 'Chọn file backup (.zip)'}
              </label>
            </div>

            <button 
              className="mgmt-btn import" 
              onClick={handleRestore}
              disabled={!restoreFile || isRestoring}
              style={{
                width: '100%',
                padding: '0.75rem',
                backgroundColor: restoreFile ? 'var(--accent)' : 'rgba(255, 255, 255, 0.05)',
                border: 'none',
                borderRadius: '8px',
                color: '#fff',
                cursor: restoreFile ? 'pointer' : 'not-allowed',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                opacity: (!restoreFile || isRestoring) ? 0.5 : 1
              }}
            >
              {isRestoring ? <Loader2 className="spinner" size={16} /> : <Upload size={16} />} 
              {isRestoring ? 'Restoring...' : 'Restore Now'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
