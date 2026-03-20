import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Layout from './components/Layout/Layout';
import Sidebar, { NavItem } from './components/Sidebar/Sidebar';
import Chat, { Message } from './components/Chat/Chat';
import IndexingPanel from './components/Indexing/IndexingPanel';
import GraphView from './components/Graph/GraphView';
import SourcePanel from './components/Chat/SourcePanel';
import { queryApi, graphApi } from './services/api';
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
  onSave: (settings: any) => void
}> = ({ currentSettings, onSave }) => {
  const [localSettings, setLocalSettings] = useState(currentSettings);

  // Sync if current settings change externally (rare)
  useEffect(() => {
    setLocalSettings(currentSettings);
  }, [currentSettings.strategy]);

  return (
    <div className="settings-page">
      <div className="settings-container glass">
        <h2>RAG Configuration</h2>
        <p className="settings-desc">Tùy chỉnh các tham số truy vấn cho hệ thống HybridRAG.</p>

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
      </div>
    </div>
  );
};

export default App;
