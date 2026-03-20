import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Send, User, Bot, Loader2, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import './Chat.css';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: {
    graph?: any[];
    chunks?: any[];
  };
}

interface ChatProps {
  messages: Message[];
  isLoading: boolean;
  onSendMessage: (content: string) => void;
  strategy: string;
  onStrategyChange: (strategy: any) => void;
  onSelectMessage?: (message: Message | null) => void;
  selectedMessage?: Message | null;
}

const Chat: React.FC<ChatProps> = ({
  messages,
  isLoading,
  onSendMessage,
  strategy,
  onStrategyChange,
  onSelectMessage,
  selectedMessage
}) => {
  const [input, setInput] = useState('');
  const [isStrategyOpen, setIsStrategyOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const strategies = [
    { id: 'full_hybrid', label: 'Graph-Vector Hybrid RAG (KG + Dense + Sparse)' },
    { id: 'semantic_hybrid', label: 'Semantic Hybrid RAG (Dense + Sparse)' },
    { id: 'dense', label: 'Vector RAG (Dense)' },
    { id: 'graph', label: 'Graph RAG (KG)' }
  ];

  const currentStrategyLabel = strategies.find(s => s.id === strategy)?.label || 'Strategy';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.length === 0 && !isLoading && (
          <div className="chat-empty">
            <Bot size={48} className="empty-icon" />
            <h3>Bắt đầu cuộc trò chuyện</h3>
            <p>Hỏi tôi bất cứ điều gì về dữ liệu đã được index trong HybridRAG.</p>
          </div>
        )}

        <AnimatePresence initial={false}>
          {messages.map((msg) => (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className={`message-wrapper ${msg.role} ${selectedMessage?.id === msg.id ? 'selected' : ''}`}
              onClick={() => msg.role === 'assistant' && onSelectMessage?.(msg)}
            >
              <div className="message-avatar">
                {msg.role === 'user' ? <User size={18} /> : <Bot size={18} />}
              </div>
              <div className="message-content glass">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.content}
                </ReactMarkdown>
                {msg.role === 'assistant' && msg.sources && (
                  <div className="source-badge">
                    <AlertCircle size={10} />
                    <span>View Sources</span>
                  </div>
                )}
                <span className="message-time">
                  {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="message-wrapper assistant loading"
          >
            <div className="message-avatar">
              <Bot size={18} />
            </div>
            <div className="message-content glass">
              <Loader2 size={18} className="spin" />
              <span className="loading-text">HybridRAG is thinking...</span>
            </div>
          </motion.div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-container glass">
        <form className="chat-input-form" onSubmit={handleSubmit}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            disabled={isLoading}
          />
          <button type="submit" disabled={!input.trim() || isLoading} className="send-btn">
            <Send size={20} />
          </button>
        </form>

        <div className="chat-input-footer">
          <div className="strategy-dropdown-wrapper">
            <button
              className={`strategy-toggle ${isStrategyOpen ? 'open' : ''}`}
              onClick={() => setIsStrategyOpen(!isStrategyOpen)}
              type="button"
            >
              <Bot size={14} />
              <span>{currentStrategyLabel}</span>
              <motion.span animate={{ rotate: isStrategyOpen ? 180 : 0 }}>
                <Loader2 size={12} className={isStrategyOpen ? '' : 'hidden-icon'} />
              </motion.span>
            </button>

            <AnimatePresence>
              {isStrategyOpen && (
                <motion.div
                  className="strategy-menu glass"
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.95 }}
                >
                  {strategies.map((strat) => (
                    <button
                      key={strat.id}
                      className={`strategy-option ${strategy === strat.id ? 'active' : ''}`}
                      onClick={() => {
                        onStrategyChange(strat.id);
                        setIsStrategyOpen(false);
                      }}
                    >
                      {strat.label}
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <div className="input-hints">
            <span>Press Enter to send</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;
