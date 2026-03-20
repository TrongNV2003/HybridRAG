import React, { useState, useEffect } from 'react';
import { X, Database, FileText, Share2, Info, ChevronDown, ChevronUp, Maximize2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { Message } from './Chat';
import { graphApi } from '../../services/api';
import './SourcePanel.css';

interface SourcePanelProps {
  message: Message | null;
  onClose: () => void;
  strategy: string;
}

const SourcePanel: React.FC<SourcePanelProps> = ({ message, onClose, strategy }) => {
  const [graphHtml, setGraphHtml] = useState<string | null>(null);
  const [expandedChunks, setExpandedChunks] = useState<Record<number, boolean>>({});
  const [isLoadingGraph, setIsLoadingGraph] = useState(false);
  const [width, setWidth] = useState(500);
  const [isResizing, setIsResizing] = useState(false);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;
      const newWidth = window.innerWidth - e.clientX;
      if (newWidth > 350 && newWidth < 900) {
        setWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      document.body.style.cursor = 'default';
    };

    if (isResizing) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);

  useEffect(() => {
    const fetchGraph = async () => {
      if (message?.sources?.graph && message.sources.graph.length > 0) {
        setIsLoadingGraph(true);
        try {
          const response = await graphApi.visualizeSubgraph(message.sources.graph);
          setGraphHtml(response.data);
        } catch (err) {
          console.error('Failed to fetch subgraph', err);
        } finally {
          setIsLoadingGraph(false);
        }
      } else {
        setGraphHtml(null);
      }
    };

    fetchGraph();
  }, [message]);

  if (!message || message.role !== 'assistant' || !message.sources) return null;

  const { graph, chunks } = message.sources;
  const showGraph = strategy === 'full_hybrid' || strategy === 'graph';
  const showChunks = strategy === 'full_hybrid' || strategy === 'semantic_hybrid' || strategy === 'dense';

  const toggleChunk = (idx: number) => {
    setExpandedChunks(prev => ({ ...prev, [idx]: !prev[idx] }));
  };

  return (
    <motion.div 
      className="source-panel glass"
      style={{ width: `${width}px` }}
      initial={{ x: width, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: width, opacity: 0 }}
      transition={{ type: 'spring', damping: 25, stiffness: 200 }}
    >
      <div 
        className="resize-handle" 
        onMouseDown={(e) => {
          e.preventDefault();
          setIsResizing(true);
        }}
      />

      <div className="source-header">
        <div className="header-title">
          <Share2 size={18} className="accent-text" />
          <h3>Retrieval Context</h3>
        </div>
        <button className="close-btn" onClick={onClose}>
          <X size={20} />
        </button>
      </div>

      <div className="source-content">
        {showGraph && graph && graph.length > 0 && (
          <section className="source-section graph-section">
            <div className="section-title">
              <Database size={16} />
              <h4>Graph Visualization</h4>
              <span className="count-badge">{graph.length} Triples</span>
            </div>
            <div className="subgraph-container glass-sm">
              {isLoadingGraph ? (
                <div className="graph-loading">
                  <div className="spinner"></div>
                  <span>Generating Graph...</span>
                </div>
              ) : graphHtml ? (
                <iframe 
                  title="Subgraph"
                  srcDoc={graphHtml}
                  className="subgraph-iframe"
                />
              ) : (
                <div className="graph-error">Failed to load graph visualization</div>
              )}
            </div>
          </section>
        )}

        {showChunks && chunks && chunks.length > 0 && (
          <section className="source-section">
            <div className="section-title">
              <FileText size={16} />
              <h4>Semantic Text Chunks</h4>
              <span className="count-badge">{chunks.length} Chunks</span>
            </div>
            <div className="chunk-list">
              {chunks.map((chunk, idx) => (
                <div key={idx} className={`chunk-card glass-sm ${expandedChunks[idx] ? 'expanded' : ''}`}>
                  <div className="chunk-header" onClick={() => toggleChunk(idx)}>
                    <div className="chunk-info">
                      <span className="source-name">{chunk.metadata?.title || `Chunk ${idx + 1}`}</span>
                      <span className="score">Similarity: {chunk.score?.toFixed(3) || 'N/A'}</span>
                    </div>
                    {expandedChunks[idx] ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </div>
                  <div className="chunk-body">
                    <p className="chunk-text">{chunk.chunk_text || chunk.text}</p>
                    {expandedChunks[idx] && chunk.metadata?.url && (
                      <a href={chunk.metadata.url} target="_blank" rel="noopener noreferrer" className="source-link">
                        View Original Source <Maximize2 size={12} />
                      </a>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {(!graph || graph.length === 0) && (!chunks || chunks.length === 0) && (
          <div className="no-sources">
            <Info size={40} opacity={0.3} />
            <p>No retrieval context available for this response.</p>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default SourcePanel;
