import axios from 'axios';

const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface IndexingResponse {
  status: string;
  message: string;
  entities_count: number;
  relationships_count: number;
  chunks_count: number;
  error?: string;
}

export interface QueryResponse {
  answer: string;
  graph_context: any[];
  chunk_context: any[];
}

export interface GraphStats {
  entities_count: number;
  relationships_count: number;
  chunks_count: number;
}

export const indexingApi = {
  wikipedia: (query_keyword: string, max_docs: number = 10, clear_old: boolean = false) =>
    api.post<IndexingResponse>('/indexing/wikipedia', { query_keyword, max_docs, clear_old }),
};

export const queryApi = {
  fullHybrid: (query: string, top_k: number = 5, threshold: number = 0.5, graph_limit: number = 10) =>
    api.post<QueryResponse>('/query/full_hybrid', { query, top_k, threshold, graph_limit }),
  
  graph: (query: string, graph_limit: number = 10) =>
    api.post<QueryResponse>('/query/graph', { query, graph_limit }),
  
  dense: (query: string, top_k: number = 5, threshold: number = 0.5) =>
    api.post<QueryResponse>('/query/dense', { query, top_k, threshold }),
    
  semanticHybrid: (query: string, top_k: number = 5, threshold: number = 0.5) =>
    api.post<QueryResponse>('/query/semantic_hybrid', { query, top_k, threshold }),
};

export const graphApi = {
  getStats: () => api.get<GraphStats>('/graph/stats'),
  getVisualizationUrl: (limit: number = 100, search?: string) => {
    let url = `/api/v1/graph/visualize?limit=${limit}`;
    if (search) url += `&search=${encodeURIComponent(search)}`;
    return url;
  },
  visualizeSubgraph: (triples: any[]) => api.post<string>('/graph/visualize_subgraph', triples),
};

export const backupApi = {
  downloadBackup: () => {
    window.location.href = '/api/v1/backup/backup';
  },
  restoreGraph: (file: File, clearExisting: boolean = true) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post(`/backup/restore?clear_existing=${clearExisting}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
};

export default api;
