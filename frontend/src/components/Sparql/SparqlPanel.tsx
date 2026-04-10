import React, { useState } from 'react';
import { Play, RotateCw, Download, BookOpen, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';
import { sparqlApi } from '../../services/api';
import './SparqlPanel.css';

interface PredefinedQuery {
  id: string;
  name: string;
  description: string;
  query: string;
}

const PREDEFINED_QUERIES: PredefinedQuery[] = [
  {
    id: 'all-entities',
    name: 'Liệt kê 10 thực thể đầu tiên',
    description: 'Lấy ra URI và nhãn của 10 thực thể bất kỳ trong hệ thống.',
    query: `PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbo: <http://dbpedia.org/ontology/>

SELECT ?entity ?label ?type
WHERE {
  ?entity rdfs:label ?label .
  ?entity rdf:type ?type .
}
LIMIT 10`
  },
  {
    id: 'dbpedia-links',
    name: 'Tìm các liên kết DBPedia tiếng Anh',
    description: 'Liệt kê các thực thể đã được liên kết với phiên bản DBPedia quốc tế thông qua owl:sameAs.',
    query: `PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT ?localEntity ?label ?dbpediaLink
WHERE {
  ?localEntity rdfs:label ?label .
  ?localEntity owl:sameAs ?dbpediaLink .
}`
  },
  {
    id: 'person-entities',
    name: 'Lọc thực thể là Con người (Person)',
    description: 'Tìm kiếm tất cả các thực thể được LLM phân loại là Person theo DBPedia Ontology.',
    query: `PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbo: <http://dbpedia.org/ontology/>

SELECT ?person ?name ?description
WHERE {
  ?person rdf:type dbo:Person .
  ?person rdfs:label ?name .
  OPTIONAL { ?person rdfs:comment ?description }
}`
  },
  {
    id: 'relationships',
    name: 'Truy vấn các quan hệ ngữ nghĩa',
    description: 'Tìm tất cả các bộ ba (Subject - Predicate - Object) trong đồ thị tri thức.',
    query: `PREFIX dbo: <http://dbpedia.org/ontology/>

SELECT ?s ?p ?o
WHERE {
  ?s ?p ?o .
  FILTER(?p != rdf:type && ?p != rdfs:label && ?p != rdfs:comment)
}
LIMIT 20`
  }
];

const SparqlPanel: React.FC = () => {
  const [query, setQuery] = useState(PREDEFINED_QUERIES[0].query);
  const [results, setResults] = useState<any[]>([]);
  const [variables, setVariables] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const handleRunQuery = async () => {
    setIsLoading(true);
    setError(null);
    setSuccessMsg(null);
    try {
      const response = await sparqlApi.query(query, true);
      setResults(response.data.results);
      setVariables(response.data.variables);
      if (response.data.sync_completed) {
        setSuccessMsg("Đã đồng bộ dữ liệu mới nhất từ Neo4j và thực thi truy vấn.");
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || "Lỗi khi thực hiện truy vấn SPARQL.");
      setResults([]);
      setVariables([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSync = async () => {
    setIsSyncing(true);
    setError(null);
    setSuccessMsg(null);
    try {
      await sparqlApi.syncRdf();
      setSuccessMsg("Đồng bộ RDF thành công!");
    } catch (err: any) {
      setError("Không thể đồng bộ dữ liệu RDF.");
    } finally {
      setIsSyncing(false);
    }
  };

  const loadPredefined = (q: PredefinedQuery) => {
    setQuery(q.query);
    setSuccessMsg(`Đã tải truy vấn mẫu: ${q.name}`);
  };

  return (
    <div className="sparql-panel animate-fade-in">
      <header className="panel-header">
        <div className="header-info">
          <h2 className="gradient-text">Semantic Hub (SPARQL)</h2>
          <p>Truy vấn tri thức chuẩn Linked Data từ đồ thị Neo4j.</p>
        </div>
        <div className="header-actions">
          <button className="btn-secondary glass" onClick={handleSync} disabled={isSyncing}>
            {isSyncing ? <Loader2 className="spinner" size={16} /> : <RotateCw size={16} />}
            <span>Sync Knowledge</span>
          </button>
          <button className="btn-secondary glass" onClick={() => window.open('/api/v1/sparql/sync-rdf', '_blank')}>
            <Download size={16} />
            <span>Export TTL</span>
          </button>
        </div>
      </header>

      <div className="sparql-main-grid">
        <section className="query-section glass">
          <div className="section-header">
            <div className="title-with-icon">
              <Play size={18} className="icon-accent" />
              <h3>SPARQL Editor</h3>
            </div>
            <div className="predefined-dropdown">
              <BookOpen size={16} />
              <select onChange={(e) => {
                const q = PREDEFINED_QUERIES.find(x => x.id === e.target.value);
                if (q) loadPredefined(q);
              }}>
                <option value="">Chọn truy vấn mẫu...</option>
                {PREDEFINED_QUERIES.map(q => (
                  <option key={q.id} value={q.id}>{q.name}</option>
                ))}
              </select>
            </div>
          </div>
          
          <div className="editor-wrapper">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              spellCheck={false}
              placeholder="Nhập câu lệnh SPARQL tại đây..."
            />
            <button className="run-btn" onClick={handleRunQuery} disabled={isLoading}>
              {isLoading ? <Loader2 className="spinner" /> : <Play size={20} />}
              <span>EXECUTE</span>
            </button>
          </div>

          {error && (
            <div className="status-msg error animate-slide-up">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}
          {successMsg && (
            <div className="status-msg success animate-slide-up">
              <CheckCircle2 size={16} />
              <span>{successMsg}</span>
            </div>
          )}
        </section>

        <section className="results-section glass">
          <div className="section-header">
            <h3>Query Results ({results.length})</h3>
          </div>
          <div className="table-responsive">
            {results.length > 0 ? (
              <table>
                <thead>
                  <tr>
                    {variables.map(v => <th key={v}>{v}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {results.map((row, i) => (
                    <tr key={i}>
                      {variables.map(v => (
                        <td key={v} title={row[v]}>
                          {row[v] && row[v].startsWith('http') ? (
                            <a href={row[v]} target="_blank" rel="noreferrer" className="uri-link">
                              {row[v].split('/').pop()?.split('#').pop()}
                            </a>
                          ) : (
                            row[v]
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="empty-results">
                <p>Chưa có kết quả. Nhấn EXECUTE để truy vấn dữ liệu.</p>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
};

export default SparqlPanel;
