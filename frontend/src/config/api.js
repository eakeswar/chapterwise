const DEFAULT_BASE = 'http://127.0.0.1:8766'

function normalizeBase(url) {
  return (url || DEFAULT_BASE).replace(/\/$/, '')
}

const API_BASE = normalizeBase(import.meta.env.VITE_API_BASE_URL)

export const API = {
  base: API_BASE,
  upload: `${API_BASE}/upload_pdf`,
  buildTopics: `${API_BASE}/build_topics`,
  topic: (topicId) => `${API_BASE}/topic/${encodeURIComponent(topicId)}`,
  tts: `${API_BASE}/tts`,
  health: `${API_BASE}/health`,
}

export function apiHeaders(extra = {}) {
  return { ...extra }
}

export default API
