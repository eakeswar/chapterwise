import API, { apiHeaders } from '../config/api'

async function parseError(response) {
  try {
    const body = await response.json()
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail)) {
      return body.detail.map((item) => item.msg || String(item)).join('; ')
    }
    return JSON.stringify(body)
  } catch {
    return `Request failed (${response.status})`
  }
}

export async function uploadPdf(file, { page_start, page_end } = {}) {
  const params = new URLSearchParams()
  if (page_start != null) params.set('page_start', String(page_start))
  if (page_end != null) params.set('page_end', String(page_end))
  const url = params.toString() ? `${API.upload}?${params.toString()}` : API.upload

  const buffer = await file.arrayBuffer()
  const response = await fetch(url, {
    method: 'POST',
    headers: apiHeaders({ 'Content-Type': 'application/pdf' }),
    body: buffer,
  })
  if (!response.ok) {
    throw new Error(await parseError(response))
  }
  return response.json()
}

export async function buildTopics(options = {}) {
  const response = await fetch(API.buildTopics, {
    method: 'POST',
    headers: apiHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(options),
  })
  if (!response.ok) {
    throw new Error(await parseError(response))
  }
  return response.json()
}

export async function fetchTopic(topicId) {
  const response = await fetch(API.topic(topicId), {
    headers: apiHeaders(),
  })
  if (!response.ok) {
    throw new Error(await parseError(response))
  }
  return response.json()
}

export async function askQuestion(topicId, question) {
  const response = await fetch(API.ask, {
    method: 'POST',
    headers: apiHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ topic_id: topicId, question }),
  })
  if (!response.ok) {
    throw new Error(await parseError(response))
  }
  return response.json()
}

export async function synthesizeSpeech(text) {
  const response = await fetch(API.tts, {
    method: 'POST',
    headers: apiHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ text }),
  })
  if (!response.ok) {
    throw new Error(await parseError(response))
  }
  return response.blob()
}

export async function checkHealth() {
  const response = await fetch(API.health)
  if (!response.ok) {
    throw new Error(`Backend unreachable (${response.status})`)
  }
  return response.json()
}
