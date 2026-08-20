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

export async function generateTopicImage(topicId, imageId) {
  const response = await fetch(API.generateImage(topicId, imageId), {
    method: 'POST',
    headers: apiHeaders(),
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

function parseSseBlock(block) {
  let event = 'message'
  const dataLines = []
  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
  }
  const raw = dataLines.join('\n')
  let data = {}
  if (raw) {
    try {
      data = JSON.parse(raw)
    } catch {
      data = { text: raw }
    }
  }
  return { event, data }
}

function dispatchSseBlock(block, handlers) {
  if (!block.trim() || block.startsWith(':')) return
  const { event, data } = parseSseBlock(block)
  if (event === 'delta' && data.text) handlers.onDelta?.(data.text)
  else if (event === 'done') handlers.onDone?.(data)
  else if (event === 'error') handlers.onError?.(data)
}

export async function consumeSse(response, handlers = {}) {
  if (!response.ok) {
    throw new Error(await parseError(response))
  }
  if (!response.body) {
    throw new Error('No response body to stream')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      if (handlers.signal?.aborted) {
        await reader.cancel()
        throw new DOMException('Aborted', 'AbortError')
      }
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const parts = buffer.split('\n\n')
      buffer = parts.pop() ?? ''
      for (const part of parts) dispatchSseBlock(part, handlers)
    }
    buffer += decoder.decode()
    if (buffer.trim()) dispatchSseBlock(buffer, handlers)
  } finally {
    reader.releaseLock?.()
  }
}

export async function streamTopicExplanation(topicId, handlers = {}) {
  const response = await fetch(API.topicExplanationStream(topicId), {
    headers: apiHeaders({ Accept: 'text/event-stream' }),
    signal: handlers.signal,
  })
  await consumeSse(response, handlers)
}

export async function streamAskQuestion(topicId, question, handlers = {}) {
  const response = await fetch(API.ask, {
    method: 'POST',
    headers: apiHeaders({
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    }),
    body: JSON.stringify({ topic_id: topicId, question }),
    signal: handlers.signal,
  })
  await consumeSse(response, handlers)
}

export async function synthesizeSpeech(text, voice = 'shimmer') {
  const response = await fetch(API.tts, {
    method: 'POST',
    headers: apiHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ text, voice }),
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
