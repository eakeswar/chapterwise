import { useEffect, useState } from 'react'
import { askQuestion } from '../utils/apiClient'

export default function AskPanel({ topicId }) {
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)

  useEffect(() => {
    setQuestion('')
    setResult(null)
    setBusy(false)
  }, [topicId])

  async function handleAsk() {
    const trimmed = question.trim()
    if (!topicId || !trimmed || busy) return
    setBusy(true)
    setResult(null)
    try {
      const payload = await askQuestion(topicId, trimmed)
      setResult(payload)
    } catch (askError) {
      setResult({
        answer: null,
        status: 'failed',
        error: askError instanceof Error ? askError.message : 'Ask failed',
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="topic-section ask-panel">
      <h2>Ask about this topic</h2>
      <textarea
        className="ask-input"
        rows={3}
        placeholder="Ask a question grounded in this section…"
        value={question}
        disabled={busy || !topicId}
        onChange={(event) => setQuestion(event.target.value)}
      />
      <button
        type="button"
        className="secondary-button"
        disabled={busy || !topicId || !question.trim()}
        onClick={handleAsk}
      >
        {busy ? 'Thinking…' : 'Ask'}
      </button>
      {busy ? <p className="topic-view-muted">Looking up an answer from this section…</p> : null}
      {!busy && result?.status === 'ready' && result.answer ? (
        <div className="topic-explanation ask-answer">{result.answer}</div>
      ) : null}
      {!busy && result?.status === 'failed' ? (
        <div className="topic-notice topic-notice-error">
          Couldn&apos;t answer
          {result.error ? `: ${result.error}` : '.'}
        </div>
      ) : null}
    </section>
  )
}
