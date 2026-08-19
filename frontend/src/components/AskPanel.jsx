import { useEffect, useRef, useState } from 'react'
import { askQuestion } from '../utils/apiClient'
import MarkdownText from './MarkdownText'

export default function AskPanel({ topicId }) {
  const inputRef = useRef(null)
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)

  function resizeInput() {
    const el = inputRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }

  useEffect(() => {
    setQuestion('')
    setResult(null)
    setBusy(false)
    resizeInput()
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
    <section className="ask-panel ask-panel-dock" aria-label="Ask about this topic">
      <h2>Ask about this topic</h2>
      <div className="ask-composer">
        <textarea
          ref={inputRef}
          className="ask-input"
          rows={1}
          placeholder="Ask a question grounded in this section…"
          value={question}
          disabled={busy || !topicId}
          onChange={(event) => {
            setQuestion(event.target.value)
            resizeInput()
          }}
        />
        <button
          type="button"
          className="ask-send-button"
          disabled={busy || !topicId || !question.trim()}
          onClick={handleAsk}
          aria-label={busy ? 'Thinking' : 'Ask question'}
        >
          {busy ? (
            <span className="ask-send-busy" aria-hidden="true">
              …
            </span>
          ) : (
            <svg className="ask-send-icon" viewBox="0 0 24 24" aria-hidden="true">
              <path
                d="M12 19V5M6 12l6-6 6 6"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          )}
        </button>
      </div>
      {busy ? <p className="topic-view-muted">Looking up an answer from this section…</p> : null}
      {!busy && result?.status === 'ready' && result.answer ? (
        <MarkdownText className="topic-explanation ask-answer">{result.answer}</MarkdownText>
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
