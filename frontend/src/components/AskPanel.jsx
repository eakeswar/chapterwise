import { useEffect, useRef, useState } from 'react'
import { streamAskQuestion } from '../utils/apiClient'
import MarkdownText from './MarkdownText'

function AskComposer({ inputRef, question, busy, topicId, onQuestionChange, onAsk }) {
  return (
    <div className="ask-composer">
      <textarea
        ref={inputRef}
        className="ask-input"
        rows={1}
        placeholder="Ask a question grounded in this section…"
        value={question}
        disabled={busy || !topicId}
        onChange={onQuestionChange}
      />
      <button
        type="button"
        className="ask-send-button"
        disabled={busy || !topicId || !question.trim()}
        onClick={onAsk}
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
  )
}

function renderTurnAnswer(turn) {
  if (turn.status === 'pending' && !turn.answer) {
    return <p className="topic-view-muted">Thinking…</p>
  }
  if (turn.status === 'pending' || turn.status === 'streaming') {
    return (
      <div className="topic-explanation ask-answer topic-explanation-streaming">
        {turn.answer || 'Writing answer…'}
        <span className="stream-cursor" aria-hidden="true" />
      </div>
    )
  }
  if (turn.status === 'incomplete') {
    return (
      <>
        {turn.answer ? (
          <div className="topic-explanation ask-answer topic-explanation-streaming">{turn.answer}</div>
        ) : null}
        <div className="topic-notice topic-notice-error">
          This answer may be incomplete
          {turn.error ? `: ${turn.error}` : '.'}
        </div>
      </>
    )
  }
  if (turn.status === 'ready' && turn.answer) {
    return <MarkdownText className="topic-explanation ask-answer">{turn.answer}</MarkdownText>
  }
  if (turn.status === 'failed') {
    return (
      <div className="topic-notice topic-notice-error">
        Couldn&apos;t answer
        {turn.error ? `: ${turn.error}` : '.'}
      </div>
    )
  }
  return null
}

export default function AskPanel({ topicId, layout = 'column' }) {
  const inputRef = useRef(null)
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [turns, setTurns] = useState([])
  const isColumnLayout = layout === 'column' || layout === 'tab'

  function resizeInput() {
    const el = inputRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }

  useEffect(() => {
    setQuestion('')
    setTurns([])
    setBusy(false)
    resizeInput()
  }, [topicId])

  async function handleAsk() {
    const trimmed = question.trim()
    if (!topicId || !trimmed || busy) return
    setBusy(true)
    setQuestion('')
    resizeInput()
    const turnIndex = turns.length
    setTurns((current) => [...current, { question: trimmed, status: 'pending' }])
    let assembled = ''
    try {
      await streamAskQuestion(topicId, trimmed, {
        onDelta(text) {
          assembled += text
          setTurns((current) => {
            const next = [...current]
            next[turnIndex] = {
              question: trimmed,
              answer: assembled,
              status: 'streaming',
            }
            return next
          })
        },
        onDone() {
          setTurns((current) => {
            const next = [...current]
            next[turnIndex] = {
              question: trimmed,
              answer: assembled,
              status: 'ready',
            }
            return next
          })
        },
        onError(payload) {
          const status = payload?.status === 'incomplete' || assembled ? 'incomplete' : 'failed'
          setTurns((current) => {
            const next = [...current]
            next[turnIndex] = {
              question: trimmed,
              answer: assembled || null,
              status,
              error: payload?.error || 'Ask failed',
            }
            return next
          })
        },
      })
    } catch (askError) {
      setTurns((current) => {
        const next = [...current]
        next[turnIndex] = {
          question: trimmed,
          answer: assembled || null,
          status: assembled ? 'incomplete' : 'failed',
          error: askError instanceof Error ? askError.message : 'Ask failed',
        }
        return next
      })
    } finally {
      setBusy(false)
    }
  }

  function handleQuestionChange(event) {
    setQuestion(event.target.value)
    resizeInput()
  }

  if (isColumnLayout) {
    return (
      <section className="ask-panel ask-panel-column" aria-label="Ask about this topic">
        <div className="ask-history">
          {turns.length === 0 && !busy ? (
            <p className="topic-view-muted ask-history-empty">Ask a question grounded in this section.</p>
          ) : null}
          {turns.map((turn, index) => (
            <div key={`${turn.question}-${index}`} className="ask-turn">
              <div className="ask-turn-question">{turn.question}</div>
              {renderTurnAnswer(turn)}
            </div>
          ))}
        </div>
        <div className="ask-input-footer">
          <AskComposer
            inputRef={inputRef}
            question={question}
            busy={busy}
            topicId={topicId}
            onQuestionChange={handleQuestionChange}
            onAsk={handleAsk}
          />
        </div>
      </section>
    )
  }

  return (
    <section className="ask-panel ask-panel-dock" aria-label="Ask about this topic">
      <h2>Ask about this topic</h2>
      <AskComposer
        inputRef={inputRef}
        question={question}
        busy={busy}
        topicId={topicId}
        onQuestionChange={handleQuestionChange}
        onAsk={handleAsk}
      />
    </section>
  )
}
