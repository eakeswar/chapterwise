import { useEffect, useMemo, useRef, useState } from 'react'
import { findTopicById, useDoc } from '../context/DocContext'
import { fetchTopic, generateTopicImage, streamTopicExplanation, synthesizeSpeech } from '../utils/apiClient'
import { mergeTopicPoll } from '../utils/mergeTopicPoll'
import TopicViewSkeleton from './TopicViewSkeleton'
import Skeleton from './Skeleton'
import AskPanel from './AskPanel'
import MarkdownText from './MarkdownText'

const TTS_CHAR_LIMIT = 3800
const CLASSIFY_POLL_MS = 3000
const DEFAULT_TTS_VOICE = 'shimmer'
const TTS_VOICES = [
  { id: 'shimmer', label: 'Shimmer (default)' },
  { id: 'alloy', label: 'Alloy' },
  { id: 'echo', label: 'Echo' },
  { id: 'fable', label: 'Fable' },
  { id: 'onyx', label: 'Onyx' },
  { id: 'nova', label: 'Nova' },
  { id: 'coral', label: 'Coral' },
  { id: 'ash', label: 'Ash' },
  { id: 'sage', label: 'Sage' },
  { id: 'verse', label: 'Verse' },
]

function imageNeedsClassifyPoll(image) {
  return image?.classify_status === 'pending' || image?.enhance_kind === 'pending'
}

function canGenerateHd(image) {
  return image?.classify_status === 'ready' && image?.enhance_role === 'decorative'
}

function pageRangeLabel(pageStart, pageEnd) {
  if (pageStart === pageEnd) return `Page ${pageStart}`
  return `Pages ${pageStart}–${pageEnd}`
}

function buildNarrationText(detail) {
  const title = detail?.topic?.title || 'Topic'
  const bodySource = (detail?.explanation || detail?.source_text || '').trim()
  const intro = `${title}. `
  const budget = Math.max(0, TTS_CHAR_LIMIT - intro.length)
  const body = bodySource.length <= budget ? bodySource : `${bodySource.slice(0, budget).trim()}…`
  return `${intro}${body}`.trim()
}

function explanationIsTerminal(detail) {
  return detail?.explanation_status === 'ready' || detail?.explanation_status === 'skipped'
}

function recordPollMerge(current, incoming, merged) {
  if (typeof window === 'undefined') return
  const entry = {
    at: Date.now(),
    incomingStatus: incoming?.explanation_status ?? null,
    incomingExplanationLen: incoming?.explanation ? incoming.explanation.length : 0,
    currentStatus: current?.explanation_status ?? null,
    currentExplanationLen: current?.explanation ? current.explanation.length : 0,
    mergedStatus: merged?.explanation_status ?? null,
    mergedExplanationLen: merged?.explanation ? merged.explanation.length : 0,
  }
  window.__chapterwisePollMerges = window.__chapterwisePollMerges || []
  window.__chapterwisePollMerges.push(entry)
}

export default function TopicView() {
  const { state, dispatch } = useDoc()
  const chapters = state.topicTree?.chapters || []
  const treeTopic = findTopicById(chapters, state.activeTopicId)
  const cached = state.activeTopicId ? state.topicCache[state.activeTopicId] : null

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [detail, setDetail] = useState(cached)
  const [askOpen, setAskOpen] = useState(false)
  const [ttsBusy, setTtsBusy] = useState(false)
  const [ttsError, setTtsError] = useState(null)
  const [ttsVoice, setTtsVoice] = useState(DEFAULT_TTS_VOICE)
  const audioRef = useRef(null)
  const audioUrlRef = useRef(null)
  const [generatingIds, setGeneratingIds] = useState(() => new Set())
  const [generateErrors, setGenerateErrors] = useState({})
  const [showOriginalIds, setShowOriginalIds] = useState(() => new Set())
  const [sourceOpen, setSourceOpen] = useState(false)

  useEffect(() => {
    setAskOpen(false)
    setGeneratingIds(new Set())
    setGenerateErrors({})
    setShowOriginalIds(new Set())
    setTtsError(null)
    setSourceOpen(false)
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.removeAttribute('src')
    }
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current)
      audioUrlRef.current = null
    }
  }, [state.activeTopicId])

  useEffect(() => {
    if (!state.activeTopicId) {
      setDetail(null)
      setError(null)
      setLoading(false)
      return
    }

    const known = state.topicCache[state.activeTopicId]
    if (known && explanationIsTerminal(known)) {
      setDetail(known)
      setError(null)
      setLoading(false)
      return
    }
    if (known?.explanation_status === 'pending') {
      setDetail(known)
      setError(null)
      setLoading(false)
      return
    }

    let cancelled = false
    setLoading(true)
    setError(null)
    setDetail(null)

    fetchTopic(state.activeTopicId)
      .then((payload) => {
        if (cancelled) return
        dispatch({ type: 'CACHE_TOPIC', payload })
        setDetail(payload)
      })
      .catch((fetchError) => {
        if (cancelled) return
        setDetail(null)
        setError(fetchError instanceof Error ? fetchError.message : 'Failed to load topic')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [state.activeTopicId, dispatch])

  const needsClassifyPoll = Boolean(detail?.images?.some(imageNeedsClassifyPoll))

  useEffect(() => {
    const topicId = state.activeTopicId
    if (!topicId || !needsClassifyPoll) return undefined

    let cancelled = false
    const timer = setInterval(() => {
      fetchTopic(topicId)
        .then((payload) => {
          if (cancelled) return
          setDetail((current) => {
            const merged = mergeTopicPoll(current, payload)
            recordPollMerge(current, payload, merged)
            dispatch({ type: 'CACHE_TOPIC', payload: merged })
            return merged
          })
        })
        .catch(() => {})
    }, CLASSIFY_POLL_MS)

    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [state.activeTopicId, needsClassifyPoll, dispatch])

  useEffect(() => {
    const topicId = state.activeTopicId
    if (!topicId || !detail || detail.topic?.id !== topicId) return undefined
    if (detail.explanation_status !== 'pending') return undefined

    const abort = new AbortController()
    let assembled = ''

    streamTopicExplanation(topicId, {
      signal: abort.signal,
      onDelta(text) {
        assembled += text
        setDetail((current) => {
          if (!current || current.topic?.id !== topicId) return current
          const next = {
            ...current,
            explanation: assembled,
            explanation_status: 'pending',
            explanation_error: null,
          }
          dispatch({ type: 'CACHE_TOPIC', payload: next })
          return next
        })
      },
      onDone() {
        setDetail((current) => {
          if (!current || current.topic?.id !== topicId) return current
          const next = {
            ...current,
            explanation: assembled,
            explanation_status: 'ready',
            explanation_error: null,
          }
          dispatch({ type: 'CACHE_TOPIC', payload: next })
          return next
        })
      },
      onError(payload) {
        const status = payload?.status === 'incomplete' || assembled ? 'incomplete' : 'failed'
        setDetail((current) => {
          if (!current || current.topic?.id !== topicId) return current
          const next = {
            ...current,
            explanation: assembled || null,
            explanation_status: status,
            explanation_error: payload?.error || 'Explanation stream failed',
          }
          dispatch({ type: 'CACHE_TOPIC', payload: next })
          return next
        })
      },
    })
      .then(() => {
        if (abort.signal.aborted) return
        setDetail((current) => {
          if (!current || current.topic?.id !== topicId) return current
          if (current.explanation_status !== 'pending') return current
          const next = {
            ...current,
            explanation: assembled || null,
            explanation_status: assembled ? 'incomplete' : 'failed',
            explanation_error: current.explanation_error || 'Stream ended unexpectedly',
          }
          dispatch({ type: 'CACHE_TOPIC', payload: next })
          return next
        })
      })
      .catch((streamError) => {
      if (abort.signal.aborted) return
      if (streamError?.name === 'AbortError') return
      setDetail((current) => {
        if (!current || current.topic?.id !== topicId) return current
        const status = assembled ? 'incomplete' : 'failed'
        const next = {
          ...current,
          explanation: assembled || null,
          explanation_status: status,
          explanation_error: streamError instanceof Error ? streamError.message : 'Explanation stream failed',
        }
        dispatch({ type: 'CACHE_TOPIC', payload: next })
        return next
      })
    })

    return () => {
      abort.abort()
    }
  }, [state.activeTopicId, detail?.topic?.id, detail?.explanation_status, dispatch])

  useEffect(() => {
    return () => {
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current)
        audioUrlRef.current = null
      }
    }
  }, [])

  const parentLabel = useMemo(() => {
    if (!treeTopic?.parentTitle) return null
    return treeTopic.level === 'subsection' ? treeTopic.parentTitle : null
  }, [treeTopic])

  async function handleListen() {
    if (!detail || ttsBusy) return
    setTtsBusy(true)
    setTtsError(null)
    try {
      const blob = await synthesizeSpeech(buildNarrationText(detail), ttsVoice)
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current)
      }
      const url = URL.createObjectURL(blob)
      audioUrlRef.current = url
      if (audioRef.current) {
        audioRef.current.src = url
        await audioRef.current.play()
      }
    } catch (listenError) {
      setTtsError(listenError instanceof Error ? listenError.message : 'TTS failed')
    } finally {
      setTtsBusy(false)
    }
  }

  function patchTopicImage(imageId, patch) {
    setDetail((current) => {
      if (!current?.images) return current
      const next = {
        ...current,
        images: current.images.map((image) =>
          image.id === imageId ? { ...image, ...patch } : image,
        ),
      }
      dispatch({ type: 'CACHE_TOPIC', payload: next })
      return next
    })
  }

  async function handleGenerateImage(image) {
    const topicId = state.activeTopicId
    if (!topicId || !image?.id || generatingIds.has(image.id)) return
    if (!canGenerateHd(image)) return
    setGeneratingIds((current) => new Set(current).add(image.id))
    setGenerateErrors((current) => {
      const next = { ...current }
      delete next[image.id]
      return next
    })
    try {
      const payload = await generateTopicImage(topicId, image.id)
      if (payload.status === 'ready' && payload.url) {
        patchTopicImage(image.id, {
          url: payload.url,
          original_url: payload.original_url || image.original_url || image.url,
          ext: payload.ext || image.ext,
          enhance_kind: 'generated',
          enhance_role: 'decorative',
        })
        setShowOriginalIds((current) => {
          const next = new Set(current)
          next.delete(image.id)
          return next
        })
        return
      }
      patchTopicImage(image.id, {
        url: payload.original_url || image.original_url || image.url,
        original_url: payload.original_url || image.original_url || image.url,
        enhance_kind: 'original',
      })
      setGenerateErrors((current) => ({
        ...current,
        [image.id]: payload.error || "Couldn't generate, try again",
      }))
    } catch (generateError) {
      patchTopicImage(image.id, {
        url: image.original_url || image.url,
        enhance_kind: image.enhance_kind === 'generated' ? 'generated' : 'original',
      })
      setGenerateErrors((current) => ({
        ...current,
        [image.id]: generateError instanceof Error ? generateError.message : "Couldn't generate, try again",
      }))
    } finally {
      setGeneratingIds((current) => {
        const next = new Set(current)
        next.delete(image.id)
        return next
      })
    }
  }

  function toggleShowOriginal(imageId) {
    setShowOriginalIds((current) => {
      const next = new Set(current)
      if (next.has(imageId)) next.delete(imageId)
      else next.add(imageId)
      return next
    })
  }

  function renderGallery() {
    if (!detail?.images?.length) {
      return <p className="topic-view-muted">No images on these pages.</p>
    }
    return (
      <div className="topic-gallery">
        {detail.images.map((image) => {
          const busy = generatingIds.has(image.id)
          const classifying = imageNeedsClassifyPoll(image)
          const showOriginal = showOriginalIds.has(image.id)
          const originalUrl = image.original_url || image.url
          const displayUrl =
            image.enhance_kind === 'generated' && !showOriginal
              ? image.url
              : image.enhance_kind === 'lanczos'
                ? image.url
                : originalUrl
          const generateError = generateErrors[image.id]
          return (
            <figure key={image.id} className="topic-gallery-item">
              {displayUrl ? (
                <img
                  src={busy ? originalUrl : displayUrl}
                  alt={`Page ${image.page} figure`}
                  loading="lazy"
                />
              ) : (
                <div className="topic-gallery-placeholder">Image data unavailable</div>
              )}
              {busy ? (
                <div className="topic-gallery-generating">Generating HD version…</div>
              ) : classifying ? (
                <div className="topic-gallery-generating">Checking image…</div>
              ) : null}
              <figcaption>
                Page {image.page} · {image.w}×{image.h}
                {image.enhance_kind ? ` · ${image.enhance_kind}` : ''}
                {canGenerateHd(image) ? (
                  <div className="topic-gallery-actions">
                    {image.enhance_kind === 'generated' ? (
                      <button
                        type="button"
                        className="topic-gallery-button"
                        onClick={() => toggleShowOriginal(image.id)}
                      >
                        {showOriginal ? 'Show HD version' : 'Show original'}
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="topic-gallery-button"
                        disabled={busy || !originalUrl}
                        onClick={() => handleGenerateImage(image)}
                      >
                        Generate HD version
                      </button>
                    )}
                  </div>
                ) : null}
                {generateError ? (
                  <div className="topic-gallery-error">Couldn&apos;t generate, try again</div>
                ) : null}
              </figcaption>
            </figure>
          )
        })}
      </div>
    )
  }

  if (!state.activeTopicId) {
    return (
      <main className="main-panel">
        <div className="main-empty">
          <h1>Select a topic</h1>
          <p>Upload a PDF, build topics, then pick a chapter, section, or subsection from the sidebar.</p>
        </div>
      </main>
    )
  }

  if (state.phase !== 'ready') {
    return (
      <main className="main-panel">
        <div className="main-empty">
          <h1>{treeTopic?.title || 'Topic'}</h1>
          <p>Topic details appear after the topic tree is built.</p>
        </div>
      </main>
    )
  }

  return (
    <main className="main-panel main-panel--topic">
      <article className="topic-view" aria-busy={loading}>
        <div className="topic-view-scroll">
            <header className="topic-view-header">
              <div className="topic-view-kicker">{detail?.topic?.level || treeTopic?.level}</div>
              <h1>{detail?.topic?.title || treeTopic?.title}</h1>
              {loading ? (
                <div className="topic-view-meta-skeleton">
                  <Skeleton className="skeleton-meta" />
                </div>
              ) : (
                <p className="topic-view-meta">
                  {pageRangeLabel(
                    detail?.topic?.page_start ?? treeTopic?.page_start,
                    detail?.topic?.page_end ?? treeTopic?.page_end,
                  )}
                  {detail?.source_char_count != null ? ` · ${detail.source_char_count.toLocaleString()} chars` : ''}
                  {detail?.image_count != null ? ` · ${detail.image_count} images` : ''}
                </p>
              )}
              {parentLabel ? <p className="topic-view-parent">In section: {parentLabel}</p> : null}

              <div className="topic-view-actions">
                <label className="topic-voice-picker">
                  <span className="topic-voice-label">Voice</span>
                  <select
                    className="topic-voice-select"
                    value={ttsVoice}
                    disabled={ttsBusy}
                    onChange={(event) => setTtsVoice(event.target.value)}
                  >
                    {TTS_VOICES.map((voice) => (
                      <option key={voice.id} value={voice.id}>
                        {voice.label}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  className="secondary-button"
                  disabled={!detail || loading || ttsBusy || detail.explanation_status !== 'ready'}
                  onClick={handleListen}
                >
                  {ttsBusy ? 'Generating audio…' : 'Listen'}
                </button>
                <audio ref={audioRef} className="topic-audio" controls preload="none" />
              </div>
              {ttsError ? <div className="error-banner">{ttsError}</div> : null}
            </header>

            {loading ? <TopicViewSkeleton /> : null}
            {error ? <div className="error-banner">{error}</div> : null}

            {!loading && !error && detail ? (
              <>
                <section className="topic-section">
                  <h2>Explanation</h2>
                  {detail.explanation_status === 'pending' ? (
                    <div className="topic-explanation topic-explanation-streaming">
                      {detail.explanation || 'Writing explanation…'}
                      <span className="stream-cursor" aria-hidden="true" />
                    </div>
                  ) : detail.explanation_status === 'incomplete' ? (
                    <>
                      {detail.explanation ? (
                        <div className="topic-explanation topic-explanation-streaming">
                          {detail.explanation}
                        </div>
                      ) : null}
                      <div className="topic-notice topic-notice-error">
                        This explanation may be incomplete
                        {detail.explanation_error ? `: ${detail.explanation_error}` : '.'}
                      </div>
                    </>
                  ) : detail.explanation_status === 'ready' && detail.explanation ? (
                    <MarkdownText className="topic-explanation">{detail.explanation}</MarkdownText>
                  ) : detail.explanation_status === 'failed' ? (
                    <div className="topic-notice topic-notice-error">
                      Could not generate an explanation
                      {detail.explanation_error ? `: ${detail.explanation_error}` : '.'}
                      {' '}Source text is shown below.
                    </div>
                  ) : detail.explanation_status === 'skipped' ? (
                    <p className="topic-view-muted">No source text to explain for this page range.</p>
                  ) : null}
                </section>

                <section className={`topic-section${sourceOpen ? '' : ' topic-section-collapsed'}`}>
                  <button
                    type="button"
                    className="topic-section-toggle"
                    aria-expanded={sourceOpen}
                    onClick={() => setSourceOpen((open) => !open)}
                  >
                    <span className="topic-section-toggle-title">
                      Source text
                      {detail.source_char_count != null
                        ? ` (${detail.source_char_count.toLocaleString()} chars)`
                        : ''}
                    </span>
                    <span className="topic-section-chevron" aria-hidden="true">
                      {sourceOpen ? '▾' : '▸'}
                    </span>
                  </button>
                  {sourceOpen ? (
                    <pre className="topic-source">{detail.source_text || '[No text in this page range]'}</pre>
                  ) : null}
                </section>

                <section className="topic-section">
                  <h2>Images</h2>
                  {renderGallery()}
                </section>
              </>
            ) : null}
        </div>
      </article>

      {askOpen ? (
        <button
          type="button"
          className="ask-drawer-backdrop"
          aria-label="Close ask drawer"
          onClick={() => setAskOpen(false)}
        />
      ) : null}

      <aside
        id="ask-drawer"
        className={`ask-drawer${askOpen ? ' ask-drawer-open' : ''}`}
        aria-label="Ask about this topic"
        aria-hidden={!askOpen}
      >
        <div className="ask-drawer-header">
          <h2>Ask about this topic</h2>
          <button
            type="button"
            className="ask-drawer-close"
            aria-label="Close ask drawer"
            onClick={() => setAskOpen(false)}
          >
            ×
          </button>
        </div>
        {!loading && !error && detail ? (
          <AskPanel topicId={state.activeTopicId} layout="column" />
        ) : (
          <p className="topic-view-muted ask-drawer-empty">Load a topic to ask questions.</p>
        )}
      </aside>

      <button
        type="button"
        className="ask-fab"
        aria-expanded={askOpen}
        aria-controls="ask-drawer"
        onClick={() => setAskOpen((open) => !open)}
      >
        <svg className="ask-fab-icon" viewBox="0 0 24 24" aria-hidden="true">
          <path
            d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4v8z"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinejoin="round"
          />
        </svg>
        <span>Ask about this topic</span>
      </button>
    </main>
  )
}
