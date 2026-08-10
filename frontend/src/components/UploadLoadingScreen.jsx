import { useEffect, useState } from 'react'
import Skeleton from './Skeleton'

const UPLOAD_HINTS = [
  'Large textbooks may take several minutes on Kaggle extract.',
  'Page images and text are parsed in document order.',
  'Topic sections will appear in the sidebar when ready.',
]

function stepState(stepId, phase) {
  if (stepId === 'upload') {
    if (phase === 'uploading') return 'active'
    if (phase === 'building') return 'done'
    return 'pending'
  }
  if (stepId === 'extract') {
    if (phase === 'uploading') return 'active'
    if (phase === 'building') return 'done'
    return 'pending'
  }
  if (stepId === 'topics') {
    if (phase === 'building') return 'active'
    return 'pending'
  }
  return 'pending'
}

function formatElapsed(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  if (minutes > 0) {
    return `${minutes}:${String(seconds).padStart(2, '0')}`
  }
  return `${seconds}s`
}

export default function UploadLoadingScreen({ phase, fileName, startedAt }) {
  const [hintIndex, setHintIndex] = useState(0)
  const [elapsedMs, setElapsedMs] = useState(0)

  useEffect(() => {
    const timer = window.setInterval(() => {
      setHintIndex((value) => (value + 1) % UPLOAD_HINTS.length)
    }, 4200)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    if (!startedAt) return undefined
    const tick = () => setElapsedMs(Date.now() - startedAt)
    tick()
    const timer = window.setInterval(tick, 250)
    return () => window.clearInterval(timer)
  }, [startedAt])

  const phaseTitle = phase === 'building' ? 'Building topics' : 'Uploading & extracting'

  return (
    <main className="main-panel upload-loading-screen" aria-live="polite" aria-busy="true">
      <div className="upload-loading-card">
        <div className="upload-loading-icon" aria-hidden="true">
          <span className="upload-loading-icon-page upload-loading-icon-page-back" />
          <span className="upload-loading-icon-page upload-loading-icon-page-front" />
        </div>

        <h1 className="upload-loading-title">{phaseTitle}</h1>
        {fileName ? <p className="upload-loading-file">{fileName}</p> : null}
        {startedAt ? (
          <p className="upload-loading-elapsed">Elapsed: {formatElapsed(elapsedMs)}</p>
        ) : null}

        <div className="upload-progress-track" aria-hidden="true">
          <div className="upload-progress-bar" />
        </div>

        <ol className="upload-steps">
          <li className={`upload-step upload-step-${stepState('upload', phase)}`}>
            <span className="upload-step-marker" />
            <span>Upload PDF to server</span>
          </li>
          <li className={`upload-step upload-step-${stepState('extract', phase)}`}>
            <span className="upload-step-marker" />
            <span>Extract text &amp; images</span>
          </li>
          <li className={`upload-step upload-step-${stepState('topics', phase)}`}>
            <span className="upload-step-marker" />
            <span>Build topic tree</span>
          </li>
        </ol>

        <p className="upload-loading-hint" key={hintIndex}>
          {UPLOAD_HINTS[hintIndex]}
        </p>
      </div>

      <div className="upload-loading-preview" aria-hidden="true">
        <div className="upload-loading-preview-sidebar">
          <Skeleton className="upload-preview-line upload-preview-line-short" />
          <Skeleton className="upload-preview-line" />
          <Skeleton className="upload-preview-line" />
          <Skeleton className="upload-preview-line upload-preview-line-medium" />
          <Skeleton className="upload-preview-line" />
          <Skeleton className="upload-preview-line upload-preview-line-short" />
        </div>
        <div className="upload-loading-preview-main">
          <Skeleton className="upload-preview-title" />
          <Skeleton className="upload-preview-block" />
          <Skeleton className="upload-preview-block upload-preview-block-tall" />
          <div className="upload-preview-gallery">
            <Skeleton className="upload-preview-thumb" />
            <Skeleton className="upload-preview-thumb" />
          </div>
        </div>
      </div>
    </main>
  )
}
