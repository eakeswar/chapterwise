import { useRef, useState } from 'react'
import { useDoc } from '../context/DocContext'
import { buildTopics, uploadPdf } from '../utils/apiClient'

function formatSeconds(value) {
  if (value == null || Number.isNaN(value)) return '—'
  if (value < 60) return `${value.toFixed(1)}s`
  const minutes = Math.floor(value / 60)
  const seconds = Math.round(value % 60)
  return `${minutes}m ${seconds}s`
}

function phaseLabel(phase) {
  if (phase === 'uploading') return 'Uploading PDF…'
  if (phase === 'building') return 'Building topics…'
  return 'Upload PDF'
}

export default function UploadPanel() {
  const { state, dispatch } = useDoc()
  const inputRef = useRef(null)
  const [testRange, setTestRange] = useState(false)
  const [pageEnd, setPageEnd] = useState('20')
  const busy = state.phase === 'uploading' || state.phase === 'building'

  async function handleFileSelected(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return

    const startedAt = Date.now()
    dispatch({ type: 'START_UPLOAD', payload: { fileName: file.name, startedAt } })

    try {
      const uploadOptions = {}
      if (testRange) {
        const end = Number.parseInt(pageEnd, 10)
        if (Number.isFinite(end) && end >= 1) {
          uploadOptions.page_start = 1
          uploadOptions.page_end = end
        }
      }

      const uploadResult = await uploadPdf(file, uploadOptions)
      dispatch({ type: 'UPLOAD_DONE', payload: uploadResult })

      const buildOptions = {}
      if (testRange) {
        buildOptions.page_start = uploadResult.page_start ?? 1
        buildOptions.page_end = uploadResult.page_end ?? uploadResult.pages_extracted
      }

      const topicsResult = await buildTopics(buildOptions)
      dispatch({ type: 'TOPICS_READY', payload: topicsResult })
    } catch (error) {
      dispatch({
        type: 'SET_ERROR',
        payload: error instanceof Error ? error.message : 'Upload failed',
      })
    }
  }

  function openFilePicker() {
    if (!busy) inputRef.current?.click()
  }

  return (
    <div className="upload-panel">
      <div className="brand-row">
        <div className="brand-mark">C</div>
        <div>
          <div className="brand-title">chapterwise</div>
          {state.pagesExtracted ? (
            <div className="brand-meta">
              {state.pagesExtracted === state.numPages
                ? `${state.numPages} pages loaded`
                : `${state.pagesExtracted} of ${state.numPages} pages extracted`}
            </div>
          ) : (
            <div className="brand-meta">Textbook → topics</div>
          )}
        </div>
      </div>

      <button
        type="button"
        className="primary-button"
        disabled={busy}
        onClick={openFilePicker}
      >
        {busy ? phaseLabel(state.phase) : 'Upload PDF'}
      </button>

      <input
        ref={inputRef}
        type="file"
        accept=".pdf,application/pdf"
        className="hidden-input"
        disabled={busy}
        onChange={handleFileSelected}
      />

      <label className="range-toggle">
        <input
          type="checkbox"
          checked={testRange}
          disabled={busy}
          onChange={(event) => setTestRange(event.target.checked)}
        />
        <span>Test first pages only</span>
      </label>

      {testRange ? (
        <label className="range-field">
          <span>Through page</span>
          <input
            type="number"
            min="1"
            value={pageEnd}
            disabled={busy}
            onChange={(event) => setPageEnd(event.target.value)}
          />
        </label>
      ) : null}

      {state.fileName ? (
        <div className="upload-meta">
          <div className="upload-meta-label">File</div>
          <div className="upload-meta-value">{state.fileName}</div>
        </div>
      ) : null}

      {state.phase === 'ready' ? (
        <div className="upload-meta success">
          {state.topicCount} topics · {state.totalChars?.toLocaleString()} chars ·{' '}
          {state.totalImages} images
          {state.uploadTiming ? (
            <>
              <br />
              Upload + extract: {formatSeconds(state.uploadTiming.total_seconds)}
              {state.uploadTiming.file_mb != null
                ? ` (${state.uploadTiming.file_mb} MB)`
                : null}
              {state.buildTiming?.total_seconds != null
                ? ` · Topics: ${formatSeconds(state.buildTiming.total_seconds)}`
                : null}
            </>
          ) : null}
        </div>
      ) : null}

      {state.error ? <div className="error-banner">{state.error}</div> : null}
    </div>
  )
}
