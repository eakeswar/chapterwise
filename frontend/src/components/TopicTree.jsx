import { useMemo, useState } from 'react'
import { useDoc } from '../context/DocContext'

function pageRangeLabel(topic) {
  if (!topic) return ''
  if (topic.page_start === topic.page_end) return `p.${topic.page_start}`
  return `pp.${topic.page_start}-${topic.page_end}`
}

function TopicNode({ topic, depth, activeTopicId, onSelect }) {
  const isActive = activeTopicId === topic.id

  return (
    <button
      type="button"
      className={`topic-node depth-${depth}${isActive ? ' active' : ''}`}
      onClick={() => onSelect(topic.id)}
    >
      <span className="topic-title">{topic.title}</span>
      <span className="topic-pages">{pageRangeLabel(topic)}</span>
    </button>
  )
}

function SectionBlock({ section, activeTopicId, onSelect, defaultOpen }) {
  const [open, setOpen] = useState(defaultOpen)
  const hasSubsections = (section.subsections || []).length > 0

  return (
    <div className="topic-group">
      <div className="topic-group-header">
        {hasSubsections ? (
          <button
            type="button"
            className="collapse-button"
            aria-expanded={open}
            onClick={() => setOpen((value) => !value)}
          >
            {open ? '▾' : '▸'}
          </button>
        ) : (
          <span className="collapse-spacer" />
        )}
        <TopicNode
          topic={section}
          depth={1}
          activeTopicId={activeTopicId}
          onSelect={onSelect}
        />
      </div>

      {open && hasSubsections ? (
        <div className="topic-children">
          {(section.subsections || []).map((subsection) => (
            <TopicNode
              key={subsection.id}
              topic={subsection}
              depth={2}
              activeTopicId={activeTopicId}
              onSelect={onSelect}
            />
          ))}
        </div>
      ) : null}
    </div>
  )
}

function ChapterBlock({ chapter, activeTopicId, onSelect, defaultOpen }) {
  const [open, setOpen] = useState(defaultOpen)
  const sections = chapter.sections || []

  return (
    <div className="topic-group chapter-group">
      <div className="topic-group-header">
        <button
          type="button"
          className="collapse-button"
          aria-expanded={open}
          onClick={() => setOpen((value) => !value)}
        >
          {open ? '▾' : '▸'}
        </button>
        <TopicNode
          topic={chapter}
          depth={0}
          activeTopicId={activeTopicId}
          onSelect={onSelect}
        />
      </div>

      {open ? (
        <div className="topic-children">
          {sections.map((section, index) => (
            <SectionBlock
              key={section.id}
              section={section}
              activeTopicId={activeTopicId}
              onSelect={onSelect}
              defaultOpen={index === 0}
            />
          ))}
        </div>
      ) : null}
    </div>
  )
}

export default function TopicTree() {
  const { state, dispatch } = useDoc()
  const chapters = state.topicTree?.chapters || []

  const emptyMessage = useMemo(() => {
    if (state.phase === 'uploading') return 'Uploading PDF…'
    if (state.phase === 'building') return 'Extracting topics from the book…'
    if (state.phase === 'error') return 'Fix the upload error and try again.'
    return 'Upload a textbook PDF to see chapters, sections, and subsections here.'
  }, [state.phase])

  function handleSelect(topicId) {
    dispatch({ type: 'SET_ACTIVE_TOPIC', payload: topicId })
  }

  return (
    <div className="topic-tree">
      <div className="topic-tree-heading">Topics</div>

      {chapters.length === 0 ? (
        <div className="topic-tree-empty">{emptyMessage}</div>
      ) : (
        <div className="topic-tree-list">
          {chapters.map((chapter, index) => (
            <ChapterBlock
              key={chapter.id}
              chapter={chapter}
              activeTopicId={state.activeTopicId}
              onSelect={handleSelect}
              defaultOpen={index === 0}
            />
          ))}
        </div>
      )}
    </div>
  )
}
