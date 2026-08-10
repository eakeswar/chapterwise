import { createContext, useContext, useReducer } from 'react'

const DocContext = createContext(null)

const initialState = {
  phase: 'idle',
  error: null,
  fileName: null,
  numPages: null,
  pagesExtracted: null,
  pageStart: null,
  pageEnd: null,
  totalChars: null,
  totalImages: null,
  topicTree: null,
  topicCount: null,
  activeTopicId: null,
  topicCache: {},
  uploadStartedAt: null,
  uploadTiming: null,
  buildTiming: null,
}

function reducer(state, action) {
  switch (action.type) {
    case 'START_UPLOAD':
      return {
        ...initialState,
        phase: 'uploading',
        fileName: action.payload.fileName,
        uploadStartedAt: action.payload.startedAt,
      }
    case 'UPLOAD_DONE':
      return {
        ...state,
        phase: 'building',
        numPages: action.payload.num_pages,
        pagesExtracted: action.payload.pages_extracted ?? action.payload.num_pages,
        pageStart: action.payload.page_start ?? 1,
        pageEnd: action.payload.page_end ?? action.payload.num_pages,
        totalChars: action.payload.total_chars,
        totalImages: action.payload.total_images,
        uploadTiming: action.payload.timing ?? null,
      }
    case 'TOPICS_READY':
      return {
        ...state,
        phase: 'ready',
        topicTree: { chapters: action.payload.chapters },
        topicCount: action.payload.topic_count,
        activeTopicId: findFirstTopicId(action.payload.chapters),
        topicCache: {},
        buildTiming: action.payload.timing ?? null,
      }
    case 'SET_ERROR':
      return {
        ...state,
        phase: 'error',
        error: action.payload,
      }
    case 'SET_ACTIVE_TOPIC':
      return {
        ...state,
        activeTopicId: action.payload,
      }
    case 'CACHE_TOPIC':
      return {
        ...state,
        topicCache: {
          ...state.topicCache,
          [action.payload.topic.id]: action.payload,
        },
      }
    default:
      return state
  }
}

function findFirstTopicId(chapters) {
  if (!chapters?.length) return null
  const chapter = chapters[0]
  const section = chapter.sections?.[0]
  const subsection = section?.subsections?.[0]
  if (subsection?.id) return subsection.id
  if (section?.id) return section.id
  return chapter.id || null
}

export function DocProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState)
  return (
    <DocContext.Provider value={{ state, dispatch }}>
      {children}
    </DocContext.Provider>
  )
}

export function useDoc() {
  const context = useContext(DocContext)
  if (!context) {
    throw new Error('useDoc must be used within DocProvider')
  }
  return context
}

export function findTopicById(chapters, topicId) {
  if (!topicId || !chapters) return null

  for (const chapter of chapters) {
    if (chapter.id === topicId) return { ...chapter, level: 'chapter' }
    for (const section of chapter.sections || []) {
      if (section.id === topicId) return { ...section, level: 'section', parentTitle: chapter.title }
      for (const subsection of section.subsections || []) {
        if (subsection.id === topicId) {
          return {
            ...subsection,
            level: 'subsection',
            parentTitle: section.title,
            chapterTitle: chapter.title,
          }
        }
      }
    }
  }
  return null
}
