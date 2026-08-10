import { DocProvider, useDoc } from './context/DocContext'
import UploadPanel from './components/UploadPanel'
import TopicTree from './components/TopicTree'
import TopicView from './components/TopicView'
import UploadLoadingScreen from './components/UploadLoadingScreen'

function MainPanel() {
  const { state } = useDoc()
  if (state.phase === 'uploading' || state.phase === 'building') {
    return (
      <UploadLoadingScreen
        phase={state.phase}
        fileName={state.fileName}
        startedAt={state.uploadStartedAt}
      />
    )
  }
  return <TopicView />
}

export default function App() {
  return (
    <DocProvider>
      <div className="app-shell">
        <aside className="sidebar">
          <UploadPanel />
          <TopicTree />
        </aside>
        <MainPanel />
      </div>
    </DocProvider>
  )
}
