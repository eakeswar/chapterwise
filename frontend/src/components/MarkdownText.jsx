import Markdown from 'react-markdown'

export default function MarkdownText({ children, className }) {
  return (
    <div className={className}>
      <Markdown>{children || ''}</Markdown>
    </div>
  )
}
