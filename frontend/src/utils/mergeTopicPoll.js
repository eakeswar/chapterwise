/** Merge a classify-poll GET /topic payload into the live topic view. */

export function mergeTopicPoll(current, incoming) {
  if (!current) return incoming
  if (!incoming) return current

  const incomingTerminal =
    incoming.explanation_status === 'ready' || incoming.explanation_status === 'skipped'
  if (incomingTerminal) {
    return incoming
  }

  return {
    ...incoming,
    explanation: current.explanation,
    explanation_status: current.explanation_status,
    explanation_error: current.explanation_error,
  }
}
