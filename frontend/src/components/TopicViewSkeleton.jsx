import Skeleton from './Skeleton'

export default function TopicViewSkeleton() {
  return (
    <div className="topic-skeleton" aria-hidden="true">
      <section className="topic-section">
        <Skeleton className="skeleton-heading" />
        <Skeleton className="skeleton-notice" />
      </section>

      <section className="topic-section">
        <Skeleton className="skeleton-heading" />
        <div className="skeleton-text-block">
          <Skeleton className="skeleton-line" />
          <Skeleton className="skeleton-line" />
          <Skeleton className="skeleton-line skeleton-line-short" />
          <Skeleton className="skeleton-line" />
          <Skeleton className="skeleton-line skeleton-line-medium" />
        </div>
      </section>

      <section className="topic-section">
        <Skeleton className="skeleton-heading" />
        <div className="topic-gallery">
          <Skeleton className="skeleton-image" />
          <Skeleton className="skeleton-image" />
          <Skeleton className="skeleton-image" />
        </div>
      </section>
    </div>
  )
}
