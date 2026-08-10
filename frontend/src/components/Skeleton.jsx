export default function Skeleton({ className = '', style = undefined }) {
  const classes = className ? `skeleton ${className}` : 'skeleton'
  return <div className={classes} style={style} aria-hidden="true" />
}
