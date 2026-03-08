ALGORITHM OF THE CODE :
for each tick t:
  maybe create a new task and push into global priority queue

  if aging enabled:
    increment wait counter for all queued tasks
    if queue length is high enough:
      compute average queued wait time
      promote at most one task that waited "too long"

  for each processor p:
    if p is idle and queue not empty:
      pop highest-priority task
      run it on p (finish time depends on p speed)