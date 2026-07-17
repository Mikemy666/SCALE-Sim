# P12: Inter-GPU communication boundary and arbitration

P12 corrects the runtime communication boundary:

- experts on `DetailedGPUId` are local and incur no inter-GPU dispatch or
  output-combine transfer;
- experts on other GPUs issue remote transfers;
- remote transfers share one simple interconnect timeline, so simultaneous
  experts cannot each consume the full configured bandwidth;
- dispatch queueing delays first compute, and combine queueing delays expert
  completion;
- initial weight loading may overlap the expert's actual dispatch interval when
  configured, including time spent waiting for other interconnect users only as
  queue wait rather than useful overlap.

The interconnect remains a system-constraint model, not a packet simulator.
Runtime reports add dispatch/combine queue wait, and the group summary adds
`TotalInterconnectQueueWait`. Communication bytes in expert and summary reports
now include remote experts only.
