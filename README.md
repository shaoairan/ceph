===========================================
Coupled Layer (CL) Code

MDS codes that uses optimal repair bandwidth, 
disk bandwidth during a node repair.

Its defined by parameters (k,m,d), k data chunks are encoded
to get m parity chunks. This code can recover any m lost chunks. 
During repair, d repair chunks will be used where each repair chunk is a 
fraction of the complete chunk. 

How to Use:
ceph osd erasure-code-profile set cl\_msr\_profile k=4 m=2 d=5 plugin=jerasure technique=cl\_msr ruleset-failure-domain=osd


****
This is only a partial code (just for the reference). Full code will be made available after an official submission.
