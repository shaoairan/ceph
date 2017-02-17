#Coupled Layer (CL) Code#
MDS codes that uses optimal repair bandwidth,
disk bandwidth during a node repair.

It is defined by parameters (k,m,d), k data chunks are encoded to get m parity chunks. 
This code can recover from loss of any m chunks out of the k+m chunks.
During repair, d repair chunks will be used where each repair chunk is a
fraction of the complete chunk.

How to Use:
ceph osd erasure-code-profile set cl\_msr\_profile k=4 m=2 d=5 plugin=jerasure technique=cl\_msr ruleset-failure-domain=osd

**This is only a partial code provided just for reference. Complete code will be made available after an official submission to Ceph Community**.

##New functions added to Ceph's ErasureCodeInterface##


###int is_repair(const set<int> &want_to_read, const set<int> &available_chunks)###
####Input Parameters:####
*want_to_read*: Chunk indices to be decoded.
*available_chunks*: Available chunk indices containing valid data.

Checks if lost chunk can be retrieved using ErasureCodeJerasureCLMSR repair algorithm. Returns 1 if it is it is a repair case, 0 otherwise.


###void get_repair_subchunks(const set<int> &to_repair, const set<int> &helper_chunks, int helper_chunk_ind, map<int, int> &repair_sub_chunks_ind)###
####Input Parameters:####
*to_repair*: Set of chunk indices  that need to be repaired
*helper_chunks*: Set of chunk indices acting as helpers for repairing lost chunks 
*helper_chunk_ind*: Current helper chunk index
####Output Parameters:####
*repair_sub_chunks_ind*: Map to be populated with repair plane indices.

Gets indices of repair planes.Returns nothing.


###int minimum_to_repair(const set<int> &want_to_read, const set<int> &available_chunks, set<int> *minimum)###
####Input Parameters:####
*want_to_read*: Chunk indices to be decoded
*available_chunks*: Available chunk indices containing valid data
####Output Parameters####:
*minimum*: Minimum chunk indices required for retrieval of lost chunks

Finds the chunk indices required to repair lost chunk. Returns minimum no of chunk indices required to retrieve lost chunks.


###int repair(const set<int> &want_to_read, const map<int, bufferlist> &chunks, map<int, bufferlist> *repaired)###
####Input parameters:####
*want_to_read*: Chunk indexes to be decoded
*chunks*: Map containing chunk indices mapped to object chunk required to retrieve lost chunk. 
####Output Parameters####:
*repaired*: Map containing chunk indices mapped to repaired data.

Repairs chunk of lost object using CLMSR algorithm. Returns 0 if repair is successful, 1 otherwise.



##New functions Added to ObjectStore


###int read(const coll_t& _cid, const ghobject_t& lost_oid, uint64_t offset, size_t len, ceph::bufferlist& bl, uint64_t chunk_size, int sub_chunk_cnt, map<int,int> &repair_sub_chunks_ind, uint32_t op_flags, bool allow_eio)###
####Input parameters:####
*_cid*: Collection index/object chunk index 
*lost_oid* : lost object index
*offset*: offset 
*size_t len*: length of data to be read
*chunk_size*: chunk size
*sub_chunk_cnt*: sub chunk count 
*repair_sub_chunks_ind*: Map containing indices mapped to repair plane indices.
*op_flags*: 
*allow_eio*:
####Output Parameters####:
*bl*: bufferlist to be populated with data read
Populates bufferlist with chunk of data of length len corresponding to lost object. Returns size of data read if reading was successful else returns -1.


##New functions Added to FileStore (To read Subchunks)
###void group_repair_nodes_ind(map<int,int> &repair_sub_chunks_ind, set<pair<int,int> > &repair_node_grps)###
####Input Parameters:####
*repair_sub_chunks_ind*: Map containing indices mapped to repair plane indices.
####Output Parameters:####
*repair_node_grps*: Set of pairs to be populated with groups of sub chunks to be read.

Groups consecutive sub chunks into one group to speed up read. Each pair consists of start sub chunk id and end sub chunk id. Returns nothing.


