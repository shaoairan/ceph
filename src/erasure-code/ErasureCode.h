// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*- 
// vim: ts=8 sw=2 smarttab
/*
 * Ceph distributed storage system
 *
 * Copyright (C) 2014 Cloudwatt <libre.licensing@cloudwatt.com>
 *
 * Author: Loic Dachary <loic@dachary.org>
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 * 
 */

#ifndef CEPH_ERASURE_CODE_H
#define CEPH_ERASURE_CODE_H

/*! @file ErasureCode.h
    @brief Base class for erasure code plugins implementors

 */ 

#include <vector>

#include "ErasureCodeInterface.h"

namespace ceph {

  class ErasureCode : public ErasureCodeInterface {
  public:
    static const unsigned SIMD_ALIGN;

    vector<int> chunk_mapping;
    ErasureCodeProfile _profile;

    ~ErasureCode() override {}

    int init(ErasureCodeProfile &profile, ostream *ss) override {
      _profile = profile;
      return 0;
    }

    const ErasureCodeProfile &get_profile() const override {
      return _profile;
    }

    int sanity_check_k(int k, ostream *ss);

    unsigned int get_coding_chunk_count() const override {
      return get_chunk_count() - get_data_chunk_count();
    }

    virtual unsigned int get_sub_chunk_count() {
      return 1;
    }

    virtual int get_repair_sub_chunk_count(const set<int> &want_to_read){
      return 1;
    }

    int minimum_to_decode(const set<int> &want_to_read,
                                  const set<int> &available_chunks,
                                  set<int> *minimum) override;

    int minimum_to_decode_with_cost(const set<int> &want_to_read,
                                            const map<int, int> &available,
                                            set<int> *minimum) override;

    int encode_prepare(const bufferlist &raw,
                       map<int, bufferlist> &encoded) const;

    int encode(const set<int> &want_to_encode,
                       const bufferlist &in,
                       map<int, bufferlist> *encoded) override;

    int encode_chunks(const set<int> &want_to_encode,
                              map<int, bufferlist> *encoded) override;

    int decode(const set<int> &want_to_read,
                       const map<int, bufferlist> &chunks,
                       map<int, bufferlist> *decoded) override;

    int decode_chunks(const set<int> &want_to_read,
                              const map<int, bufferlist> &chunks,
                              map<int, bufferlist> *decoded) override;

    virtual int minimum_to_repair(const set<int> &want_to_read, 
                                  const set<int> &available_chunks, 
                                  set<int> *minimum);

    virtual int is_repair(const set<int> &want_to_read,
                       const set<int> &available_chunks);

    
    virtual void get_repair_subchunks(const set<int> &to_repair,
                                   const set<int> &helper_chunks,
                                   int helper_chunk_ind,
                                   map<int, int> &repair_sub_chunks_ind);

    virtual int repair(const set<int> &want_to_read,
                       const map<int, bufferlist> &chunks,
                       map<int, bufferlist> *repaired);

    const vector<int> &get_chunk_mapping() const override;

    int to_mapping(const ErasureCodeProfile &profile,
		   ostream *ss);

    static int to_int(const std::string &name,
		      ErasureCodeProfile &profile,
		      int *value,
		      const std::string &default_value,
		      ostream *ss);

    static int to_bool(const std::string &name,
		       ErasureCodeProfile &profile,
		       bool *value,
		       const std::string &default_value,
		       ostream *ss);

    static int to_string(const std::string &name,
			 ErasureCodeProfile &profile,
			 std::string *value,
			 const std::string &default_value,
			 ostream *ss);

    int decode_concat(const map<int, bufferlist> &chunks,
			      bufferlist *decoded) override;

  protected:
    int parse(const ErasureCodeProfile &profile,
	      ostream *ss);

  private:
    int chunk_index(unsigned int i) const;
  };
}

#endif
