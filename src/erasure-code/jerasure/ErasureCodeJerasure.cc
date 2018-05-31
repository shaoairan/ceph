// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*- 
// vim: ts=8 sw=2 smarttab
/*
 * Ceph distributed storage system
 *
 * Copyright (C) 2013,2014 Cloudwatt <libre.licensing@cloudwatt.com>
 * Copyright (C) 2014 Red Hat <contact@redhat.com>
 *
 * Author: Loic Dachary <loic@dachary.org>
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 * 
 */
#include <dlfcn.h>
#include "common/debug.h"
#include "ErasureCodeJerasure.h"
#include "crush/CrushWrapper.h"
#include "osd/osd_types.h"
#include <sstream>
#include "math.h"
#include "gpu/mylibrary.h"
extern "C" {
#include "jerasure.h"
#include "reed_sol.h"
#include "galois.h"
#include "cauchy.h"
#include "liberation.h"
}

#define LARGEST_VECTOR_WORDSIZE 16

#define dout_context g_ceph_context
#define dout_subsys ceph_subsys_osd
#undef dout_prefix
#define dout_prefix _prefix(_dout)

#define talloc(type, num) (type *) malloc(sizeof(type)*(num))

#define FT(A) FunctionTest printFunctionName(#A)

class FunctionTest
{
  static int tabs;
  std::string a;
  public:
    FunctionTest( std::string a_ ):a(a_)
    {
      
      for( int i = 0; i < tabs; i ++ )
      {
          printf("\t");
      }
      std::cout << "entering:: " << a << "\n";
      tabs ++;
    }

    ~FunctionTest()
    {
      tabs --;
      for( int i = 0; i < tabs; i ++ )
      {
          printf("\t");
      }
      std::cout << "leave:: " << a << "\n";
    }
};

int FunctionTest::tabs = 2;

static ostream& _prefix(std::ostream* _dout)
{
  return *_dout << "ErasureCodeJerasure: ";
}

static int pow_int(int a, int x){
  int power = 1;

  while (x)
    {
      if (x & 1)power *= a;
      x /= 2;
      a *= a;
    }
  return power;
}

int ErasureCodeJerasure::create_ruleset(const string &name,
					CrushWrapper &crush,
					ostream *ss) const
{
  int ruleid = crush.add_simple_ruleset(name, ruleset_root, ruleset_failure_domain,
					"indep", pg_pool_t::TYPE_ERASURE, ss);
  if (ruleid < 0)
    return ruleid;
  else {
    crush.set_rule_mask_max_size(ruleid, get_chunk_count());
    return crush.get_rule_mask_ruleset(ruleid);
  }
}

int ErasureCodeJerasure::init(ErasureCodeProfile& profile, ostream *ss)
{
  FT(ErasureCodeJerasure::init);
  int err = 0;
  dout(10) << "technique=" << technique << dendl;
  profile["technique"] = technique;
  err |= to_string("ruleset-root", profile,
		   &ruleset_root,
		   DEFAULT_RULESET_ROOT, ss);
  err |= to_string("ruleset-failure-domain", profile,
		   &ruleset_failure_domain,
		   DEFAULT_RULESET_FAILURE_DOMAIN, ss);
  err |= parse(profile, ss);
  if (err)
    return err;
  prepare();
  ErasureCode::init(profile, ss);
  return err;
}

int ErasureCodeJerasure::parse(ErasureCodeProfile &profile,
			       ostream *ss)
{
  int err = ErasureCode::parse(profile, ss);
  err |= to_int("k", profile, &k, DEFAULT_K, ss);
  err |= to_int("m", profile, &m, DEFAULT_M, ss);
  err |= to_int("w", profile, &w, DEFAULT_W, ss);
  if (chunk_mapping.size() > 0 && (int)chunk_mapping.size() != k + m) {
    *ss << "mapping " << profile.find("mapping")->second
	<< " maps " << chunk_mapping.size() << " chunks instead of"
	<< " the expected " << k + m << " and will be ignored" << std::endl;
    chunk_mapping.clear();
    err = -EINVAL;
  }
  err |= sanity_check_k(k, ss);
  return err;
}

unsigned int ErasureCodeJerasure::get_chunk_size(unsigned int object_size) const
{
  unsigned alignment = get_alignment();
  if (per_chunk_alignment) {
    unsigned chunk_size = object_size / k;
    if (object_size % k)
      chunk_size++;
    dout(20) << "get_chunk_size: chunk_size " << chunk_size
	     << " must be modulo " << alignment << dendl; 
    assert(alignment <= chunk_size);
    unsigned modulo = chunk_size % alignment;
    if (modulo) {
      dout(10) << "get_chunk_size: " << chunk_size
	       << " padded to " << chunk_size + alignment - modulo << dendl;
      chunk_size += alignment - modulo;
    }
    return chunk_size;
  } else {
    unsigned tail = object_size % alignment;
    unsigned padded_length = object_size + ( tail ?  ( alignment - tail ) : 0 );
    assert(padded_length % (k*sub_chunk_no) == 0);
    return padded_length / k;
  }
}

int ErasureCodeJerasure::encode_chunks(const set<int> &want_to_encode,
				       map<int, bufferlist> *encoded)
{

  FT( ErasureCodeJerasure::encode_chunks );
  int chunk_size = (*encoded->begin()).second.length();
  char *chunks[k + m + nu];

  for (int i = 0; i < k + m; i++){
    if (i < k) {
      chunks[i] = (*encoded)[i].c_str();
    } else {
    chunks[i+nu] = (*encoded)[i].c_str();
    }
  }

  for(int i = k; i < k+nu; i++){
    //create buffers that will be cleaned later   
    chunks[i] = (char*) malloc(chunk_size);
    memset(chunks[i],0, chunk_size);
  }

  jerasure_encode(&chunks[0], &chunks[k+nu], (*encoded)[0].length());

  //clean the memory allocated for nu shortened chunks
  for(int i=k ; i < k+nu; i++){
    free(chunks[i]);
  }

  return 0;
}

int ErasureCodeJerasure::decode_chunks(const set<int> &want_to_read,
				       const map<int, bufferlist> &chunks,
				       map<int, bufferlist> *decoded)
{
  FT(ErasureCodeJerasure::decode_chunks);
  unsigned blocksize = (*chunks.begin()).second.length();
  int erasures[k + m + 1];
  int erasures_count = 0;
  char *data[k+nu];
  char *coding[m];
  for (int i =  0; i < k + m ; i++) {
    if (chunks.find(i) == chunks.end()) {
      if(i < k)
        erasures[erasures_count] = i;
      else
        erasures[erasures_count] = i+nu;
      erasures_count++;
    }
    if (i < k)
      data[i] = (*decoded)[i].c_str();
    else
      coding[i - k] = (*decoded)[i].c_str();
  }

  for(int i=k; i < k+nu; i++){
    data[i] = (char*)malloc(blocksize);
    if(data[i]==NULL){
      assert(0);
    }
    memset(data[i], 0, blocksize);
  }

  erasures[erasures_count] = -1;

  assert(erasures_count > 0);

  int res = jerasure_decode(erasures, data, coding, blocksize);

  for (int i=k ; i < k+nu; i++){
    free(data[i]);
  }

  return res;
}

int ErasureCodeJerasure::minimum_to_decode2(const set<int> &want_to_read,
                                  const set<int> &available,
                                  map<int, list<pair<int,int>>> *minimum){
  return ErasureCode::minimum_to_decode2(want_to_read, available, minimum);
}

int ErasureCodeJerasure::decode2(const set<int> &want_to_read,
                const map<int, bufferlist> &chunks,
                map<int, bufferlist> *decoded, int chunk_size)
{
  FT(ErasureCodeJerasure::decode2);
  return ErasureCode::decode(want_to_read, chunks, decoded);
}

bool ErasureCodeJerasure::is_prime(int value)
{
  int prime55[] = {
    2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,
    73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,
    151,157,163,167,173,179,
    181,191,193,197,199,211,223,227,229,233,239,241,251,257
  };
  int i;
  for (i = 0; i < 55; i++)
    if (value == prime55[i])
      return true;
  return false;
}

// 
// ErasureCodeJerasureReedSolomonVandermonde
//
void ErasureCodeJerasureReedSolomonVandermonde::jerasure_encode(char **data,
                                                                char **coding,
                                                                int blocksize)
{
  jerasure_matrix_encode(k, m, w, matrix, data, coding, blocksize);
}

int ErasureCodeJerasureReedSolomonVandermonde::jerasure_decode(int *erasures,
                                                                char **data,
                                                                char **coding,
                                                                int blocksize)
{
  return jerasure_matrix_decode(k, m, w, matrix, 1,
				erasures, data, coding, blocksize);
}

unsigned ErasureCodeJerasureReedSolomonVandermonde::get_alignment() const
{
  if (per_chunk_alignment) {
    return w * LARGEST_VECTOR_WORDSIZE;
  } else {
    unsigned alignment = k*w*sizeof(int);
    if ( ((w*sizeof(int))%LARGEST_VECTOR_WORDSIZE) )
      alignment = k*w*LARGEST_VECTOR_WORDSIZE;
    return alignment;
  }
}

int ErasureCodeJerasureReedSolomonVandermonde::parse(ErasureCodeProfile &profile,
						     ostream *ss)
{
  int err = 0;
  err |= ErasureCodeJerasure::parse(profile, ss);
  if (w != 8 && w != 16 && w != 32) {
    *ss << "ReedSolomonVandermonde: w=" << w
	<< " must be one of {8, 16, 32} : revert to " << DEFAULT_W << std::endl;
    profile["w"] = "8";
    err |= to_int("w", profile, &w, DEFAULT_W, ss);
    err = -EINVAL;
  }
  err |= to_bool("jerasure-per-chunk-alignment", profile,
		 &per_chunk_alignment, "false", ss);
  return err;
}

void ErasureCodeJerasureReedSolomonVandermonde::prepare()
{
  matrix = reed_sol_vandermonde_coding_matrix(k, m, w);
}

// 
// ErasureCodeJerasureReedSolomonRAID6
//
void ErasureCodeJerasureReedSolomonRAID6::jerasure_encode(char **data,
                                                                char **coding,
                                                                int blocksize)
{
  reed_sol_r6_encode(k, w, data, coding, blocksize);
}

int ErasureCodeJerasureReedSolomonRAID6::jerasure_decode(int *erasures,
							 char **data,
							 char **coding,
							 int blocksize)
{
  return jerasure_matrix_decode(k, m, w, matrix, 1, erasures, data, coding, blocksize);
}

unsigned ErasureCodeJerasureReedSolomonRAID6::get_alignment() const
{
  if (per_chunk_alignment) {
    return w * LARGEST_VECTOR_WORDSIZE;
  } else {
    unsigned alignment = k*w*sizeof(int);
    if ( ((w*sizeof(int))%LARGEST_VECTOR_WORDSIZE) )
      alignment = k*w*LARGEST_VECTOR_WORDSIZE;
    return alignment;
  }
}

int ErasureCodeJerasureReedSolomonRAID6::parse(ErasureCodeProfile &profile,
					       ostream *ss)
{
  int err = ErasureCodeJerasure::parse(profile, ss);
  profile.erase("m");
  m = 2;
  if (w != 8 && w != 16 && w != 32) {
    *ss << "ReedSolomonRAID6: w=" << w
	<< " must be one of {8, 16, 32} : revert to 8 " << std::endl;
    profile["w"] = "8";
    err |= to_int("w", profile, &w, DEFAULT_W, ss);
    err = -EINVAL;
  }
  return err;
}

void ErasureCodeJerasureReedSolomonRAID6::prepare()
{
  matrix = reed_sol_r6_coding_matrix(k, w);
}

// 
// ErasureCodeJerasureCauchy
//
void ErasureCodeJerasureCauchy::jerasure_encode(char **data,
						char **coding,
						int blocksize)
{
  jerasure_schedule_encode(k, m, w, schedule,
			   data, coding, blocksize, packetsize);
}

int ErasureCodeJerasureCauchy::jerasure_decode(int *erasures,
					       char **data,
					       char **coding,
					       int blocksize)
{
  return jerasure_schedule_decode_lazy(k, m, w, bitmatrix,
				       erasures, data, coding, blocksize, packetsize, 1);
}

unsigned ErasureCodeJerasureCauchy::get_alignment() const
{
  if (per_chunk_alignment) {
    unsigned alignment = w * packetsize;
    unsigned modulo = alignment % LARGEST_VECTOR_WORDSIZE;
    if (modulo)
      alignment += LARGEST_VECTOR_WORDSIZE - modulo;
    return alignment;
  } else {
    unsigned alignment = k*w*packetsize*sizeof(int);
    if ( ((w*packetsize*sizeof(int))%LARGEST_VECTOR_WORDSIZE) )
      alignment = k*w*packetsize*LARGEST_VECTOR_WORDSIZE;
    return alignment;
  }  
}

int ErasureCodeJerasureCauchy::parse(ErasureCodeProfile &profile,
				     ostream *ss)
{
  int err = ErasureCodeJerasure::parse(profile, ss);
  err |= to_int("packetsize", profile, &packetsize, DEFAULT_PACKETSIZE, ss);
  err |= to_bool("jerasure-per-chunk-alignment", profile,
		 &per_chunk_alignment, "false", ss);
  return err;
}

void ErasureCodeJerasureCauchy::prepare_schedule(int *matrix)
{
  bitmatrix = jerasure_matrix_to_bitmatrix(k, m, w, matrix);
  schedule = jerasure_smart_bitmatrix_to_schedule(k, m, w, bitmatrix);
}

// 
// ErasureCodeJerasureCauchyOrig
//
void ErasureCodeJerasureCauchyOrig::prepare()
{
  int *matrix = cauchy_original_coding_matrix(k, m, w);
  prepare_schedule(matrix);
  free(matrix);
}

// 
// ErasureCodeJerasureCauchyGood
//
void ErasureCodeJerasureCauchyGood::prepare()
{
  int *matrix = cauchy_good_general_coding_matrix(k, m, w);
  prepare_schedule(matrix);
  free(matrix);
}

// 
// ErasureCodeJerasureLiberation
//
ErasureCodeJerasureLiberation::~ErasureCodeJerasureLiberation()
{
  if (bitmatrix)
    free(bitmatrix);
  if (schedule)
    jerasure_free_schedule(schedule);
}

void ErasureCodeJerasureLiberation::jerasure_encode(char **data,
                                                    char **coding,
                                                    int blocksize)
{
  jerasure_schedule_encode(k, m, w, schedule, data,
			   coding, blocksize, packetsize);
}

int ErasureCodeJerasureLiberation::jerasure_decode(int *erasures,
                                                    char **data,
                                                    char **coding,
                                                    int blocksize)
{
  return jerasure_schedule_decode_lazy(k, m, w, bitmatrix, erasures, data,
				       coding, blocksize, packetsize, 1);
}

unsigned ErasureCodeJerasureLiberation::get_alignment() const
{
  unsigned alignment = k*w*packetsize*sizeof(int);
  if ( ((w*packetsize*sizeof(int))%LARGEST_VECTOR_WORDSIZE) )
    alignment = k*w*packetsize*LARGEST_VECTOR_WORDSIZE;
  return alignment;
}

bool ErasureCodeJerasureLiberation::check_k(ostream *ss) const
{
  if (k > w) {
    *ss << "k=" << k << " must be less than or equal to w=" << w << std::endl;
    return false;
  } else {
    return true;
  }
}

bool ErasureCodeJerasureLiberation::check_w(ostream *ss) const
{
  if (w <= 2 || !is_prime(w)) {
    *ss <<  "w=" << w << " must be greater than two and be prime" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool ErasureCodeJerasureLiberation::check_packetsize_set(ostream *ss) const
{
  if (packetsize == 0) {
    *ss << "packetsize=" << packetsize << " must be set" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool ErasureCodeJerasureLiberation::check_packetsize(ostream *ss) const
{
  if ((packetsize%(sizeof(int))) != 0) {
    *ss << "packetsize=" << packetsize
	<< " must be a multiple of sizeof(int) = " << sizeof(int) << std::endl;
    return false;
  } else {
    return true;
  }
}

int ErasureCodeJerasureLiberation::revert_to_default(ErasureCodeProfile &profile,
						     ostream *ss)
{
  int err = 0;
  *ss << "reverting to k=" << DEFAULT_K << ", w="
      << DEFAULT_W << ", packetsize=" << DEFAULT_PACKETSIZE << std::endl;
  profile["k"] = DEFAULT_K;
  err |= to_int("k", profile, &k, DEFAULT_K, ss);
  profile["w"] = DEFAULT_W;
  err |= to_int("w", profile, &w, DEFAULT_W, ss);
  profile["packetsize"] = DEFAULT_PACKETSIZE;
  err |= to_int("packetsize", profile, &packetsize, DEFAULT_PACKETSIZE, ss);
  return err;
}

int ErasureCodeJerasureLiberation::parse(ErasureCodeProfile &profile,
					 ostream *ss)
{
  int err = ErasureCodeJerasure::parse(profile, ss);
  err |= to_int("packetsize", profile, &packetsize, DEFAULT_PACKETSIZE, ss);

  bool error = false;
  if (!check_k(ss))
    error = true;
  if (!check_w(ss))
    error = true;
  if (!check_packetsize_set(ss) || !check_packetsize(ss))
    error = true;
  if (error) {
    revert_to_default(profile, ss);
    err = -EINVAL;
  }
  return err;
}

void ErasureCodeJerasureLiberation::prepare()
{
  bitmatrix = liberation_coding_bitmatrix(k, w);
  schedule = jerasure_smart_bitmatrix_to_schedule(k, m, w, bitmatrix);
}

// 
// ErasureCodeJerasureBlaumRoth
//
bool ErasureCodeJerasureBlaumRoth::check_w(ostream *ss) const
{
  // back in Firefly, w = 7 was the default and produced useable 
  // chunks. Tolerate this value for backward compatibility.
  if (w == 7)
    return true;
  if (w <= 2 || !is_prime(w+1)) {
    *ss <<  "w=" << w << " must be greater than two and "
	<< "w+1 must be prime" << std::endl;
    return false;
  } else {
    return true;
  }
}

void ErasureCodeJerasureBlaumRoth::prepare()
{
  bitmatrix = blaum_roth_coding_bitmatrix(k, w);
  schedule = jerasure_smart_bitmatrix_to_schedule(k, m, w, bitmatrix);
}

// 
// ErasureCodeJerasureLiber8tion
//
int ErasureCodeJerasureLiber8tion::parse(ErasureCodeProfile &profile,
					 ostream *ss)
{
  int err = ErasureCodeJerasure::parse(profile, ss);
  profile.erase("m");
  err |= to_int("m", profile, &m, DEFAULT_M, ss);
  profile.erase("w");
  err |= to_int("w", profile, &w, DEFAULT_W, ss);
  err |= to_int("packetsize", profile, &packetsize, DEFAULT_PACKETSIZE, ss);

  bool error = false;
  if (!check_k(ss))
    error = true;
  if (!check_packetsize_set(ss))
    error = true;
  if (error) {
    revert_to_default(profile, ss);
    err = -EINVAL;
  }
  return err;
}

void ErasureCodeJerasureLiber8tion::prepare()
{
  bitmatrix = liber8tion_coding_bitmatrix(k);
  schedule = jerasure_smart_bitmatrix_to_schedule(k, m, w, bitmatrix);
}

//
// ErasureCodeJerasureCLMSR
//

void ErasureCodeJerasureCLMSR::jerasure_encode(char **data, char **coding, int blocksize)
{
  FT(ErasureCodeJerasureCLMSR::jerasure_encode);
  if(encode_systematic(data, coding, blocksize) == -1){
      dout(0) << "error in encode_systematic" << dendl;
  }
}

int ErasureCodeJerasureCLMSR::jerasure_decode(int *erasures, char **data,
                                                             char **coding,
                                                             int chunksize)
{
  FT(ErasureCodeJerasureCLMSR::jerasure_decode);
  int r = decode_layered(erasures, data, coding, chunksize);
  return r;
}

unsigned ErasureCodeJerasureCLMSR::get_alignment() const
{
  if (per_chunk_alignment) {
    return w * LARGEST_VECTOR_WORDSIZE;
  } else {
    unsigned alignment = k*sub_chunk_no*w*sizeof(int);
    if ( ((w*sizeof(int))%LARGEST_VECTOR_WORDSIZE) )
      alignment = k*sub_chunk_no*w*LARGEST_VECTOR_WORDSIZE;
    return alignment;
  }
}

int ErasureCodeJerasureCLMSR::create_ruleset(const string &name,
                                   CrushWrapper &crush,
                                   ostream *ss) const
{
  if (crush.rule_exists(name)) {
    *ss << "rule " << name << " exists";
    return -EEXIST;
  }
  if (!crush.name_exists(ruleset_root)) {
    *ss << "root item " << ruleset_root << " does not exist";
    return -ENOENT;
  }
  int root = crush.get_item_id(ruleset_root);

  int ruleset = 0;
  int rno = 0;
  for (rno = 0; rno < crush.get_max_rules(); rno++) {
    if (!crush.rule_exists(rno) && !crush.ruleset_exists(rno))
       break;
  }
  ruleset = rno;

  int steps = 4 + ruleset_steps.size();
  int min_rep = 3;
  int max_rep = get_chunk_count();
  int ret;
  ret = crush.add_rule(steps, ruleset, pg_pool_t::TYPE_ERASURE,
                  min_rep, max_rep, rno);
  assert(ret == rno);
  int step = 0;

  ret = crush.set_rule_step(rno, step++, CRUSH_RULE_SET_CHOOSELEAF_TRIES, 5, 0);
  assert(ret == 0);
  ret = crush.set_rule_step(rno, step++, CRUSH_RULE_SET_CHOOSE_TRIES, 100, 0);
  assert(ret == 0);
  ret = crush.set_rule_step(rno, step++, CRUSH_RULE_TAKE, root, 0);
  assert(ret == 0);
  // [ [ "choose", "rack", 2 ],
  //   [ "chooseleaf", "host", 5 ] ]
  for (vector<Step>::const_iterator i = ruleset_steps.begin();
       i != ruleset_steps.end();
       ++i) {
    int op = i->op == "chooseleaf" ?
      CRUSH_RULE_CHOOSELEAF_INDEP : CRUSH_RULE_CHOOSE_INDEP;
    int type = crush.get_type_id(i->type);
    if (type < 0) {
      *ss << "unknown crush type " << i->type;
      return -EINVAL;
    }
    ret = crush.set_rule_step(rno, step++, op, i->n, type);
    assert(ret == 0);
  }
  ret = crush.set_rule_step(rno, step++, CRUSH_RULE_EMIT, 0, 0);
  assert(ret == 0);
  crush.set_rule_name(rno, name);
  return ruleset;
}

int ErasureCodeJerasureCLMSR::parse_ruleset(ErasureCodeProfile &profile,
                                  ostream *ss)
{
  int err = 0;
  err |= to_string("ruleset-root", profile,
                   &ruleset_root,
                   "default", ss);

  if (profile.count("ruleset-steps") != 0) {
    ruleset_steps.clear();
    string str = profile.find("ruleset-steps")->second;
    json_spirit::mArray description;
    try {
      json_spirit::mValue json;
      json_spirit::read_or_throw(str, json);

      if (json.type() != json_spirit::array_type) {
        *ss << "ruleset-steps='" << str
            << "' must be a JSON array but is of type "
            << json.type() << " instead" << std::endl;
        return ERROR_LRC_ARRAY;
      }
      description = json.get_array();
    } catch (json_spirit::Error_position &e) {
      *ss << "failed to parse ruleset-steps='" << str << "'"
          << " at line " << e.line_ << ", column " << e.column_
          << " : " << e.reason_ << std::endl;
      return ERROR_LRC_PARSE_JSON;
    }

    int position = 0;
    for (vector<json_spirit::mValue>::iterator i = description.begin();
         i != description.end();
         ++i, position++) {
      if (i->type() != json_spirit::array_type) {
        stringstream json_string;
        json_spirit::write(*i, json_string);
        *ss << "element of the array "
            << str << " must be a JSON array but "
            << json_string.str() << " at position " << position
            << " is of type " << i->type() << " instead" << std::endl;
        return ERROR_LRC_ARRAY;
      }
      int r = parse_ruleset_step(str, i->get_array(), ss);
      if (r)
        return r;
    }
  }
  return 0;
}

int ErasureCodeJerasureCLMSR::parse_ruleset_step(string description_string,
                                       json_spirit::mArray description,
                                       ostream *ss)
{
  stringstream json_string;
  json_spirit::write(description, json_string);
  string op;
  string type;
  int n = 0;
  int position = 0;
  for (vector<json_spirit::mValue>::iterator i = description.begin();
       i != description.end();
       ++i, position++) {
    if ((position == 0 || position == 1) &&
        i->type() != json_spirit::str_type) {
      *ss << "element " << position << " of the array "
          << json_string.str() << " found in " << description_string
          << " must be a JSON string but is of type "
         << i->type() << " instead" << std::endl;
      return position == 0 ? ERROR_LRC_RULESET_OP : ERROR_LRC_RULESET_TYPE;
    }
    if (position == 2 && i->type() != json_spirit::int_type) {
      *ss << "element " << position << " of the array "
          << json_string.str() << " found in " << description_string
          << " must be a JSON int but is of type "
          << i->type() << " instead" << std::endl;
      return ERROR_LRC_RULESET_N;
    }

    if (position == 0)
      op = i->get_str();
    if (position == 1)
      type = i->get_str();
    if (position == 2)
      n = i->get_int();
  }   
  ruleset_steps.push_back(Step(op, type, n));
  return 0;
} 

int ErasureCodeJerasureCLMSR::parse(ErasureCodeProfile &profile,
						     ostream *ss)
{
  FT(ErasureCodeJerasureCLMSR::parse);
  printf("======================doing cuda!!");
/*  void *myGpuLibrary = dlopen("libcephCudaLibTest.so", RTLD_NOW);
  if (!myGpuLibrary) {
    *ss << "load dlopen(" << " ==libcephCudaLibTest.so== " << "): " << dlerror();
    return -EIO;
  }

  const int (*docalDlopen)() =
    (const int (*)())dlsym(myGpuLibrary, "docal");
  docalDlopen();*/
  //docal666();
  
  docal();
  printf("======================cuda done!!");

  int err = 0;
  err |= ErasureCodeJerasure::parse(profile, ss);
  err |= to_int("d", profile, &d, std::to_string(k+m-1), ss);
  
  //*ss << "parse of cl_msr value of d is " << d << std::endl;

  //check for mds block input
  if (profile.find("mds_type") == profile.end() ||
      profile.find("mds_type")->second.size() == 0){
    mds_block = VANDERMONDE_RS;
    *ss << "mds_type not found in profile" << std::endl;
  } else {
    std::string p = profile.find("mds_type")->second;
    *ss << "recieved mds_type as " << p << std::endl;

    if(p == "reed_sol_van") mds_block = VANDERMONDE_RS;
    else if(p == "cauchy_mds") mds_block = CAUCHY_MDS;
    else{
      mds_block = VANDERMONDE_RS;
      *ss << "mds_type " << p << "is not currently supported using the default reed_sol_van" << std::endl;
    }

  }
   
  //*ss  << "finished checking mds_type" << std::endl;
  if (w != 8 ){//&& w != 16 && w != 32) {
    *ss << "CLMSR: w=" << w
	<< " must be one of {8, 16, 32} : revert to " << DEFAULT_W << std::endl;
    profile["w"] = "8";
    err |= to_int("w", profile, &w, DEFAULT_W, ss);
    err = -EINVAL;
    return err;
  }



  if ((d < k) || (d > k+m-1)){
    *ss << "value of d " << d
	<< " must be within [ " << k << "," << k+m-1 << "]" << std::endl;
    err = -EINVAL;
    return err;
  }
  q = d-k+1;//this will be m when d=(n-1)=k+m-1

  if( (k+m)%q ){
    nu = q - (k+m)%q;
    *ss << "CLMSR: (k+m)%q=" << (k+m)%q
	 << "q doesn't divide k+m, to use shortening" << std::endl;
  } else {
    nu = 0;
  }

  if(k+m+nu > 254){
    err = -EINVAL;
    return err;
  }

  err |= to_bool("jerasure-per-chunk-alignment", profile,
		 &per_chunk_alignment, "false", ss);

  /*if (profile.find("ruleset-failure-domain") != profile.end())
    ruleset_failure_domain = profile.find("ruleset-failure-domain")->second;*/

  if (ruleset_failure_domain != "") {
    ruleset_steps.clear();
    ruleset_steps.push_back(Step("chooseleaf", ruleset_failure_domain, 0));
  }

  err |= parse_ruleset(profile, ss);
  return err;
}

void ErasureCodeJerasureCLMSR::prepare()
{
  dout(10) << __func__ << " k:" << k << " m: " << m << " w:" << w << dendl;
  
  //get the value for gamma needed for gamma transform
  int u;
  for(u = 2; ; u++){
    if(galois_single_multiply(u,u,w) != 1){
      gamma = u;
      break;
    }
  }

  //get the matrix needed based on mds_block;
  switch(mds_block){
    case VANDERMONDE_RS:
      matrix = reed_sol_vandermonde_coding_matrix(k+nu, m, w);
      break;  
    case CAUCHY_MDS:
      matrix = cauchy_original_coding_matrix(k+nu, m, w);
      break;
    default:
      dout(0) << __func__ << "mds_block " << mds_block << " not supported" << dendl;
      assert(0);
  }

  t = (k+m+nu)/q;
  sub_chunk_no = pow_int(q, t);
  dout(10) << __func__ << " q:" << q << " nu:" << nu << " t:" << t << dendl;

  //create B_buf here
  B_buf = talloc(char*, q*t);
  assert(B_buf);
  for(int i = 0; i < q*t; i++)B_buf[i] = NULL;
}

int ErasureCodeJerasureCLMSR::minimum_to_decode2(const set<int> &want_to_read,
                                  const set<int> &available,
                                  map<int, list<pair<int,int>>> *minimum){
  set<int> minimum_shard_ids;

  if(is_repair(want_to_read, available)) {
    dout(10) << __func__ << "is_repair is true " << dendl;
    int r = minimum_to_repair(want_to_read, available, &minimum_shard_ids);
    map<int, int> repair_subchunks;
    get_repair_subchunks(want_to_read, minimum_shard_ids,
                           0, repair_subchunks);

    list<pair<int,int>> grouped_repair_subchunks;
    group_repair_subchunks(repair_subchunks, grouped_repair_subchunks);

    for(set<int>::iterator i=minimum_shard_ids.begin();
        i != minimum_shard_ids.end(); ++i){
      minimum->insert(make_pair(*i, grouped_repair_subchunks));
    }
    return r;
  } else {
    dout(10) << __func__ << " is_repair is false"<<dendl;
    return ErasureCode::minimum_to_decode2(want_to_read, available, minimum);
  }
}

int ErasureCodeJerasureCLMSR::decode2(const set<int> &want_to_read,
                const map<int, bufferlist> &chunks,
                map<int, bufferlist> *decoded, int chunk_size){
  FT(ErasureCodeJerasureCLMSR::decode2);
  set<int> avail;
  for(map<int, bufferlist>::const_iterator i = chunks.begin();
      i != chunks.end(); ++i){
    avail.insert(i->first);
  }

  printf("is_repair result: %d\n",is_repair(want_to_read, avail));
  printf("want_to_read:\n");
  for(set<int>::const_iterator i = want_to_read.begin();
      i != want_to_read.end(); ++i){
    printf("**: %d\n", *i);
  }

  printf( "avail: \n" );
  for(set<int>::const_iterator i = avail.begin();
      i != avail.end(); ++i){
    printf("==: %d\n", *i);
  }

/*  //debug
  set<int> temp;
  temp.insert(2);*/

  if(is_repair(want_to_read, avail)){
    printf("repair chosen!!!!!!!!!!!!\n");
    return repair(want_to_read, chunks, decoded);
  } else {
    printf("decode chosen!!!!!!!!!!!!\n");
    return ErasureCode::decode(want_to_read, chunks, decoded);
  }
}

int ErasureCodeJerasureCLMSR::minimum_to_repair(const set<int> &want_to_read,
                                   const set<int> &available_chunks,
                                   set<int> *minimum)
{
  int lost_node_index = 0;
  int rep_node_index = 0;


  if((available_chunks.size() >= (unsigned)d)){//&& (want_to_read.size() <= (unsigned)k+m-d) ){

    for(set<int>::iterator i=want_to_read.begin();
        i != want_to_read.end(); ++i){
        lost_node_index = (*i < k) ? (*i) : (*i+nu);
      for(int j = 0; j < q; j++){  
        if(j != lost_node_index%q) {
           rep_node_index = (lost_node_index/q)*q+j;//add all the nodes in lost node's y column.
           if(rep_node_index < k){
             if(want_to_read.find(rep_node_index) == want_to_read.end())minimum->insert(rep_node_index);
           }
           else if(rep_node_index >= k+nu){
             if(want_to_read.find(rep_node_index-nu) == want_to_read.end())minimum->insert(rep_node_index-nu);
           }
        }
      }
    }
    if (includes(
        available_chunks.begin(), available_chunks.end(), minimum->begin(), minimum->end())) {
      for(set<int>::iterator i = available_chunks.begin();
          i != available_chunks.end(); ++i){
        if(minimum->size() < (unsigned)d){
          if(minimum->find(*i) == minimum->end())minimum->insert(*i);
        } else break;
      }
    } else {
      dout(0) << "minimum_to_repair: shouldn't have come here" << dendl;
      assert(0);
    }

  } else{
    if( d == k+m-1){
     assert(available_chunks.size() + want_to_read.size() == (unsigned)k+m);
     lost_node_index = *(want_to_read.begin());// < k) ? (*(want_to_read.begin.())): (*(want_to_read.begin())+nu);
     lost_node_index = (lost_node_index < k) ? lost_node_index : lost_node_index+nu;
     int y_0 = lost_node_index/q; 
     for(set<int>::iterator i=want_to_read.begin();
        i != want_to_read.end(); ++i){
        lost_node_index =  (*i < k) ? (*i) : (*i+nu); 
        assert(lost_node_index/q == y_0);
     }
     dout(10) << __func__ << " picking all the availbale chunks " << dendl;
     *minimum = available_chunks;
     return 0;
    } else{
      dout(0) << "available_chunks: " << available_chunks << " want_to_read:" <<  want_to_read << dendl;
      assert(0);
    }
    //return -EIO;
  }
  assert(minimum->size() == (unsigned)d);
  return 0;
}

/*int ErasureCodeJerasureCLMSR::minimum_to_repair(const set<int> &want_to_read,
                                   const set<int> &available_chunks,
                                   set<int> *minimum)
{
  int lost_node_index = 0;
  int rep_node_index = 0;




    for(set<int>::iterator i=want_to_read.begin();
        i != want_to_read.end(); ++i){
        lost_node_index = (*i < k) ? (*i) : (*i+nu);
      for(int j = 0; j < q; j++){
        if(j != lost_node_index%q) {
           rep_node_index = (lost_node_index/q)*q+j;//add all the nodes in lost node's y column.
           if(rep_node_index < k){
             if(want_to_read.find(rep_node_index) == want_to_read.end())minimum->insert(rep_node_index);
           }
           else if(rep_node_index >= k+nu){
             if(want_to_read.find(rep_node_index-nu) == want_to_read.end())minimum->insert(rep_node_index-nu);
           }
        }
      }
    }
    if (includes(
        available_chunks.begin(), available_chunks.end(), minimum->begin(), minimum->end())) {
      for(set<int>::iterator i = available_chunks.begin();
          i != available_chunks.end(); ++i){
        if(minimum->size() < (unsigned)d){
          if(minimum->find(*i) == minimum->end())minimum->insert(*i);
        } else break;
      }
    } else {
      dout(0) << "minimum_to_repair: shouldn't have come here" << dendl;
      assert(0);
    }

  } else{
    dout(0) << "available_chunks: " << available_chunks << " want_to_read:" <<  want_to_read << dendl;
    assert(0);
    //return -EIO;
  }
  assert(minimum->size() == (unsigned)d);

  return 0;
}*/

//this will be called by ECBackend when a helper node needs to know what subchunks to read,
//the result will be sent as input to filestore read.
void ErasureCodeJerasureCLMSR::get_repair_subchunks(const set<int> &to_repair,
                                   const set<int> &helper_chunks,
                                   int helper_chunk_ind,
                                   map<int, int> &repair_sub_chunks_ind)
{
  int z_vec[t];
  int count = 0;
  int repair_sub_chunk_no = 0;
  int lost_node = 0;
  
  for(int z=0; z < sub_chunk_no; z++){
    get_plane_vector(z, z_vec);
    count = 0;
    for(set<int>::iterator i = to_repair.begin(); 
        i != to_repair.end(); ++i){
      lost_node = (*i < k) ? (*i) :(*i+nu); 
      if(z_vec[lost_node/q] == lost_node%q){
        count++;
        break;
      }
    }
    if(count > 0) {
      repair_sub_chunks_ind[repair_sub_chunk_no] = z;
      repair_sub_chunk_no++;
    }
  }
}

void ErasureCodeJerasureCLMSR::group_repair_subchunks(map<int,int> &repair_subchunks, list<pair<int,int> > &grouped_subchunks) {
  set<int> temp;
  for(map<int,int>:: iterator r = repair_subchunks.begin(); r!= repair_subchunks.end();r++) {
    temp.insert(r->second);
  }
  int start = -1;
  int end =  -1 ;
  for(set<int>::iterator r = temp.begin(); r!= temp.end();r++) {
    if(start == -1) {
      start = *r;
      end = *r;
    }
    else if(*r == end+1) {
      end = *r;
    }
    else {
      grouped_subchunks.push_back(make_pair(start,end-start+1));
      start = *r;
      end = *r;
    }
  }
  if(start != -1) {
    grouped_subchunks.push_back(make_pair(start,end-start+1));
  }
}   

int ErasureCodeJerasureCLMSR::is_repair(const set<int> &want_to_read,
                                   const set<int> &available_chunks){

  FT(ErasureCodeJerasureCLMSR::is_repair);
  //dout(10)<<__func__<< "want_to_read:" << want_to_read<<"available"<<available_chunks<<dendl;
  if(includes(
        available_chunks.begin(), available_chunks.end(), want_to_read.begin(), want_to_read.end()) ) return 0;

  if((d-1)*get_repair_sub_chunk_count(want_to_read) >= (k-1)*sub_chunk_no) return 0;

  //for d=n-1 will be able to handle erasures when all the non erased symbols are available
  //and when the erasures are within a y-crossection.
  if((d < k+m-1) && (available_chunks.size() < (unsigned)d)) return 0;
  if(d == k+m-1){
    if(available_chunks.size() + want_to_read.size() < (unsigned)k+m ) return 0;
    else if( want_to_read.size() > (unsigned)m) return 0;
    //else return 1;
  }
  
  //in every plane the number of erasures in B can't exceed m
  int erasures_weight_vector[t];
  int min_y = q+1;
  memset(erasures_weight_vector, 0, t*sizeof(int));

  for(set<int>::iterator i=want_to_read.begin();
      i != want_to_read.end(); ++i){
    int lost_node_id = (*i < k) ? *i: *i+nu;
    erasures_weight_vector[lost_node_id/q]++;
    for(int x=0; x < q; x++){
      int node = (lost_node_id/q)*q+x;
      node = (node < k) ? node : node-nu;
      if(want_to_read.find(node) == want_to_read.end()){//node from same group not erased
        if(available_chunks.find(node) == available_chunks.end()) return 0;//node from same group not available as well
      }
    }
  }
  for(int y = 0; y < t; y++){
    if((erasures_weight_vector[y] > 0) && (erasures_weight_vector[y] < min_y)) min_y = erasures_weight_vector[y];
  }
  assert((min_y > 0) && (min_y != (q+1) ));
  /*if(d == k+m-1){
    if(hw_erasures == 1){
      dout(10) << __func__<< "repairing erasures within y-cross for the case of d=n-1"<<dendl;
      return 1; 
    } else return 0;
  }*/
  if((q+(int)want_to_read.size()-min_y) <= m) return 1;
  return 0;
}

/*int ErasureCodeJerasureCLMSR::is_repair(const set<int> &want_to_read,
                                   const set<int> &available_chunks){

  //dout(10)<<__func__<< "want_to_read:" << want_to_read<<"available"<<available_chunks<<dendl;
  if(includes(
        available_chunks.begin(), available_chunks.end(), want_to_read.begin(), want_to_read.end()) ) return 0;

  if((d-1)*get_repair_sub_chunk_count(want_to_read) >= (k-1)*sub_chunk_no) return 0;
  if(available_chunks.size() < (unsigned)d) return 0;
  //else if(want_to_read.size() > (unsigned) (k+m-d))return 0;
  //in every plane the number of erasures in B can't exceed m

  int erasures_weight_vector[t];
  int min_y = q+1;
  memset(erasures_weight_vector, 0, t*sizeof(int));

  for(set<int>::iterator i=want_to_read.begin();
      i != want_to_read.end(); ++i){
    int lost_node_id = (*i < k) ? *i: *i+nu;
    erasures_weight_vector[lost_node_id/q]++;
    for(int x=0; x < q; x++){
      int node = (lost_node_id/q)*q+x;
      node = (node < k) ? node : node-nu;
      if(want_to_read.find(node) == want_to_read.end()){//node from same group not erased
        if(available_chunks.find(node) == available_chunks.end()) return 0;//node from same group not available as well
      }
    }
  }
  for(int y = 0; y < t; y++){
    if((erasures_weight_vector[y] > 0) && (erasures_weight_vector[y] < min_y)) min_y = erasures_weight_vector[y];
  }
  assert((min_y > 0) && (min_y != (q+1) ));

  if((q+(int)want_to_read.size()-min_y) <= m) return 1;
  return 0;
}
*/

int ErasureCodeJerasureCLMSR::get_repair_sub_chunk_count(const set<int> &want_to_read)
{
  int repair_subchunks_count = 1;

  int weight_vector[t];
  memset(weight_vector, 0, t*sizeof(int));

  for(set<int>::iterator i = want_to_read.begin();
      i != want_to_read.end(); ++i){
    weight_vector[(*i)/q]++;
  }

  for(int y = 0; y < t; y++) repair_subchunks_count = repair_subchunks_count*(q-weight_vector[y]);

  dout(20) << __func__ << " number of repair subchunks:" << sub_chunk_no - repair_subchunks_count << " for repair of want_to_read:"<< want_to_read <<dendl;
  //return sub_chunk_no - pow_int(q-1, want_to_read.size()) * pow_int(q, t - want_to_read.size());
  return sub_chunk_no - repair_subchunks_count;
}

//for any d <=n-1
int ErasureCodeJerasureCLMSR::repair(const set<int> &want_to_read,
                        const map<int, bufferlist> &chunks,
                        map<int, bufferlist> *repaired)
{
  FT(ErasureCodeJerasureCLMSR::repair);
  //dout(10) << __func__ << " want_to_read: " << want_to_read.size() << " chunk size: "<< chunks.size() << dendl;
  //dout(10) << __func__ << " want_to_read " << want_to_read << " hlper chunks: "<< chunks << dendl;
  if(d< k+m-1) {
    if((chunks.size() != (unsigned)d) || (want_to_read.size() > (unsigned)k+m-d)){
      dout(0) << __func__ << "chunk size not sufficient for repair"<< dendl;
      assert(0);
      return -EIO;
    }
  } else{
    assert(want_to_read.size()+chunks.size() == (unsigned)k+m);
  }

  int repair_sub_chunk_no = get_repair_sub_chunk_count(want_to_read);

  //if chunks include want_to_read just point repaired to them and return
 //XXXXXXXXXXXXX ADD THAT PART XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


  map<int, int> repair_sub_chunks_ind;

  //dout(10) << __func__ << " want_to_read:"<<want_to_read<<" sub_chunk_count:"<<repair_sub_chunk_no<< " list of indices list of indices::"<<repair_sub_chunks_ind<<" size of sub chunk list:"<< repair_sub_chunks_ind.size()<<dendl;
  get_repair_subchunks(want_to_read, want_to_read, 0, repair_sub_chunks_ind);
  assert(repair_sub_chunks_ind.size() == (unsigned)repair_sub_chunk_no); 

  unsigned repair_blocksize = (*chunks.begin()).second.length();
  assert(repair_blocksize%repair_sub_chunk_no == 0);

  unsigned sub_chunksize = repair_blocksize/repair_sub_chunk_no;
  unsigned chunksize = sub_chunk_no*sub_chunksize;

  //only the lost_node's have chunk_size allocated.
  map<int, char*> repaired_data;
  map<int, char*> helper_data;
  set<int> aloof_nodes;
  map<int, bufferlist> temp;

  for (int i =  0; i < k + m; i++) {
    //included helper data only for d+nu nodes.
    if (chunks.find(i) != chunks.end()) {//i is a helper 
      temp[i] = chunks.find(i)->second;
      temp[i].rebuild_aligned(SIMD_ALIGN);

      if(i<k) helper_data[i] = temp[i].c_str();
      else helper_data[i+nu] = temp[i].c_str();

      
    } else{
      if(want_to_read.find(i) == want_to_read.end()){//aloof node case.
        int aloof_node_id = (i < k) ? i: i+nu;
        aloof_nodes.insert(aloof_node_id);
      }else{
        bufferptr ptr(buffer::create_aligned(chunksize, SIMD_ALIGN));
      
        int lost_node_id = (i < k) ? i : i+nu;
   
        (*repaired)[i].push_front(ptr);
        repaired_data[lost_node_id] = (*repaired)[i].c_str();
        memset(repaired_data[lost_node_id], 0, chunksize);
      }
    }
  }

  //this is for shortened codes i.e., when nu > 0
  for(int i=k; i < k+nu; i++){
    helper_data[i] = (char*)malloc(repair_blocksize);
    if(helper_data[i] == NULL){
      dout(0) << "memory allocation failed for shortened case" << dendl;
      assert(0);
    }
    memset(helper_data[i], 0, repair_blocksize);
  }

  assert(helper_data.size()+aloof_nodes.size()+repaired_data.size() == (unsigned) q*t);
  
  int r = repair_lost_chunks(repaired_data, aloof_nodes,
                           helper_data, repair_blocksize, repair_sub_chunks_ind);

  //clear buffers created for the purpose of shortening
  for(int i = k; i < k+nu; i++){
    free(helper_data[i]);
  }

  return r;
}

int ErasureCodeJerasureCLMSR::repair_lost_chunks(map<int,char*> &repaired_data, set<int> &aloof_nodes,
                           map<int, char*> &helper_data, int repair_blocksize, map<int,int> &repair_sub_chunks_ind)
{
 FT(ErasureCodeJerasureCLMSR::repair_lost_chunks);
 unsigned sub_chunksize = repair_blocksize/repair_sub_chunks_ind.size();

  int z_vec[t];
  map<int, set<int> > ordered_planes;
  map<int, int> repair_plane_to_ind;
  int order = 0;
  int x,y, node_xy, node_sw, z_sw;
  char *A1, *A2, *B1, *B2;
  int count_retrieved_sub_chunks = 0;
  int num_erased = 0;


  //cal the order
  //dout(10) << " lost_nodes " << repaired_data << " aloof_nodes " << aloof_nodes << " helper nodes " << helper_data << "repair_blockssize" << repair_blocksize<< dendl;
  // z_vec : z's q0,q1,q2,q3,q4....qt
  for(map<int,int>::iterator i = repair_sub_chunks_ind.begin();
      i != repair_sub_chunks_ind.end(); ++i){
    get_plane_vector(i->second, z_vec);
    order = 0;
    //check across all erasures
    for(map<int,char*>::iterator j = repaired_data.begin();
        j != repaired_data.end(); ++j)
    {
      if(j->first%q == z_vec[j->first/q])order++;
    }
    assert(order>0);
    ordered_planes[order].insert(i->second);
    repair_plane_to_ind[i->second] = i->first;
  }

  int plane_count = 0;
  int erasure_locations[q*t];

  //going to use the global B_buf
  assert(B_buf);
  for(int i = 0; i < q*t; i++){
    if(B_buf[i]==NULL) B_buf[i] = talloc(char, sub_chunksize*sub_chunk_no);
    assert(B_buf[i]);
  }

  //repair planes in order
  for(order=1; ;order++){
    if(ordered_planes.find(order) == ordered_planes.end())break;
    else{
      //dout(10) << "decoding planes of order " << order <<dendl;

      plane_count += ordered_planes[order].size();
      for(set<int>::iterator z=ordered_planes[order].begin();
          z != ordered_planes[order].end(); ++z)
      {
        get_plane_vector(*z, z_vec);

        num_erased = 0;
        for(y=0; y < t; y++){
          for(x = 0; x < q; x++){

            node_xy = y*q + x;

            if( (repaired_data.find(node_xy) != repaired_data.end()) ||
                (aloof_nodes.find(node_xy) != aloof_nodes.end()) )
            {//case of erasure, aloof node can't get a B.
             erasure_locations[num_erased] = node_xy;
             //dout(10)<< num_erased<< "'th erasure of node " << node_xy << " = (" << x << "," << y << ")" << dendl;
             num_erased++;
            } else{//should be in helper data
              assert(helper_data.find(node_xy) != helper_data.end());
              //so A1 is available, need to check if A2 is available.
              A1 = &helper_data[node_xy][repair_plane_to_ind[*z]*sub_chunksize];

	      z_sw = (*z) + (x - z_vec[y])*pow_int(q,t-1-y);
              node_sw = y*q + z_vec[y];
              //dout(10) << "current node=" << node_xy << " plane="<< *z << " node_sw=" << node_sw << " plane_sw="<< z_sw << dendl;
              //consider this as an erasure, if A2 not found.
              if(repair_plane_to_ind.find(z_sw) == repair_plane_to_ind.end())
              {
                erasure_locations[num_erased] = node_xy;
                //dout(10)<< num_erased<< "'th erasure of node " << node_xy << " = (" << x << "," << y << ")" << dendl;
                num_erased++;
              } else {
                if(repaired_data.find(node_sw) != repaired_data.end()){
                  assert(z_sw < sub_chunk_no);
                  A2 = &repaired_data[node_sw][z_sw*sub_chunksize];
                  get_B1_fromA1A2(&B_buf[node_xy][repair_plane_to_ind[*z]*sub_chunksize], A1, A2,sub_chunksize);
                } else if(aloof_nodes.find(node_sw) != aloof_nodes.end()){
                  B2 = &B_buf[node_sw][repair_plane_to_ind[z_sw]*sub_chunksize];
                  get_B1_fromA1B2(&B_buf[node_xy][repair_plane_to_ind[*z]*sub_chunksize], A1, B2, sub_chunksize);
                } else{

                  assert(helper_data.find(node_sw) != helper_data.end());
                  //dout(10) << "obtaining B1 from A1 A2 for node: " << node_xy << " on plane:" << *z << dendl;
                  A2 = &helper_data[node_sw][repair_plane_to_ind[z_sw]*sub_chunksize];
                  if( z_vec[y] != x){
                    get_B1_fromA1A2(&B_buf[node_xy][repair_plane_to_ind[*z]*sub_chunksize], A1, A2, sub_chunksize);
                  } else memcpy(&B_buf[node_xy][repair_plane_to_ind[*z]*sub_chunksize], A1, sub_chunksize);
                }
              }

            }


          }//y
        }//x
        erasure_locations[num_erased] = -1;
        //int erasuresxy[num_erased];
        //get_erasure_coordinates(erasure_locations, erasuresxy, num_erased);
        //we obtained all the needed B's
        assert(num_erased <= m);
        //dout(10) << "going to decode for B's in repair plane "<< *z << " at index " << repair_plane_to_ind[*z] << dendl;
        jerasure_matrix_decode_substripe(k+nu, m, w, matrix, 0, erasure_locations, &B_buf[0], &B_buf[k+nu], repair_plane_to_ind[*z], sub_chunksize);
    for(int i = 0; i < num_erased; i++){
          x = erasure_locations[i]%q;
          y = erasure_locations[i]/q;
          //dout(10) << "B symbol recovered at (x,y) = (" << x <<","<<y<<")"<<dendl;
          //dout(10) << "erasure location " << erasure_locations[i] << dendl;
          node_sw = y*q+z_vec[y];
          z_sw = (*z) + (x - z_vec[y]) * pow_int(q,t-1-y);

          B1 = &B_buf[erasure_locations[i]][repair_plane_to_ind[*z]*sub_chunksize];

          //make sure it is not an aloof node before you retrieve repaired_data
          if( aloof_nodes.find(erasure_locations[i]) == aloof_nodes.end()){

            if(x == z_vec[y] ){//hole-dot pair (type 0)
              //dout(10) << "recovering the hole dot pair/lost node in repair plane" << dendl;
              A1 = &repaired_data[erasure_locations[i]][*z*sub_chunksize];
              memcpy(A1, B1, sub_chunksize);
              count_retrieved_sub_chunks++;
            }//can recover next case (type 2) only after obtaining B's for all the planes with same order
            else {

              if(repaired_data.find(erasure_locations[i]) != repaired_data.end() ){//this is a hole (lost node)
                A1 = &repaired_data[erasure_locations[i]][*z*sub_chunksize];
                //check if type-2
                if( repaired_data.find(node_sw) != repaired_data.end()){
                  if(x < z_vec[y]){//recover both A1 and A2 here
                    A2 = &repaired_data[node_sw][z_sw*sub_chunksize];
                    B2 = &B_buf[node_sw][repair_plane_to_ind[z_sw]*sub_chunksize];
                    gamma_inverse_transform(A1, A2, B1, B2, sub_chunksize);
                    count_retrieved_sub_chunks = count_retrieved_sub_chunks + 2;
                  }
                } else{

                  //dout(10) << "repaired_data" << repaired_data << dendl;
                  //A2 for this particular node is available
                  assert(helper_data.find(node_sw) != helper_data.end());
                  assert(repair_plane_to_ind.find(z_sw) !=  repair_plane_to_ind.end());

                  A2 = &helper_data[node_sw][repair_plane_to_ind[z_sw]*sub_chunksize];
                  get_type1_A(A1, B1, A2, sub_chunksize);
                  count_retrieved_sub_chunks++;
                }
              } else {//not a hole and has an erasure in the y-crossection.
                assert(repaired_data.find(node_sw) != repaired_data.end());
                if(repair_plane_to_ind.find(z_sw) == repair_plane_to_ind.end()){
                  //i got to recover A2, if z_sw was already there
                  //dout(10) << "recovering A2 of node:" << node_sw << " at location " << z_sw << dendl;
                  A1 = &helper_data[erasure_locations[i]][repair_plane_to_ind[*z]*sub_chunksize];
                  A2 = &repaired_data[node_sw][z_sw*sub_chunksize];
                  get_type2_A(A2, B1, A1, sub_chunksize);
                  count_retrieved_sub_chunks++;
                }
              }

            }//type-1 erasure recovered.
          }//not an aloof node
        }//erasures

       //dout(10) << "repaired data after decoding at plane: " << *z << " "<< repaired_data << dendl;
       //dout(10) << "helper data after decoding at plane: " << *z << " "<< helper_data << dendl;

      }//planes of a particular order

    }
  }
  assert(repair_sub_chunks_ind.size() == (unsigned)plane_count);
  assert(sub_chunk_no*repaired_data.size() == (unsigned)count_retrieved_sub_chunks);

  //dout(10) << "repaired_data = " << repaired_data << dendl;

  //for(int i=0; i<q*t ; i++)free(B_buf[i]);
  return 0;
}

int ErasureCodeJerasureCLMSR::encode_systematic(char** data_ptrs, char** code_ptrs, int size)
{
  FT(ErasureCodeJerasureCLMSR::encode_systematic);
  int i;

  //Need the chunk size to be aligned to sub_chunk_no
  assert(size%sub_chunk_no == 0);

  //Now we encode for the nodes k, k+1, ..., k+m-1
  // This is done by calling the decode function to recover the above m nodes.
  int erasure_locations[k+m+nu];

  int numerased = 0;
  for(i=k+nu; i< k+nu+m; i++){
    erasure_locations[numerased] = i;
    numerased++;
  }
  erasure_locations[numerased] = -1;

  int ret = decode_layered(erasure_locations, data_ptrs, code_ptrs, size);
  return ret;
}

int ErasureCodeJerasureCLMSR::decode_layered(int* erasure_locations, char** data_ptrs, char** code_ptrs, int size)
{

  printf("******************\ngamma:\t%d\nq:\t%d\nt:\t%d\nd:\t%d\n****************\n", gamma,q,t,d );
  FT(ErasureCodeJerasureCLMSR::decode_layered);
  int i;
  char* A1 = NULL;
  char* A2 = NULL;
  int x, y;
  int hm_w;
  int z, z_sw, node_xy, node_sw;
  int num_erasures = 0;

  assert(size%sub_chunk_no == 0);
  int ss_size = size/sub_chunk_no;
  
  for(i=0; i < q*t; i++){
    if(erasure_locations[i]==-1)break;
    else num_erasures++;
  }

  if(!num_erasures) assert(0);

  int* erased = jerasure_erasures_to_erased(k+nu,m, erasure_locations);
  if(erased==NULL) assert(0);

  i = 0;

  //when we see less than m erasures,we assume m erasures and work
  while((num_erasures < m) && (i < q*t)) {
      if(erased[i] != 1){     
        erasure_locations[num_erasures] = i;
        erased[i] = 1;
        num_erasures++;
      }
      i++;
  }
  erasure_locations[num_erasures] = -1;

  assert(num_erasures == m);
  
  erasure_t erasures[m];
  int weight_vec[t];

  get_erasure_coordinates(erasure_locations, erasures);
  get_weight_vector( erasures, weight_vec);

  int max_weight = get_hamming_weight(weight_vec);

  int order[sub_chunk_no];
  int z_vec[t];

  assert(B_buf);

  for(i = 0; i < q*t; i++) {
    if(B_buf[i] == NULL)B_buf[i] = talloc(char, size);    
    assert(B_buf[i]);
  } 

  set_planes_sequential_decoding_order( order, erasures);

  for(hm_w = 0; hm_w <= max_weight; hm_w++){
    for(z = 0; z<sub_chunk_no; z++){
      if(order[z]==hm_w){
	decode_erasures(erasure_locations, z, z_vec, data_ptrs, code_ptrs, ss_size, B_buf);
      }
    }

    /* Need to get A's from B's*/
    for(z = 0; z<sub_chunk_no; z++){
      if(order[z]==hm_w){
	get_plane_vector(z,z_vec);
	for(i = 0; i<num_erasures; i++){
	  x = erasures[i].x;
	  y = erasures[i].y;
	  node_xy = y*q+x;
          node_sw = y*q+z_vec[y];
          z_sw = z + ( x - z_vec[y] ) * pow_int(q,t-1-y);

	  A1 = (node_xy < k+nu) ? &data_ptrs[node_xy][z*ss_size] : &code_ptrs[node_xy-k-nu][z*ss_size];
          A2 = (node_sw < k+nu) ? &data_ptrs[node_sw][z_sw*ss_size] : &code_ptrs[node_sw-k-nu][z_sw*ss_size];

	  if(z_vec[y] != x){ //not a hole-dot pair
	    if(is_erasure_type_1(i, erasures, z_vec)){
	      get_type1_A(A1, &B_buf[node_xy][z*ss_size], A2, ss_size);
	    } else{
	      // case for type-2 erasure, there is a hole-dot pair in this y column
              assert(erased[node_sw]==1);
              get_A1_fromB1B2(A1, &B_buf[node_xy][z*ss_size], &B_buf[node_sw][z_sw*ss_size], ss_size);
	    }
	  } else { //for type 0 erasure (hole-dot pair)  copy the B1 to A1
            memcpy(A1, &B_buf[node_xy][z*ss_size], ss_size);
          }

	}//get A's from B's
      }
    }//plane

  }//hm_w, order

  free(erased);
  return 0;
}

void ErasureCodeJerasureCLMSR::set_planes_sequential_decoding_order(int* order, erasure_t* erasures){
  int z, i;

  int z_vec[t];

  for(z = 0; z< sub_chunk_no; z++){
    get_plane_vector(z,z_vec);
    order[z] = 0;
    //check across all m erasures
    for(i = 0; i<m; i++){
      if(erasures[i].x == z_vec[erasures[i].y]){
	order[z] = order[z]+1;
      }
    }
  }
}

void ErasureCodeJerasureCLMSR::decode_erasures(int* erasure_locations, int z, int* z_vec,
                            char** data_ptrs, char** code_ptrs, int ss_size, char** B_buf)
{
  int x, y;
  int node_xy;
  int node_sw;
  int z_sw;

  char* A1 = NULL;
  char* A2 = NULL;

  int* erased = jerasure_erasures_to_erased(k+nu,m, erasure_locations);
  if(erased==NULL) return;

  get_plane_vector(z,z_vec);

  // Need to get the B's to do a jerasure_decode
  for(x=0; x < q; x++){
    for(y=0; y<t; y++){
      node_xy = y*q+x; 
      node_sw = y*q+z_vec[y];
      z_sw = z + (x - z_vec[y]) * pow_int(q,t-1-y);

      A1 = (node_xy < k+nu) ? &data_ptrs[node_xy][z*ss_size] : &code_ptrs[node_xy-k-nu][z*ss_size];
      A2 = (node_sw < k+nu) ? &data_ptrs[node_sw][z_sw*ss_size] : &code_ptrs[node_sw-k-nu][z_sw*ss_size];

      if(erased[node_xy] == 0){ //if not an erasure 
	if(z_vec[y] != x){//not a dot
          get_B1_fromA1A2(&B_buf[node_xy][z*ss_size], A1, A2, ss_size);
	} else { //dot
          memcpy(&B_buf[node_xy][z*ss_size], A1, ss_size);
        }
      }
    }
  }

  //Decode in B's
  jerasure_matrix_decode_substripe(k+nu, m, w, matrix, 0, erasure_locations, 
                                   &B_buf[0], &B_buf[k+nu], z, ss_size);

  free(erased);
}

void ErasureCodeJerasureCLMSR::get_plane_vector(int z, int* z_vec)
{
  int i ;

  for(i = 0; i<t; i++ ){
    z_vec[t-1-i] = z%q;
    z = (z - z_vec[t-1-i])/q;
  }
  return;
}

void ErasureCodeJerasureCLMSR::get_erasure_coordinates(int* erasure_locations, erasure_t* erasures)
{
  int i;

  for(i = 0; i<m; i++){
    if(erasure_locations[i]==-1)break;
    erasures[i].x = erasure_locations[i]%q;
    erasures[i].y = erasure_locations[i]/q;
  }
}

void ErasureCodeJerasureCLMSR::get_weight_vector(erasure_t* erasures, int* weight_vec)
{
  int i;

  memset(weight_vec, 0, sizeof(int)*t);
  for( i = 0; i< m; i++)
    {
      weight_vec[erasures[i].y]++;
    }
  return;
}

int ErasureCodeJerasureCLMSR::get_hamming_weight( int* weight_vec)
{
  int i;
  int weight = 0;

  for(i=0;i<t;i++){
    if(weight_vec[i] != 0) weight++;
  }
  return weight;
}


extern int ErasureCodeJerasureCLMSR::is_erasure_type_1(int ind, erasure_t* erasures, int* z_vec){

  // Need to look for the column of where erasures[i] is and search to see if there is a hole dot pair.
  int i;

  if(erasures[ind].x == z_vec[erasures[ind].y]) return 0; //type-0 erasure

  for(i=0; i < m; i++){
    if(erasures[i].y == erasures[ind].y){
      if(erasures[i].x == z_vec[erasures[i].y]){
	return 0;
      }
    }
  }
  return 1;

}

void ErasureCodeJerasureCLMSR::gamma_transform(char* dest1, char* dest2, char* code_symbol_1, char* code_symbol_2,  int size)
{
  int tmatrix[4];
  tmatrix[0] = 1;
  tmatrix[1] = gamma;
  tmatrix[2] = gamma;
  tmatrix[3] = 1;

  char* A[2];
  A[0] = code_symbol_1;
  A[1] = code_symbol_2;
  
  char* dest[2];
  dest[0] = dest1;
  dest[1] = dest2;

  jerasure_matrix_dotprod(2, w, &tmatrix[0], NULL, 2, A, dest, size);
  jerasure_matrix_dotprod(2, w, &tmatrix[2], NULL, 3, A, dest, size);

}

void ErasureCodeJerasureCLMSR::gamma_inverse_transform(char* dest1, char* dest2, char* code_symbol_1, char* code_symbol_2,  int size)
{
  int gamma_square = galois_single_multiply(gamma, gamma, w);
  int gamma_det_inv = galois_single_divide(1, 1 ^ (gamma_square), w);

  int tmatrix[4];
  tmatrix[0] = gamma_det_inv;
  tmatrix[1] = galois_single_multiply(gamma,gamma_det_inv,w);
  tmatrix[2] = tmatrix[1];
  tmatrix[3] = tmatrix[0];

  char* A[2];
  A[0] = code_symbol_1;
  A[1] = code_symbol_2;
 
  char* dest[2];
  dest[0] = dest1;
  dest[1] = dest2;

  jerasure_matrix_dotprod(2, w, &tmatrix[0], NULL, 2, A, dest, size);
  jerasure_matrix_dotprod(2, w, &tmatrix[2], NULL, 3, A, dest, size);

}

void ErasureCodeJerasureCLMSR::get_type1_A(char* A1, char* B1, char* A2, int size){
  int tmatrix[2];
  tmatrix[0] = 1;
  tmatrix[1] = gamma;
  
  char* in_dot[2];
  in_dot[0] = B1;
  in_dot[1] = A2;

  char* dest[1];
  dest[0] = A1;

  jerasure_matrix_dotprod(2, w, &tmatrix[0], NULL, 2, in_dot, dest, size);
}

void ErasureCodeJerasureCLMSR::get_type2_A(char* A2, char* B1, char* A1, int size){
  int tmatrix[2];
  tmatrix[0] = galois_single_divide(1,gamma,w);
  tmatrix[1] = tmatrix[0];
  
  char* in_dot[2];
  in_dot[0] = B1;
  in_dot[1] = A1;

  char* dest[1];
  dest[0] = A2;

  jerasure_matrix_dotprod(2, w, &tmatrix[0], NULL, 2, in_dot, dest, size);
}

void ErasureCodeJerasureCLMSR::get_B1_fromA1B2(char* B1, char* A1, char* B2, int size)
{
  int gamma_square = galois_single_multiply(gamma, gamma, w);

  int tmatrix[2];
  tmatrix[0] = (1 ^ gamma_square);
  tmatrix[1] = gamma;

  char* in_dot[2];
  in_dot[0] = A1;
  in_dot[1] = B2;

  char* dest[1];
  dest[0] = B1;
  
  jerasure_matrix_dotprod(2, w, &tmatrix[0], NULL, 2, in_dot, dest, size);
}

void ErasureCodeJerasureCLMSR::get_B1_fromA1A2(char* B1, char* A1, char* A2, int size)
{
  int tmatrix[2];
  
  tmatrix[0] = 1;
  tmatrix[1] = gamma;
 
  char* in_dot[2];
  in_dot[0] = A1;
  in_dot[1] = A2;

  char* dest[1];
  dest[0] = B1;
  
  jerasure_matrix_dotprod(2, w, &tmatrix[0], NULL, 2, in_dot, dest, size); 
}

void ErasureCodeJerasureCLMSR::get_A1_fromB1B2(char* A1, char* B1, char* B2,  int size)
{
  int gamma_square = galois_single_multiply(gamma, gamma, w);
  int gamma_det_inv = galois_single_divide(1, 1 ^ (gamma_square), w);

  int tmatrix[2];
  tmatrix[0] = gamma_det_inv;
  tmatrix[1] = galois_single_multiply(gamma,gamma_det_inv,w);

  char* in_dot[2];
  in_dot[0] = B1;
  in_dot[1] = B2;
 
  char* dest[1];
  dest[0] = A1;

  jerasure_matrix_dotprod(2, w, &tmatrix[0], NULL, 2, in_dot, dest, size);

}

void ErasureCodeJerasureCLMSR::jerasure_matrix_dotprod_substripe(int k, int w, int *matrix_row,
                          int *src_ids, int dest_id,
                          char **data_ptrs, char **coding_ptrs, int z, int ss_size)
{
  int init;
  char *dptr, *sptr;
  int i;

  if (w!=1 && w != 8 && w != 16 && w != 32) {
    fprintf(stderr,"ERROR: jerasure_matrix_dotprod() called and w is not 1, 8, 16 or 32\n");
    assert(0);
  }

  init = 0;
  //printf("jerasure_matrix_dotprod_substripe: assigning dptr for plane %d ss_size %d\n",z,ss_size);
  dptr = (dest_id < k) ? &data_ptrs[dest_id][z*ss_size] : &coding_ptrs[dest_id-k][z*ss_size];

  /* First copy or xor any data that does not need to be multiplied by a factor */

  for (i = 0; i < k; i++) {
    if (matrix_row[i] == 1) {
      if (src_ids == NULL) {
        sptr = &data_ptrs[i][z*ss_size];
      } else if (src_ids[i] < k) {
        sptr = &data_ptrs[src_ids[i]][z*ss_size];
      } else {
        sptr = &coding_ptrs[src_ids[i]-k][z*ss_size];
      }
      if (init == 0) {
        memcpy(dptr, sptr, ss_size);
        //jerasure_total_memcpy_bytes += ss_size;
        init = 1;
      } else {
        galois_region_xor(sptr, dptr, ss_size);
        //jerasure_total_xor_bytes += ss_size;
      }
    }
  }
  /* Now do the data that needs to be multiplied by a factor */

  for (i = 0; i < k; i++) {
    if (matrix_row[i] != 0 && matrix_row[i] != 1) {
      //printf("matrix_row[%d] = %d\n",i,matrix_row[i]);
      if (src_ids == NULL) {
        sptr = &data_ptrs[i][z*ss_size];
      } else if (src_ids[i] < k) {
        sptr = &data_ptrs[src_ids[i]][z*ss_size];
      } else {
        sptr = &coding_ptrs[src_ids[i]-k][z*ss_size];
      }
      switch (w) {
        case 8:  galois_w08_region_multiply(sptr, matrix_row[i], ss_size, dptr, init); break;
        case 16: galois_w16_region_multiply(sptr, matrix_row[i], ss_size, dptr, init); break;
        case 32: galois_w32_region_multiply(sptr, matrix_row[i], ss_size, dptr, init); break;
      }
      //jerasure_total_gf_bytes += ss_size;
      init = 1;
      //printf("sptr[0] = %d, dptr[0] = %d\n", sptr[0], dptr[0]);
    }
  }
  //printf("jerasure_matrix_dotprod_substripe: exiting\n");
}

int ErasureCodeJerasureCLMSR::jerasure_matrix_decode_substripe(int k, int m, int w, int *matrix, int row_k_ones, int *erasures,
                          char **data_ptrs, char **coding_ptrs, int z, int ss_size)
{
  int i, edd, lastdrive;
  int *tmpids;
  int *erased, *decoding_matrix, *dm_ids;

  if (w != 8 && w != 16 && w != 32) return -1;

  erased = jerasure_erasures_to_erased(k, m, erasures);
  if (erased == NULL) return -1;

  /* Find the number of data drives failed */

  lastdrive = k;

  edd = 0;
  for (i = 0; i < k; i++) {
    if (erased[i]) {
      edd++;
      lastdrive = i;
    }
  }
 /* You only need to create the decoding matrix in the following cases:

      1. edd > 0 and row_k_ones is false.
      2. edd > 0 and row_k_ones is true and coding device 0 has been erased.
      3. edd > 1

      We're going to use lastdrive to denote when to stop decoding data.
      At this point in the code, it is equal to the last erased data device.
      However, if we can't use the parity row to decode it (i.e. row_k_ones=0
         or erased[k] = 1, we're going to set it to k so that the decoding
         pass will decode all data.
   */

  if (!row_k_ones || erased[k]) lastdrive = k;

  dm_ids = NULL;
  decoding_matrix = NULL;

  if (edd > 1 || (edd > 0 && (!row_k_ones || erased[k]))) {
    dm_ids = talloc(int, k);
    if (dm_ids == NULL) {
      free(erased);
      return -1;
    }

    decoding_matrix = talloc(int, k*k);
    if (decoding_matrix == NULL) {
      free(erased);
      free(dm_ids);
      return -1;
    }

    if (jerasure_make_decoding_matrix(k, m, w, matrix, erased, decoding_matrix, dm_ids) < 0) {
      free(erased);
      free(dm_ids);
      free(decoding_matrix);
      return -1;
    }
  }

  /* Decode the data drives.
     If row_k_ones is true and coding device 0 is intact, then only decode edd-1 drives.
     This is done by stopping at lastdrive.
     We test whether edd > 0 so that we can exit the loop early if we're done.
   */
  //int alpha = pow_int(m, k/m + 1);
  //print_coding(k, m, alpha, w, ss_size*alpha, data_ptrs, coding_ptrs);

  for (i = 0; edd > 0 && i < lastdrive; i++) {
    if (erased[i]) {
      jerasure_matrix_dotprod_substripe(k, w, decoding_matrix+(i*k), dm_ids, i, data_ptrs, coding_ptrs, z, ss_size);
      edd--;
    }
  }

  /* Then if necessary, decode drive lastdrive */

  if (edd > 0) {
    tmpids = talloc(int, k);
    for (i = 0; i < k; i++) {
      tmpids[i] = (i < lastdrive) ? i : i+1;
    }
    jerasure_matrix_dotprod_substripe(k, w, matrix, tmpids, lastdrive, data_ptrs, coding_ptrs, z, ss_size);
    free(tmpids);
  }

  /* Finally, re-encode any erased coding devices */

  for (i = 0; i < m; i++) {
    if (erased[k+i]) {
      jerasure_matrix_dotprod_substripe(k, w, matrix+(i*k), NULL, i+k, data_ptrs, coding_ptrs, z, ss_size);
    }
  }
  //print_coding(k,m,alpha,w,ss_size*alpha, data_ptrs, coding_ptrs);

  free(erased);
  if (dm_ids != NULL) free(dm_ids);
  if (decoding_matrix != NULL) free(decoding_matrix);

  return 0;
}
