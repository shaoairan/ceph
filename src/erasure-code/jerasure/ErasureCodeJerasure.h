// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*- 
// vim: ts=8 sw=2 smarttab
/*
 * Ceph distributed storage system
 *
 * Copyright (C) 2013, 2014 Cloudwatt <libre.licensing@cloudwatt.com>
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

#ifndef CEPH_ERASURE_CODE_JERASURE_H
#define CEPH_ERASURE_CODE_JERASURE_H

#include "include/err.h"
#include "json_spirit/json_spirit.h"
#include "erasure-code/ErasureCode.h"

extern "C"{
  #include "gf_bridgeout.h"
}

#define DEFAULT_RULESET_ROOT "default"
#define DEFAULT_RULESET_FAILURE_DOMAIN "host"
#define ERROR_LRC_ARRAY                 -(MAX_ERRNO + 1)
#define ERROR_LRC_OBJECT                -(MAX_ERRNO + 2)
#define ERROR_LRC_INT                   -(MAX_ERRNO + 3)
#define ERROR_LRC_STR                   -(MAX_ERRNO + 4)
#define ERROR_LRC_PLUGIN                -(MAX_ERRNO + 5)
#define ERROR_LRC_DESCRIPTION           -(MAX_ERRNO + 6)
#define ERROR_LRC_PARSE_JSON            -(MAX_ERRNO + 7)
#define ERROR_LRC_MAPPING               -(MAX_ERRNO + 8)
#define ERROR_LRC_MAPPING_SIZE          -(MAX_ERRNO + 9)
#define ERROR_LRC_FIRST_MAPPING         -(MAX_ERRNO + 10)
#define ERROR_LRC_COUNT_CONSTRAINT      -(MAX_ERRNO + 11)
#define ERROR_LRC_CONFIG_OPTIONS        -(MAX_ERRNO + 12)
#define ERROR_LRC_LAYERS_COUNT          -(MAX_ERRNO + 13)
#define ERROR_LRC_RULESET_OP            -(MAX_ERRNO + 14)
#define ERROR_LRC_RULESET_TYPE          -(MAX_ERRNO + 15)
#define ERROR_LRC_RULESET_N             -(MAX_ERRNO + 16)
#define ERROR_LRC_ALL_OR_NOTHING        -(MAX_ERRNO + 17)
#define ERROR_LRC_GENERATED             -(MAX_ERRNO + 18)
#define ERROR_LRC_K_M_MODULO            -(MAX_ERRNO + 19)
#define ERROR_LRC_K_MODULO              -(MAX_ERRNO + 20)
#define ERROR_LRC_M_MODULO              -(MAX_ERRNO + 21)

typedef enum mds_block{
  VANDERMONDE_RS=0,
  CAUCHY_MDS=1
}mds_block_t;

typedef struct erasure
{
  int x;
  int y;
}erasure_t;

class ErasureCodeJerasure : public ErasureCode {
public:
  int k;
  std::string DEFAULT_K;
  int m;
  std::string DEFAULT_M;
  int w;
  std::string DEFAULT_W;
  int nu; 
  //nu is shortening parameter used for cl_msr code.
  const char *technique;
  string ruleset_root;
  string ruleset_failure_domain;
  bool per_chunk_alignment;
  int sub_chunk_no;

  explicit ErasureCodeJerasure(const char *_technique) :
    k(0),
    DEFAULT_K("2"),
    m(0),
    DEFAULT_M("1"),
    w(0),
    DEFAULT_W("8"),    
    nu(0),
    technique(_technique),
    ruleset_root(DEFAULT_RULESET_ROOT),
    ruleset_failure_domain(DEFAULT_RULESET_FAILURE_DOMAIN),
    per_chunk_alignment(false),
    sub_chunk_no(1)
  {}

  ~ErasureCodeJerasure() override {}
  
  int create_ruleset(const string &name,
			     CrushWrapper &crush,
			     ostream *ss) const override;

  unsigned int get_chunk_count() const override {
    return k + m;
  }

  unsigned int get_data_chunk_count() const override {
    return k;
  }

  unsigned int get_chunk_size(unsigned int object_size) const override;

  virtual int get_sub_chunk_count() {
    return sub_chunk_no;
  }

  virtual int minimum_to_decode2(const set<int> &want_to_read,
                                   const set<int> &available_chunks,
                                   map<int, list<pair<int, int>>> *minimum);

  virtual int decode2(const set<int> &want_to_read,
                const map<int, bufferlist> &chunks,
                map<int, bufferlist> *decoded, int chunk_size);

  int encode_chunks(const set<int> &want_to_encode,
			    map<int, bufferlist> *encoded) override;

  int decode_chunks(const set<int> &want_to_read,
			    const map<int, bufferlist> &chunks,
			    map<int, bufferlist> *decoded) override;

  int init(ErasureCodeProfile &profile, ostream *ss) override;

  virtual void jerasure_encode(char **data,
                               char **coding,
                               int blocksize) = 0;
  virtual int jerasure_decode(int *erasures,
                               char **data,
                               char **coding,
                               int blocksize) = 0;
  virtual unsigned get_alignment() const = 0;
  virtual void prepare() = 0;
  static bool is_prime(int value);
protected:
  virtual int parse(ErasureCodeProfile &profile, ostream *ss);
};

class ErasureCodeJerasureReedSolomonVandermonde : public ErasureCodeJerasure {
public:
  int *matrix;

  ErasureCodeJerasureReedSolomonVandermonde() :
    ErasureCodeJerasure("reed_sol_van"),
    matrix(0)
  {
    DEFAULT_K = "7";
    DEFAULT_M = "3";
    DEFAULT_W = "8";
  }
  ~ErasureCodeJerasureReedSolomonVandermonde() override {
    if (matrix)
      free(matrix);
  }

  void jerasure_encode(char **data,
                               char **coding,
                               int blocksize) override;
  int jerasure_decode(int *erasures,
                               char **data,
                               char **coding,
                               int blocksize) override;
  unsigned get_alignment() const override;
  void prepare() override;
private:
  int parse(ErasureCodeProfile &profile, ostream *ss) override;
};

class ErasureCodeJerasureReedSolomonRAID6 : public ErasureCodeJerasure {
public:
  int *matrix;

  ErasureCodeJerasureReedSolomonRAID6() :
    ErasureCodeJerasure("reed_sol_r6_op"),
    matrix(0)
  {
    DEFAULT_K = "7";
    DEFAULT_W = "8";
  }
  ~ErasureCodeJerasureReedSolomonRAID6() override {
    if (matrix)
      free(matrix);
  }

  void jerasure_encode(char **data,
                               char **coding,
                               int blocksize) override;
  int jerasure_decode(int *erasures,
                               char **data,
                               char **coding,
                               int blocksize) override;
  unsigned get_alignment() const override;
  void prepare() override;
private:
  int parse(ErasureCodeProfile &profile, ostream *ss) override;
};

#define DEFAULT_PACKETSIZE "2048"

class ErasureCodeJerasureCauchy : public ErasureCodeJerasure {
public:
  int *bitmatrix;
  int **schedule;
  int packetsize;

  explicit ErasureCodeJerasureCauchy(const char *technique) :
    ErasureCodeJerasure(technique),
    bitmatrix(0),
    schedule(0),
    packetsize(0)
  {
    DEFAULT_K = "7";
    DEFAULT_M = "3";
    DEFAULT_W = "8";
  }
  ~ErasureCodeJerasureCauchy() override {
    if (bitmatrix)
      free(bitmatrix);
    if (schedule)
      free(schedule);
  }

  void jerasure_encode(char **data,
                               char **coding,
                               int blocksize) override;
  int jerasure_decode(int *erasures,
                               char **data,
                               char **coding,
                               int blocksize) override;
  unsigned get_alignment() const override;
  void prepare_schedule(int *matrix);
private:
  int parse(ErasureCodeProfile &profile, ostream *ss) override;
};

class ErasureCodeJerasureCauchyOrig : public ErasureCodeJerasureCauchy {
public:
  ErasureCodeJerasureCauchyOrig() :
    ErasureCodeJerasureCauchy("cauchy_orig")
  {}

  void prepare() override;
};

class ErasureCodeJerasureCauchyGood : public ErasureCodeJerasureCauchy {
public:
  ErasureCodeJerasureCauchyGood() :
    ErasureCodeJerasureCauchy("cauchy_good")
  {}

  void prepare() override;
};

class ErasureCodeJerasureLiberation : public ErasureCodeJerasure {
public:
  int *bitmatrix;
  int **schedule;
  int packetsize;

  explicit ErasureCodeJerasureLiberation(const char *technique = "liberation") :
    ErasureCodeJerasure(technique),
    bitmatrix(0),
    schedule(0),
    packetsize(0)
  {
    DEFAULT_K = "2";
    DEFAULT_M = "2";
    DEFAULT_W = "7";
  }
  ~ErasureCodeJerasureLiberation() override;

  void jerasure_encode(char **data,
                               char **coding,
                               int blocksize) override;
  int jerasure_decode(int *erasures,
                               char **data,
                               char **coding,
                               int blocksize) override;
  unsigned get_alignment() const override;
  virtual bool check_k(ostream *ss) const;
  virtual bool check_w(ostream *ss) const;
  virtual bool check_packetsize_set(ostream *ss) const;
  virtual bool check_packetsize(ostream *ss) const;
  virtual int revert_to_default(ErasureCodeProfile &profile,
				ostream *ss);
  void prepare() override;
private:
  int parse(ErasureCodeProfile &profile, ostream *ss) override;
};

class ErasureCodeJerasureBlaumRoth : public ErasureCodeJerasureLiberation {
public:
  ErasureCodeJerasureBlaumRoth() :
    ErasureCodeJerasureLiberation("blaum_roth")
  {
  }

  bool check_w(ostream *ss) const override;
  void prepare() override;
};

class ErasureCodeJerasureLiber8tion : public ErasureCodeJerasureLiberation {
public:
  ErasureCodeJerasureLiber8tion() :
    ErasureCodeJerasureLiberation("liber8tion")
  {
    DEFAULT_K = "2";
    DEFAULT_M = "2";
    DEFAULT_W = "8";
  }

  void prepare() override;
private:
  int parse(ErasureCodeProfile &profile, ostream *ss) override;
};

class ErasureCodeJerasureCLMSR : public ErasureCodeJerasure {
public:
  int *matrix;
  int gamma; //defines 2X2 transform
  int q;//this is d-k+1 (q = m for d=n-1)
  // n = k+m = qt-nu, k = d+1-q;
  // alpha = q^t
  int t;//t=n/q;
  //this parameter is used by regenerating codes
  //specifically cl_msr here.
  int d;
  mds_block_t mds_block;
  char** B_buf;//need to be super careful on how this is used
  //we might have to add mutexes eventually while using this buffer.

  struct Step {
    Step(string _op, string _type, int _n) :
      op(_op),
      type(_type),
      n(_n) {}
    string op;
    string type;
    int n;
  };
  vector<Step> ruleset_steps;

  ErasureCodeJerasureCLMSR() :
    ErasureCodeJerasure("cl_msr"),
    matrix(0),
    gamma(0),
    d(0),
    mds_block(VANDERMONDE_RS),
    B_buf(0)
  {
    table_p_global = get_w8_log_tables_bridgeOut();
    DEFAULT_K = "6";
    DEFAULT_M = "3";
    DEFAULT_W = "8";
    ruleset_steps.push_back(Step("chooseleaf", "host", 0));
  }

  virtual ~ErasureCodeJerasureCLMSR() {
    if (matrix)
      free(matrix);
    if(B_buf){
      for(int i=0; i<q*t; i++){
        if(B_buf[i]) free(B_buf[i]);
      }
      free(B_buf);
    }
  }

  int create_ruleset(const string &name,
                             CrushWrapper &crush,
                             ostream *ss) const override;

  virtual int minimum_to_decode2(const set<int> &want_to_read,
                                   const set<int> &available_chunks,
                                   map<int, list<pair<int, int>>> *minimum);
  virtual int decode2(const set<int> &want_to_read,
                const map<int, bufferlist> &chunks,
                map<int, bufferlist> *decoded, int chunk_size);

  int is_repair(const set<int> &want_to_read,
                                   const set<int> &available_chunks);

  int minimum_to_repair(const set<int> &want_to_read,
                                   const set<int> &available_chunks,
                                   set<int> *minimum);

  void get_repair_subchunks(const set<int> &to_repair,
                                   const set<int> &helper_chunks,
                                   int helper_chunk_ind,
                                   map<int, int> &repair_sub_chunks_ind);

  int get_repair_sub_chunk_count(const set<int> &want_to_read);

  int repair(const set<int> &want_to_read,
                       const map<int, bufferlist> &chunks,
                       map<int, bufferlist> *repaired);

  virtual void jerasure_encode(char **data,
                               char **coding,
                               int blocksize);
  virtual int jerasure_decode(int *erasures,
                               char **data,
                               char **coding,
                               int blocksize);
  virtual unsigned get_alignment() const;
  virtual void prepare();
private:
  virtual int parse(ErasureCodeProfile &profile, ostream *ss);
  int parse_ruleset(ErasureCodeProfile &profile, ostream *ss);
  int parse_ruleset_step(string description_string,
                         json_spirit::mArray description,
                         ostream *ss);
 void group_repair_subchunks(map<int,int> &repair_subchunks, list<pair<int,int> > &grouped_subchunks);
  int encode_systematic(char** data_ptrs, char** code_ptrs, int size);
  int decode_layered(int* erasure_locations, char** data_ptrs, char** code_ptrs, int size);
  int repair_lost_chunks(map<int,char*> &repaired_data, set<int> &aloof_nodes,
                          map<int, char*> &helper_data, int repair_blocksize, map<int,int> &repair_sub_chunks_ind);
  void decode_erasures(int* erasure_locations, int z, int* z_vec,
                            char** data_ptrs, char** code_ptrs, int ss_size, char** B_buf);


  void set_planes_sequential_decoding_order(int* order, erasure_t* erasures);
  void gamma_inverse_transform(char* dest1, char* dest2, char* code_symbol_1, char* code_symbol_2,  int size);
  void gamma_transform(char* dest1, char* dest2, char* code_symbol_1, char* code_symbol_2,  int size);
  void get_type1_A(char* A1, char* B1, char* A2, int size);
  void get_type2_A(char* A2, char* B1, char* A1, int size);
  void get_B1_fromA1B2(char* B1, char* A1, char* B2, int size);
  int is_erasure_type_1(int ind, erasure_t* erasures, int* z_vec);
  void get_plane_vector(int z, int* z_vec);
  void get_erasure_coordinates( int* erasure_locations, erasure_t* erasures);
  void get_weight_vector(erasure_t* erasures, int* weight_vec);
  int get_hamming_weight(int* weight_vec);
  void get_A1_fromB1B2(char* A1, char* B1, char* B2, int size);
  void get_B1_fromA1A2(char* B1, char* A1, char* A2, int size);

  int jerasure_matrix_decode_substripe(int k, int m, int w,
                          int *matrix, int row_k_ones, int *erasures,
                          char **data_ptrs, char **coding_ptrs, int z, int ss_size);
  
  void jerasure_matrix_dotprod_substripe(int k, int w, int *matrix_row,
                          int *src_ids, int dest_id,
                          char **data_ptrs, char **coding_ptrs, int z, int ss_size);

  void jerasure_matrix_dotprod_substripe_bridgeOut(int k, int w, int *matrix_row,
                          int *src_ids, int dest_id,
                          char **data_ptrs, char **coding_ptrs, int z, int ss_size);
};

//hou
class ErasureCodeJerasureCLMSR_GPU : public ErasureCodeJerasure {
public:
  int *matrix;
  int gamma; //defines 2X2 transform
  int q;//this is d-k+1 (q = m for d=n-1)
  // n = k+m = qt-nu, k = d+1-q;
  // alpha = q^t
  int t;//t=n/q;
  //this parameter is used by regenerating codes
  //specifically cl_msr here.
  int d;
  mds_block_t mds_block;
  char** B_buf;//need to be super careful on how this is used
  //we might have to add mutexes eventually while using this buffer.


  struct Step {
    Step(string _op, string _type, int _n) :
      op(_op),
      type(_type),
      n(_n) {}
    string op;
    string type;
    int n;
  };
  vector<Step> ruleset_steps;

  ErasureCodeJerasureCLMSR_GPU() :
    ErasureCodeJerasure("cl_msr_gpu"),
    matrix(0),
    gamma(0),
    d(0),
    mds_block(VANDERMONDE_RS),
    B_buf(0)
  {
    DEFAULT_K = "6";
    DEFAULT_M = "3";
    DEFAULT_W = "8";
    ruleset_steps.push_back(Step("chooseleaf", "host", 0));
  }

  virtual ~ErasureCodeJerasureCLMSR_GPU() {
    if (matrix)
      free(matrix);
    if(B_buf){
      for(int i=0; i<q*t; i++){
        if(B_buf[i]) free(B_buf[i]);
      }
      free(B_buf);
    }
  }

  int create_ruleset(const string &name,
                             CrushWrapper &crush,
                             ostream *ss) const override;

  virtual int minimum_to_decode2(const set<int> &want_to_read,
                                   const set<int> &available_chunks,
                                   map<int, list<pair<int, int>>> *minimum);
  virtual int decode2(const set<int> &want_to_read,
                const map<int, bufferlist> &chunks,
                map<int, bufferlist> *decoded, int chunk_size);

  int is_repair(const set<int> &want_to_read,
                                   const set<int> &available_chunks);

  int minimum_to_repair(const set<int> &want_to_read,
                                   const set<int> &available_chunks,
                                   set<int> *minimum);

  void get_repair_subchunks(const set<int> &to_repair,
                                   const set<int> &helper_chunks,
                                   int helper_chunk_ind,
                                   map<int, int> &repair_sub_chunks_ind);

  int get_repair_sub_chunk_count(const set<int> &want_to_read);

  int repair(const set<int> &want_to_read,
                       const map<int, bufferlist> &chunks,
                       map<int, bufferlist> *repaired);

  virtual void jerasure_encode(char **data,
                               char **coding,
                               int blocksize);
  virtual int jerasure_decode(int *erasures,
                               char **data,
                               char **coding,
                               int blocksize);
  virtual unsigned get_alignment() const;
  virtual void prepare();
private:
  virtual int parse(ErasureCodeProfile &profile, ostream *ss);
  int parse_ruleset(ErasureCodeProfile &profile, ostream *ss);
  int parse_ruleset_step(string description_string,
                         json_spirit::mArray description,
                         ostream *ss);
 void group_repair_subchunks(map<int,int> &repair_subchunks, list<pair<int,int> > &grouped_subchunks);
  int encode_systematic(char** data_ptrs, char** code_ptrs, int size);
  int decode_layered    (int* erasure_locations, char** data_ptrs, char** code_ptrs, int size);
  int decode_layered_gpu(int* erasure_locations, char** data_ptrs, char** code_ptrs, int size);
  int repair_lost_chunks(map<int,char*> &repaired_data, set<int> &aloof_nodes,
                          map<int, char*> &helper_data, int repair_blocksize, map<int,int> &repair_sub_chunks_ind);
  int repair_lost_chunks_gpu(map<int,char*> &repaired_data, set<int> &aloof_nodes,
                          map<int, char*> &helper_data, int repair_blocksize, map<int,int> &repair_sub_chunks_ind);
  void decode_erasures(int* erasure_locations, int z, int* z_vec,
                            char** data_ptrs, char** code_ptrs, int ss_size, char** B_buf);


  void set_planes_sequential_decoding_order(int* order, erasure_t* erasures);
  void gamma_inverse_transform(char* dest1, char* dest2, char* code_symbol_1, char* code_symbol_2,  int size);
  void gamma_transform(char* dest1, char* dest2, char* code_symbol_1, char* code_symbol_2,  int size);
  void get_type1_A(char* A1, char* B1, char* A2, int size);
  void get_type2_A(char* A2, char* B1, char* A1, int size);
  void get_B1_fromA1B2(char* B1, char* A1, char* B2, int size);
  int is_erasure_type_1(int ind, erasure_t* erasures, int* z_vec);
  void get_plane_vector(int z, int* z_vec);
  void get_erasure_coordinates( int* erasure_locations, erasure_t* erasures);
  void get_weight_vector(erasure_t* erasures, int* weight_vec);
  int get_hamming_weight(int* weight_vec);
  void get_A1_fromB1B2(char* A1, char* B1, char* B2, int size);
  void get_B1_fromA1A2(char* B1, char* A1, char* A2, int size);

  int jerasure_matrix_decode_substripe(int k, int m, int w,
                          int *matrix, int row_k_ones, int *erasures,
                          char **data_ptrs, char **coding_ptrs, int z, int ss_size);  

  void jerasure_matrix_dotprod_substripe(int k, int w, int *matrix_row,
                          int *src_ids, int dest_id,
                          char **data_ptrs, char **coding_ptrs, int z, int ss_size);

  //int init_gf_log_w8_gpu( cudaStream_t stream );
};

#endif
