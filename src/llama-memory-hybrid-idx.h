#pragma once

#include "llama-memory-hybrid.h"

#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    // The pooled keys the scoring reads are cached: the indexer cache's V plane, which attention
    // never reads, holds one finished block key (mean of the members, norm, rope) per complete
    // block, as F32 [idx_dim] rows indexed by the compact block id. A graph pools only a window
    // of blocks and set_rows them into the plane; the window is the blocks this ubatch completes
    // or touches, or every block after a change the host tables cannot follow.
    //   win_cells I32 [ratio*n_win, ns] member cells of each window block (pads repeat an entry)
    //   win_pos   I32 [4*n_win*ns]      mrope position rows of each window block's first token
    //   win_blk   I32 [n_win, ns]       row each window block is written to
    // k_idxs is the ubatch's own cells (set_input_k_idxs must have run): a cell written now
    // invalidates its block even when the same cells make it up, since its key changed.
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * bias,
                       ggml_tensor * win_cells, ggml_tensor * win_pos, ggml_tensor * win_blk,
                       const ggml_tensor * k_idxs,
                       const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias) const;

    // window slots the next graph needs: n_tps while the rows of every sequence in the ubatch
    // are trusted, else every block. A graph shape parameter, so it is asked at build time
    int64_t qsa_pool_n_win(const llama_ubatch & ubatch, uint32_t n_ns, int64_t n_blocks) const;

    // 0 when the V plane does not hold pooled keys (unsupported ratio, or LLAMA_QSA_POOL_CACHE=0);
    // every graph then pools every block into a window of its own and scores that directly
    uint32_t qsa_pool_ratio() const { return pool_ratio; }

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // every pooled row of every sequence is repooled by its next graph
    void qsa_pool_invalidate() const;

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    // the common compress ratio of the indexer layers, set before mem_idx is built
    uint32_t pool_ratio = 0;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // what each pooled row of a sequence holds. Streams map one to one onto sequences when the
    // cache is not unified, and a unified stream trusts its rows only while it holds one sequence
    struct pool_seq {
        bool dirty = true;             // repool everything at the next graph
        int32_t n_valid = 0;           // rows [0, n_valid) may hold a key
        std::vector<int32_t> pb;       // block index of the key each row holds, -1 for none
        std::vector<int32_t> cells;    // its ratio member cells, in slot order
    };
    mutable std::vector<pool_seq> pool_rows;   // indexed by seq id
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * bias,
                       ggml_tensor * win_cells, ggml_tensor * win_pos, ggml_tensor * win_blk,
                       const ggml_tensor * k_idxs,
                       const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias) const;

    int64_t  qsa_pool_n_win(const llama_ubatch & ubatch, int64_t n_blocks) const;
    uint32_t qsa_pool_ratio() const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
