#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"


#include <algorithm>
#include <cinttypes>
#include <cassert>
#include <cmath>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
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
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // The V plane is never read as values (scoring uses K alone), so it becomes the pooled
        // key plane: F32 cells of idx_dim/ratio, so that the ratio cells of a block hold one
        // finished key of idx_dim. That needs one ratio across the indexer layers dividing
        // idx_dim; otherwise V stays unused and every graph pools every block.
        static const bool pool_env = [] {
            const char * e = getenv("LLAMA_QSA_POOL_CACHE");
            return e == nullptr || atoi(e) != 0;
        }();

        uint32_t ratio = 0;
        bool uniform = true;

        // n_layer_all: an MTP draft's only indexer layer sits past the trunk
        for (uint32_t il = 0; il < model.hparams.n_layer_all; ++il) {
            if (!filter_idx(il)) {
                continue;
            }

            const uint32_t r = model.hparams.dsv4_compress_ratios[il];

            if (ratio == 0) {
                ratio = r;
            } else if (r != ratio) {
                uniform = false;
            }
        }

        const uint32_t idx_dim = model.hparams.indexer_head_size;

        ggml_type type_v_idx  = type_v;
        bool      v_trans_idx = v_trans;

        if (pool_env && uniform && ratio > 0 && ratio <= 64 && idx_dim % ratio == 0) {
            pool_ratio = ratio;

            hparams_idx.n_embd_head_v_full = idx_dim/ratio;

            type_v_idx  = GGML_TYPE_F32;
            v_trans_idx = false;

            LLAMA_LOG_INFO("%s: pooled indexer keys cached in the V plane, ratio = %u\n", __func__, ratio);
        } else {
            LLAMA_LOG_WARN("%s: pooled indexer keys not cached (ratio %u, uniform %d, env %d)\n",
                    __func__, ratio, uniform ? 1 : 0, pool_env ? 1 : 0);
        }

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v_idx, v_trans_idx, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }

    qsa_pool_invalidate();
}

void llama_memory_hybrid_idx::qsa_pool_invalidate() const {
    for (auto & ps : pool_rows) {
        ps.dirty = true;
    }
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        // removing a suffix keeps every lower block and its row where it is; anything else
        // can move a block to another row, so its rows are repooled by the next graph
        const bool suffix = p1 < 0 || p1 > mem_idx->seq_pos_max(seq_id);

        if (!suffix && seq_id >= 0 && (size_t) seq_id < pool_rows.size()) {
            pool_rows[seq_id].dirty = true;
        }

        mem_idx->seq_rm(seq_id, p0, p1);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }

    qsa_pool_invalidate();
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }

    qsa_pool_invalidate();
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }

    qsa_pool_invalidate();
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }

    qsa_pool_invalidate();
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }

    // the restored plane holds whatever the writer had pooled; the tables do not know it
    qsa_pool_invalidate();
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

// an mrope model carries several position rows for every token, so llama_ubatch::is_pos_2d
// says nothing about images; an image is a token whose rows disagree (its height and width
// positions differ from its temporal one), and only those make the block ranking unstable
static bool qsa_ubatch_has_2d(const llama_ubatch & ubatch) {
    if (ubatch.n_pos < 3 || ubatch.pos == nullptr) {
        return false;
    }

    const int64_t n = ubatch.n_tokens;

    for (int64_t i = 0; i < n; ++i) {
        if (ubatch.pos[i + n] != ubatch.pos[i] || ubatch.pos[i + 2*n] != ubatch.pos[i]) {
            return true;
        }
    }

    return false;
}

bool llama_memory_hybrid_idx::qsa_single_seq(const llama_ubatch & ubatch, uint32_t n_ns) const {
    if (mem_idx == nullptr || qsa_ubatch_has_2d(ubatch)) {
        return false;
    }

    GGML_ASSERT(n_ns > 0 && ubatch.n_tokens % n_ns == 0);
    const int64_t n_tps = ubatch.n_tokens/n_ns;

    for (uint32_t s = 0; s < n_ns; ++s) {
        const llama_seq_id seq = ubatch.seq_id[s*n_tps][0];

        const auto & cells = mem_idx->get_cells(seq);

        int n_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_present++;
            }
        }

        if (n_present > 1) {
            return false;
        }
    }

    return true;
}

int64_t llama_memory_hybrid_idx::qsa_pool_n_win(const llama_ubatch & ubatch, uint32_t n_ns, int64_t n_blocks) const {
    // 2d positions are ranked, and ranks shift with every insertion: never trusted
    if (pool_ratio == 0 || qsa_ubatch_has_2d(ubatch)) {
        return n_blocks;
    }

    GGML_ASSERT(n_ns > 0 && ubatch.n_tokens % n_ns == 0);
    const int64_t n_tps = ubatch.n_tokens/n_ns;

    for (uint32_t s = 0; s < n_ns; ++s) {
        const llama_seq_id seq = ubatch.seq_id[s*n_tps][0];

        if (seq < 0 || (size_t) seq >= pool_rows.size() || pool_rows[seq].dirty) {
            return n_blocks;
        }

        // a unified stream holding several sequences renumbers blocks as they complete
        const auto & cells = mem_idx->get_cells(seq);

        int n_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_present++;
            }
        }

        if (n_present > 1) {
            return n_blocks;
        }
    }

    // n_tps tokens touch at most n_tps blocks, complete or not
    return std::min<int64_t>(n_blocks, n_tps);
}

// LLAMA_QSA_POOL_TRACE=1 logs every window decision and fill: n_win against n_blocks, and
// how many blocks were actually stale
static bool qsa_pool_trace() {
    static const bool on = getenv("LLAMA_QSA_POOL_TRACE") != nullptr && atoi(getenv("LLAMA_QSA_POOL_TRACE")) != 0;
    return on;
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * bias,
        ggml_tensor * win_cells,
        ggml_tensor * win_pos,
        ggml_tensor * win_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pad,
        const ggml_tensor * k_idxs,
        int64_t n_kv,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);
    GGML_ASSERT((blk_cells == nullptr) == (blk_pad == nullptr));

    GGML_ASSERT(cell_blk == nullptr || ggml_backend_buffer_is_host(cell_blk->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(k_idxs->buffer));

    const int64_t n_ns     = win_cells->ne[1];       // streams in this ubatch
    const int64_t r        = ratio;

    GGML_ASSERT(cell_blk == nullptr || (cell_blk->ne[0] == n_kv && cell_blk->ne[1] == n_ns));
    const int64_t n_blocks = (n_kv + r - 1)/r;
    const int64_t n_win    = win_cells->ne[0]/r;
    const int64_t n_tokens = ubatch->n_tokens;

    // win_blk exists only when the plane is written: an input no node reads has no memory
    GGML_ASSERT(win_cells->ne[0] == r*n_win && win_cells->ne[1] == n_ns);
    GGML_ASSERT(win_blk == nullptr || (win_blk->ne[0] == n_win && win_blk->ne[1] == n_ns));
    GGML_ASSERT(win_pos->ne[0] == 4*n_win*n_ns);
    GGML_ASSERT(n_win >= 1 && n_win <= n_blocks);
    GGML_ASSERT((win_blk != nullptr) == (pool_ratio != 0));

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = cell_blk != nullptr ? (int32_t *) cell_blk->data : nullptr;
    float   * dst_bias      = (float   *) bias->data;
    int32_t * dst_win_cells = (int32_t *) win_cells->data;
    int32_t * dst_win_pos   = (int32_t *) win_pos->data;
    int32_t * dst_win_blk   = win_blk != nullptr ? (int32_t *) win_blk->data : nullptr;
    int32_t * dst_blk_cells = blk_cells != nullptr ? (int32_t *) blk_cells->data : nullptr;
    float   * dst_blk_pad   = blk_pad   != nullptr ? (float   *) blk_pad->data   : nullptr;

    if (blk_cells != nullptr) {
        GGML_ASSERT(blk_cells->ne[0] == r*(n_blocks + 1) && blk_cells->ne[1] == n_ns);
        GGML_ASSERT(blk_pad->ne[0]   == r*(n_blocks + 1) && blk_pad->ne[1]   == n_ns);
    }

    const int32_t * ub_cells = (const int32_t *) k_idxs->data;

    // every graph pools every block when the window covers them all
    const bool full = n_win >= n_blocks;

    const int64_t n_blocks_max = ((int64_t) mem_idx->get_size() + r - 1)/r;

    std::vector<uint8_t> touched;
    std::vector<int32_t> bid_cells;

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // TODO: this runs per ubatch and is O(n_kv) per stream, about 865 us at 33k context. the cost
    //       is the per-cell scan rather than these allocations, so hoisting them buys nothing
    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    std::fill(dst_win_pos, dst_win_pos + 4*n_win*n_ns, 0);

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk != nullptr ? dst_cell_blk + s*n_kv : nullptr;
        int32_t * cur_win_cells = dst_win_cells + s*(r*n_win);
        int32_t * cur_win_blk   = dst_win_blk != nullptr ? dst_win_blk + s*n_win : nullptr;

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(blk_of.begin(),   blk_of.end(),   -1);
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        group_cells();

        // mrope repeats one position across an image, so rank cells instead of using the position
        if (dup && ubatch->is_pos_2d() && one_seq) {
            order.clear();
            order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            });

            rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                rank[order[k]] = (int32_t) k;
            }

            ranked = true;

            group_cells();
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        int32_t n_bid = 0;

        for (int64_t pb = 0; pb < n_blocks; ++pb) {
            for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                if (grp_slots[g] != slots_full) {
                    continue;
                }

                grp_bid[g] = n_bid++;

                bid_idx  .push_back((int32_t) (pb*r));
                bid_cell .push_back(grp_first[g]);
                bid_slot0.push_back(grp_slot0[g]);
            }
        }

        GGML_ASSERT(n_bid <= n_blocks);

        // the rope position rows of a block: its first token's, in the four mrope sections
        auto blk_sec_pos = [&](int32_t b, int32_t * sec_pos) {
            sec_pos[0] = sec_pos[1] = sec_pos[2] = sec_pos[3] = bid_idx[b];

            if (ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }
        };

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? n_bid : n_blocks - 1;

        bid_cells.assign((size_t) n_bid*r, 0);

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = cell_grp[j];

            blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (blk_of[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                bid_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
            }

            if (cur_cell_blk != nullptr) {
                cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j];
            }
        }

        // the block table for two-stage selection: every complete block's cells, then the spare
        // block holding the unpooled cells (the tail of a single sequence), pads marked -inf
        if (dst_blk_cells != nullptr) {
            int32_t * cur_blk_cells = dst_blk_cells + s*(r*(n_blocks + 1));
            float   * cur_blk_pad   = dst_blk_pad   + s*(r*(n_blocks + 1));

            std::fill(cur_blk_cells, cur_blk_cells + r*(n_blocks + 1), 0);
            std::fill(cur_blk_pad,   cur_blk_pad   + r*(n_blocks + 1), -INFINITY);

            std::copy(bid_cells.begin(), bid_cells.end(), cur_blk_cells);
            std::fill(cur_blk_pad, cur_blk_pad + (size_t) n_bid*r, 0.0f);

            if (have_dead) {
                int32_t * dead_cells = cur_blk_cells + (size_t) dead_bid*r;
                float   * dead_pad   = cur_blk_pad   + (size_t) dead_bid*r;

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (cells.is_empty(j) || blk_of[j] >= 0) {
                        continue;
                    }

                    const int64_t idx  = ranked ? rank[j] : cells.pos_get(j);
                    const int64_t slot = idx%r;

                    // more than one unpooled group means several sequences: a slot already
                    // taken keeps the first, the caller uses the single-stage path there
                    if (dead_pad[slot] == 0.0f) {
                        continue;
                    }

                    dead_cells[slot] = (int32_t) j;
                    dead_pad[slot]   = 0.0f;
                }
            }
        }

        // the window: which complete blocks this graph pools, and where their keys go.
        // a block is stale when a cell of it was written by this ubatch (its key changed even
        // if the cells did not), when the row holds another block, or when nothing is trusted
        {
            touched.assign(n_kv, 0);

            for (int64_t ii = 0; ii < n_tps; ++ii) {
                const int32_t c = ub_cells[s*n_tps + ii];

                GGML_ASSERT(c >= 0 && c < n_kv);
                touched[c] = 1;
            }

            pool_seq * ps = nullptr;

            if (pool_ratio != 0) {
                if ((size_t) seq_of_stream >= pool_rows.size()) {
                    pool_rows.resize((size_t) seq_of_stream + 1);
                }

                ps = &pool_rows[seq_of_stream];

                if (ps->pb.empty()) {
                    ps->pb   .assign(n_blocks_max, -1);
                    ps->cells.assign((size_t) n_blocks_max*r, -1);
                    ps->n_valid = 0;
                    ps->dirty   = true;
                }

                GGML_ASSERT(full || !ps->dirty);
            }

            // a ranking that appeared without a 2d ubatch (it cannot, by construction) would
            // move every row; refuse loudly rather than pool a stale window
            GGML_ASSERT((full || !ranked) && "qsa: ranked positions in a windowed graph");

            int64_t n_stale = 0;

            for (int32_t b = 0; b < n_bid; ++b) {
                const int32_t * bc = bid_cells.data() + (size_t) b*r;

                bool stale = full;

                if (!stale) {
                    stale = b >= ps->n_valid || ps->pb[b] != bid_idx[b];

                    for (int64_t k = 0; !stale && k < r; ++k) {
                        stale = ps->cells[(size_t) b*r + k] != bc[k] || touched[bc[k]];
                    }
                }

                if (!stale) {
                    continue;
                }

                if (n_stale >= n_win) {
                    GGML_ABORT("qsa: %" PRId64 " blocks need pooling but the window holds %" PRId64 "\n", n_stale + 1, n_win);
                }

                std::copy(bc, bc + r, cur_win_cells + n_stale*r);

                if (cur_win_blk != nullptr) {
                    cur_win_blk[n_stale] = b;
                }

                int32_t sec_pos[4];
                blk_sec_pos(b, sec_pos);

                for (int64_t sec = 0; sec < 4; ++sec) {
                    dst_win_pos[sec*(n_win*n_ns) + s*n_win + n_stale] = sec_pos[sec];
                }

                n_stale++;
            }

            // pads repeat the last entry, so writing them again changes nothing. with no entry
            // the pads recompute a valid row from its own cells, or land on a row no query reads
            {
                int32_t pad_cells[64];
                int32_t pad_blk = 0;
                int32_t pad_pos[4] = { 0, 0, 0, 0 };

                if (n_stale > 0) {
                    std::copy(cur_win_cells + (n_stale - 1)*r, cur_win_cells + n_stale*r, pad_cells);
                    pad_blk = cur_win_blk != nullptr ? cur_win_blk[n_stale - 1] : 0;

                    for (int64_t sec = 0; sec < 4; ++sec) {
                        pad_pos[sec] = dst_win_pos[sec*(n_win*n_ns) + s*n_win + n_stale - 1];
                    }
                } else if (n_bid > 0) {
                    std::copy(bid_cells.data(), bid_cells.data() + r, pad_cells);
                    blk_sec_pos(0, pad_pos);
                } else {
                    std::fill(pad_cells, pad_cells + r, 0);
                }

                for (int64_t w = n_stale; w < n_win; ++w) {
                    std::copy(pad_cells, pad_cells + r, cur_win_cells + w*r);

                    if (cur_win_blk != nullptr) {
                        cur_win_blk[w] = pad_blk;
                    }

                    for (int64_t sec = 0; sec < 4; ++sec) {
                        dst_win_pos[sec*(n_win*n_ns) + s*n_win + w] = pad_pos[sec];
                    }
                }
            }

            if (qsa_pool_trace()) {
                LLAMA_LOG_INFO("qsa-pool: seq %d n_tokens %" PRId64 " n_kv %" PRId64 " n_blocks %" PRId64 " n_bid %d n_win %" PRId64 " stale %" PRId64 " full %d dirty %d\n",
                        (int) seq_of_stream, n_tokens, n_kv, n_blocks, n_bid, n_win, n_stale, full ? 1 : 0, ps != nullptr && ps->dirty ? 1 : 0);
            }

            if (ps != nullptr) {
                for (int32_t b = 0; b < n_bid; ++b) {
                    ps->pb[b] = bid_idx[b];
                    std::copy(bid_cells.data() + (size_t) b*r, bid_cells.data() + (size_t) (b + 1)*r, ps->cells.data() + (size_t) b*r);
                }

                for (int32_t b = n_bid; b < ps->n_valid; ++b) {
                    ps->pb[b] = -1;
                }

                ps->n_valid = n_bid;

                // ranks move with every insertion and a shared stream renumbers blocks: keep
                // asking for every block until the stream is one sequence on plain positions
                ps->dirty = ranked || !one_seq;
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    // a block that starts after the query is future in every cell: the cell
                    // mask hides it anyway, and a block-level pick (two-stage) must not spend
                    // a candidate slot on it in place of a visible block
                    if (b >= n_bid || !cells.seq_has((uint32_t) bid_cell[b], seq_id) || bid_idx[b] > q) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = bid_idx[b] >= tail_start ? 1e9f : 0.0f;
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so
                // it gets the tail value. it must stay finite: a sequence with fewer than
                // `ratio` cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * bias,
        ggml_tensor * win_cells,
        ggml_tensor * win_pos,
        ggml_tensor * win_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pad,
        const ggml_tensor * k_idxs,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(mem != nullptr);

    GGML_ASSERT(get_idx() != nullptr);

    mem->set_input_qsa(cell_blk, bias, win_cells, win_pos, win_blk, blk_cells, blk_pad, k_idxs, get_idx()->get_n_kv(), ubatch, ratio, blk_bias);
}

bool llama_memory_hybrid_idx_context::qsa_single_seq(const llama_ubatch & ubatch) const {
    GGML_ASSERT(mem != nullptr);

    return mem->qsa_single_seq(ubatch, get_n_stream());
}

int64_t llama_memory_hybrid_idx_context::qsa_pool_n_win(const llama_ubatch & ubatch, int64_t n_blocks) const {
    GGML_ASSERT(mem != nullptr);

    return mem->qsa_pool_n_win(ubatch, get_n_stream(), n_blocks);
}

uint32_t llama_memory_hybrid_idx_context::qsa_pool_ratio() const {
    return mem != nullptr ? mem->qsa_pool_ratio() : 0;
}
