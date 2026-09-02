#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// Greedy speculative decoding is only useful if it reproduces the non-speculative
// output. The target model sees the same tokens either way, but in a different
// *shape*: one token per decode without a draft, a chunk of (1 + n_draft) with one.
// This walks the difference apart:
//
//   [det]      the same path twice, to separate a wrong answer from an unstable one
//   [shape]    same tokens, chunked vs one-at-a-time, no rollback at all
//   [ckpt]     saving and restoring a sequence, across every source/destination state
//   [ckpt-roll] a restore followed by a trim, which is how the prompt cache reuses
//   [roll-ar]  rolling back further than the last ubatch was wide
//   [roll]     the real speculative loop - chunk, partial accept, roll back, continue
//   [trace]    (SPEC_TRACE=1) which graph nodes move when the shape changes
//
// Every phase reports the numeric spread and the only thing greedy sampling
// actually sees: argmax flips.
//
// Tuning: SPEC_N tokens, SPEC_PREFILL prefill, SPEC_CHUNK chunk width, SPEC_RS
// snapshot slots, SPEC_VERBOSE per-position detail.
//
// Note that a generated dummy model cannot stand in for a real one here: on the
// qwen4exp test model every phase scores an exact zero even with the snapshot
// planes deliberately left unwritten, so it detects none of this.


// ---------------------------------------------------------------------------
// [trace] mode: record the leading values of every named graph tensor for one
// decode, replay the same decode at a different chunk width, and report every
// node whose column-0 values move. Column 0 is the same token in both runs, so
// a correct backend reproduces it regardless of how many tokens sit beside it.
// I32 rows are flagged as selections: a changed index is a different choice,
// not a rounding difference, which is how the expert routing shows up.
struct trace_entry {
    std::string name;
    std::string op;
    int64_t     ne[4];
    float       v[8];
    int         n_v;
    bool        is_idx;   // an I32 row: a selection, so any change is a different choice
};

static std::vector<trace_entry> g_trace;
static bool                     g_trace_on = false;

static bool trace_cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    if (ask) {
        return g_trace_on;
    }
    if (!g_trace_on) {
        return true;
    }
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_I32) {
        return true;
    }
    if (!ggml_is_contiguous(t) || t->data == nullptr) {  // strided views are not worth chasing here
        return true;
    }
    // ggml names unlabelled nodes by graph index, and the two graphs do not have
    // the same node count, so "node_27" is a different node in each. Only nodes the
    // model named through cb() can be matched across runs.
    const char * tname = ggml_get_name(t);
    if (strncmp(tname, "node_", 5) == 0 || tname[0] == '\0' || tname[0] == ' ') {
        return true;
    }
    // Recurrent-cache writes are not comparable this way. A per-token activation is
    // [n_embd, n_tokens], so its element 0 is the chunk's first token in both runs.
    // A cache row is the rollback ring, whose slot 0 holds the state after the *last*
    // token of the ubatch - a different token in each run. Comparing them measures one
    // step of the recurrence, not a batch-invariance failure.
    if (strncmp(tname, "cache_", 6) == 0) {
        return true;
    }
    trace_entry e;
    e.name = ggml_get_name(t);
    e.op   = ggml_op_name(t->op);
    for (int i = 0; i < 4; ++i) {
        e.ne[i] = t->ne[i];
    }
    e.n_v = (int) std::min<int64_t>(8, (int64_t) (ggml_nbytes(t)/ggml_type_size(t->type)));
    e.n_v = (int) std::min<int64_t>(e.n_v, t->ne[0]);
    if (e.n_v <= 0) {
        return true;
    }
    e.is_idx = t->type == GGML_TYPE_I32;
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, e.v, 0, e.n_v*sizeof(float));
    } else if (t->type == GGML_TYPE_I32) {
        int32_t tmp[8];
        ggml_backend_tensor_get(t, tmp, 0, e.n_v*sizeof(int32_t));
        for (int i = 0; i < e.n_v; ++i) {
            e.v[i] = (float) tmp[i];
        }
    } else {
        ggml_fp16_t tmp[8];
        ggml_backend_tensor_get(t, tmp, 0, e.n_v*sizeof(ggml_fp16_t));
        for (int i = 0; i < e.n_v; ++i) {
            e.v[i] = ggml_fp16_to_fp32(tmp[i]);
        }
    }
    g_trace.push_back(std::move(e));
    return true;
}

static uint32_t env_u32(const char * name, uint32_t def) {
    const char * e = getenv(name);
    return e ? (uint32_t) atoi(e) : def;
}

struct ctx_cfg {
    uint32_t n_rs_seq;
    uint32_t n_ctx;
    uint32_t n_batch;
};

static llama_context * make_ctx(const common_params & params, llama_model * model, const ctx_cfg & cfg) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = cfg.n_rs_seq;
    cparams.n_ctx     = cfg.n_ctx;
    cparams.n_batch   = cfg.n_batch;
    cparams.n_ubatch  = cfg.n_batch;
    return llama_init_from_model(model, cparams);
}

// decode [pos0, pos0 + n) requesting logits for every token, or only the last one: a prefill
// only needs the last, and a span of more than INT32_MAX/n_vocab rows (8648 at 248320) makes
// the CUDA output path write past its tensor (illegal access, or silent corruption of whatever
// sits next to it, which showed as a width-1 path that differed between two fresh contexts)
static bool decode_span(llama_context * ctx, const std::vector<llama_token> & toks, uint32_t pos0, uint32_t n, bool logits_all = true) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (uint32_t i = 0; i < n; ++i) {
        common_batch_add(batch, toks[pos0 + i], (llama_pos) (pos0 + i), { 0 }, logits_all || i + 1 == n);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

struct diff_stats {
    double   max_abs   = 0.0;
    uint32_t n_flips   = 0;   // positions whose argmax differs
    int32_t  first_pos = -1;  // first position with a flipped argmax
};

static int argmax(const float * l, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (l[i] > l[best]) {
            best = i;
        }
    }
    return best;
}

static void accumulate(diff_stats & st, const float * a, const float * b, int n_vocab, uint32_t pos) {
    for (int i = 0; i < n_vocab; ++i) {
        st.max_abs = std::max(st.max_abs, (double) std::fabs(a[i] - b[i]));
    }
    if (argmax(a, n_vocab) != argmax(b, n_vocab)) {
        st.n_flips++;
        if (st.first_pos < 0) {
            st.first_pos = (int32_t) pos;
        }
    }
}

static void report(const char * label, const diff_stats & st, uint32_t n_cmp) {
    printf("  %-46s max|dlogit| = %-11.6g  argmax flips = %u/%u", label, st.max_abs, st.n_flips, n_cmp);
    if (st.first_pos >= 0) {
        printf("  first at pos %d", st.first_pos);
    }
    printf("\n");
    fflush(stdout);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict     = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    params.cb_eval           = trace_cb;
    params.cb_eval_user_data = nullptr;

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    // a non-recurrent model still answers the shape question, and is the control for it
    const bool has_rs = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    const uint32_t n_tok    = env_u32("SPEC_N",      64);  // tokens after the prefill
    const uint32_t n_pre    = env_u32("SPEC_PREFILL", 8);  // prefill length
    const uint32_t chunk    = env_u32("SPEC_CHUNK",   7);  // 1 + n_draft
    const uint32_t n_rs_seq = has_rs ? env_u32("SPEC_RS", 6) : 0;
    const bool     verbose_shape = env_u32("SPEC_VERBOSE", 0) != 0;

    const ctx_cfg cfg = { n_rs_seq, n_pre + n_tok + 2*chunk + 8, n_pre + n_tok + 2*chunk + 8 };

    std::vector<llama_token> toks;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        for (uint32_t i = 0; i < n_pre + n_tok; ++i) {
            toks.push_back((llama_token) ((7*i + 1) % (uint32_t) n_vocab));
        }
    } else {
        toks = common_tokenize(vocab, "The quick brown fox jumps over the lazy dog. "
                                      "Meanwhile, the compiler emitted a diagnostic about an unused "
                                      "variable, and the linker complained about a duplicate symbol "
                                      "in the translation unit that nobody had touched for years.", true);
        while (toks.size() < n_pre + n_tok) {
            toks.push_back(toks[toks.size() % 17]);
        }
        toks.resize(n_pre + n_tok);
    }

    printf("\nconfig: prefill=%u tokens=%u chunk=%u n_rs_seq=%u n_vocab=%d\n\n",
           n_pre, n_tok, chunk, n_rs_seq, n_vocab);

    // Every phase produces logits for the same n_tok positions, so any two phases
    // are directly comparable. Capturing them lets each path be checked twice:
    // once against the single-token baseline (is it right?) and once against a
    // second run of itself (is it even stable?).
    using run_t = std::vector<std::vector<float>>;

    // one token per decode: exactly what the server does with no draft model
    const auto run_ar = [&]() {
        run_t out(n_tok);
        llama_context * ctx = make_ctx(params, model, cfg);
        if (ctx == nullptr || !decode_span(ctx, toks, 0, n_pre, false)) {
            fprintf(stderr, "%s : baseline prefill failed\n", __func__);
            exit(1);
        }
        for (uint32_t i = 0; i < n_tok; ++i) {
            if (!decode_span(ctx, toks, n_pre + i, 1)) {
                fprintf(stderr, "%s : baseline decode failed at %u\n", __func__, i);
                exit(1);
            }
            const float * l = llama_get_logits_ith(ctx, 0);
            out[i].assign(l, l + n_vocab);
        }
        llama_free(ctx);
        return out;
    };

    // fixed-size chunks, no rollback: only the decode shape differs from run_ar
    const auto run_chunked = [&]() {
        run_t out(n_tok);
        llama_context * ctx = make_ctx(params, model, cfg);
        if (ctx == nullptr || !decode_span(ctx, toks, 0, n_pre, false)) {
            exit(1);
        }
        for (uint32_t i = 0; i < n_tok; i += chunk) {
            const uint32_t n = std::min(chunk, n_tok - i);
            if (!decode_span(ctx, toks, n_pre + i, n)) {
                fprintf(stderr, "%s : chunked decode failed at %u\n", __func__, i);
                exit(1);
            }
            for (uint32_t j = 0; j < n; ++j) {
                const float * l = llama_get_logits_ith(ctx, (int32_t) j);
                out[i + j].assign(l, l + n_vocab);
            }
        }
        llama_free(ctx);
        return out;
    };

    // the real loop: decode a chunk, accept only part of it, roll the rest back,
    // continue from the accepted position. Only accepted slots are recorded,
    // because only those are the tokens the server actually emits.
    // accept_all=false cycles the acceptance count so every rollback depth is hit.
    const auto run_rollback = [&](bool accept_one) {
        run_t out(n_tok);
        llama_context * ctx = make_ctx(params, model, cfg);
        if (ctx == nullptr || !decode_span(ctx, toks, 0, n_pre, false)) {
            exit(1);
        }
        uint32_t i = 0, round = 0;
        while (i < n_tok) {
            const uint32_t n = std::min(chunk, n_tok - i);
            if (!decode_span(ctx, toks, n_pre + i, n)) {
                fprintf(stderr, "%s : roll decode failed at %u\n", __func__, i);
                exit(1);
            }
            const uint32_t n_acc = accept_one ? 1 : 1 + (round % n);
            for (uint32_t j = 0; j < n_acc; ++j) {
                const float * l = llama_get_logits_ith(ctx, (int32_t) j);
                out[i + j].assign(l, l + n_vocab);
            }
            if (n_acc < n &&
                !llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) (n_pre + i + n_acc), -1)) {
                fprintf(stderr, "%s : rollback of %u refused at %u\n", __func__, n - n_acc, i);
                exit(1);
            }
            i += n_acc;
            round++;
        }
        llama_free(ctx);
        return out;
    };

    int rc = 0;

    const auto cmp = [&](const char * label, const run_t & a, const run_t & b, bool detail) {
        diff_stats st;
        uint32_t n_cmp = 0;
        for (uint32_t i = 0; i < n_tok; ++i) {
            if (a[i].empty() || b[i].empty()) {
                continue;
            }
            diff_stats one;
            accumulate(one, a[i].data(), b[i].data(), n_vocab, i);
            st.max_abs = std::max(st.max_abs, one.max_abs);
            if (one.n_flips) {
                st.n_flips++;
                if (st.first_pos < 0) {
                    st.first_pos = (int32_t) i;
                }
            }
            n_cmp++;
            if (detail) {
                printf("        pos %2u (chunk %u slot %u) max|dlogit| = %-11.6g %s\n",
                       i, i/chunk, i%chunk, one.max_abs, one.n_flips ? "ARGMAX FLIP" : "");
            }
        }
        report(label, st, n_cmp);
        return st;
    };

    if (env_u32("SPEC_TRACE", 0)) {
        // one decode at chunk 1 and one at chunk n, from the identical prefill
        const auto capture = [&](uint32_t n) {
            g_trace.clear();
            llama_context * ctx = make_ctx(params, model, cfg);
            if (ctx == nullptr || !decode_span(ctx, toks, 0, n_pre, false)) {
                exit(1);
            }
            g_trace_on = true;
            const bool ok = decode_span(ctx, toks, n_pre, n);
            g_trace_on = false;
            llama_free(ctx);
            if (!ok) {
                exit(1);
            }
            return g_trace;
        };
        const auto a = capture(1);
        const auto b = capture(chunk);
        printf("  [trace] %zu nodes at chunk 1, %zu at chunk %u\n", a.size(), b.size(), chunk);

        // the two graphs do not have the same node count, so match by name and
        // compare the k-th occurrence of each name in one against the other
        std::map<std::string, std::vector<const trace_entry *>> by_name;
        for (const auto & e : b) {
            by_name[e.name].push_back(&e);
        }
        std::map<std::string, size_t> seen;
        size_t n_shown = 0, n_idx = 0;
        for (const auto & ea : a) {
            const auto it = by_name.find(ea.name);
            if (it == by_name.end()) {
                continue;
            }
            const size_t k = seen[ea.name]++;
            if (k >= it->second.size()) {
                continue;
            }
            const trace_entry & eb = *it->second[k];
            double d = 0.0;
            for (int i = 0; i < std::min(ea.n_v, eb.n_v); ++i) {
                d = std::max(d, (double) std::fabs(ea.v[i] - eb.v[i]));
            }
            if (d == 0.0) {
                continue;
            }
            if (ea.is_idx) {
                n_idx++;
            }
            if (n_shown < 16 || (ea.is_idx && n_idx <= 8)) {
                printf("  [trace] %-44s %-12s [%5lld x %5lld] %s max|d| = %g\n",
                       ea.name.c_str(), ea.op.c_str(),
                       (long long) ea.ne[0], (long long) ea.ne[1],
                       ea.is_idx ? "SELECTION" : "         ", d);
            }
            n_shown++;
        }
        printf("  [trace] %zu traced nodes moved, %zu of them selections (expert ids and the like)\n",
               n_shown, n_idx);
        printf("\n");
    }

    const run_t ar = run_ar();
    printf("baseline: %u single-token decodes captured\n\n", n_tok);

    printf("  reproducibility (same path twice, fresh context each time)\n");
    if (cmp("[det] 1-at-a-time", ar, run_ar(), false).n_flips) {
        printf("        ^ the AR path is not reproducible; nothing below is meaningful\n");
        rc = 1;
    }
    const run_t ch = run_chunked();
    cmp("[det] chunked", ch, run_chunked(), false);
    run_t rl;
    if (has_rs) {
        rl = run_rollback(false);
        cmp("[det] chunked+rollback", rl, run_rollback(false), false);
    }

    // ----------------------------------------------------------------- [ckpt]
    // The server restores a saved sequence state whenever a prompt is reused. Two
    // things vary and both have to be neutral: what the *source* looked like when
    // it was saved (plain, or with a rollback still pending, so the live state sits
    // in a snapshot plane), and what the *destination* was doing beforehand (empty,
    // holding an unrelated prompt, or holding one with its own pending rollback).
    // Every combination replays the same tokens and must land on the same logits.
    if (has_rs) {
        printf("\n  checkpoint restore (replayed one token at a time)\n");

        // an unrelated prompt, to dirty the destination
        std::vector<llama_token> other(toks.size());
        for (size_t i = 0; i < toks.size(); ++i) {
            other[i] = (llama_token) ((toks[i] + 5077) % n_vocab);
        }
        const uint32_t n_dirty = n_pre + std::min(n_tok, chunk);
        const uint32_t n_back  = std::min(chunk - 1, n_rs_seq);

        const auto replay = [&](llama_context * ctx) {
            run_t out(n_tok);
            for (uint32_t i = 0; i < n_tok; ++i) {
                if (!decode_span(ctx, toks, n_pre + i, 1)) {
                    fprintf(stderr, "%s : ckpt replay failed at %u\n", __func__, i);
                    exit(1);
                }
                const float * l = llama_get_logits_ith(ctx, 0);
                out[i].assign(l, l + n_vocab);
            }
            return out;
        };

        for (uint32_t src_mode = 0; src_mode < 2; ++src_mode) {
            common_prompt_checkpoint ckpt;
            run_t ref;
            {
                llama_context * ctx = make_ctx(params, model, cfg);
                if (ctx == nullptr) {
                    exit(1);
                }
                if (src_mode == 0) {
                    // saved from a settled state
                    if (!decode_span(ctx, toks, 0, n_pre, false)) {
                        exit(1);
                    }
                } else {
                    // saved while a rollback is pending: the live state is in a snapshot plane
                    if (!decode_span(ctx, toks, 0, n_pre + n_back, false) ||
                        !llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) n_pre, -1)) {
                        fprintf(stderr, "%s : could not save from a rollback plane\n", __func__);
                        exit(1);
                    }
                }
                ckpt.update_tgt(ctx, 0, 0);
                ref = replay(ctx);   // the source's own continuation is the reference
                llama_free(ctx);
            }
            if (src_mode == 0) {
                cmp("[ckpt] plain source, own replay", ar, ref, false);
            }

            for (uint32_t dst_mode = 0; dst_mode < 3; ++dst_mode) {
                llama_context * ctx = make_ctx(params, model, cfg);
                if (ctx == nullptr) {
                    exit(1);
                }
                if (dst_mode > 0 && !decode_span(ctx, other, 0, n_dirty, false)) {
                    exit(1);
                }
                if (dst_mode == 2 &&
                    !llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) (n_dirty - n_back), -1)) {
                    fprintf(stderr, "%s : could not leave a pending rollback\n", __func__);
                    exit(1);
                }

                ckpt.load_tgt(ctx, 0, 0);
                const run_t out = replay(ctx);
                llama_free(ctx);

                static const char * dst_names[] = { "into a fresh context",
                                                    "over another prompt",
                                                    "over a pending rollback" };
                char label[96];
                snprintf(label, sizeof(label), "[ckpt] %s src -> %s",
                         src_mode == 0 ? "plain" : "rolled-back", dst_names[dst_mode]);
                if (cmp(label, ref, out, false).n_flips) {
                    rc = 1;
                }
            }
        }
    }

    // ------------------------------------------------------------ [ckpt-roll]
    // The server reaches a rollback through the prompt cache, not only through the
    // accept/reject loop: on a partial prefix match it restores a context checkpoint
    // and then trims the few positions the new prompt does not want. A restore only
    // rewrites the live state, so the snapshot planes still belong to whatever the
    // destination context was doing before, and the trim reads one of them.
    if (has_rs) {
        printf("\n  checkpoint restore followed by a trim\n");

        std::vector<llama_token> other(toks.size());
        for (size_t i = 0; i < toks.size(); ++i) {
            other[i] = (llama_token) ((toks[i] + 5077) % n_vocab);
        }
        const uint32_t n_back  = std::min(chunk - 1, n_rs_seq);
        const uint32_t n_dirty = n_pre + std::min(n_tok, chunk);

        common_prompt_checkpoint ckpt;
        run_t ref;
        {
            llama_context * ctx = make_ctx(params, model, cfg);
            if (ctx == nullptr || !decode_span(ctx, toks, 0, n_pre + n_back, false)) {
                exit(1);
            }
            ckpt.update_tgt(ctx, 0, 0);
            // the source trims the same positions from the state it just built
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) n_pre, -1)) {
                fprintf(stderr, "%s : source trim refused\n", __func__);
                exit(1);
            }
            ref.resize(n_tok);
            for (uint32_t i = 0; i < n_tok; ++i) {
                if (!decode_span(ctx, toks, n_pre + i, 1)) {
                    exit(1);
                }
                const float * l = llama_get_logits_ith(ctx, 0);
                ref[i].assign(l, l + n_vocab);
            }
            llama_free(ctx);
        }

        for (uint32_t dst_mode = 0; dst_mode < 2; ++dst_mode) {
            llama_context * ctx = make_ctx(params, model, cfg);
            if (ctx == nullptr) {
                exit(1);
            }
            if (dst_mode == 1 && !decode_span(ctx, other, 0, n_dirty, false)) {
                exit(1);
            }
            ckpt.load_tgt(ctx, 0, 0);
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) n_pre, -1)) {
                fprintf(stderr, "%s : trim after restore refused (mode %u)\n", __func__, dst_mode);
                llama_free(ctx);
                rc = 1;
                continue;
            }
            run_t out(n_tok);
            for (uint32_t i = 0; i < n_tok; ++i) {
                if (!decode_span(ctx, toks, n_pre + i, 1)) {
                    exit(1);
                }
                const float * l = llama_get_logits_ith(ctx, 0);
                out[i].assign(l, l + n_vocab);
            }
            llama_free(ctx);

            if (cmp(dst_mode == 0 ? "[ckpt-roll] into a fresh context"
                                  : "[ckpt-roll] over another prompt", ref, out, false).n_flips) {
                rc = 1;
            }
        }
    }

    // -------------------------------------------------------------- [roll-ar]
    // Rolling back further than the last ubatch was wide. The snapshot ring is
    // filled by the decode that writes it, so a run of single-token decodes only
    // ever refreshes the newest slot; a deeper rollback then has to reach slots
    // that an *earlier*, wider ubatch wrote. Advancing R tokens one at a time and
    // rolling all R back must return to exactly where it started.
    if (has_rs) {
        printf("\n  rollback deeper than the last ubatch\n");
        for (uint32_t r = 1; r <= std::min(n_rs_seq, n_tok - 1); ++r) {
            llama_context * ctx = make_ctx(params, model, cfg);
            if (ctx == nullptr || !decode_span(ctx, toks, 0, n_pre, false)) {
                exit(1);
            }
            for (uint32_t i = 0; i < r; ++i) {           // r single-token decodes
                if (!decode_span(ctx, toks, n_pre + i, 1)) {
                    exit(1);
                }
            }
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) n_pre, -1)) {
                fprintf(stderr, "%s : rollback of %u refused\n", __func__, r);
                llama_free(ctx);
                continue;
            }
            if (!decode_span(ctx, toks, n_pre, 1)) {     // back where we started
                exit(1);
            }
            run_t out(n_tok);
            const float * l = llama_get_logits_ith(ctx, 0);
            out[0].assign(l, l + n_vocab);
            llama_free(ctx);

            char label[64];
            snprintf(label, sizeof(label), "[roll-ar] %u single-token step%s back", r, r == 1 ? "" : "s");
            if (cmp(label, ar, out, false).n_flips) {
                rc = 1;
            }
        }
    }

    printf("\n  correctness (vs the single-token baseline)\n");
    cmp("[shape] chunked", ar, ch, verbose_shape);
    if (has_rs) {
        cmp("[roll] cycling acceptance", ar, rl, false);
        cmp("[roll] accept-one-per-round", ar, run_rollback(true), false);
    }

    printf("\n");
    return rc;
}
