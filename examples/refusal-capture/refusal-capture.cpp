// Capture final-prompt-token residual activations (<prefix>-<il>) over a set of
// prompts, for refusal-direction extraction. Stage 1 of the refusal pipeline.
//
// Uses the existing cb_eval hook (no core changes) to grab the last column of
// each layer's tapped tensor — the </think> decision-token residual, matching
// the ds4 capture. Writes raw f32 [n_prompts][n_layer][n_embd].
//
// Env:
//   CAPTURE_PROMPTS  file, one prompt per line (# comments skipped)
//   CAPTURE_OUT      output .f32 path
//   CAPTURE_OUT_THINK optional second output; both modes from one model load
//   CAPTURE_TENSOR   tensor name prefix to capture (default "l_out")
//   CAPTURE_TEMPLATE chat template: "ds4" (default) or "qwen4exp"
//   CAPTURE_GEN      generated tokens to run (default 0 = prompt-final position)
//   CAPTURE_BAND     "lo,hi" generated-token half-open band to average over
//                    (default "0,<CAPTURE_GEN>"); ignored when CAPTURE_GEN=0
//
// POSITION MATTERS. Measured on Qwen3.8-27B held-out captures: a direction
// derived at the final prompt token scores 0/64 anti-selective layers there and
// 22/64 at the response position -- 16 of them inside its own L29-63 operating
// band. The reverse holds too (a response-derived direction degrades at the
// prompt position), so this is a property of the method, not one derivation.
// CAPTURE_GEN>0 samples greedily and averages the residual over the band, which
// is the "mid-response" position that comparison says we are blind to.
//
// PICK THE TAP DELIBERATELY. This grabs column ne[1]-1, i.e. it assumes ne[1] is
// the token axis. On a hyper-connection arch (ds4, qwen4exp) the post-block
// residual `l_last` is [n_embd, hc, n_tokens], so ne[1] is the HC-stream axis and
// tapping it silently captures stream hc-1 of token 0. That produced 192/400
// byte-identical vectors on ds4 and collapsed separation 108 -> 13 with no error.
// Use a narrow [n_embd, n_tokens] tensor: `ffn_out` on ds4 and qwen4exp.
//
// The prefix match is exact up to the final '-', so "ffn_out" matches ffn_out-12
// and rejects ffn_moe_out-12.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct capture_state {
    std::string prefix;          // e.g. "ffn_out"
    int n_layer = 0;
    int n_embd  = 0;
    std::vector<float> cur;      // [n_layer*n_embd] for the in-flight prompt
    std::vector<uint8_t> scratch;
    // Number of layer-columns actually written for the in-flight prompt. A tap
    // that never matches, or that matches a tensor of the wrong width, writes
    // nothing and leaves a zero row -- indistinguishable from a real capture in
    // the .f32. Counted so the run can abort loudly instead.
    int hits = 0;
    int rejected_shape = 0;
    // Token count of the prompt currently being decoded. A tap whose ne[1] is
    // neither this nor 1 is not indexed by token, so its "last column" is not the
    // final prompt token. ne[1] == 1 is legitimate: on the last layer the
    // inp_out_ids reduction runs before the FFN, and with llama_batch_get_one only
    // the final token is an output row -- so ffn_out-<n_layer-1> is [n_embd, 1] and
    // column 0 IS the final prompt token.
    int n_tokens_expected = 0;
    // Response-band accumulation. Held in double because a 40-step sum of f32
    // residuals at these norms loses low bits otherwise.
    std::vector<double> accum;
    int accum_n = 0;
};

static bool capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (capture_state *) user_data;

    // name is "<prefix>-<il>"
    const char * dash = std::strrchr(t->name, '-');
    if (!dash) {
        return false;
    }
    const size_t plen = st->prefix.size();
    if ((size_t)(dash - t->name) != plen || std::strncmp(t->name, st->prefix.c_str(), plen) != 0) {
        return false;
    }

    if (ask) {
        return true; // yes, deliver this tensor's data
    }

    const int il = std::atoi(dash + 1);
    if (il < 0 || il >= st->n_layer) {
        return true;
    }
    const int n_embd   = (int) t->ne[0];
    const int n_tokens = (int) t->ne[1];
    if (n_embd != st->n_embd || n_tokens < 1) {
        // n_tokens == 0 is legitimate: on the last layer the inp_out_ids reduction
        // runs before the FFN, so earlier ubatches carry no output rows. A width
        // mismatch is not legitimate and means the tap is the wrong tensor.
        if (n_embd != st->n_embd) {
            st->rejected_shape++;
        }
        return true;
    }
    // A 3-D tap means ne[1] is not the token axis (hyper-connection residual).
    // Refuse rather than silently capturing a stream slice of the wrong token.
    if (t->ne[2] != 1 || t->ne[3] != 1) {
        st->rejected_shape++;
        return true;
    }
    // ne[2] == 1 alone is not sufficient: a single-token decode collapses an HC
    // residual to [n_embd, hc, 1], which passes the check above while ne[1] is
    // still the stream axis. Require the second axis to actually be tokens.
    if (n_tokens != 1 && n_tokens != st->n_tokens_expected) {
        st->rejected_shape++;
        return true;
    }

    // last column = final prompt token (the </think> decision position)
    const size_t col_bytes = (size_t) n_embd * sizeof(float);
    const size_t off       = (size_t) (n_tokens - 1) * col_bytes;

    float * dst = st->cur.data() + (size_t) il * n_embd;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        std::memcpy(dst, (const uint8_t *) t->data + off, col_bytes);
    } else {
        ggml_backend_tensor_get(t, dst, off, col_bytes);
    }
    st->hits++;
    return true;
}

static std::string render_ds4(const std::string & user, bool think) {
    // BOS + User + prompt + Assistant + (<think> | </think>)
    // nothink (</think>) = "about to answer directly"; think (<think>) = "about to reason"
    return "<｜begin▁of▁sentence｜><｜User｜>" + user +
           (think ? "<｜Assistant｜><think>" : "<｜Assistant｜></think>");
}

static std::string render_qwen4exp(const std::string & user, bool think) {
    // No BOS: this tokenizer sets add_bos_token = false.
    // nothink closes the block (final token "\n\n", id 271, the "answer directly"
    // decision position); think leaves it open (final token "\n", id 198).
    // Neither form injects a reasoning-effort system message -- that is what the
    // template's 'medium' does, and mixing efforts across arms changes the prefix.
    return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n" +
           (think ? "<think>\n" : "<think>\n\n</think>\n\n");
}

static std::string render(const std::string & tmpl, const std::string & user, bool think) {
    if (tmpl == "qwen4exp") {
        return render_qwen4exp(user, think);
    }
    return render_ds4(user, think);
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * pf = std::getenv("CAPTURE_PROMPTS");
    const char * of = std::getenv("CAPTURE_OUT");        // nothink output
    const char * oft = std::getenv("CAPTURE_OUT_THINK"); // optional think output (both modes, one load)
    const char * tn = std::getenv("CAPTURE_TENSOR");
    const char * tm = std::getenv("CAPTURE_TEMPLATE");
    if (!pf || !of) {
        fprintf(stderr, "set CAPTURE_PROMPTS and CAPTURE_OUT env vars\n");
        return 1;
    }
    const std::string tmpl = tm ? tm : "ds4";
    if (tmpl != "ds4" && tmpl != "qwen4exp") {
        fprintf(stderr, "CAPTURE_TEMPLATE must be 'ds4' or 'qwen4exp' (got '%s')\n", tmpl.c_str());
        return 1;
    }

    std::vector<std::string> prompts;
    {
        std::ifstream in(pf);
        std::string line;
        while (std::getline(in, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) line.pop_back();
            if (!line.empty() && line[0] != '#') prompts.push_back(line);
        }
    }
    if (prompts.empty()) { fprintf(stderr, "no prompts in %s\n", pf); return 1; }

    capture_state st;
    st.prefix = tn ? tn : "l_out";

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    params.warmup   = false;
    params.cb_eval  = capture_cb;
    params.cb_eval_user_data = &st;

    auto init = common_init_from_params(params);
    if (!init || !init->model() || !init->context()) { fprintf(stderr, "model load failed\n"); return 1; }
    llama_context * ctx   = init->context();
    const llama_model * model = init->model();

    st.n_layer = llama_model_n_layer(model);
    st.n_embd  = llama_model_n_embd(model);
    st.cur.assign((size_t) st.n_layer * st.n_embd, 0.0f);
    fprintf(stderr, "capture: %zu prompts, n_layer=%d n_embd=%d tensor=%s-* template=%s\n",
            prompts.size(), st.n_layer, st.n_embd, st.prefix.c_str(), tmpl.c_str());
    {
        // Echo the exact rendered prefix and its token ids. A wrong template
        // tokenizes as garbage rather than failing, so this is the only cheap
        // way to see it before spending a full capture run.
        const std::string sample = render(tmpl, "PROMPT", false);
        std::vector<llama_token> stoks = common_tokenize(ctx, sample, true, true);
        fprintf(stderr, "capture: nothink render = %s\n", sample.c_str());
        fprintf(stderr, "capture: nothink ids    = [");
        for (size_t i = 0; i < stoks.size(); ++i) fprintf(stderr, "%s%d", i ? ", " : "", stoks[i]);
        fprintf(stderr, "] (final token %d)\n", stoks.empty() ? -1 : stoks.back());
    }

    FILE * out  = std::fopen(of, "wb");
    if (!out) { fprintf(stderr, "cannot open %s\n", of); return 1; }
    FILE * outt = oft ? std::fopen(oft, "wb") : nullptr;
    if (oft && !outt) { fprintf(stderr, "cannot open %s\n", oft); return 1; }

    const int gen_tokens = std::getenv("CAPTURE_GEN") ? std::atoi(std::getenv("CAPTURE_GEN")) : 0;
    int band_lo = 0, band_hi = gen_tokens;
    if (const char * bs = std::getenv("CAPTURE_BAND")) {
        if (std::sscanf(bs, "%d,%d", &band_lo, &band_hi) != 2) {
            fprintf(stderr, "CAPTURE_BAND must be \"lo,hi\"\n");
            return 1;
        }
    }
    if (gen_tokens > 0) {
        if (band_lo < 0 || band_hi > gen_tokens || band_lo >= band_hi) {
            fprintf(stderr, "CAPTURE_BAND %d,%d invalid for CAPTURE_GEN=%d\n",
                    band_lo, band_hi, gen_tokens);
            return 1;
        }
        st.accum.assign(st.cur.size(), 0.0);
        fprintf(stderr, "capture: position=response gen=%d band=[%d,%d)\n",
                gen_tokens, band_lo, band_hi);
    } else {
        fprintf(stderr, "capture: position=prompt_final\n");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    auto mem = llama_get_memory(ctx);
    // capture one mode's final-token activation into st.cur, write to f
    auto capture_one = [&](const std::string & prompt, bool think, FILE * f) -> bool {
        llama_memory_clear(mem, true);
        std::fill(st.cur.begin(), st.cur.end(), 0.0f);
        st.hits = 0;
        st.rejected_shape = 0;
        const std::string text = render(tmpl, prompt, think);
        std::vector<llama_token> toks = common_tokenize(ctx, text, true, true);
        if (toks.empty()) return true; // skip, leave zeros
        st.n_tokens_expected = (int) toks.size();
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), toks.size()))) return false;

        if (gen_tokens > 0) {
            // Greedy continuation. st.cur is overwritten by each decode, so the
            // band average is accumulated as we go rather than kept per step.
            std::fill(st.accum.begin(), st.accum.end(), 0.0);
            st.accum_n = 0;
            for (int t = 0; t < gen_tokens; ++t) {
                const float * logits = llama_get_logits_ith(ctx, -1);
                if (!logits) return false;
                llama_token next = 0;
                float best = logits[0];
                for (int v = 1; v < n_vocab; ++v) {
                    if (logits[v] > best) { best = logits[v]; next = v; }
                }
                if (llama_vocab_is_eog(vocab, next)) break;

                st.n_tokens_expected = 1;
                if (llama_decode(ctx, llama_batch_get_one(&next, 1))) return false;
                if (t >= band_lo && t < band_hi) {
                    for (size_t k = 0; k < st.cur.size(); ++k) st.accum[k] += st.cur[k];
                    st.accum_n++;
                }
            }
            if (st.accum_n == 0) {
                // Generation stopped before reaching the band. Leave a zero row
                // rather than a partial one, and say so -- a short continuation
                // is a property of the prompt, not a failure.
                std::fill(st.cur.begin(), st.cur.end(), 0.0f);
            } else {
                for (size_t k = 0; k < st.cur.size(); ++k) {
                    st.cur[k] = (float) (st.accum[k] / st.accum_n);
                }
            }
        }

        std::fwrite(st.cur.data(), sizeof(float), st.cur.size(), f);
        return true;
    };

    bool aborted = false;
    for (size_t i = 0; i < prompts.size(); ++i) {
        if (!capture_one(prompts[i], false, out)) { fprintf(stderr, "  [%zu] nothink decode failed\n", i); break; }
        if (i == 0 && st.hits < st.n_layer) {
            // Every layer must contribute a column. Anything less means the tap
            // name is wrong for this architecture, or its shape is not
            // [n_embd, n_tokens]. Writing the remaining prompts would produce a
            // file of zeros that looks exactly like a successful capture.
            fprintf(stderr,
                    "capture: ABORT -- tap '%s-*' filled %d of %d layers"
                    " (%d rejected on shape).\n"
                    "  The tensor name is wrong for this arch, or it is not"
                    " [n_embd, n_tokens].\n"
                    "  ds4/qwen4exp: use CAPTURE_TENSOR=ffn_out. 'l_out' does not"
                    " exist on qwen4exp and\n"
                    "  'l_last' is [n_embd, hc, n_tokens] -- see the header note.\n",
                    st.prefix.c_str(), st.hits, st.n_layer, st.rejected_shape);
            aborted = true;
            break;
        }
        if (outt && !capture_one(prompts[i], true, outt)) { fprintf(stderr, "  [%zu] think decode failed\n", i); break; }
        if ((i + 1) % 25 == 0) fprintf(stderr, "  %zu/%zu\n", i + 1, prompts.size());
    }
    if (aborted) {
        std::fclose(out);
        if (outt) std::fclose(outt);
        std::remove(of);
        if (oft) std::remove(oft);
        return 1;
    }
    std::fclose(out);
    if (outt) std::fclose(outt);
    fprintf(stderr, "wrote %s%s [%zu x %d x %d f32]\n", of, oft ? " + think" : "",
            prompts.size(), st.n_layer, st.n_embd);
    return 0;
}
