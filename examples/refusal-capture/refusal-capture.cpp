// Capture final-prompt-token residual activations (l_out-<il>) over a set of
// prompts, for refusal-direction extraction. Stage 1 of the refusal pipeline.
//
// Uses the existing cb_eval hook (no core changes) to grab the last column of
// each layer's l_out tensor — the </think> decision-token residual, matching
// the ds4 capture. Writes raw f32 [n_prompts][n_layer][n_embd].
//
// Env:
//   CAPTURE_PROMPTS  file, one prompt per line (# comments skipped)
//   CAPTURE_OUT      output .f32 path
//   CAPTURE_TENSOR   tensor name prefix to capture (default "l_out")
// DS4 chat template is applied automatically (BOS/User/Assistant/</think>).

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct capture_state {
    std::string prefix;          // e.g. "l_out"
    int n_layer = 0;
    int n_embd  = 0;
    std::vector<float> cur;      // [n_layer*n_embd] for the in-flight prompt
    std::vector<uint8_t> scratch;
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
    return true;
}

static std::string render_ds4(const std::string & user, bool think) {
    // BOS + User + prompt + Assistant + (<think> | </think>)
    // nothink (</think>) = "about to answer directly"; think (<think>) = "about to reason"
    return "<｜begin▁of▁sentence｜><｜User｜>" + user +
           (think ? "<｜Assistant｜><think>" : "<｜Assistant｜></think>");
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
    if (!pf || !of) {
        fprintf(stderr, "set CAPTURE_PROMPTS and CAPTURE_OUT env vars\n");
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
    fprintf(stderr, "capture: %zu prompts, n_layer=%d n_embd=%d tensor=%s-*\n",
            prompts.size(), st.n_layer, st.n_embd, st.prefix.c_str());

    FILE * out  = std::fopen(of, "wb");
    if (!out) { fprintf(stderr, "cannot open %s\n", of); return 1; }
    FILE * outt = oft ? std::fopen(oft, "wb") : nullptr;
    if (oft && !outt) { fprintf(stderr, "cannot open %s\n", oft); return 1; }

    auto mem = llama_get_memory(ctx);
    // capture one mode's final-token activation into st.cur, write to f
    auto capture_one = [&](const std::string & prompt, bool think, FILE * f) -> bool {
        llama_memory_clear(mem, true);
        std::fill(st.cur.begin(), st.cur.end(), 0.0f);
        const std::string text = render_ds4(prompt, think);
        std::vector<llama_token> toks = common_tokenize(ctx, text, true, true);
        if (toks.empty()) return true; // skip, leave zeros
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), toks.size()))) return false;
        std::fwrite(st.cur.data(), sizeof(float), st.cur.size(), f);
        return true;
    };

    for (size_t i = 0; i < prompts.size(); ++i) {
        if (!capture_one(prompts[i], false, out)) { fprintf(stderr, "  [%zu] nothink decode failed\n", i); break; }
        if (outt && !capture_one(prompts[i], true, outt)) { fprintf(stderr, "  [%zu] think decode failed\n", i); break; }
        if ((i + 1) % 25 == 0) fprintf(stderr, "  %zu/%zu\n", i + 1, prompts.size());
    }
    std::fclose(out);
    if (outt) std::fclose(outt);
    fprintf(stderr, "wrote %s%s [%zu x %d x %d f32]\n", of, oft ? " + think" : "",
            prompts.size(), st.n_layer, st.n_embd);
    return 0;
}
