#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"adaptive_target",           sampling.adaptive_target},
            {"adaptive_decay",            sampling.adaptive_decay},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_parser_params.format)},
            {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
            {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
            {"generation_prompt",         chat_parser_params.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"adaptive_target",           sampling.adaptive_target},
        {"adaptive_decay",            sampling.adaptive_decay},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
        {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
        {"generation_prompt",         chat_parser_params.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(const common_chat_parser_params & chat_parser_params)
    : chat_parser_params(chat_parser_params)
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
    if (chat_parser_params.is_continuation && !chat_parser_params.echo) {
        // initialize chat_msg to avoid emitting a delta containing the assistant prefill
        chat_msg = common_chat_parse("", true, chat_parser_params);
    }
}

common_chat_msg task_result_state::update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    generated_text += text_added;
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    auto new_msg = common_chat_parse(
        generated_text,
        is_partial,
        chat_parser_params);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // the JSON library converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             stats.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (stats.is_set()) {
        deltas.back()["timings"] = stats.to_json();
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    if (stats.is_set()) {
        server_sent_events.back().at("data")["timings"] = stats.to_json();
    }

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !state.oai_resp_created && (is_progress || n_decoded == 1)) {
        state.oai_resp_created = true;
    }

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (stats.is_set()) {
            last_json["timings"] = stats.to_json();
        }
        if (is_progress) {
            last_json["prompt_progress"] = progress.to_json();
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    } else if (is_progress) {
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", json {
                        {"id",        "fc_" + diff.tool_call_delta.id},
                        {"arguments", ""},
                        {"call_id",   "call_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (stats.is_set()) {
            data["timings"] = stats.to_json();
        }
        if (is_progress) {
            data["prompt_progress"] = progress.to_json();
        }
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_slots::to_json() {
    return slots_data;
}

json server_task_result_metrics::to_json() {
    // not used, /metrics renders prometheus text via to_metrics()
    return json{};
}

// metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
std::string server_task_result_metrics::to_metrics() {
    const std::vector<metric_item> counters = {
        {
            "prompt_tokens_total",
            "Number of prompt tokens processed, excluding cached tokens",
            (double) metrics.prompt.count
        }, {
            "prompt_tokens_cached_total",
            "Number of prompt tokens reused from the cache",
            (double) metrics.n_prompt_cached
        }, {
            "prompt_seconds_total",
            "Total time spent processing prompts",
            metrics.prompt.time / 1.e6
        }, {
            "tokens_predicted_total",
            "Number of generation tokens processed",
            (double) metrics.predict.count
        }, {
            "tokens_predicted_seconds_total",
            "Total time spent generating tokens",
            metrics.predict.time / 1.e6
        }, {
            "n_decode_total",
            "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding",
            (double) metrics.n_decode
        }, {
            "n_tokens_max",
            "Largest observed sequence length (prompt + generation)",
            (double) metrics.n_tokens_max
        }, {
            "spec_decode_num_draft_tokens_total",
            "Speculative: Total draft tokens generated",
            (double) metrics.n_draft_tokens
        }, {
            "spec_decode_num_accepted_tokens_total",
            "Speculative: Total draft tokens accepted by the target model",
            (double) metrics.n_draft_accepted
        }, {
            "spec_decode_num_drafts_total",
            "Speculative: Total speculative decoding verification steps",
            (double) metrics.n_draft_verif_steps
        },
    };

    const std::vector<metric_item> gauges = {
        {
            "prompt_tokens_seconds",
            "Average prompt throughput in tokens/s",
            metrics.prompt_bucket.n_per_second()
        }, {
            "predicted_tokens_seconds",
            "Average generation throughput in tokens/s",
            metrics.predict_bucket.n_per_second()
        }, {
            "requests_processing",
            "Number of requests processing",
            (double) n_processing_slots
        }, {
            "requests_deferred",
            "Number of requests deferred",
            (double) n_tasks_deferred
        }, {
            "n_busy_slots_per_decode",
            "Average number of busy slots per llama_decode() call",
            (double) metrics.n_busy_slots / std::max((double) metrics.n_decode, 1.0)
        },
    };

    std::stringstream prometheus;

    auto add_items = [&prometheus](const char * type, const std::vector<metric_item> & items) {
        for (const auto & item : items) {
            prometheus << "# HELP llamacpp:" << item.name << " " << item.description << "\n"
                       << "# TYPE llamacpp:" << item.name << " " << type             << "\n"
                       << "llamacpp:"        << item.name << " " << item.value       << "\n";
        }
    };

    add_items("counter", counters);
    add_items("gauge",   gauges);

    // labeled counter: one time series per draft position
    if (!metrics.n_accepted_per_pos.empty()) {
        prometheus << "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
                      " Accepted tokens per draft position\n"
                   << "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n";
        for (size_t i = 0; i < metrics.n_accepted_per_pos.size(); i++) {
            prometheus << "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\""
                       << i << "\"} " << metrics.n_accepted_per_pos[i] << "\n";
        }
    }

    return prometheus.str();
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//
server_prompt_cache::~server_prompt_cache() {
    // spill the surviving RAM tier to the disk tier so it outlives the process. Spilling only
    // queues (pure moves of already-serialized bytes, no llama_context access) — the disk cache
    // dtor, which runs after this body as a member destructor, drains the write queue before
    // joining its writer thread, so every queued state lands on disk before shutdown completes.
    if (!disk) {
        return;
    }

    const size_t n_total = states.size();

    size_t n_spilled = 0;
    for (auto & state : states) {
        if (disk->spill(std::move(state))) {
            n_spilled++;
        }
    }
    states.clear();

    SRV_INF("prompt cache: shutdown spill queued %zu/%zu RAM-tier prompts to the disk cache\n", n_spilled, n_total);
}

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt.size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // skip over-limit entries to avoid disturbing the RAM cache — unless a disk tier can take
    // them. A large state (e.g. a 256k-ctx slot) never fits --cache-ram, but that is exactly the
    // prompt worth persisting, so allocate it transiently here and let update()/evict_front spill
    // it to disk after the caller fills the buffers.
    if (limit_size > 0 && state_size_new > limit_size) {
        const bool disk_eligible = disk &&
            prompt.tokens.size() >= disk->min_tokens_to_store() &&
            state_size_new <= disk->entry_cap();
        if (!disk_eligible) {
            SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                    state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
            return nullptr;
        }
        SRV_TRC(" - prompt state size %.3f MiB exceeds RAM limit; diverting to disk tier\n",
                state_size_new / (1024.0 * 1024.0));
    }

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->prompt.tokens.size()) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit.
        // don't evict to make room for an oversize (disk-bound) entry — it will be spilled
        // straight back out by update()/evict_front once the caller fills it, so evicting the
        // rest of the RAM cache for it would just churn.
        const bool oversize = state_size_new > limit_size;
        while (!oversize && !states.empty() && size() + state_size_new > limit_size) {
            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                    states.front().size() / (1024.0 * 1024.0));

            evict_front();
        }
    }

    std::vector<uint8_t> state_data_tgt;
    std::vector<uint8_t> state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.prompt =*/ {
            /*.tokens      =*/ prompt.tokens.clone(),
            /*.checkpoints =*/ prompt.checkpoints,
        },
        /*.data   =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
    });

    return &states.back();
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
        }
    }

    if (it_best != states.end()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

        {
            auto & data = it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);

                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = it_best->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\n", size);

                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }
        }

        prompt = std::move(it_best->prompt);

        states.erase(it_best);

        return true;
    }

    // no RAM winner — consult the L2 disk tier (streams the state back in on a hit).
    if (disk) {
        return disk->load(prompt, tokens_new, ctx_tgt, ctx_dft, id_slot);
    }

    return true;
}

void server_prompt_cache::evict_front() {
    if (states.empty()) {
        return;
    }

    // demote to the disk tier if enabled (spill takes ownership), otherwise drop.
    if (disk) {
        disk->spill(std::move(states.front()));
    }

    states.pop_front();
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (!states.empty() && size() > limit_size) {
            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            evict_front();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            evict_front();
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}

//
// server_prompt_disk_cache (L2 on-disk tier)
//

namespace {

// .dkv fixed header layout (v2 — all fields little-endian, written back-to-back, no padding):
//   off  0  u32 magic          "DKV2"
//   off  4  u64 compat_hash
//   off 12  u32 hits           \ rewritten in place on every hit (ds4_kvstore_touch_file
//   off 16  i64 last_hit_unix  / semantics) so the decay score survives restarts
//   off 24  u64 n_tokens, then tokens + blobs (see write_entry)
// v1 files lack the hits/last_hit fields; the magic gate skips them and a later spill of
// the same prefix overwrites the file in place (self-healing, no back-compat reader).
constexpr uint32_t DKV_MAGIC   = 0x32564B44; // "DKV2"
constexpr std::streamoff DKV_OFF_HITS = sizeof(uint32_t) + sizeof(uint64_t); // = 12, after magic + compat_hash
constexpr double   DKV_HALFLIFE_SECONDS = 6.0 * 3600.0; // 6h hit half-life (ds4_kvstore)
// cap the pending-spill queue: each entry is a full (main+drft) state blob,
// GiB-scale at long context. Unbounded, it would hold in RAM the very states
// eviction is trying to free (and stall shutdown draining them). Drop the
// oldest-queued spill when full — best-effort persistence, a miss just recomputes.
//
// Bound by BYTES as well as count. A count-only cap is meaningless here because
// entry size grows with the conversation: an agent session storing an ever-longer
// prefix reaches ~350 MiB/entry at 36k tokens and ~2.3 GiB at 384k, so 4 queued
// entries is 1.4 GiB early and 9 GiB deep. On a UMA box whose post-load headroom
// is single-digit GiB that walks straight into the OOM killer — it did, five times
// on the GB10, each time as a clean SIGTERM from earlyoom mid-agent-session.
constexpr size_t   DKV_MAX_WRITE_Q       = 4;
constexpr size_t   DKV_MAX_WRITE_Q_BYTES = 1024ull * 1024 * 1024; // 1 GiB of pending spills

uint64_t dkv_fnv1a(const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// common-prefix length of two raw token vectors
size_t dkv_common_prefix(const llama_tokens & a, const llama_tokens & b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        i++;
    }
    return i;
}

template <typename T> void dkv_w(std::ostream & os, const T & v) {
    os.write(reinterpret_cast<const char *>(&v), sizeof(T));
}
template <typename T> bool dkv_r(std::istream & is, T & v) {
    return (bool) is.read(reinterpret_cast<char *>(&v), sizeof(T));
}
void dkv_w_blob(std::ostream & os, const std::vector<uint8_t> & b) {
    uint64_t n = b.size();
    dkv_w(os, n);
    if (n) {
        os.write(reinterpret_cast<const char *>(b.data()), n);
    }
}
// readable bytes remaining in a seekable stream from the current position
// (-1 if the stream can't report it). Used to reject corrupt/truncated length
// prefixes before they drive a huge resize() -> bad_alloc.
int64_t dkv_stream_remaining(std::istream & is) {
    const std::streampos cur = is.tellg();
    if (cur < 0) { return -1; }
    is.seekg(0, std::ios::end);
    const std::streampos end = is.tellg();
    is.seekg(cur, std::ios::beg);
    if (end < 0 || !is) { return -1; }
    return (int64_t) (end - cur);
}
bool dkv_r_blob(std::istream & is, std::vector<uint8_t> & b) {
    uint64_t n = 0;
    if (!dkv_r(is, n)) {
        return false;
    }
    const int64_t rem = dkv_stream_remaining(is);
    if (rem < 0 || n > (uint64_t) rem) {
        return false; // corrupt/truncated length — refuse to allocate
    }
    b.resize(n);
    if (n) {
        return (bool) is.read(reinterpret_cast<char *>(b.data()), n);
    }
    return true;
}

// rewrite just the hits/last_hit_unix header fields of an existing entry in place —
// a full-file rewrite would be GiB-scale I/O for a counter bump. Best-effort: a lost
// update (entry replaced/removed concurrently) only costs decay-score accuracy.
void dkv_touch_file(const std::string & path, uint32_t hits, int64_t last_hit_unix) {
    std::fstream fs(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!fs) {
        return;
    }
    fs.seekp(DKV_OFF_HITS);
    dkv_w(fs, hits);
    dkv_w(fs, last_hit_unix);
}

} // namespace

server_prompt_disk_cache::server_prompt_disk_cache(std::string dir, std::string compat_desc,
                                                   size_t limit_bytes, size_t min_tokens, size_t max_entry_bytes)
    : dir(std::move(dir)), compat_desc(std::move(compat_desc)),
      limit_bytes(limit_bytes), min_tokens(min_tokens), max_entry_bytes(max_entry_bytes) {
    compat_hash = dkv_fnv1a(this->compat_desc.data(), this->compat_desc.size());

    std::error_code ec;
    std::filesystem::create_directories(this->dir, ec);
    if (ec) {
        SRV_ERR("disk prompt cache: cannot create dir '%s': %s\n", this->dir.c_str(), ec.message().c_str());
    }

    scan_existing();

    writer = std::thread([this]() { writer_loop(); });

    SRV_INF("disk prompt cache enabled: dir='%s', budget=%zu MiB, %zu existing entries (%.3f MiB)\n",
            this->dir.c_str(), limit_bytes / (1024 * 1024), index.size(), disk_size() / (1024.0 * 1024.0));
}

server_prompt_disk_cache::~server_prompt_disk_cache() {
    stop.store(true);
    cv.notify_all();
    if (writer.joinable()) {
        writer.join();
    }
}

std::string server_prompt_disk_cache::path_for(const llama_tokens & toks) const {
    uint64_t h = compat_hash;
    h ^= dkv_fnv1a(toks.data(), toks.size() * sizeof(llama_token));
    char name[32];
    snprintf(name, sizeof(name), "%016llx.dkv", (unsigned long long) h);
    return (std::filesystem::path(dir) / name).string();
}

size_t server_prompt_disk_cache::disk_size() const {
    size_t res = 0;
    for (const auto & e : index) {
        res += e.file_size;
    }
    return res;
}

double server_prompt_disk_cache::score(const entry & e, int64_t now_unix) const {
    const double age  = std::max<double>(0.0, double(now_unix - e.last_hit_unix));
    const double decay = std::pow(0.5, age / DKV_HALFLIFE_SECONDS);
    const double eff_hits = e.hits * decay;
    return (eff_hits + 1.0) * double(e.tokens.size()) / std::max<size_t>(1, e.file_size);
}

bool server_prompt_disk_cache::spill(server_prompt_cache_state && state) {
    // only text-token prompts can be keyed/replayed by token prefix
    if (state.prompt.tokens.has_mtmd || state.prompt.tokens.size() < min_tokens || state.data.size() == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lk(mtx);

    // dedup: skip if an identical (or longer) prefix is already persisted
    const llama_tokens & toks = state.prompt.tokens.get_tokens();
    for (const auto & e : index) {
        if (dkv_common_prefix(e.tokens, toks) == toks.size()) {
            return false;
        }
    }

    // a single entry larger than the whole byte budget would evict the queue and still
    // not fit; skip it outright rather than dropping useful pending writes for nothing
    const size_t bytes_new = state.size();
    if (bytes_new > DKV_MAX_WRITE_Q_BYTES) {
        SRV_WRN("disk prompt cache: %zu-token spill (%.3f MiB) exceeds the pending-write budget (%zu MiB), skipping\n",
                toks.size(), bytes_new / (1024.0 * 1024.0), DKV_MAX_WRITE_Q_BYTES / (1024 * 1024));
        return false;
    }

    size_t bytes_q = 0;
    for (const auto & e : write_q) {
        bytes_q += e.size();
    }

    while (!write_q.empty() &&
           (write_q.size() >= DKV_MAX_WRITE_Q || bytes_q + bytes_new > DKV_MAX_WRITE_Q_BYTES)) {
        SRV_WRN("disk prompt cache: spill queue full (%zu entries, %.3f MiB), dropping oldest pending write\n",
                write_q.size(), bytes_q / (1024.0 * 1024.0));
        bytes_q -= write_q.front().size();
        write_q.pop_front();
    }
    write_q.push_back(std::move(state));
    cv.notify_one();

    return true;
}

void server_prompt_disk_cache::writer_loop() {
    for (;;) {
        server_prompt_cache_state p;
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [this]() { return stop.load() || !write_q.empty(); });
            if (write_q.empty()) {
                if (stop.load()) {
                    return;
                }
                continue;
            }
            p = std::move(write_q.front());
            write_q.pop_front();
        }

        // file I/O without the lock (may be multi-GiB)
        write_entry(p);
    }
}

bool server_prompt_disk_cache::write_entry(const server_prompt_cache_state & p) {
    const llama_tokens & toks = p.prompt.tokens.get_tokens();
    const std::string path = path_for(toks);
    const std::string tmp  = path + ".tmp";

    // the serialization below is deterministic, so the entry's on-disk size is known exactly
    // up front: reserve it by evicting BEFORE the write lands — evicting after would
    // transiently overshoot the budget by up to a full (GiB-scale) entry
    size_t bytes_new = 2*sizeof(uint32_t) + 2*sizeof(uint64_t)              // magic, compat_hash, hits, last_hit_unix
                     + sizeof(uint64_t) + toks.size()*sizeof(llama_token)   // n_toks + tokens
                     + sizeof(uint64_t) + p.data.main.size()                // main blob
                     + sizeof(uint64_t) + p.data.drft.size()                // drft blob
                     + sizeof(uint32_t);                                    // n_ckpt
    for (const auto & c : p.prompt.checkpoints) {
        bytes_new += sizeof(int64_t) + 2*sizeof(int32_t)
                   + sizeof(uint64_t) + c.data_tgt.size()
                   + sizeof(uint64_t) + c.data_dft.size()
                   + sizeof(uint64_t) + c.data_spec.size();
    }

    {
        std::lock_guard<std::mutex> lk(mtx);
        if (limit_bytes > 0 && bytes_new > limit_bytes) {
            SRV_WRN("disk prompt cache: %zu-token entry (%.3f MiB) exceeds the disk budget (%zu MiB), skipping\n",
                    toks.size(), bytes_new / (1024.0 * 1024.0), limit_bytes / (1024 * 1024));
            return false;
        }
        evict_locked(bytes_new);
    }

    const int64_t now = (int64_t) std::time(nullptr);

    {
        std::ofstream os(tmp, std::ios::binary | std::ios::trunc);
        if (!os) {
            SRV_ERR("disk prompt cache: cannot open '%s' for write\n", tmp.c_str());
            return false;
        }

        dkv_w(os, DKV_MAGIC);
        dkv_w(os, compat_hash);
        dkv_w(os, (uint32_t) 0); // hits — rewritten in place on each hit (DKV_OFF_HITS)
        dkv_w(os, now);          // last_hit_unix

        uint64_t n_toks = toks.size();
        dkv_w(os, n_toks);
        os.write(reinterpret_cast<const char *>(toks.data()), n_toks * sizeof(llama_token));

        dkv_w_blob(os, p.data.main);
        dkv_w_blob(os, p.data.drft);

        uint32_t n_ckpt = (uint32_t) p.prompt.checkpoints.size();
        dkv_w(os, n_ckpt);
        for (const auto & c : p.prompt.checkpoints) {
            dkv_w(os, (int64_t) c.n_tokens);
            dkv_w(os, (int32_t) c.pos_min);
            dkv_w(os, (int32_t) c.pos_max);
            dkv_w_blob(os, c.data_tgt);
            dkv_w_blob(os, c.data_dft);
            dkv_w_blob(os, c.data_spec);
        }

        if (!os) {
            SRV_ERR("disk prompt cache: write failed for '%s'\n", tmp.c_str());
            os.close();
            std::filesystem::remove(tmp);
            return false;
        }
    }

    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        SRV_ERR("disk prompt cache: rename '%s' failed: %s\n", tmp.c_str(), ec.message().c_str());
        std::filesystem::remove(tmp, ec);
        return false;
    }

    const size_t fsize = (size_t) std::filesystem::file_size(path, ec);

    {
        std::lock_guard<std::mutex> lk(mtx);
        // replace any existing index entry for this path
        index.erase(std::remove_if(index.begin(), index.end(),
                        [&](const entry & e) { return e.path == path; }),
                    index.end());
        entry e;
        e.path          = path;
        e.tokens        = toks;
        e.file_size     = fsize;
        e.hits          = 0;
        e.last_hit_unix = now;
        index.push_back(std::move(e));

        // no post-write evict: the pre-write reserve above is exact, and only load() can
        // touch the index between the reserve and here (it only shrinks it)
    }

    SRV_TRC("disk prompt cache: wrote %zu-token entry (%.3f MiB) -> %s\n",
            toks.size(), fsize / (1024.0 * 1024.0), path.c_str());
    return true;
}

void server_prompt_disk_cache::evict_locked(size_t bytes_incoming) {
    if (limit_bytes == 0) {
        return;
    }
    const int64_t now = (int64_t) std::time(nullptr);
    while (disk_size() + bytes_incoming > limit_bytes && !index.empty()) {
        // drop the lowest-scoring entry
        auto worst = index.begin();
        double worst_score = score(*worst, now);
        for (auto it = index.begin() + 1; it != index.end(); ++it) {
            const double s = score(*it, now);
            if (s < worst_score) {
                worst_score = s;
                worst = it;
            }
        }
        std::error_code ec;
        std::filesystem::remove(worst->path, ec);
        SRV_TRC("disk prompt cache: evicted %zu-token entry (%.3f MiB, score %.4g)\n",
                worst->tokens.size(), worst->file_size / (1024.0 * 1024.0), worst_score);
        index.erase(worst);
    }
}

bool server_prompt_disk_cache::load(server_prompt & prompt, const server_tokens & tokens_new,
                                    llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    if (tokens_new.has_mtmd) {
        return true; // nothing to restore for multimodal prompts
    }

    std::string path;
    uint32_t hits_new     = 0;
    int64_t  last_hit_new = 0;
    {
        std::lock_guard<std::mutex> lk(mtx);
        if (index.empty()) {
            return true;
        }

        const llama_tokens & toks_new = tokens_new.get_tokens();

        // pick the entry that preserves the most of its own context while matching the new prompt,
        // mirroring the RAM tier's f_keep / sim heuristic.
        float f_keep_best = -1.0f;
        float sim_best    = 0.0f;
        auto  it_best     = index.end();
        for (auto it = index.begin(); it != index.end(); ++it) {
            const size_t lcp = dkv_common_prefix(it->tokens, toks_new);
            if (lcp == 0) {
                continue;
            }
            const float f_keep = float(lcp) / it->tokens.size();
            const float sim    = float(lcp) / std::max<size_t>(1, toks_new.size());
            if (f_keep < 0.25f) {
                continue; // don't restore a large prompt for a tiny overlap
            }
            if (f_keep_best < f_keep && sim_best < sim) {
                f_keep_best = f_keep;
                sim_best    = sim;
                it_best     = it;
            }
        }

        if (it_best == index.end()) {
            return true; // miss: leave prompt untouched
        }

        path = it_best->path;
        it_best->hits         += 1;
        it_best->last_hit_unix = (int64_t) std::time(nullptr);

        hits_new     = it_best->hits;
        last_hit_new = it_best->last_hit_unix;
    }

    // persist the hit bump into the file header so the decay score survives restarts
    dkv_touch_file(path, hits_new, last_hit_new);

    // read + restore without the lock
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        SRV_WRN("disk prompt cache: entry vanished '%s'\n", path.c_str());
        std::lock_guard<std::mutex> lk(mtx);
        index.erase(std::remove_if(index.begin(), index.end(),
                        [&](const entry & e) { return e.path == path; }), index.end());
        return true;
    }

    uint32_t magic = 0;
    uint64_t chash = 0;
    if (!dkv_r(is, magic) || magic != DKV_MAGIC || !dkv_r(is, chash) || chash != compat_hash) {
        SRV_WRN("disk prompt cache: incompatible entry '%s', discarding\n", path.c_str());
        std::lock_guard<std::mutex> lk(mtx);
        index.erase(std::remove_if(index.begin(), index.end(),
                        [&](const entry & e) { return e.path == path; }), index.end());
        std::error_code ec; std::filesystem::remove(path, ec);
        return true;
    }

    uint32_t f_hits     = 0; // the in-RAM index is authoritative while running —
    int64_t  f_last_hit = 0; // these are only consumed by scan_existing() at startup
    if (!dkv_r(is, f_hits) || !dkv_r(is, f_last_hit)) {
        return true;
    }

    uint64_t n_toks = 0;
    if (!dkv_r(is, n_toks)) {
        return true;
    }
    {   // reject a corrupt token count before it drives a huge allocation
        const int64_t rem = dkv_stream_remaining(is);
        if (rem < 0 || n_toks > (uint64_t) rem / sizeof(llama_token)) {
            SRV_WRN("disk prompt cache: corrupt token count in '%s'\n", path.c_str());
            return true;
        }
    }
    llama_tokens toks(n_toks);
    if (n_toks && !is.read(reinterpret_cast<char *>(toks.data()), n_toks * sizeof(llama_token))) {
        return true;
    }

    server_prompt_cache_state loaded;
    loaded.prompt.tokens = server_tokens(toks, /*has_mtmd=*/false);
    if (!dkv_r_blob(is, loaded.data.main) || !dkv_r_blob(is, loaded.data.drft)) {
        SRV_WRN("disk prompt cache: truncated entry '%s'\n", path.c_str());
        return true;
    }

    uint32_t n_ckpt = 0;
    dkv_r(is, n_ckpt);
    for (uint32_t i = 0; i < n_ckpt; i++) {
        common_prompt_checkpoint c;
        int64_t nt = 0; int32_t pmin = 0, pmax = 0;
        dkv_r(is, nt); dkv_r(is, pmin); dkv_r(is, pmax);
        c.n_tokens = nt; c.pos_min = pmin; c.pos_max = pmax;
        if (!dkv_r_blob(is, c.data_tgt) || !dkv_r_blob(is, c.data_dft) || !dkv_r_blob(is, c.data_spec)) {
            SRV_WRN("disk prompt cache: truncated checkpoint in '%s'\n", path.c_str());
            return true;
        }
        loaded.prompt.checkpoints.push_back(std::move(c));
    }

    // restore into the live contexts (whole-sequence, position offset 0)
    {
        const size_t sz = loaded.data.main.size();
        const size_t n  = llama_state_seq_set_data_ext(ctx_tgt, loaded.data.main.data(), sz, id_slot, 0);
        if (n != sz) {
            SRV_ERR("disk prompt cache: failed to restore target state from '%s'\n", path.c_str());
            std::lock_guard<std::mutex> lk(mtx);
            index.erase(std::remove_if(index.begin(), index.end(),
                            [&](const entry & e) { return e.path == path; }), index.end());
            std::error_code ec; std::filesystem::remove(path, ec);
            return false; // caller will clear the slot
        }
    }
    if (ctx_dft && !loaded.data.drft.empty()) {
        const size_t sz = loaded.data.drft.size();
        const size_t n  = llama_state_seq_set_data_ext(ctx_dft, loaded.data.drft.data(), sz, id_slot, 0);
        if (n != sz) {
            SRV_WRN("disk prompt cache: failed to restore draft state from '%s'\n", path.c_str());
            return false;
        }
    }

    SRV_INF("disk prompt cache: restored %zu-token prefix from disk (f_keep %.3f)\n",
            toks.size(), (double) dkv_common_prefix(toks, tokens_new.get_tokens()) / std::max<size_t>(1, toks.size()));

    prompt = std::move(loaded.prompt);
    return true;
}

// file size that never throws and never trips the caller's error_code — a size we cannot
// read just means this file goes unaccounted, which must not abort the scan
static size_t dkv_file_size(const std::filesystem::directory_entry & de) {
    std::error_code ec;
    const auto sz = std::filesystem::file_size(de.path(), ec);
    return ec ? 0 : (size_t) sz;
}

void server_prompt_disk_cache::scan_existing() {
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) {
        return;
    }

    size_t n_incompat = 0, n_corrupt = 0, n_foreign = 0, n_unreadable = 0;
    // bytes held by files we skipped. They are deliberately NOT deleted (a later run with
    // the old -c / kv type / model matches them again), but they are also not in the index,
    // so eviction cannot see them and they do not count against limit_bytes. Left unreported
    // that is a silent overshoot: changing -c once orphans the entire previous cache on disk
    // while the budget still believes it has the whole allowance free.
    size_t bytes_skipped = 0;
    for (const auto & de : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) { break; }
        if (!de.is_regular_file()) {
            continue;
        }
        // sweep orphaned .tmp files (a write killed mid-rename) so they don't
        // accumulate forever — they're never indexed and never evicted otherwise
        if (de.path().extension() == ".tmp") {
            std::error_code rec; std::filesystem::remove(de.path(), rec);
            continue;
        }
        if (de.path().extension() != ".dkv") {
            continue;
        }
        std::ifstream is(de.path(), std::ios::binary);
        if (!is) { ++n_unreadable; bytes_skipped += dkv_file_size(de); continue; }

        uint32_t magic    = 0;
        uint64_t chash    = 0;
        uint32_t hits     = 0;
        int64_t  last_hit = 0;
        uint64_t n_toks   = 0;
        // split from one combined test so the reason a file was skipped is
        // reportable: an incompatible entry (changed -c, kv type, model) is
        // expected and recoverable, a corrupt one is not, and reporting "0
        // existing entries" for both hides a whole cache silently vanishing
        if (!dkv_r(is, magic) || magic != DKV_MAGIC) {
            ++n_foreign;  bytes_skipped += dkv_file_size(de); continue;
        }
        if (!dkv_r(is, chash)) {
            ++n_corrupt;  bytes_skipped += dkv_file_size(de); continue;
        }
        if (chash != compat_hash) {
            ++n_incompat; bytes_skipped += dkv_file_size(de); continue; // leave the file; a later run may match again
        }
        if (!dkv_r(is, hits) || !dkv_r(is, last_hit) || !dkv_r(is, n_toks)) {
            ++n_corrupt;  bytes_skipped += dkv_file_size(de); continue;
        }
        {   // reject a corrupt token count before it drives a huge allocation
            const int64_t rem = dkv_stream_remaining(is);
            if (rem < 0 || n_toks > (uint64_t) rem / sizeof(llama_token)) {
                ++n_corrupt; bytes_skipped += dkv_file_size(de); continue;
            }
        }
        llama_tokens toks(n_toks);
        if (n_toks && !is.read(reinterpret_cast<char *>(toks.data()), n_toks * sizeof(llama_token))) {
            ++n_corrupt; bytes_skipped += dkv_file_size(de); continue;
        }

        entry e;
        e.path          = de.path().string();
        e.tokens        = std::move(toks);
        e.file_size     = (size_t) std::filesystem::file_size(de.path(), ec);
        // restore the persisted hit counters — resetting them here would make the
        // decay-based eviction score meaningless across restarts. A bogus/future
        // last_hit is harmless: score() clamps the age at 0.
        e.hits          = hits;
        e.last_hit_unix = last_hit;
        index.push_back(std::move(e));
    }

    if (n_incompat || n_corrupt || n_foreign || n_unreadable) {
        // an incompatible sweep is the normal consequence of changing -c, the KV
        // types or the model, and it means the next sessions start cold. Say so
        // rather than silently reporting an empty cache.
        SRV_WRN("disk prompt cache: skipped %zu file(s) holding %.3f MiB - %zu incompatible "
                "(context/kv/model changed), %zu corrupt, %zu foreign, %zu unreadable\n",
                n_incompat + n_corrupt + n_foreign + n_unreadable, bytes_skipped / (1024.0 * 1024.0),
                n_incompat, n_corrupt, n_foreign, n_unreadable);

        // skipped files are kept on purpose (restoring the old -c makes them valid again) but
        // they are outside the index, so eviction cannot reclaim them and the budget does not
        // know they exist. Only say so when they actually push real usage past the allowance —
        // that is the point at which "budget 64 GiB" stops describing what is on the disk.
        if (limit_bytes > 0 && bytes_skipped + disk_size() > limit_bytes) {
            SRV_WRN("disk prompt cache: %.3f MiB of skipped files sits outside the %zu MiB budget and "
                    "cannot be evicted; delete them to reclaim the space, or restore the previous "
                    "settings to make them usable again\n",
                    bytes_skipped / (1024.0 * 1024.0), limit_bytes / (1024 * 1024));
        }
    }

    std::lock_guard<std::mutex> lk(mtx);
    evict_locked();
}
