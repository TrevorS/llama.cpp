# GB10 / DGX Spark: thermal wedges, and what to actually do about them

Written 2026-08-02 after the 13th hard lockup on this box. Everything numeric here is
measured locally or cited from a source at the bottom. The telemetry is 53,869
`spark-monitor` samples across two sessions — **both of which end in a wedge** (§3.0),
so this is run-up data, not survivor data. Where we were wrong earlier, that is called
out; the wrong turns cost days and are worth not repeating.

Scripts alongside this file: `soak.sh` (sustained-load soak with watchdog),
`cap-sweep.sh` (find the highest safe clock cap), `sched-ab.sh` (CUDA sync-policy A/B),
`gov-probe.sh` (CPU governor cost).

---

## 1. Recognising the failure

The box stops. Completely.

- no ping, no SSH, no console
- **nothing in the journal** — it simply ends mid-line on an ordinary message
- no OOM kill, no `Xid`, no `NVRM` error, no thermal message, no shutdown sequence
- processes keep writing to disk for **~60 s after journald goes quiet** (check file
  mtimes against the last journal timestamp — that gap is the tell)
- only a hard power cycle recovers it

Time to failure under sustained load: **4–60 minutes** in the field; ours have ranged
from 38 minutes to 11 hours.

This is not specific to llama.cpp. The same signature is reported under vLLM.

**It is not an OOM.** Confirm by checking that `earlyoom` never targeted the inference
process — on our wedge it only SIGTERMed three desktop helpers of 6, 5 and 0 MiB while
`MemAvailable` sat flat at ~2.88 GiB.

---

## 2. First question: is the hardware faulty?

Do this **before** building any software mitigation. Multiple users with this exact
symptom have failed field diagnostics and been RMA'd by NVIDIA:

> `082-000-1-020000021139` — *"Acceptable temperature limits exceeded or the thermal
> sensor is broken or miscalibrated."*

One reporter's failure persisted through a complete factory OS reinstall, and NVIDIA
approved the RMA on the diagnostic result alone. Others hit `PowerStress FAIL`
(`MODS-020000600139`) with the same outcome.

Install it:

```bash
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-archive-keyring.gpg | sudo tee /usr/share/keyrings/cuda-archive-keyring.gpg > /dev/null
```

```bash
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa /" | sudo tee /etc/apt/sources.list.d/cuda-sbsa-ubuntu2404.list
```

```bash
sudo apt-get update && sudo apt-get install -y dgx-spark-fieldiag
```

Running it needs **Secure Boot disabled** and a text-mode console. It stops the display
manager, so run it from a physical console or expect the SSH session to survive `init 3`:

```bash
sudo init 3
```

```bash
cd /opt/nvidia/dgx-spark-fieldiag && sudo ./partnerdiag --field
```

If the thermal sensor or PowerStress test fails, stop here and open a support case.
Everything in section 4 is symptom management for a box that may simply be broken.

---

## 3. The thermal model, measured

### 3.0 What two actual wedges look like — the ground truth

`experiments/profiles/thermal/*.csv` are not survivor traces. **Both end in NUL padding**
— the power-loss write signature — and their last timestamps match recorded wedges
(07-28 10:12:53 and 07-28 18:20:23). They are run-up data for two failures.

| | T-60→30m | T-30→10m | T-10→2m | final 2 min |
| --- | --- | --- | --- | --- |
| **wedge 10:12** SoC | 53.5 | 60.6 | 85.2 | **88.4 mean / 98.3 max** |
| power | 10.4 W | 20.8 W | 56.9 W | 63.2 W |
| **wedge 18:20** SoC | 54.3 | 60.4 | 93.3 | **94.0 mean / 96.7 max** |
| power | 11.4 W | 19.9 W | 64.9 W | 63.9 W |

Both die identically: idle → ramp → **~10 minutes sustained in the mid-90s** → gone. The
final ten samples of each read 94–98 °C. Neither shows a spike, a runaway, or any
distinguishing transient — they sit in the band and stop.

**The lethal condition is sustained residence at ~94–98 °C, not a peak.** Momentary
excursions past 90 are survivable; minutes in the mid-90s are not.

Both wedges occurred at only **56–65 W mean** against a 140 W TDP. That is the
quantitative death of watts as a tripwire, and it matches the field reports of fan
curves failing to ramp.

*Parser note:* anything reading these files must skip NUL rows or it throws partway
through. That the data survives at all is the point — a 1 Hz CSV keeps writing after
journald goes silent, which is why `spark-monitor --log` is non-optional (§5).

### 3.1 The SoC is a shared budget — this is the key fact

CPU and GPU sit on one package under one cooling solution. Load on **either** heats the
same ACPI zones. From our telemetry:

| load | SoC mean | SoC p95 |
| --- | --- | --- |
| gpu 0-20%, cpu 0-20% | 51.4 °C | 63.5 °C |
| gpu 0-20%, **cpu 20-60%** | **78.5 °C** | **94.6 °C** |
| gpu 20-80%, cpu 0-20% | 86.5 °C | 95.4 °C |
| gpu 80-100%, cpu 0-20% | 88.7 °C | 96.5 °C |

**CPU at 20–60% is thermally comparable to the GPU at 80–100%.** A compile, a container,
a busy Python process — each spends the same budget the GPU needs. This is the real
reason for "never build during model residency"; the memory argument is secondary.

### 3.2 GPU die temperature does not represent the SoC

Across 3,668 busy samples:

- `SoC − GPU die`: mean **+9.8 °C**, p95 +14.8, max +19.4
- In the 2,131 samples with **SoC ≥ 90 °C**, the GPU die's *maximum* was **89.0 °C**

Anything keyed on GPU die temperature is looking at the wrong component.

### 3.3 Watts are not a proxy either

| gpu power | SoC mean | SoC max |
| --- | --- | --- |
| 0–60 W | 80.2 °C | **94.0 °C** |
| 60–70 W | 91.3 °C | 97.0 °C |
| 70–80 W | 93.2 °C | 98.3 °C |

You can be at low power and still be at 94 °C. A watt threshold will not protect you.

### 3.4 The performance cliff and the danger zone are the same place

| SoC | sm_mhz mean | sm_mhz p10 |
| --- | --- | --- |
| ≤ 70 °C | 2511 | 2489 |
| 85–88 °C | 2479 | 2457 |
| 88–90 °C | 2465 | 2437 |
| 90–92 °C | 2450 | 2418 |
| **92–95 °C** | **2390** | **2281** |
| 95 °C+ | 2373 | 2262 |

Everything up to 92 °C costs **2.4 % of clocks**. Past 92 °C it falls off a cliff — and
that is the same band where the box wedges. **Holding the SoC under ~90 °C gives up
almost nothing.** Any argument that thermal headroom costs performance has this backwards.

### 3.5 Heating is ~20× faster than cooling

- rising: median 0.2 °C/s, p95 **2.85 °C/s**
- idle cooling: median **0.1 °C/s** → ~100 s to shed 10 °C

A 4 °C guard band can be crossed in about a second. Reactive control must back off hard
and recover slowly. Temporal filters (median-5/9, EMA) do **not** clean the signal —
the excursions are real CPU-driven heating, not sensor glitches.

### 3.6 Platform constraints

- 140 W TDP in a 1.13 L chassis, firmware-controlled cooling
- **no fan control** — no `pwm*` nodes; `nvidia-smi` reports fan `N/A`
- **no power limit** — `nvidia-smi -q -d POWER` returns `N/A` throughout
- ACPI zones: 7 × `acpitz`, critical trip at **105 °C** — far above where we die, so the
  kernel's own protection will never save you
- The same silicon in a chassis with a better vapour chamber (MSI EdgeXpert) runs 10–15 °C
  cooler and reportedly never throttles. **The chassis is the bottleneck, not the GPU.**

---

## 4. Levers, in the order worth applying

### 4.1 Cap the GPU clock — the single best lever

`-d POWER` being `N/A` does **not** mean there is no clock control. Locked graphics
clocks work:

```bash
sudo nvidia-smi -lgc 300,2200
```

Reported effect in the field: a node went from *78–88 °C with periodic thermal shutdowns*
to *65–69 °C at sustained 96 % utilisation with zero shutdowns*, at negligible throughput
cost — inference here is memory-bandwidth bound, not clock bound. Our own clock table
(§3.4) agrees: the whole usable range is only ~5 % wide.

**Measured locally 2026-08-02** — same binary, model, prompt and DSpark config; the only
difference is the power strategy:

| | P85 duty cycling, no cap | **no duty cycling, clocks capped 2200** |
| --- | --- | --- |
| tg | 23.11 / 23.48 t/s | **27.25 / 27.67 t/s** (**+18 %**) |
| acceptance | 178/280 = 0.6357 | 178/280 = 0.6357 (identical) |
| SM clock under load | 2373–2511 | 2190, pinned |
| SoC max | — | **67.7 °C** |
| power max | — | 33.0 W |

The clock cap is **faster and cooler at the same time**, and leaves numerics untouched
(identical accepted/generated counts). The reason is mechanical: duty cycling throttles by
inserting host sleeps, discarding wall-clock; a clock cap lowers the V/f operating point,
so the GPU stays continuously busy at lower power per unit work. Voltage scaling is
superlinear, so a ~10 % clock reduction buys a disproportionate thermal saving.

**`GGML_CUDA_POWER` is therefore superseded for steady-state serving.** Caveat: this was a
short two-leg test. It establishes throughput and short-run thermals, **not** wedge
immunity — the wedges appear after sustained load, so a soak is still owed before claiming
the problem is solved.

Persist it across reboots (persistence mode is already `Enabled` on this box):

```bash
sudo python3 -c 'open("/etc/systemd/system/gb10-clock-cap.service","w").write("[Unit]\nDescription=Cap GB10 GPU clocks to keep the SoC out of the wedge band\nAfter=nvidia-persistenced.service\nWants=nvidia-persistenced.service\n\n[Service]\nType=oneshot\nRemainAfterExit=yes\nExecStart=/usr/bin/nvidia-smi -lgc 300,2200\nExecStop=/usr/bin/nvidia-smi -rgc\n\n[Install]\nWantedBy=multi-user.target\n")'
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now gb10-clock-cap.service
```

Undo at any time:

```bash
sudo nvidia-smi -rgc
```

Tune the cap against your own workload — 2200 is the published figure, and our observed
operating range is 2373–2511, so 2200 is a real but small reduction.

### 4.2 Stop burning a CPU core on nothing

The CPU frequency governor is `performance` on all 20 policies, pinned at 2808 MHz. Every
CPU task runs flat out, maximising heat per unit work — while llama.cpp uses only
**5.2 % mean / 7.8 % p95** of the CPU during GPU-heavy work.

```bash
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo schedutil | sudo tee "$p/scaling_governor" > /dev/null; done
```

Verify:

```bash
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
```

**Measured 2026-08-02** (`gov-probe.sh`), and it is smaller than it looks:

| | `performance` | `schedutil` |
| --- | --- | --- |
| idle SoC | 46.8 °C | **46.8 °C — identical** |
| idle clocks | 3900 / 2808 MHz | 1911 / 552 MHz |
| load SoC max | 72.1 °C | **67.9 °C** |
| fixed CPU work | 10.4 s | 10.7 s (+2.9 %) |

**There is no idle win.** An idle ARM core clock-gates regardless of its *permitted*
frequency; `scaling_cur_freq` reports what it may run at, not switching activity. The
pinning was cosmetic at idle, and the "20 cores burning heat doing nothing" framing was
wrong.

The win is **4.2 °C under CPU load for 2.9 % CPU throughput** — a good trade here, because
we do not care about CPU throughput and do care about the shared budget. It targets the
co-consumer case (containers, builds, agents alongside the model), not inference itself,
where llama.cpp uses ~5 % CPU and there is little to throttle.

It does **not** help the busy-wait core (§4.4): a spin loop presents as 100 % utilisation,
so schedutil clocks it to maximum exactly as `performance` did.

Not persistent across reboots — needs a systemd unit or `cpufrequtils` if wanted
permanently.

### 4.3 Constrain concurrent CPU work

Given §3.1, treat CPU work during model residency as spending GPU thermal budget.

- Builds: `-j1` or `-j2`, never `-j4`+ with a model resident. A single-TU incremental
  rebuild measured 18.7 s and 821 MiB, which is safe; a full parallel build is not.
- Containers: budget CPU explicitly, e.g. `--cpus=1`, not `--cpus=4`.
- Anything long-running and CPU-hungry: run it when the model is unloaded.

### 4.4 The busy-wait: known, deliberate, and aimed at this chip

`ggml/src/ggml-cuda/ggml-cuda.cu` sets spin-wait **only** for compute capability 12.1 —
which is exactly GB10:

```c
// Temporary performance fix:
// Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays
// in cuda synchronize calls.
// TODO: ... remove this call again when cudaDeviceScheduleSpin is default.
if (prop.major == 12 && prop.minor == 1) {
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
}
```

That is one host thread busy-polling inside `cudaStreamSynchronize` — **not** work that
could be spread across cores. Spreading it would multiply the heat, not divide it. It
accounts for the 5.2 % of 20 cores ≈ **1.04 cores** we measure: a core at 2.8 GHz
producing pure heat and zero throughput, on a shared thermal budget.

Upstream's general fix is `cudaDeviceScheduleBlockingSync`, measured at 30.2 vs 30.1 tps
on a *discrete* GPU. **That number does not transfer.** Measured here via
`GGML_CUDA_SCHED` (added to `ggml-cuda.cu`, default unchanged) with `sched-ab.sh`:

| policy | tg | cores in decode | SoC rise | TTFT |
| --- | --- | --- | --- | --- |
| **spin** (upstream default) | **27.78** | 1.00 | +16.1 °C | 463 ms |
| yield | 27.52 | 1.00 | +16.3 °C | 476 ms |
| blocking | 18.49 | 0.16 | +8.9 °C | 542 ms |

**`blocking` costs 33 % throughput.** Decode is latency-bound — every token is a ~36 ms
round trip and futex wake latency lands on each one. Upstream's cc121 special-case is
well-founded; **do not "fix" the spin.** As a thermal lever it is ~7 °C for 33 % of
throughput, against the clock cap's ~21 °C for an 18 % *gain* — roughly 25× worse value.
`yield` is useless: it yields and is immediately rescheduled.

**What does work — pin the spin to a little core.** It polls a flag; it has no need of a
3.9 GHz X925 when a 2.8 GHz A725 polls identically:

```bash
taskset -c 0,1,2,3,4,10,11,12,13,14 build/bin/llama-server ...
```

| | tg | SoC rise |
| --- | --- | --- |
| spin, all cores | 27.78 | +16.1 °C |
| spin, little cores only | 27.00 | **+10.9 °C** |

**5.2 °C for 2.8 %**, keeping the low-latency sync. Cluster map: little = 0-4, 10-14;
big = 5-9, 15-19. Caveat: measured on decode at 16k context; prefill does more host-side
work and may pay more.

Also note `--poll` defaults to **50** (the ggml CPU threadpool polls too). For a fully
offloaded model, `--poll 0` is worth testing:

```bash
build/bin/llama-server --poll 0 ...
```

### 4.5 Firmware

A fan-curve regression after EC/UEFI updates is reported, with rollback as the fix, across
multiple OEM variants (ASUS GX10, Gigabyte, MSI). Ours is BIOS `5.36_0ACUM018`
(08/06/2025). Check yours before updating anything:

```bash
cat /sys/class/dmi/id/bios_version /sys/class/dmi/id/bios_date
```

---

## 5. Operating recipe

**Always** run telemetry alongside real work. Every number in this document came from
these CSVs; the one session we ran without it is the one we cannot diagnose:

```bash
spark-monitor --log ~/thermal/$(date +%Y%m%d-%H%M).csv --interval 1
```

Guard rails that have proven necessary:

- **SoC abort at 93 °C**, sampling `max` across `/sys/class/thermal/thermal_zone*/temp`.
  Calibrated against §3.0: both wedges lived at **94–98 °C for ~10 minutes**, so 93 is
  just below the lethal band with real margin. A momentary touch of 90 is *not* a
  failure signal — our first two soaks aborted at 90.1 and 90.6 and so never learned
  whether they would have plateaued at 91 or climbed to 96, which is the whole question.
- **What to watch is time-in-band, not peak.** Track seconds above 90 °C; a config that
  spikes to 92 and falls back is fine, one that settles at 94 is not.
- After any wedge: **cold drain** (full power-down, not a reboot). A warm reboot does not
  clear the latch, and a health check passing on a warm-rebooted box means nothing —
  ours passed 38 minutes before the next wedge.
- Deep prefills need a genuinely cold chassis. **Idle time predicts fill capacity; die
  temperature does not** — the die cools in minutes, the heatsink does not. A 9-minute
  cooldown at 49.8 °C performed *worse* than a 10-minute one at 60.7 °C.

Quick SoC read:

```bash
cat /sys/class/thermal/thermal_zone*/temp | sort -rn | head -1 | awk '{printf "SoC %.1f C\n", $1/1000}'
```

---

## 6. What we got wrong

Recorded so the next person doesn't spend the time again.

- **Built a duty-cycle governor instead of using `nvidia-smi -lgc`.** `GGML_CUDA_POWER` is
  a software sleep-loop reimplementation of a hardware lever that already existed. We
  concluded "no clock control" from `-d POWER` showing `N/A` without checking `-lgc`.
- **`GGML_CUDA_POWER_ADAPT` keyed on NVML throttle bits.** Those are GPU-die referenced
  (§3.2), so it is a lagging indicator at best. It wedged and was demoted to a fixed duty.
- **Treated this as a GPU problem for months.** It is an SoC problem; the CPU is a
  first-class contributor (§3.1) and no GPU-side governor can see or control it.
- **Used watts as a tripwire.** Falsified twice — once by a wedge at 81.3 W peak with zero
  samples ≥ 84 W, and again by §3.3.
- **Ran a long agent session without telemetry**, which is why wedge #13's cause is
  inferred rather than known.
- **Never ran field diagnostics**, despite a matching, RMA-able signature being public.
- **Read the two thermal CSVs as survivor traces for a whole session.** They end in NUL
  padding and stop at recorded wedge times — they were the run-up data all along. That
  error produced a confident "the box survives 98.7 °C, our abort is too conservative"
  when the truth is it reached 98.7 °C *and then died*. Check for a NUL tail before
  concluding anything from a trace on this box.
- **Aborted both soaks at 90 °C**, which §3.0 now shows is below the lethal band. Both
  runs were killed just short of answering the question they were built to answer.
- **Predicted an idle thermal win from the CPU governor.** There is none (§4.2); idle
  cores clock-gate regardless of permitted frequency.
- **Assumed the busy-wait was free to remove.** It costs 33 % throughput on this iGPU
  (§4.4). Upstream's odd-looking special-case was correct.

---

## 7. Open questions

- Does fieldiag pass on this unit? Unknown — untested. Given §3.0 shows two wedges at
  56-65 W against a 140 W TDP, a fan curve that never ramps is a live hypothesis.
- What does `-lgc 300,2200` actually cost *our* workload? The published "negligible" is
  from someone else's model and batch shape.
- Does `cudaDeviceScheduleBlockingSync` regress GB10 sync latency enough to matter?
- Is there an EC/UEFI version pairing that fixes the fan curve on this specific unit?
- **What clock cap holds SoC out of the 94-98 C band under sustained prefill?** Unanswered:
  both soaks were aborted at 90 before they could plateau. `cap-sweep.sh` sweeps
  2200/2000/1800/1600 on one model load; it needs `/etc/sudoers.d/nvidia-clocks`.
- Is the wedge purely thermal? Now better supported than it was -- §3.0 shows both wedges
  ending after ~10 minutes at 94-98 C -- but still not proven causal. Nothing rules out
  temperature being a correlate of the real trigger.
- Does little-core pinning (§4.4) still pay during prefill, where host-side work is heavier?

---

## Sources

- [DGX Spark hangs under vLLM load, fieldiag fails on the thermal sensor](https://forums.developer.nvidia.com/t/dgx-spark-hangs-under-vllm-load-fieldiag-fails-on-the-thermal-sensor/369381)
- [GB10 thermal throttling after EC/UEFI updates, ACPI zones 96-97 C, fans not ramping](https://forums.developer.nvidia.com/t/dgx-spark-gb10-thermal-throttling-after-ec-uefi-updates-acpi-zones-96-97c-fans-not-ramping/377044)
- [Your DGX Spark Is Cooking Itself — Wild Pines AI](https://www.wildpines.ai/blog/your-dgx-spark-is-cooking-itself/)
- [llama.cpp discussion #22238 — 100% single-core usage while fully offloaded](https://github.com/ggml-org/llama.cpp/discussions/22238)
- [DGX Spark FieldDiag PowerStress FAIL (MODS-020000600139) → RMA](https://forums.developer.nvidia.com/t/dgx-spark-fielddiag-powerstress-fail-mods-020000600139-thermal-sensor-requesting-rma/373266)
- [NVIDIA DGX Spark Field Diagnostics User Guide](https://docs.nvidia.com/pdf/userguide-dgx-spark-fieldiag.pdf)
- [DGX Spark Thermal throttling (main thread)](https://forums.developer.nvidia.com/t/dgx-spark-thermal-throttling/349647)
- [spark-doctor — community diagnostic CLI for GB10](https://github.com/joeynyc/spark-doctor)
