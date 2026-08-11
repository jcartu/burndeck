/* burndeck demo feed — a synthetic multi-GPU rig speaking the exact stream
   contract (full snapshot, then keyed delta frames at 4 Hz with 1 Hz history
   appends). Loaded by index.html only when no backend is meant to exist:
   ?demo / ?gpus=N query, *.github.io hosting, or file://.

   The rig is deliberately alive: TP job envelopes breathe, one GPU runs hot
   every couple of minutes, the queue backs up and drains, a model gets pulled
   into VRAM with live load telemetry and released again, PCIe drops width on
   one card and recovers — so every gauge, badge, ticker event and theme pulse
   path gets exercised without hardware. */
window.BurndeckDemo = (() => {
  const T0 = Date.now();
  const GPU_NAME = 'RTX PRO 6000 Blackwell Workstation Edition';
  const MEM_TOTAL = 97887, POWER_LIMIT = 600;
  const HIST_MAX = 180;

  const wave = (i, t, period, phase = 0) =>
    0.5 + 0.5 * Math.sin(t * 2 * Math.PI / period + i * 0.9 + phase);

  /* ── per-tick rig state ── */
  function gpuState(i, t, n){
    const job = 0.42 + 0.5 * wave(Math.floor(i / 8), t, 210);
    let util = Math.min(100, Math.max(2,
      100 * job * (0.86 + 0.14 * wave(i, t, 7)) + 5 * wave(i, t, 2.3, 1.7)));
    let mem = MEM_TOTAL * Math.min(0.97, 0.55 + 0.32 * job + 0.02 * wave(i, t, 31));
    let power = POWER_LIMIT * (0.09 + 0.82 * Math.pow(util / 100, 1.35));
    let temp = 33 + 46 * (power / POWER_LIMIT) + 3 * wave(i, t, 47);

    /* drama: the last GPU spikes hot for ~18 s every ~2 min */
    const hotPhase = (t + 30) % 130;
    if (i === n - 1 && hotPhase < 18) temp += 14 * Math.sin(Math.PI * hotPhase / 18);

    /* drama: GPU 1 drops PCIe width for ~14 s every ~3 min */
    const pciePhase = (t + 75) % 190;
    const degraded = i === 1 && n > 1 && pciePhase < 14;

    const procs = [{ pid: 41000 + i, process_name: 'vllm::EngineCore',
                     used_memory_mib: mem * 0.985 }];
    /* drama: a short-lived compute process visits GPU 0 every ~50 s */
    const spawnPhase = (t + 12) % 50;
    if (i === 0 && spawnPhase < 9)
      procs.push({ pid: 52000 + (Math.floor(t / 50) % 7), process_name: 'python3',
                   used_memory_mib: 2048 });

    return {
      index: i, name: GPU_NAME,
      uuid: `GPU-demo${String(i).padStart(2, '0')}-0000-0000-0000-000000000000`,
      pcie_gen_max: 5, pcie_width_max: 16,
      bus_id: `00000000:${(0x18 + i).toString(16).toUpperCase().padStart(2, '0')}:00.0`,
      power_limit_w: POWER_LIMIT,
      temperature_c: +temp.toFixed(1),
      fan_pct: +Math.min(100, Math.max(0, (temp - 40) * 2.6)).toFixed(1),
      graphics_clock_mhz: Math.round(900 + 1900 * util / 100),
      memory_clock_mhz: 14001,
      pstate: util > 40 ? 'P0' : util > 8 ? 'P2' : 'P8',
      pcie_gen: 5, pcie_width: degraded ? 8 : 16,
      pcie_aer: { root_port: '', correctable: {}, total_correctable: Math.floor(t / 500),
                  total_fatal: 0, total_nonfatal: 0 },
      processes: procs,
      utilization_gpu_pct: +util.toFixed(1),
      utilization_memory_pct: +(util * 0.8).toFixed(1),
      memory_used_mib: mem, memory_total_mib: MEM_TOTAL,
      memory_pct: mem / MEM_TOTAL * 100,
      power_draw_w: +power.toFixed(1),
      power_pct: power / POWER_LIMIT * 100,
    };
  }

  function modelStates(t, n){
    const load = 0.42 + 0.5 * wave(0, t, 210);
    const gust = 0.75 + 0.25 * wave(0, t, 23);
    const out = (30 + 240 * load * gust) * Math.max(n, 1) / 8;
    const inp = out * (2.1 + 1.2 * wave(0, t, 61));
    const total = Math.floor((t + 3 * 86400) * (out + inp) * 0.4);
    const gpus = Array.from({ length: n }, (_, i) => i);
    const models = [{
      id: 'demo-vllm', endpoint: 'http://127.0.0.1:8000', port: 8000,
      label: 'Qwen3-235B-A22B FP8 · vLLM',
      gpu_indices: gpus, cfg_gpu_indices: gpus,
      accent: '#45d1ff', accent2: '#8b7cff',
      healthy: true, error: '',
      model_names: ['qwen3-235b-a22b-fp8'],
      model_meta: [{ id: 'qwen3-235b-a22b-fp8', root: '', max_model_len: 262144, created: T0 / 1000 | 0 }],
      max_model_len: 262144,
      running: Math.round(1 + 5 * load),
      waiting: Math.max(0, Math.round((wave(0, t, 97) - 0.62) * 36)),
      kv_cache_pct: +(22 + 58 * load + 6 * wave(1, t, 17)).toFixed(1),
      prompt_tokens_total: Math.floor(total * 0.7),
      generation_tokens_total: Math.floor(total * 0.3),
      requests_total: Math.floor(total / 900), request_rate: 0,
      input_tps: +inp.toFixed(2), output_tps: +out.toFixed(2),
      total_tps: +(inp + out).toFixed(2),
      ttft_ms: null, inter_token_ms: null,
      engine: 'vLLM', engine_version: '0.10.2',
      srv_model_path: '', srv_ctx: 262144,
      worker_pids: gpus.map(g => 41000 + g),
      vram_mib: 61000 * Math.max(n, 1),
      intel: {
        display_name: 'Qwen3 235B A22B', family: 'Qwen',
        creator: 'Alibaba Cloud', origin: 'Hangzhou · CN',
        blurb: "Alibaba's Tongyi Qianwen family — open-weight dense and MoE models spanning chat, coding, math and vision.",
        license: 'apache-2.0', params: '235B', moe: { experts: 128, active: 8 },
        ctx_serving: 262144, ctx_native: 262144, quant: 'FP8', dtype: 'bfloat16',
        layers: 94, hidden: 4096, heads: 64, attn: 'GQA', weights_gb: 235,
        engine: 'vLLM', engine_version: '0.10.2', tp: n,
        uptime_s: t + 3 * 86400, vram_mib: 61000 * Math.max(n, 1),
      },
    }];
    if (n >= 2){
      const dg = [0, 1].slice(0, n);
      models.push({
        id: 'demo-sglang', endpoint: 'http://127.0.0.1:30000', port: 30000,
        label: 'Llama-3.3-70B AWQ · SGLang',
        gpu_indices: dg, cfg_gpu_indices: dg,
        accent: '#4de6a8', accent2: '#57c8ff',
        healthy: true, error: '',
        model_names: ['llama-3.3-70b-instruct-awq'],
        model_meta: [{ id: 'llama-3.3-70b-instruct-awq', root: '', max_model_len: 131072, created: T0 / 1000 | 0 }],
        max_model_len: 131072,
        running: Math.round(2 * load), waiting: 0,
        kv_cache_pct: +(30 + 40 * wave(2, t, 41)).toFixed(1),
        prompt_tokens_total: Math.floor(total * 0.2),
        generation_tokens_total: Math.floor(total * 0.08),
        requests_total: Math.floor(total / 3000), request_rate: 0,
        input_tps: +(inp * 0.18).toFixed(2), output_tps: +(out * 0.28).toFixed(2),
        total_tps: +(inp * 0.18 + out * 0.28).toFixed(2),
        ttft_ms: null, inter_token_ms: null,
        engine: 'SGLang', engine_version: '0.4.9',
        srv_model_path: '', srv_ctx: 131072,
        worker_pids: dg.map(g => 41000 + g),
        vram_mib: 21000 * dg.length,
        intel: {
          display_name: 'Llama 3.3 70B Instruct', family: 'Llama',
          creator: 'Meta AI', origin: 'Menlo Park · US',
          blurb: "Meta's Llama herd — the open-weight family that kicked off the local-LLM era.",
          license: 'llama-3.3', params: '70B',
          ctx_serving: 131072, ctx_native: 131072, quant: 'AWQ INT4', dtype: 'bfloat16',
          layers: 80, hidden: 8192, heads: 64, attn: 'GQA', weights_gb: 40,
          engine: 'SGLang', engine_version: '0.4.9', tp: dg.length,
          uptime_s: t + 7200, vram_mib: 21000 * dg.length,
        },
      });
    }
    return models;
  }

  /* ── a model cycles: loading (35 s) → resident (75 s) → released (70 s) ── */
  function sniffedStates(t){
    const phase = (t + 20) % 180;
    if (phase >= 110) return [];
    const loading = phase < 35;
    const pct = loading ? phase / 35 * 100 : null;
    const s = {
      id: 'sniff-gptoss', name: 'gpt-oss-120b', app: 'llama.cpp',
      state: loading ? 'loading' : 'resident',
      pids: [61044], gpu_indices: [0], ports: loading ? [] : [8080],
      served_names: loading ? [] : ['gpt-oss-120b'],
      vram_mib: loading ? 61000 * (phase / 35) : 61000,
      disk_gb: 61, n_files: 2, uptime_s: phase,
      intel: {
        family: 'GPT', creator: 'OpenAI', origin: 'San Francisco · US',
        blurb: "OpenAI's GPT lineage — gpt-oss brings the o-series reasoning stack to open weights.",
        license: 'apache-2.0', params: '117B', moe: { experts: 128, active: 4 },
        ctx_native: 131072, quant: 'MXFP4', dtype: 'bfloat16',
        layers: 36, hidden: 2880, heads: 64, attn: 'GQA',
      },
    };
    if (loading){
      s.loading = {
        stages: ['init', 'weights', 'warmup', 'api'],
        stage_idx: pct < 8 ? 0 : pct < 82 ? 1 : pct < 96 ? 2 : 3,
        pct, detail: pct < 82 ? 'mapping safetensors shards' : 'compiling CUDA graphs',
        disk_read_mbps: pct < 82 ? 5200 + 800 * wave(0, t, 5) : 40,
        vram_rate_mibps: pct < 82 ? 1740 : 60,
        cpu_pct: 220 + 60 * wave(1, t, 7),
        rss_mib: 18000 * Math.min(1, phase / 30),
        eta_s: Math.max(0, 35 - phase),
      };
    }
    return [s];
  }

  /* Mirror build_snapshot's per-GPU TPS attribution: each model's rate is
     split evenly across its GPUs, and the model chips ride along. */
  function attachTps(gs, ms){
    const per = new Map(gs.map(g => [g.index, { input: 0, output: 0, total: 0, models: [] }]));
    for (const m of ms){
      if (!m.healthy) continue;
      const idxs = m.gpu_indices.filter(i => per.has(i));
      for (const i of idxs){
        const t = per.get(i);
        t.input += m.input_tps / idxs.length;
        t.output += m.output_tps / idxs.length;
        t.total += m.total_tps / idxs.length;
        t.models.push({ id: m.id, label: m.label, names: m.model_names, accent: m.accent });
      }
    }
    for (const g of gs) g.tps = per.get(g.index);
  }

  function totalsState(gpus, models){
    const tps = models.reduce((a, m) => a + m.total_tps, 0);
    return {
      input_tps: models.reduce((a, m) => a + m.input_tps, 0),
      output_tps: models.reduce((a, m) => a + m.output_tps, 0),
      total_tps: tps,
      gpu_util_avg: gpus.reduce((a, g) => a + g.utilization_gpu_pct, 0) / gpus.length,
      memory_pct_avg: gpus.reduce((a, g) => a + g.memory_pct, 0) / gpus.length,
      power_draw_w: gpus.reduce((a, g) => a + g.power_draw_w, 0),
      power_limit_w: POWER_LIMIT * gpus.length,
      power_pct: gpus.reduce((a, g) => a + g.power_draw_w, 0) / (POWER_LIMIT * gpus.length) * 100,
    };
  }

  function start(ingest, qs){
    const n = Math.max(1, Math.min(32,
      parseInt(qs.get('gpus') || qs.get('demo') || '8', 10) || 8));
    const now = () => (Date.now() - T0) / 1000;

    /* full snapshot with the whole history window prefilled, so charts and
       sparklines are rich from the very first paint */
    const t = now();
    const gpus = Array.from({ length: n }, (_, i) => gpuState(i, t, n));
    const hist = { total_tps: [], gpu_util_avg: [], mem_pct_avg: [], power_total: [] };
    for (const g of gpus) g.history = { util: [], mem: [], power: [], temp: [] };
    for (let k = HIST_MAX - 1; k >= 0; k--){
      const tk = t - k;
      const gs = Array.from({ length: n }, (_, i) => gpuState(i, tk, n));
      const ts = totalsState(gs, modelStates(tk, n));
      gs.forEach((g, i) => {
        gpus[i].history.util.push(g.utilization_gpu_pct);
        gpus[i].history.mem.push(g.memory_pct);
        gpus[i].history.power.push(g.power_pct);
        gpus[i].history.temp.push(g.temperature_c);
      });
      hist.total_tps.push(ts.total_tps);
      hist.gpu_util_avg.push(ts.gpu_util_avg);
      hist.mem_pct_avg.push(ts.memory_pct_avg);
      hist.power_total.push(ts.power_draw_w);
    }
    const models = modelStates(t, n);
    attachTps(gpus, models);
    const totals = totalsState(gpus, models);
    totals.history = hist;
    ingest({
      generated_at: new Date().toISOString(),
      host: 'demo-rig', driver: '580.82', cuda: '13.0',
      uptime_seconds: 86400 * 12 + t,
      poll_seconds: 0.25, llm_poll_seconds: 1,
      gpus, models, sniffed: sniffedStates(t), totals,
    });

    /* 4 Hz deltas, 1 Hz history appends + model refresh */
    let tick = 0;
    setInterval(() => {
      tick++;
      const tc = now();
      const gs = Array.from({ length: n }, (_, i) => gpuState(i, tc, n));
      const ms = modelStates(tc, n);
      attachTps(gs, ms);
      const ts = totalsState(gs, ms);
      const frame = {
        type: 'update',
        generated_at: new Date().toISOString(),
        uptime_seconds: 86400 * 12 + tc,
        gpus: gs, totals: ts,
      };
      if (tick % 4 === 0){
        frame.models = ms;
        frame.sniffed = sniffedStates(tc);
        frame.history = {
          max_points: HIST_MAX,
          gpus: Object.fromEntries(gs.map(g => [String(g.index), {
            util: g.utilization_gpu_pct, mem: g.memory_pct,
            power: g.power_pct, temp: g.temperature_c,
          }])),
          totals: { total_tps: ts.total_tps, gpu_util_avg: ts.gpu_util_avg,
                    mem_pct_avg: ts.memory_pct_avg, power_total: ts.power_draw_w },
        };
      }
      ingest(frame);
    }, 250);
  }

  return { start };
})();
