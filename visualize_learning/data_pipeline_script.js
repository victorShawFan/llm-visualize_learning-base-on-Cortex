document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');

    if (!prevBtn || !nextBtn || !resetBtn || !infoBox || !visualContent || !codeSnippet) {
        console.error("Required elements not found in Data Pipeline script");
        return;
    }

    let currentStep = 0;
    let currentInterval = null; // Track intervals

    // Offline 数据工程总览：process_data.py → .pkl/.npy → dataset.py
    const steps = [
      {
        title: 'Step 0: 环境初始化与清洗工具',
        description:
          '在任何离线预处理开始前，<code>_init()</code> 会调用 <code>utils.init_env</code> 设置随机种子、并行参数等，同时将 <code>TOKENIZERS_PARALLELISM</code> 设为 true。之后一系列小工具 (<code>_remove_urls</code> / <code>_remove_brackets</code> / <code>_filter_content</code>) 用于统一清洗文本并替换占位符。<span class="step-badge">process_data.py:12-40</span>',
        code: `def _init():\n    from utils import init_env\n    init_env()\n    os.environ["TOKENIZERS_PARALLELISM"] = "true"\n\ncontent = _filter_content(raw_text)`,
        render: () => renderInit(),
      },
      {
        title: 'Step 1: 通用语料预编码 (Wikipedia / CMM-Math / Github-Code)',
        description:
          '预训练前，Wikipedia、CMM-Math 与 Github-Code 等通用语料会被统一编码为 Token 序列并写入 <code>./data/tmp/*.pkl</code>。每个函数都遵循：<code>raw → 组装字符串 → append(text_end) → tokenizer.encode → 列表 or np.ndarray → pickle.dump</code> 的模式。<span class="step-badge">process_data.py:57-101,103-127</span>',
        code: `with open('./data/raw/wikipedia-cn-20230720-filtered.json') as f:\n    for item in json_:\n        ids = TrainerTools().tokenizer.encode(f"{item['completion']}{text_end}")\n        encoded.append(ids)\n\nwith open('./data/tmp/wikipedia.pkl', 'wb') as f:\n    pickle.dump(encoded, f)`,
        render: () => renderGenericSources(),
      },
      {
        title: 'Step 2: 预训练 SFT JSONL → 长短分桶 (preprocess_pretrain_data)',
        description:
          '对带有 <code>history/input/output</code> 字段的 JSONL 进行统一处理：先将历史对话扁平化为 <code>history</code> 字符串，再拼上当前 input/output，用 <code>_filter_content</code> 清洗，并追加 <code>text_end</code>。编码得到的 token 序列以长度为 기준分到 <b>短桶 (short)</b> 与 <b>长桶 (long)</b>，并分别在 tokens 计数达到 4e8 时落盘滚动。<span class="step-badge">process_data.py:130-188</span>',
        code: `if len(history) == 0:\n    item = _filter_content(f"{json_['input']}\n{json_['output']}{text_end}")\nelse:\n    item = _filter_content(f"{history}{json_['input']}\n{json_['output']}{text_end}")\nids = TrainerTools().tokenizer.encode(item.strip())\nif len(ids) > short_thresholds[file_idx]:\n    result_long.append(ids)\nelse:\n    result_short.append(ids)`,
        render: () => renderPretrainSplit(),
      },
      {
        title: 'Step 3: Think / Answer 自认知样本 (get_self_cognition)',
        description:
          '自认知数据 (<code>self_cognition.jsonl</code>) 会根据 <code>add_think_tag</code> 配置决定是否在用户 query 后追加 <code>/no think</code> 控制指令。每条样本通过 <code>Tokenizer.apply_chat_template</code> 包装成 <code>&lt;system&gt; / &lt;user&gt; / &lt;assistant&gt;&lt;think&gt;&lt;/think&gt;&lt;answer&gt;&lt;/answer&gt;</code> 结构，并直接返回编码后的 Token 张量列表，供 SFT 或 RLHF 模块使用。<span class="step-badge">process_data.py:191-213, tokenizer.py:300-356</span>',
        code: `chat_template = [\n  {'role': 'system', 'content': ' '},\n  {'role': 'user', 'content': user},\n  {'role': 'assistant', 'think': ' ', 'content': content.strip()},\n]\nencoded = TrainerTools().tokenizer.apply_chat_template(chat_template)`,
        render: () => renderSelfCognition(),
      },
      {
        title: 'Step 4: 预训练短序列多源混合 (merge_pretrain_data)',
        description:
          '不同语言与来源的短序列预训练数据 (.pkl) 会在 <code>merge_pretrain_data</code> 中进行多源混合：例如将英文 <code>pretrain_short_en_0</code> 分成两半分别 merge 到 <code>pretrain_short_zh_0/1</code>，对结果做 shuffle 并展平成 <code>flat_result</code> 写入最终的 <code>./data/pretrain_short_*.pkl</code>。<span class="step-badge">process_data.py:216-237</span>',
        code: `with open('./data/tmp/pretrain_short_en_0.pkl', 'rb') as f:\n    en = pickle.load(f)\nen_0, en_1 = en[:mid], en[mid:]\n...\nflat_result = list(itertools.chain.from_iterable(shuffle(result)))\nwith open('./data/pretrain_short_0.pkl', 'wb') as f:\n    pickle.dump(flat_result, f)`,
        render: () => renderMerge(),
      },
      {
        title: 'Step 5: 从 .pkl/.npy 到 PretrainDataset',
        description:
          '当训练正式开始，<code>PretrainDataset</code> 会依据 <code>_get_file_type</code> 选择加载方式：<br>• <code>.npy</code> → <b>内存映射</b>：避免一次性载入全部 token。<br>• <code>.pkl</code> → 直接 <code>pickle.load</code> 得到 token list。然后在 <code>__getitem__</code> 中使用滑动窗口 <code>[start_idx:end_idx]</code> 切分成固定长度 block，用于自回归预训练。<span class="step-badge">dataset.py:15-35,40-52,86-120</span>',
        code: `file_type = _get_file_type(file_path)\nif file_type == 'npy':\n    self.data = np.load(file_path, mmap_mode='r')\n...\nstart_idx = item * self.stride\nend_idx = start_idx + self.block_size\nseq = self.data[start_idx:end_idx]`,
        render: () => renderPretrainDataset(),
      },
      {
        title: 'Step 6: SFT / DPO / RL Dataset 与离线预处理的衔接',
        description:
          'SFTDataset / DPODataset / RLDataset 既可以直接读取预编码好的 <code>.npy/.pkl</code>（例如已经 apply_chat_template + encode 的序列），也可以从 JSONL/on-the-fly 组装文本再 encode。Offline Pipeline 负责“洗干净 +切分好 +打平混合”，Online Dataset 则负责在 train loop 中优雅地提供 batch。<span class="step-badge">dataset.py:123-197,223-309,312-400</span>',
        code: `if file_type in ['npy', 'pkl']:\n    ids = self.data[item]\nelse:\n    text = build_text_from_jsonl(record)\n    ids = tokenizer.encode(text)\nreturn torch.tensor(ids, dtype=torch.long)`,
        render: () => renderDatasetBridge(),
      },
      {
        title: 'Step 7: 训练入口中的文件级循环',
        description:
          '在 <code>Trainer.train()</code> 中，<code>self.train_config.file_dataset</code> 会列出所有预处理好的数据文件 (.pkl/.npy/.jsonl)。训练主循环对这些文件逐个调用 <code>_create_dataset(file_idx)</code>，利用 <code>TrainerTools().parallel.process_dataloader</code> 构造带分布式采样的 DataLoader，从而形成「文件 → Dataset → DataLoader → Batch」的闭环。<span class="step-badge">trainer.py:500-514,771-777</span>',
        code: `for file_idx in range(len(self.train_config.file_dataset)):\n    dataset, file_path = self._create_dataset(file_idx)\n    train_loader = TrainerTools().parallel.process_dataloader(\n        dataset=dataset,\n        data_loader_kwargs=self.data_loader_kwargs,\n        sampler_kwargs=self.sampler_kwargs\n    )`,
        render: () => renderTrainerLoop(),
      },
    ];

    function updateUI() {
      // Clear previous interval if any
      if (currentInterval) {
        // No intervals used in this script's animations, but good practice.
        // Actually renderPretrainDataset doesn't use interval, dataset_script.js did.
        // Let's check the render functions.
        // None of the render functions below use setInterval. They are static or CSS animations.
      }

      if (currentStep < 0) currentStep = 0;
      if (currentStep >= steps.length) currentStep = steps.length - 1;

      const step = steps[currentStep];
      infoBox.innerHTML = `<div class="step-badge">Data Pipeline</div><strong>${step.title}</strong><br>${step.description}`;
      codeSnippet.textContent = step.code;
      
      if (window.hljs) {
        hljs.highlightElement(codeSnippet);
      }

      visualContent.innerHTML = '';
      visualContent.className = 'fade-in';
      
      try {
        step.render();
      } catch(e) {
        console.error("Render failed", e);
      }

      updateButtons();
    }

    // --- Renderers ---

    function renderInit() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:20px; align-items:center; width:100%;">
          <div class="dict-entry" style="max-width:500px;">
            <strong>Env Init</strong><br>
            init_env() + TOKENIZERS_PARALLELISM = "true"
          </div>
          <div style="display:flex; gap:10px; flex-wrap:wrap; justify-content:center;">
            <div class="tok-box">_remove_urls</div>
            <div class="tok-box">_remove_brackets</div>
            <div class="tok-box">_filter_content</div>
          </div>
        </div>
      `;
    }

    function renderGenericSources() {
      visualContent.innerHTML = `
        <div style="display:flex; gap:20px; flex-wrap:wrap; justify-content:center; align-items:flex-start;">
          <div class="file-icon" style="border-color:#3498db; color:#3498db;">
            wikipedia-cn.json
            <div style="font-size:10px; margin-top:5px;">completion 字段</div>
          </div>
          <div class="file-icon" style="border-color:#e67e22; color:#e67e22;">
            CMM-Math.jsonl
            <div style="font-size:10px; margin-top:5px;">question / options / analysis / answer</div>
          </div>
          <div class="file-icon" style="border-color:#16a085; color:#16a085;">
            github-code.parquet
            <div style="font-size:10px; margin-top:5px;">content 列 (1/4 采样)</div>
          </div>
        </div>
        <div class="arrow" style="margin-top:20px;">↓ Tokenizer.encode + text_end</div>
        <div class="dict-entry" style="max-width:520px;">→ ./data/tmp/*.pkl (list[list[int]])</div>
      `;
    }

    function renderPretrainSplit() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:20px; align-items:center; width:100%;">
          <div class="code-block-dark" style="max-width:640px;">
            history + input + output + &lt;/s&gt; → ids (len = N)
          </div>
          <div style="display:flex; gap:20px; flex-wrap:wrap; justify-content:center;">
            <div class="dict-entry" style="border-left:4px solid #2ecc71;">
              <strong>short_bucket</strong><br>
              len(ids) ≤ threshold → pretrain_short_*.pkl
            </div>
            <div class="dict-entry" style="border-left:4px solid #e74c3c;">
              <strong>long_bucket</strong><br>
              len(ids) &gt; threshold → pretrain_long_*.pkl
            </div>
          </div>
        </div>
      `;
    }

    function renderSelfCognition() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:15px; align-items:center; width:100%;">
          <div class="dict-entry" style="max-width:520px;">
            原始 JSONL：{ "query": "你是谁?", "response": "我是 {{NAME}}" }
          </div>
          <div class="arrow">↓ 替换 {{AUTHOR}} / {{NAME}} + 可选 /no think</div>
          <div class="code-block-dark" style="max-width:520px;">
            &lt;system&gt;  &lt;/s&gt;\n
            &lt;user&gt; 你是谁? /no think &lt;/s&gt;\n
            &lt;assistant&gt;&lt;think&gt; &lt;/think&gt;&lt;answer&gt;我是 Cortex&lt;/answer&gt;&lt;/s&gt;
          </div>
        </div>
      `;
    }

    function renderMerge() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:20px; align-items:center; width:100%;">
          <div style="display:flex; gap:10px; flex-wrap:wrap; justify-content:center;">
            <div class="file-icon" style="border-color:#3498db; color:#3498db;">pretrain_short_en_0.pkl</div>
            <div class="file-icon" style="border-color:#3498db; color:#3498db;">pretrain_short_en_1.pkl</div>
            <div class="file-icon" style="border-color:#e67e22; color:#e67e22;">pretrain_short_zh_0.pkl</div>
            <div class="file-icon" style="border-color:#e67e22; color:#e67e22;">pretrain_short_zh_1.pkl</div>
          </div>
          <div class="arrow">↓ shuffle + chain.from_iterable</div>
          <div class="dict-entry" style="max-width:520px;">→ ./data/pretrain_short_0.pkl / pretrain_short_1.pkl (混合中英短序列)</div>
        </div>
      `;
    }

    function renderPretrainDataset() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:20px; align-items:center; width:100%;">
          <div style="display:flex; gap:20px; flex-wrap:wrap; justify-content:center;">
            <div class="dict-entry">file_type = 'npy' → np.load(mmap_mode='r')</div>
            <div class="dict-entry">file_type = 'pkl' → pickle.load</div>
          </div>
          <div class="arrow">↓ PretrainDataset.__getitem__ (block_size / stride)</div>
          <div class="code-block-dark" style="max-width:520px;">start_idx = item * stride\nend_idx = start_idx + block_size\nseq = data[start_idx:end_idx]</div>
        </div>
      `;
    }

    function renderDatasetBridge() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:15px; align-items:center; width:100%;">
          <div class="dict-entry" style="max-width:520px;">
            <strong>SFTDataset / DPODataset / RLDataset</strong><br>
            既可以直接消费 <code>.npy/.pkl</code>（预编码 token），也可以在线从 JSONL 构造文本再 encode。
          </div>
          <div class="arrow">↓</div>
          <div class="code-block-dark" style="max-width:520px;">if file_type in ['npy', 'pkl']:\n    ids = self.data[item]\nelse:\n    text = build_text_from_jsonl(record)\n    ids = tokenizer.encode(text)</div>
        </div>
      `;
    }

    function renderTrainerLoop() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:20px; align-items:center; width:100%;">
          <div class="dict-entry" style="max-width:520px;">TrainConfig.file_dataset = [\n  'pretrain_short_0.pkl',\n  'pretrain_short_1.pkl',\n  ...\n]</div>
          <div class="arrow">↓ Trainer.train()</div>
          <div class="code-block-dark" style="max-width:520px;">for file_idx in range(len(file_dataset)):\n    dataset, file_path = self._create_dataset(file_idx)\n    loader = parallel.process_dataloader(dataset, ...)</div>
        </div>
      `;
    }

    function goNext() {
        if (currentStep < steps.length - 1) {
            currentStep++;
            updateUI();
        }
    }

    function goPrev() {
        if (currentStep > 0) {
            currentStep--;
            updateUI();
        }
    }

    function updateButtons() {
        if (prevBtn) prevBtn.disabled = currentStep === 0;
        if (nextBtn) nextBtn.disabled = currentStep === steps.length - 1;
    }

    if (nextBtn) nextBtn.addEventListener('click', goNext);
    if (prevBtn) prevBtn.addEventListener('click', goPrev);

    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            currentStep = 0;
            updateUI();
        });
    }

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });

    updateUI();
});
