document.addEventListener('DOMContentLoaded', () => {
  const prevBtn = document.getElementById('prevStep');
  const nextBtn = document.getElementById('nextStep');
  const resetBtn = document.getElementById('reset');
  const infoBox = document.getElementById('infoBox');
  const visualContent = document.getElementById('visualContent');
  const codeSnippet = document.getElementById('codeSnippet');

  if (!prevBtn || !nextBtn || !resetBtn || !infoBox || !visualContent || !codeSnippet) {
    console.error("Required elements not found in SFT script");
    return;
  }

  let currentStep = 0;

  // SFT 训练可视化：从数据配置到梯度更新与 checkpoint
  const steps = [
    {
      title: "步骤 0: SFT 配置与文件类型",
      description:
        "SFTDataset 会根据后缀自动选择加载策略：<code>.jsonl</code>（对话文本）、<code>.npy</code>/<code>.pkl</code>（已编码 token）。同时记录 <code>max_len</code> 等关键信息。<br><span class='step-badge'>dataset.py:145-156</span>",
      code: `file_type = _get_file_type(file_path)\nif file_type == 'jsonl':\n    self.plain_text = True\nself.max_len = max_len`,
      state: 'config',
    },
    {
      title: "步骤 1: VLM 图像标签与 tokens_per_image",
      description:
        "在 VLM SFT 场景下，<code>SFTDataset</code> 还会额外读取 <code>image_tags_file_dataset</code>，并根据 <code>tokens_per_image</code> 对 <code><image></code> 标记展开成多个图像 token，用于 Vision-LLM 对齐。<br><span class='step-badge'>dataset.py:SFTDataset.__init__ / __getitem__</span>",
      code: `if isinstance(train_config.model_config, VLMConfig):\n    image_tag_file_path = image_tags_file_dataset[file_idx]\n    tokens_per_image = model_config.tokens_per_image\n# __getitem__ 中：inputs = repeat_image_tok(inputs, tokens_per_image)`,
      state: 'vlm',
    },
    {
      title: "步骤 2: 加载 SFT 训练数据",
      description:
        "从 JSONL 加载对话样本。包含系统提示词、用户指令和助手回答。<br><span class='step-badge'>SFTDataset.__getitem__</span>",
      code: `sample = dataset[i]\n# {"role": "user", "content": "..."}`,
      state: 'input',
    },
    {
      title: "步骤 3: 构建训练序列",
      description:
        "应用 <code>Tokenizer.apply_chat_template</code> 拼接对话，并生成 Token 序列。角色标签使用 <code>&lt;system&gt;/&lt;user&gt;/&lt;assistant&gt;</code>，回答部分会被包上 <code>&lt;answer&gt;</code>，可选思维链使用 <code>&lt;think&gt;</code>。每轮对话以 <code>&lt;/s&gt;</code> 结尾。<br><span class='step-badge'>tokenizer.py:300-356</span>",
      code: `text = tokenizer.apply_chat_template(messages)\nids = tokenizer.encode(text, add_special_tokens=False)`,
      state: 'tokenize',
    },
    {
      title: "步骤 4: 截断到 max_len",
      description:
        "过长的对话会被截断到 <code>max_len</code>。这一截断在 inputs 与 labels 上保持对齐，防止越界访问。",
      code: `inputs = inputs[:max_len]\nlabels = labels[:max_len]`,
      state: 'truncate',
    },
    {
      title: "步骤 5: Loss Masking (Crucial!)",
      description:
        "这是 SFT 的核心逻辑。<code>get_sft_collate_fn</code> 调用 <code>_mask_prompt</code>，将 <code>inputs</code> 复制给 <code>labels</code>，然后将 **非回答部分**（如 Prompt、System Message、User Query）的 Label 设为 <code>-100</code>。只有 <code>&lt;answer&gt;...&lt;/answer&gt;</code>（以及可选的 <code>&lt;think&gt;</code>）内的 Token 会计算 Loss。<br><span class='step-badge'>utils.py:get_sft_collate_fn / _mask_prompt</span>",
      code: `labels = pad_sequence(batch_train_data, padding_value=-100)\nif mask_prompt:\n    labels = _mask_prompt(labels)\nloss = F.cross_entropy(logits, labels)`,
      state: 'mask',
    },
    {
      title: "步骤 6: 梯度累积 (Grad Accumulation)",
      description:
        "为了模拟大 Batch Size，在多次的前向/反向传播中累积梯度，每隔 <code>N</code> 步执行一次参数更新。<br><span class='step-badge'>trainer.py:757-866</span>",
      code: `loss = loss / grad_acc_steps\nloss.backward()\nif (step + 1) % grad_acc_steps == 0:\n    optimizer.step()`,
      state: 'acc',
    },
    {
      title: "步骤 7: 检查点保存 (Checkpoint)",
      description:
        "定期将当前训练状态落盘：<code>save_checkpoint</code> 保存模型参数和（可选）优化器状态，<code>save_steps</code> 单独记录 <code>global_steps</code> 与学习率调度器信息，用于断点续训。<br><span class='step-badge'>checkpoint.py:15-38, 140-151, trainer.py:887-907</span>",
      code: `save_steps(global_steps=global_steps, lr_scheduler=lr_scheduler)\nsave_checkpoint(model=train_model, optimizer=optimizer)`,
      state: 'checkpoint',
    },
    {
      title: "步骤 8: 学习率与优化器状态监控",
      description:
        "在 SFT 阶段通常使用带 Warmup 的 Cosine LR。通过记录当前 step 对应的 LR 与 Optimizer 中一阶/二阶动量，可以判断模型是否处于稳定训练阶段。",
      code: `lr = optimizer.param_groups[0]['lr']\nexp_avg = optimizer.state[param]['exp_avg']`,
      state: 'lr_monitor',
    },
    {
      title: "步骤 9: Eval & 指标观测",
      description:
        "Trainer 会定期通过 <code>_eval</code> 钩子在验证样本上做生成，将结果写入 <code>gen.txt</code> 并在多卡间同步，便于观察困惑度或主观质量，再由开发者根据曲线手动决定是否调整训练计划。<span class='step-badge'>trainer.py:665-690, eval.py:13-52</span>",
      code: `self._on_batch_end(tag=f'epoch:{epoch}/batch:{batch}')\n# submit_gen_task(...) → 写入 gen.txt`,
      state: 'eval',
    },
    {
      title: "步骤 10: 作为 RLHF 的初始化点",
      description:
        "训练完成后，SFT 权重会被拷贝为 Policy / Ref 初始化值：Ref Model 冻结用于 KL 参照，Policy Model 在此基础上继续执行 DPO / PPO / GRPO 等强化阶段。",
      code: `policy_model.load_state_dict(sft_ckpt)\nref_model.load_state_dict(sft_ckpt)\nref_model.requires_grad_(False)`,
      state: 'rlhf_init',
    },
    {
      title: "实战模拟: Next Token Prediction",
      description: "<b>SFT 核心机制演示：</b> 观察模型如何基于给定的 Context 逐个预测下一个 Token，并计算 Cross Entropy Loss。这就是 SFT (Teacher Forcing) 的本质：最大化 Ground Truth Token 的概率。",
      code: `Context: "The capital of France is" -> Label: " Paris"`,
      state: 'sim',
    }
  ];

  function updateUI() {
    // Boundary checks
    if (currentStep < 0) currentStep = 0;
    if (currentStep >= steps.length) currentStep = steps.length - 1;

    const step = steps[currentStep];
    infoBox.innerHTML = `<div class="step-badge">Phase ${currentStep}</div><strong>${step.title}</strong><p style="margin-top:10px">${step.description}</p>`;
    codeSnippet.textContent = step.code;
    
    if (window.hljs) {
      hljs.highlightElement(codeSnippet);
    }
    
    render();
    updateButtons();
  }

  function render() {
    visualContent.innerHTML = '';
    const state = steps[currentStep].state;

    if (state === 'sim') {
        renderSim();
        return;
    }

    if (state === 'config') {
      const cfg = document.createElement('div');
      cfg.style.display = 'flex';
      cfg.style.gap = '20px';
      cfg.style.justifyContent = 'center';
      cfg.innerHTML = `
        <div class="batch-mini" style="background:#edf2f7; border-color:#4299e1;">
          file_type
          <div style="font-size:0.8em; color:#4a5568;">.jsonl / .npy / .pkl</div>
        </div>
        <div class="batch-mini" style="background:#f0fff4; border-color:#48bb78;">
          max_len
          <div style="font-size:0.8em; color:#276749;">例如 4096</div>
        </div>
      `;
      visualContent.appendChild(cfg);
    } 
    else if (state === 'vlm') {
        const vlmBox = document.createElement('div');
        vlmBox.innerHTML = `
          <div style="display:flex; align-items:center; gap:20px; background:white; padding:20px; border-radius:10px; border:1px solid #ddd;">
              <div style="text-align:center;">
                  <div style="font-size:40px;">🖼️</div>
                  <div>Raw Image</div>
              </div>
              <div style="font-size:24px;">➜</div>
              <div style="text-align:center;">
                  <div style="display:grid; grid-template-columns:repeat(4, 10px); gap:2px;">
                      ${Array(16).fill(0).map(()=>`<div style="width:10px; height:10px; background:#e67e22;"></div>`).join('')}
                  </div>
                  <div style="font-size:0.8em; color:#e67e22;">tokens_per_image</div>
              </div>
              <div style="font-size:24px;">➜</div>
              <div class="token-box" style="background:#fefcbf;">&lt;image&gt; x N</div>
          </div>
        `;
        visualContent.appendChild(vlmBox);
    }
    else if (state === 'input') {
      const sample = document.createElement('div');
      sample.className = 'data-sample';
      sample.innerHTML = `
        <div class='sample-label'>[ Raw JSONL ]</div>
        <div style='font-family:monospace; font-size:0.8em; background:#f8f9fa; padding:10px;'>
          {"role": "system", "content": "你是有帮助的助手"}<br>
          {"role": "user", "content": "你好"}<br>
          {"role": "assistant", "content": "我是 Cortex"}
        </div>
      `;
      visualContent.appendChild(sample);
    } else if (state === 'tokenize' || state === 'truncate') {
      const row = document.createElement('div');
      row.className = 'token-row';
      const tokens = [
        '<system>', '你是一个有帮助的助手', '</s>',
        '<user>', '你好', '</s>',
        '<assistant>', '<answer>', '我是 Cortex', '</answer>', '</s>',
      ];
      tokens.forEach((t, i) => {
        const unit = document.createElement('div');
        unit.className = 'token-unit';
        const box = document.createElement('div');
        box.className = 'token-box';
        box.innerText = t;

        if (state === 'truncate' && i >= 9) {
          box.style.background = '#fff5f5';
          box.style.borderColor = '#f56565';
          unit.appendChild(box);
          unit.innerHTML += "<small style='color:#e53e3e'>截断</small>";
        } else {
          unit.appendChild(box);
        }
        row.appendChild(unit);
      });
      visualContent.appendChild(row);
    } else if (state === 'mask') {
      const container = document.createElement('div');
      container.style.display = 'flex';
      container.style.flexDirection = 'column';
      container.style.gap = '16px';

      // Case 1: Standard SFT
      const case1 = document.createElement('div');
      case1.className = 'token-row';
      const title1 = document.createElement('div');
      title1.innerText = 'Case 1: Standard SFT (Mask Prompt)';
      title1.style.fontSize = '0.8em';
      title1.style.marginBottom = '4px';
      container.appendChild(title1);
      const tokens1 = [
        '<system>', '...', '</s>',
        '<user>', 'Q', '</s>',
        '<assistant>', '<answer>', 'A1', 'A2', '</answer>', '</s>',
      ];
      
      tokens1.forEach((t, i) => {
        const unit = document.createElement('div');
        unit.className = 'token-unit';
        const box = document.createElement('div');
        box.className = 'token-box';
        box.innerText = t;
        
        const isLoss = (t === '<answer>' || t === 'A1' || t === 'A2' || t === '</answer>' || t === '</s>' && i > 6);
        
        if (!isLoss) {
          box.style.background = '#edf2f7';
          box.style.color = '#a0aec0';
          unit.appendChild(box);
          unit.innerHTML += "<small style='color:#e53e3e'>-100</small>";
        } else {
          box.style.background = '#f0fff4';
          box.style.borderColor = '#48bb78';
          unit.appendChild(box);
          unit.innerHTML += "<small style='color:#2f855a'>Loss</small>";
        }
        case1.appendChild(unit);
      });
      container.appendChild(case1);

      // Case 2: Reasoning SFT
      const title2 = document.createElement('div');
      title2.innerText = 'Case 2: Reasoning SFT (Train on Think + Answer)';
      title2.style.fontSize = '0.8em';
      title2.style.marginBottom = '4px';
      container.appendChild(title2);
      const case2 = document.createElement('div');
      case2.className = 'token-row';
      const tokens2 = [
        '<user>', 'Q', '</s>',
        '<assistant>', '<think>', 'T1', '</think>', '<answer>', 'A1', '</answer>', '</s>',
      ];
      tokens2.forEach((t, i) => {
        const unit = document.createElement('div');
        unit.className = 'token-unit';
        const box = document.createElement('div');
        box.className = 'token-box';
        box.innerText = t;

        const isPrompt = i <= 3;

        if (!isPrompt) {
          box.style.background = '#f0fff4';
          box.style.borderColor = '#48bb78';
          unit.appendChild(box);
          unit.innerHTML += "<small style='color:#2f855a'>Loss</small>";
        } else {
          box.style.background = '#edf2f7';
          box.style.color = '#a0aec0';
          unit.appendChild(box);
          unit.innerHTML += "<small style='color:#e53e3e'>-100</small>";
        }
        case2.appendChild(unit);
      });
      container.appendChild(case2);

      visualContent.appendChild(container);
    } else if (state === 'acc') {
      const accBox = document.createElement('div');
      accBox.style.width = '100%';
      accBox.innerHTML = `
        <div style="display:flex; justify-content:around; gap:10px;">
          <div class="batch-mini active">Batch 1<br>Grad ↑</div>
          <div class="batch-mini active">Batch 2<br>Grad ↑</div>
          <div class="batch-mini">Batch 3<br>Grad ...</div>
          <div class="batch-mini" style="border-style:dashed">Step update</div>
        </div>
        <p style="font-size:0.8em; color:#718096; margin-top:15px; text-align:center;">梯度在参数中累加，Batch 4 后才真正执行 step()</p>
      `;
      visualContent.appendChild(accBox);
    } else if (state === 'checkpoint') {
      const ckpt = document.createElement('div');
      ckpt.innerHTML = `
        <div style="padding:20px; border:2px solid #2d3748; background:#1a202c; color:white; border-radius:10px; text-align:center;">
          <div style="font-size:2em;">💾</div>
          <b>Checkpoint-Step-500</b><br>
          <small>model.safetensors | optimizer.pt</small>
        </div>
      `;
      visualContent.appendChild(ckpt);
    }
    else if (state === 'lr_monitor') {
        const monitor = document.createElement('div');
        monitor.className = 'tensor-box';
        monitor.innerHTML = `
          <div style="display:flex; gap:20px; align-items:center;">
              <div>
                  <div>LR Schedule</div>
                  <div style="width:100px; height:40px; background:linear-gradient(90deg, #e74c3c, #f1c40f, #3498db); border-radius:4px;"></div>
              </div>
              <div>
                  <div>Optimizer State</div>
                  <div style="font-family:monospace; font-size:0.8em;">
                      exp_avg: [0.01, -0.02...]<br>
                      exp_avg_sq: [0.001, 0.004...]
                  </div>
              </div>
          </div>
        `;
        visualContent.appendChild(monitor);
    }
    else if (state === 'eval') {
        const evalBox = document.createElement('div');
        evalBox.className = 'chat-box template-view';
        evalBox.innerHTML = `
          <div><strong>gen.txt</strong> (Rank 0)</div>
          <hr style="margin:5px 0; opacity:0.3;">
          <div style="font-family:monospace; font-size:0.85em;">
              [Epoch 1] Query: 1+1=? <br>
              Gen: 1+1=2. This is a basic arithmetic...<br>
              <br>
              [Epoch 1] Query: Who are you? <br>
              Gen: I am Cortex, a LLM trained by...
          </div>
        `;
        visualContent.appendChild(evalBox);
    }
    else if (state === 'rlhf_init') {
        const initBox = document.createElement('div');
        initBox.style.display='flex'; initBox.style.gap='30px'; initBox.style.justifyContent='center';
        
        initBox.innerHTML = `
          <div style="text-align:center;">
              <div class="batch-mini" style="background:#3498db; color:white;">SFT Weights</div>
              <div style="font-size:24px;">↙ &nbsp; ↘</div>
              <div style="display:flex; gap:20px;">
                  <div class="batch-mini" style="background:#e74c3c; color:white;">Policy Model<br>(Trainable)</div>
                  <div class="batch-mini" style="background:#95a5a6; color:white;">Ref Model<br>(Frozen)</div>
              </div>
          </div>
        `;
        visualContent.appendChild(initBox);
    }
  }

  function renderSim() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; align-items:center; gap:20px; width:100%;">
            <div style="font-family:monospace; font-size:1.2em;">
                Context: "The capital of France is"
            </div>
            
            <div style="display:flex; gap:10px; align-items:center;">
                <div class="arrow">➜</div>
                <div class="token-box" style="background:#fff; border:2px solid #3498db; width:80px;">Model</div>
                <div class="arrow">➜</div>
                <div style="display:flex; flex-direction:column; align-items:center;">
                    <div style="font-size:0.8em; color:#666;">Logits</div>
                    <div style="display:flex; gap:2px;">
                        <div style="width:10px; height:20px; background:#ddd;"></div>
                        <div style="width:10px; height:40px; background:#e74c3c;" title="Paris (High Prob)"></div>
                        <div style="width:10px; height:10px; background:#ddd;"></div>
                    </div>
                </div>
            </div>
            
            <div style="display:flex; gap:20px; align-items:center; background:#f0fff4; padding:15px; border-radius:8px; border:1px solid #48bb78;">
                <div>
                    <strong>Target: " Paris"</strong>
                </div>
                <div style="font-size:20px;">⚡</div>
                <div>
                    <strong>Loss = -log(P(Paris))</strong><br>
                    <span style="color:#276749;">Minimize this to 0</span>
                </div>
            </div>
            
            <div style="font-size:0.9em; color:#555; max-width:400px; text-align:center;">
                SFT 本质上是在每一个位置上做多分类任务。只有 Ground Truth 对应的 Token 会产生梯度信号。
            </div>
        </div>
      `;
  }

  function updateButtons() {
    if (prevBtn) prevBtn.disabled = currentStep === 0;
    if (nextBtn) nextBtn.disabled = currentStep === steps.length - 1;
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

  nextBtn.addEventListener('click', goNext);
  prevBtn.addEventListener('click', goPrev);

  resetBtn.addEventListener('click', () => {
    currentStep = 0;
    updateUI();
  });

  // Decoupled Keyboard Navigation
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') goPrev();
    if (e.key === 'ArrowRight') goNext();
  });

  // Initial render
  updateUI();
});
