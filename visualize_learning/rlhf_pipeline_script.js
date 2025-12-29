document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');
    
    // Guard
    if (!visualContent) {
        console.error("Required elements not found in rlhf_pipeline_script");
        return;
    }

    let currentStep = 0;

    // RLHF 总览：从 SFT → DPO → PPO / GRPO
    const steps = [
      {
        title: 'Phase 0: 预训练基座与 Tokenizer',
        description:
          '一切从通用 LLM 基座与 Tokenizer 开始：预训练权重通过 <code>Trainer._new_model</code> 与 <code>TrainConfig.model_config</code> 加载；<code>TrainerTools().tokenizer</code> 则作为全局入口被 SFT / DPO / PPO / GRPO 复用。<br><span class="step-badge">trainer.py:78-139, tokenizer.py:16-83</span>',
        code: 'self.model = self._new_model(self.train_config)\nself.tokenizer = Tokenizer()  # TrainerTools 全局单例',
        render: () => renderPhase0(),
      },
      {
        title: 'Phase 1: SFT 监督微调 (Supervised Fine-Tuning)',
        description:
          '使用带有 System / User / Assistant 标签的指令数据做监督微调：<code>SFTDataset</code> 负责从 jsonl / npy / pkl 读取样本，<code>SFTTrainer</code> 通过 <code>get_sft_collate_fn</code> 调用 <code>_mask_prompt</code> 只在 &lt;think&gt; / &lt;answer&gt; 区域计算交叉熵。<br><span class="step-badge">sft_trainer.py:15-60, dataset.py:SFTDataset, utils.py:get_sft_collate_fn</span>',
        code: 'trainer = SFTTrainer(train_config, eval_prompts)\nloss = LMLoss(ignore_index=-100)(logits, labels_masked)',
        render: () => renderSFT(),
      },
      {
        title: 'Phase 2: SFT 模型作为 RLHF 初始化 (Policy / Ref)',
        description:
          'SFT 结束后会把权重同时拷贝给 Policy 与 Ref：Policy 继续参与 DPO / PPO / GRPO 的更新，Ref 保持冻结，用于 KL 或相对 logprob 约束。<br><span class="step-badge">ppo_trainer.py:120-127, dpo_trainer.py:66-72, grpo_trainer.py:93-109</span>',
        code: 'policy_model.load_state_dict(sft_ckpt)\nref_model.load_state_dict(sft_ckpt)\nfor p in ref_model.parameters(): p.requires_grad = False',
        render: () => renderCopy(),
      },
      {
        title: 'Phase 3: DPO 偏好对齐 (Direct Preference Optimization)',
        description:
          '使用 DPODataset 提供的 (prompt, chosen, rejected) 对进行偏好学习：<code>DPOTrainer</code> 并行运行 Policy 与 Ref，对 chosen / rejected 计算 log 概率差，再送入 <code>DPOLoss</code> 或 IPO 变体。<br><span class="step-badge">dpo_trainer.py:28-90,156-190, loss.py:151-230</span>',
        code: 'pi_c, pi_r = logπ(policy, chosen/reject)\nref_c, ref_r = logπ(ref, chosen/reject)\nlogits = (pi_c - pi_r) - (ref_c - ref_r)\nloss = DPOLoss(beta, label_smoothing, ipo)(logits)',
        render: () => renderDPO(),
      },
      {
        title: 'Phase 4: PPO-RM 强化阶段 (Policy + Value + Ref + RM)',
        description:
          '在 PPO 配置中，Cortex 使用 <code>ValueModel</code> 包装 base model 输出 V(s)，并维护一个 Ref Model。通过 <code>batch_generate</code> 采样 rollout，再用外部 Reward Model 打分，结合 KL 惩罚与 GAE 计算优势，送入 <code>PPOLoss</code>。<br><span class="step-badge">ppo_trainer.py:31-190, loss.py:233-283</span>',
        code: 'policy_out, values = policy_model(...), value_model(...)\nref_out = ref_model(...)\nrewards = env_reward + kl_penalty\nadvantages, returns = compute_gae(rewards, values, last_values)\nloss = PPOLoss(clip_eps, vf_coef)(log_probs, old_log_probs, values, old_values, returns, advantages)',
        render: () => renderPPO(),
      },
      {
        title: 'Phase 5: GRPO 群体相对优势 (Group Relative Policy Optimization)',
        description:
          'GRPO 移除了 Value 网络，改为对每个 Prompt 一次性采样 group_size 个回答，并使用组内标准化奖励作为优势。Ref Model 仍用于 KL 与重要性采样，损失由 <code>GRPOLoss</code> 控制。<br><span class="step-badge">grpo_trainer.py:48-152, loss.py:359-386</span>',
        code: 'rewards = reward_func(prompts, completions)\navg, std = group_mean_std(rewards, group_size)\nadvantages = (rewards - avg) / (std + 1e-4)\nloss = GRPOLoss(...)(log_probs, old_log_probs, advantages, ref_log_probs)',
        render: () => renderGRPO(),
      },
      {
        title: 'Phase 6: Loss 家族与 TrainConfig 选择',
        description:
          '不同阶段复用同一套 Trainer 基础设施，仅通过 <code>train_type</code>、<code>loss_config</code> 与专用 config 切换 Loss 家族：SFT → LMLoss/KDLoss, DPO → DPOLoss, PPO → PPOLoss, GRPO → GRPOLoss。<br><span class="step-badge">train_configs.py, loss.py</span>',
        code: 'if train_type == "sft": loss = LMLoss(... )\nelif train_type == "dpo": loss = DPOLoss(...)\nelif train_type == "ppo": loss = PPOLoss(...)\nelif train_type == "grpo": loss = GRPOLoss(...)',
        render: () => renderConfigPanel(),
      },
      {
        title: 'Phase 7: 整体 Pipeline 思维导图',
        description:
          '最终可以把整个 Cortex RLHF 流程看成一条时间轴：预训练 → SFT → (可选) DPO → PPO/GRPO。每一阶段都共享同一套并行/调度/日志基础设施，只切换 Trainer 与 Loss。',
        code: 'Pretrain → SFT → [DPO] → [PPO / GRPO]\n# train_configs.py 中通过 train_type / rlhf_config 进行组合',
        render: () => renderSummary(),
      },
    ];

    function updateUI() {
      const step = steps[currentStep];
      if (infoBox) infoBox.innerHTML = `<div class="step-badge">Phase ${currentStep}</div><strong>${step.title}</strong><br>${step.description}`;
      if (codeSnippet) {
          codeSnippet.textContent = step.code;
          if(window.hljs) hljs.highlightElement(codeSnippet);
      }
      visualContent.innerHTML = '';
      visualContent.className = 'rlhf-pipeline-viz fade-in';
      try {
          step.render();
      } catch(e) {
          console.error("Render failed", e);
      }
      updateButtons();
    }

    function renderPhase0() {
      const wrap = document.createElement('div');
      wrap.style.display = 'flex';
      wrap.style.flexDirection = 'column';
      wrap.style.alignItems = 'center';
      wrap.style.gap = '20px';

      const top = document.createElement('div');
      top.style.display = 'flex';
      top.style.gap = '20px';

      top.innerHTML = `
        <div class="model-box policy" style="border-color:#4a5568; background:#edf2f7;">Pretrained LLM<br><span style="font-size:0.8em;opacity:0.8;">Base Weights</span></div>
        <div class="model-box value" style="border-color:#4a5568; background:#edf2f7;">Tokenizer<br><span style="font-size:0.8em;opacity:0.8;">Vocab & Template</span></div>
      `;

      const bottom = document.createElement('div');
      bottom.className = 'dict-viz';
      bottom.innerHTML = `
        <div class="dict-entry" style="max-width:420px; text-align:center;">
          TrainConfig.model_config → 模型结构 / ckpt 路径<br>
          TrainerTools().tokenizer → chat_template / encode / decode
        </div>
      `;

      wrap.appendChild(top);
      wrap.appendChild(bottom);
      visualContent.appendChild(wrap);
    }

    function renderSFT() {
      const wrap = document.createElement('div');
      wrap.style.display = 'flex';
      wrap.style.flexDirection = 'column';
      wrap.style.gap = '20px';
      wrap.style.alignItems = 'center';

      wrap.innerHTML = `
        <div class="dict-viz">
          <div class="dict-entry">Dataset → <b>SFTDataset</b><br><span style="font-size:0.8em;">jsonl / npy / pkl, max_seq_len, image_tags</span></div>
          <div class="dict-entry">Trainer → <b>SFTTrainer</b><br><span style="font-size:0.8em;">get_sft_collate_fn(mask_prompt) → _mask_prompt(labels == -100)</span></div>
        </div>
        <div class="tensor-viz">
          <div class="tensor-cell" style="background:#e2e8f0; color:#a0aec0;">Prompt (-100)</div>
          <div class="tensor-cell" style="background:#c6f6d5; border-color:#48bb78;">&lt;think&gt; (Loss)</div>
          <div class="tensor-cell" style="background:#c6f6d5; border-color:#48bb78;">&lt;answer&gt; (Loss)</div>
        </div>
        <div class="reward-calc" style="max-width:420px;">
          <div class="reward-row">
            <div class="r-label">Prompt 区域</div>
            <div class="r-bar negative" style="background:#e2e8f0;color:#718096; width:100px;">Masked</div>
          </div>
          <div class="reward-row">
            <div class="r-label">Think / Answer 区域</div>
            <div class="r-bar positive" style="width:180px;">CrossEntropy Loss</div>
          </div>
        </div>
      `;

      visualContent.appendChild(wrap);
    }

    function renderCopy() {
      const wrap = document.createElement('div');
      wrap.style.display = 'flex';
      wrap.style.flexDirection = 'column';
      wrap.style.gap = '25px';
      wrap.style.alignItems = 'center';

      wrap.innerHTML = `
        <div class="seq-viz">
          <div class="token prompt" style="background:#3182ce; color:white;">SFT Checkpoint</div>
          <div class="arrow">⤡ ⤢</div>
          <div style="display:flex; gap:40px;">
              <div class="token gen" style="background:#e53e3e; color:white;">Policy (Train)</div>
              <div class="token gen" style="background:#718096; color:white;">Ref (Frozen)</div>
          </div>
        </div>
        <div style="text-align:center; color:#4a5568; font-size:0.9em;">
            Ref 模型用于计算 KL 散度或 DPO 的概率比率，防止策略崩塌。
        </div>
      `;

      visualContent.appendChild(wrap);
    }

    function renderDPO() {
      const wrap = document.createElement('div');
      wrap.style.display = 'flex';
      wrap.style.flexDirection = 'column';
      wrap.style.gap = '20px';
      wrap.style.alignItems = 'center';

      wrap.innerHTML = `
        <div class="dict-viz">
          <div class="dict-entry">Dataset → <b>DPODataset</b> (prompt, chosen, rejected)</div>
          <div class="dict-entry">Trainer → <b>DPOTrainer</b> 调用 <code>_logprobs</code> 聚合 token logP</div>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; width:100%; max-width:520px;">
          <div class="box" style="border-radius:12px; padding:16px; border:1px solid #e53e3e; background:#fff5f5;">
            <div style="font-weight:700; margin-bottom:8px; color:#c53030;">Policy Model</div>
            chosen: -12.5<br>rejected: -15.2
          </div>
          <div class="box" style="border-radius:12px; padding:16px; border:1px solid #718096; background:#edf2f7;">
            <div style="font-weight:700; margin-bottom:8px; color:#4a5568;">Ref Model</div>
            chosen: -13.0<br>rejected: -14.8
          </div>
        </div>
        <div class="math-row" style="background:#fff; padding:10px; border-radius:8px; border:1px dashed #cbd5e0;">
            logits = (π_c − π_r) − (π_ref_c − π_ref_r)
        </div>
      `;

      visualContent.appendChild(wrap);
    }

    function renderPPO() {
      const wrap = document.createElement('div');
      wrap.style.display = 'flex';
      wrap.style.flexDirection = 'column';
      wrap.style.gap = '25px';
      wrap.style.alignItems = 'center';

      wrap.innerHTML = `
        <div style="display:flex; gap:20px;">
          <div class="model-box policy">Policy (π)</div>
          <div class="model-box value">Value (V)</div>
          <div class="model-box ref">Ref (π_ref)</div>
        </div>
        <div class="reward-calc">
          <div class="reward-row">
            <div class="r-label">Env Reward</div>
            <div class="r-bar positive" style="width:150px;">From Reward Model</div>
          </div>
          <div class="reward-row">
            <div class="r-label">KL Penalty</div>
            <div class="r-bar negative" style="width:80px;">-β · KL</div>
          </div>
          <div class="sum-line"></div>
          <div style="font-size:0.85em; color:#4a5568; text-align:center; padding:10px;">
            R_t = R_env + R_KL ➜ GAE(γ, λ) ➜ PPOLoss
          </div>
        </div>
      `;

      visualContent.appendChild(wrap);
    }

    function renderGRPO() {
      const wrap = document.createElement('div');
      wrap.style.display = 'flex';
      wrap.style.flexDirection = 'column';
      wrap.style.gap = '20px';
      wrap.style.alignItems = 'center';

      wrap.innerHTML = `
        <div class="prompt-box">Prompt × group_size = 4</div>
        <div class="completion-row" style="gap:10px;">
          <div class="comp-card">C1 (0.8)</div>
          <div class="comp-card">C2 (0.9)</div>
          <div class="comp-card">C3 (0.4)</div>
          <div class="comp-card">C4 (0.7)</div>
        </div>
        <div class="adv-results" style="gap:10px;">
          <div class="adv-card positive"><div class="id">C1</div><div class="res">A≈+0.46</div></div>
          <div class="adv-card positive"><div class="id">C2</div><div class="res">A≈+0.92</div></div>
          <div class="adv-card negative"><div class="id">C3</div><div class="res">A≈-1.38</div></div>
          <div class="adv-card neutral"><div class="id">C4</div><div class="res">A≈0.00</div></div>
        </div>
        <div style="font-size:0.8em; color:#666;">No Critic Model Needed</div>
      `;

      visualContent.appendChild(wrap);
    }

    function renderConfigPanel() {
      const wrap = document.createElement('div');
      wrap.style.display = 'flex';
      wrap.style.gap = '20px';
      wrap.style.flexWrap = 'wrap';
      wrap.style.justifyContent = 'center';

      const types = [
        { name: 'SFT', desc: 'LMLoss + (Optional KDLoss)', color: '#3182ce' },
        { name: 'DPO', desc: 'DPOLoss / IPO', color: '#ed8936' },
        { name: 'PPO', desc: 'PPOLoss + ValueModel', color: '#38a169' },
        { name: 'GRPO', desc: 'GRPOLoss (group_size)', color: '#805ad5' },
      ];

      types.forEach(t => {
        const card = document.createElement('div');
        card.className = 'dict-entry';
        card.style.borderLeft = `4px solid ${t.color}`;
        card.style.background = '#fff';
        card.style.boxShadow = '0 2px 4px rgba(0,0,0,0.05)';
        card.innerHTML = `<strong>${t.name}</strong><br><span style="font-size:0.85em;">${t.desc}</span>`;
        wrap.appendChild(card);
      });

      visualContent.appendChild(wrap);
    }

    function renderSummary() {
      const wrap = document.createElement('div');
      wrap.style.display = 'flex';
      wrap.style.flexDirection = 'column';
      wrap.style.gap = '20px';
      wrap.style.alignItems = 'center';

      wrap.innerHTML = `
        <div class="seq-viz" style="justify-content:center;">
          <div class="token prompt" style="background:#4a5568; color:white;">Pretrain</div>
          <div class="arrow">➜</div>
          <div class="token gen" style="background:#3182ce; color:white;">SFT</div>
          <div class="arrow">➜</div>
          <div class="token gen" style="background:#ed8936; color:white;">DPO</div>
          <div class="arrow">➜</div>
          <div class="token gen" style="background:#38a169; color:white;">PPO / GRPO</div>
        </div>
        <div class="dict-viz">
          <div class="dict-entry" style="max-width:520px; text-align:center; background:#ebf8ff; border-color:#3182ce;">
            同一套 Trainer 基础设施：<br>
            并行 (parallel.py) • 调度器 (scheduler.py) • 日志 (log.py) • 采样 (generate_utils.py)<br>
            只通过 <code>train_type</code> 与各自 config 切换具体对齐算法。
          </div>
        </div>
      `;

      visualContent.appendChild(wrap);
    }

    function updateButtons() {
      if(prevBtn) prevBtn.disabled = currentStep === 0;
      if(nextBtn) nextBtn.disabled = currentStep === steps.length - 1;
    }

    if(nextBtn) nextBtn.addEventListener('click', () => {
      if (currentStep < steps.length - 1) {
        currentStep++;
        updateUI();
      }
    });

    if(prevBtn) prevBtn.addEventListener('click', () => {
      if (currentStep > 0) {
        currentStep--;
        updateUI();
      }
    });

    if(resetBtn) resetBtn.addEventListener('click', () => {
      currentStep = 0;
      updateUI();
    });

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft') goPrev();
      if (e.key === 'ArrowRight') goNext();
    });

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

    updateUI();
});
