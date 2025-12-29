document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');

    if (!prevBtn || !nextBtn || !resetBtn || !infoBox || !visualContent || !codeSnippet) {
        console.error("Required elements not found in Batch Gen script");
        return;
    }

    let currentStep = 0;

    const steps = [
        {
            title: "步骤 0: Left-Padding 对齐 (Batch Prep)",
            description: "在批量生成中，不同长度的 Prompt 必须对齐。生成任务通常使用<b>左填充 (Left-Padding)</b>，因为模型是从右向左自回归生成，这样能保证最新的 token 始终在序列的最右侧，方便取 logits。<span class='step-badge'>utils.py:329</span>",
            code: `padded = pad_sequence(reversed_seqs, batch_first=True, padding_value=pad_id)\nfinal_input = padded.flip(dims=(1,))`,
            state: "pad"
        },
        {
            title: "步骤 1: 动态位置编码 (Position IDs)",
            description: "Left-Padding 引入了大量的 PAD token。为了不让这些无效 token 干扰位置编码，<code>calc_position_ids</code> 会忽略 PAD，只对有效 token 从 0 开始递增计数。<span class='step-badge'>utils.py:128</span>",
            code: `mask_cumsum = attention_mask.long().cumsum(-1)\nposition_ids = (mask_cumsum - 1).masked_fill(mask==0, 0)`,
            state: "pos"
        },
        {
            title: "步骤 2: 缓冲区分配 (Buffer Allocation)",
            description: "为了效率，预先分配一个形状为 <code>[batch, max_new_tokens]</code> 的全 PAD 缓冲区。这避免了每一步都进行 <code>torch.cat</code> 产生的内存碎片和性能开销。<span class='step-badge'>generate_utils.py:600</span>",
            code: `buffer = torch.full((batch, max_new), pad_id, device=device)`,
            state: "alloc"
        },
        {
            title: "步骤 3: 循环生成与 Logits 记录 (Generation Loop)",
            description: "进入自回归循环。每一步只对最新的 <code>current_tokens</code> (1列) 进行前向计算，利用 KV Cache 加速。产生的 logits 被存入 <code>padded_logits</code> 供后续分析。<span class='step-badge'>generate_utils.py:625</span>",
            code: `logits = model(curr_tokens, past_key_values=kv_cache).logits[:, -1, :]\npadded_logits[:, i, :] = logits`,
            state: "loop_start"
        },
        {
            title: "步骤 4: 结束状态追踪 (Done Tracking)",
            description: "维护一个布尔向量 <code>done</code>。一旦某个样本生成了 <code>EOS</code> (End of Sequence) token，该位置被标记为 True。已完成的样本后续不再更新。<span class='step-badge'>generate_utils.py:710</span>",
            code: `new_done = (next_token == eos_token_id)\ndone = done | new_done`,
            state: "done_track"
        },
        {
            title: "步骤 5: 强制 PAD 填充 (Masked Fill)",
            description: "对于已经 <code>done</code> 的样本，虽然模型还在跑（因为 batch 中其他样本可能没停），但我们会强制将其输出替换为 <code>PAD</code>，防止生成无效内容。<span class='step-badge'>generate_utils.py:699</span>",
            code: `next_token = torch.where(done, pad_token, next_token_active)`,
            state: "force_pad"
        },
        {
            title: "步骤 6: Attention Mask 动态更新",
            description: "每一步都会追加一列新的 Mask。对于未完成的样本是 1，已完成的是 0。这确保了后续的 Position ID 计算和 Self-Attention 机制能正确忽略 PAD 部分。<span class='step-badge'>generate_utils.py:718</span>",
            code: `new_mask_bit = (~done).long()\nattention_mask = torch.cat([attention_mask, new_mask_bit], dim=-1)`,
            state: "mask_update"
        },
        {
            title: "步骤 7: 最终拼接 (Final Concat)",
            description: "循环结束后，将原始 Prompt 与生成的缓冲区拼接，得到完整的生成序列。同时返回对应的 Logits 立方体。<span class='step-badge'>generate_utils.py:731</span>",
            code: `final_seq = torch.cat([orig_tokens, generated_buffer], dim=1)\nreturn final_seq, padded_logits`,
            state: "final"
        }
    ];

    function updateUI() {
        if (currentStep < 0) currentStep = 0;
        if (currentStep >= steps.length) currentStep = steps.length - 1;

        const step = steps[currentStep];
        infoBox.innerHTML = `<div class="step-badge">Batch Generation Logic</div><strong>${step.title}</strong><p style="margin-top:10px;">${step.description}</p>`;
        codeSnippet.textContent = step.code;
        
        if (window.hljs) {
            hljs.highlightElement(codeSnippet);
        }

        try {
            render();
        } catch(e) {
            console.error("Render failed", e);
        }

        updateButtons();
    }

    function render() {
        visualContent.innerHTML = '';
        const state = steps[currentStep].state;

        // Mock Batch Data
        const batchData = [
            { id: 0, prompt: ["Hi", "AI"], gen: ["Hello", "!"], len: 2, eos: false },
            { id: 1, prompt: ["A"], gen: ["B", "C", "D"], len: 3, eos: false },
            { id: 2, prompt: ["Yes"], gen: ["Ok", "."], len: 2, eos: true } // Early stop
        ];
        const maxPrompt = 2;
        const maxGen = 3;

        const table = document.createElement('div');
        table.style.display = 'flex';
        table.style.flexDirection = 'column';
        table.style.gap = '10px';

        batchData.forEach((row, rIdx) => {
            const rowDiv = document.createElement('div');
            rowDiv.className = 'sequence-row';
            
            // Render Prompt (Left Padding)
            const pLen = row.prompt.length;
            const padCount = maxPrompt - pLen;
            
            for(let i=0; i<maxPrompt; i++) {
                const isPad = i < padCount;
                const cell = document.createElement('div');
                cell.className = `token-cell ${isPad ? 'pad' : 'prompt'}`;
                cell.innerText = isPad ? "PAD" : row.prompt[i - padCount];
                if (state === 'pos') {
                    const posId = isPad ? 0 : (i - padCount);
                    cell.innerHTML += `<div style="font-size:9px; color:#666;">pos:${posId}</div>`;
                }
                rowDiv.appendChild(cell);
            }

            // Render Generation Buffer
            if (state !== 'pad' && state !== 'pos') {
                const sep = document.createElement('div');
                sep.style.width='2px'; sep.style.background='#ccc';
                rowDiv.appendChild(sep);

                for(let j=0; j<maxGen; j++) {
                    const cell = document.createElement('div');
                    let type = 'alloc';
                    let txt = "";
                    
                    if (state === 'alloc') {
                        txt = "PAD"; type = "pad";
                    } else {
                        // Logic Simulation
                        if (j < row.len) {
                            txt = row.gen[j];
                            type = "gen";
                            if (row.eos && j === row.len-1) { txt = "EOS"; type = "eos"; }
                        } else {
                            // After stop
                            txt = "PAD"; type = "pad";
                            if (state === 'force_pad' || state === 'mask_update') {
                                cell.style.opacity = '0.5';
                            }
                        }
                    }
                    
                    cell.className = `token-cell ${type}`;
                    cell.innerText = txt;
                    
                    if (state === 'mask_update') {
                        const maskVal = (j < row.len) ? 1 : 0;
                        cell.innerHTML += `<div style="font-size:9px; color:#blue;">m:${maskVal}</div>`;
                    }

                    rowDiv.appendChild(cell);
                }
            }
            
            // Status Badge
            if (state === 'done_track' || state === 'force_pad') {
                const status = document.createElement('div');
                const isDone = row.eos; // simplify for visualization
                status.className = 'status-badge ' + (isDone ? 'done' : 'active');
                status.innerText = isDone ? 'DONE' : 'ACTIVE';
                rowDiv.appendChild(status);
            }

            table.appendChild(rowDiv);
        });

        visualContent.appendChild(table);
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
