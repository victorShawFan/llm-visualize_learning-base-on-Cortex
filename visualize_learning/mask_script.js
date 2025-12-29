document.addEventListener('DOMContentLoaded', () => {
    const seqLenInput = document.getElementById('seqLen');
    const padLenInput = document.getElementById('padLen');
    const pastLenInput = document.getElementById('pastLen');
    const docLensInput = document.getElementById('docLens');
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');

    // Guard
    if (!visualContent) {
        console.error("Required elements not found in mask_script");
        return;
    }

    let currentStep = 0;
    let seqLen = 4;
    let padLen = 1;
    let pastLen = 0;
    let docLens = [2, 2];

    const steps = [
        {
            title: "步骤 0: 原始掩码初始化 (Init)",
            description: "首先创建一个全为极小值的矩阵 <code>(tgt_len, tgt_len)</code>。这一步是所有掩码生成的起点。<span class='step-badge'>attention_masks.py:60</span>",
            code: `mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min)`,
            state: "init"
        },
        {
            title: "步骤 1: 因果下三角生成 (Causal Mask)",
            description: "利用 <code>mask_cond</code> 生成下三角矩阵。<code>masked_fill_</code> 将下三角（当前及历史位置）填充为 0（可见），上三角保持极小值（不可见）。<span class='step-badge'>attention_masks.py:97</span>",
            code: `mask_cond = torch.arange(mask.size(-1))
mask.masked_fill_(mask_cond < (mask_cond + 1).view(-1, 1), 0)`,
            state: "causal"
        },
        {
            title: "步骤 2: 缓存长度扩展 (Past KV Extension)",
            description: "如果存在 <code>past_key_values</code>，掩码需要向左扩展。历史 Token 对当前所有 Token 都是可见的，因此拼接一个全 0 矩阵。<span class='step-badge'>attention_masks.py:104</span>",
            code: `if past_key_values_length > 0:
    mask = torch.cat([torch.zeros(tgt_len, past_len), mask], dim=-1)`,
            state: "cache"
        },
        {
            title: "步骤 3: 维度扩展与广播 (Expand & Broadcast)",
            description: "将掩码扩展到 4D <code>[bsz, 1, tgt_len, src_len]</code>。这里利用了 Broadcasting 机制，维度 1 (num_heads) 为 1，表示所有 Head 共享同一掩码。<span class='step-badge'>attention_masks.py:107</span>",
            code: `mask = mask[None, None, :, :].expand(bsz, 1, tgt_len, total_len)`,
            state: "expand_causal"
        },
        {
            title: "步骤 4: Padding 掩码准备 (Pad Mask)",
            description: "处理输入的 <code>attention_mask</code> (通常 1 表示有效，0 表示 Padding)。将其反转并扩展为 4D 形状，Padding 位置填为极小值。<span class='step-badge'>attention_masks.py:26</span>",
            code: `expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
inverted_mask = 1.0 - expanded_mask
return inverted_mask.masked_fill(inverted_mask.to(torch.bool), min_val)`,
            state: "pad"
        },
        {
            title: "步骤 5: 文档 ID 生成 (Doc IDs)",
            description: "为了支持 Packed Sequence 训练，我们根据 EOT 位置生成 Doc IDs：先对 <code>is_eot</code> 做 <code>cumsum</code> 得到 <code>doc_ids_ending</code>，再通过 <code>F.pad</code> 右移一位，让 EOT 本身仍然属于前一个文档。<span class='step-badge'>utils.py:57-70</span>",
            code: `is_eot = (input_ids == tokenizer.end)
doc_ids_ending = torch.cumsum(is_eot, dim=-1)
doc_ids = F.pad(doc_ids_ending[:, :-1], (1, 0), value=0)`,
            state: "doc_ids"
        },
        {
            title: "步骤 6: 文档边界掩码 (Doc Boundary Mask, Query>Key)",
            description: "生成 <code>boundary_mask</code>：当 Query 的 Doc ID 大于 Key 的 Doc ID（<code>id_q &gt; id_k</code>）时，表示 Query 属于更后的文档而 Key 属于更前的文档，此时屏蔽对应注意力，避免跨文档『回看』。<span class='step-badge'>utils.py:71-91</span>",
            code: `query_doc_ids = doc_ids.unsqueeze(2)
key_doc_ids = doc_ids.unsqueeze(1)
boundary_mask = query_doc_ids > key_doc_ids
final_mask = torch.zeros((bsz, seq_len, seq_len), dtype=dtype, device=input_ids.device)
final_mask.masked_fill_(boundary_mask, torch.finfo(dtype).min)
doc_boundary_mask = final_mask.unsqueeze(1)  # (bsz, 1, seq_len, seq_len)`,
            state: "doc_mask"
        },
        {
            title: "步骤 7: 最终叠加 (Final Combination)",
            description: "将 因果掩码 + Padding 掩码 + 文档边界掩码 叠加。由于是加法且掩码值为极小值（负无穷），任意一个掩码生效都会导致该位置被屏蔽。<span class='step-badge'>attention_masks.py:162</span>",
            code: `combined_attention_mask = causal_mask
if pad_mask is not None:
    combined_attention_mask = pad_mask if combined_attention_mask is None else pad_mask + combined_attention_mask
if doc_boundary_mask is not None:
    combined_attention_mask = doc_boundary_mask if combined_attention_mask is None else doc_boundary_mask + combined_attention_mask`,
            state: "final"
        }
    ];

    function init() {
        if (!visualContent) return;
        
        currentStep = 0;
        if (seqLenInput) seqLen = parseInt(seqLenInput.value) || 4;
        if (padLenInput) padLen = parseInt(padLenInput.value) || 1;
        if (pastLenInput) pastLen = parseInt(pastLenInput.value) || 0;
        
        if (docLensInput) {
            const docStr = docLensInput.value || "2,2";
            docLens = docStr.split(',').map(Number);
        }
        
        updateUI();
    }

    function updateUI() {
        const step = steps[currentStep];
        if (infoBox) infoBox.innerHTML = `<div class="step-badge">Iteration 7 - Attention Masking</div><strong>${step.title}</strong><p style="margin-top:10px; font-size:0.95em;">${step.description}</p>`;
        if (codeSnippet) {
            codeSnippet.textContent = step.code;
            if(window.hljs) hljs.highlightElement(codeSnippet);
        }
        
        try {
            render();
        } catch(e) {
            console.error("Render failed", e);
        }
        updateButtons();
    }

    function updateButtons() {
        if(prevBtn) prevBtn.disabled = currentStep === 0;
        if(nextBtn) nextBtn.disabled = currentStep === steps.length - 1;
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

    function render() {
        if (!visualContent) return;
        visualContent.innerHTML = '';
        const state = steps[currentStep].state;
        const totalLen = seqLen + pastLen;

        if (state === "init") {
            const grid = createGrid(seqLen, seqLen);
            addGridHeaders(grid, seqLen, seqLen);
            fillGrid(grid, seqLen, seqLen, () => ({text: "-inf", cls: "masked", tooltip: "Initial State: All Masked"}));
            visualContent.appendChild(createLabel(`Initial Matrix [${seqLen}x${seqLen}]`));
            visualContent.appendChild(grid);
        }
        else if (state === "causal") {
            const grid = createGrid(seqLen, seqLen);
            addGridHeaders(grid, seqLen, seqLen);
            fillGrid(grid, seqLen, seqLen, (i, j) => {
                const isMasked = j > i;
                return {
                    text: isMasked ? "-inf" : "0",
                    cls: isMasked ? "masked" : "valid",
                    tooltip: isMasked ? `Masked: j(${j}) > i(${i}) (Future Token)` : `Visible: j(${j}) <= i(${i}) (Past/Current)`
                };
            });
            visualContent.appendChild(createLabel(`Causal Lower Triangular [${seqLen}x${seqLen}]`));
            visualContent.appendChild(grid);
        }
        else if (state === "cache") {
            const grid = createGrid(seqLen, totalLen);
            addGridHeaders(grid, seqLen, totalLen);
            fillGrid(grid, seqLen, totalLen, (i, j) => {
                if (j < pastLen) return {text: "0", cls: "past", tooltip: "Visible: Past Key/Value Cache"};
                const currJ = j - pastLen;
                const isMasked = currJ > i;
                return {
                    text: isMasked ? "-inf" : "0",
                    cls: isMasked ? "masked" : "valid",
                    tooltip: isMasked ? `Masked: Future Token` : `Visible`
                };
            });
            visualContent.appendChild(createLabel(`Extended with Past KV [${seqLen}x${totalLen}]`));
            visualContent.appendChild(grid);
        }
        else if (state === "expand_causal") {
            const wrapper = document.createElement('div');
            wrapper.style.display='flex'; wrapper.style.gap='20px'; wrapper.style.alignItems='center';
            
            const dimBox = document.createElement('div');
            dimBox.className = "tensor-box";
            dimBox.innerHTML = `
                <div style="font-family:monospace; margin-bottom:5px;">[Batch, 1, Tgt, Src]</div>
                <div class="dim-viz">
                    <div style="border:2px solid var(--primary-color); padding:10px; border-radius:8px;">
                        Batch
                        <div style="border:2px dashed #ecc94b; padding:5px; margin-top:5px; border-radius:4px;">
                            Head (Broadcast 1 -> N)
                            <div style="border:2px solid #48bb78; padding:5px; margin-top:5px; background:white; border-radius:4px;">
                                ${seqLen} x ${totalLen} Mask
                            </div>
                        </div>
                    </div>
                </div>
            `;
            wrapper.appendChild(dimBox);
            visualContent.appendChild(createLabel("Broadcasting to 4D Tensor"));
            visualContent.appendChild(wrapper);
        }
        else if (state === "pad") {
            const grid = createGrid(1, totalLen);
            for(let j=0; j<totalLen; j++) {
                const isPad = j >= (totalLen - padLen);
                grid.appendChild(createCell(
                    isPad ? "-inf" : "0", 
                    isPad ? "masked-pad" : "valid", 
                    isPad ? "Masked: Padding Token" : "Visible: Real Token"
                ));
            }
            visualContent.appendChild(createLabel("Padding Mask (Broadcasts to all queries)"));
            visualContent.appendChild(grid);
        }
        else if (state === "doc_ids") {
            const container = document.createElement('div');
            container.style.display = 'flex'; container.style.flexDirection = 'column'; container.style.gap = '10px';
            
            const row = document.createElement('div');
            row.style.display = 'flex'; row.style.gap = '5px';
            
            let currentDocId = 0;
            let docMap = []; 
            
            for(let len of docLens) {
                for(let k=0; k<len; k++) {
                    docMap.push(currentDocId);
                }
                currentDocId++;
            }
            while(docMap.length < seqLen) docMap.push(currentDocId);
            docMap = docMap.slice(0, seqLen);

            docMap.forEach((id, idx) => {
                const cell = createCell(`T${idx}`, 'valid', `Token ${idx} belongs to Doc ${id}`);
                cell.style.flexDirection='column';
                cell.innerHTML = `<span style="font-size:0.7em">Tok${idx}</span><strong style="font-size:1.2em">ID:${id}</strong>`;
                cell.style.background = id % 2 === 0 ? '#e6fffa' : '#fff5f5';
                cell.style.borderColor = id % 2 === 0 ? '#38b2ac' : '#fc8181';
                row.appendChild(cell);
            });
            
            container.appendChild(row);
            visualContent.appendChild(createLabel("Generated Document IDs (from cumsum)"));
            visualContent.appendChild(container);
        }
        else if (state === "doc_mask") {
            let currentDocId = 0;
            let docMap = []; 
            for(let len of docLens) {
                for(let k=0; k<len; k++) docMap.push(currentDocId);
                currentDocId++;
            }
            while(docMap.length < seqLen) docMap.push(currentDocId);
            docMap = docMap.slice(0, seqLen);

            const grid = createGrid(seqLen, totalLen);
            addGridHeaders(grid, seqLen, totalLen);
            fillGrid(grid, seqLen, totalLen, (i, j) => {
                if (j < pastLen) return {text: "0", cls: "past", tooltip: "Past KV assumed same doc or valid"};
                
                const qIdx = i;
                const kIdx = j - pastLen;
                
                if (kIdx >= seqLen) return {text: "0", cls: "valid", tooltip: "Padding area (ignored here)"};

                const idQ = docMap[qIdx];
                const idK = docMap[kIdx];
                
                const isBlocked = idQ > idK;
                return {
                    text: isBlocked ? "-inf" : "0",
                    cls: isBlocked ? "masked-doc" : "valid",
                    tooltip: isBlocked ? `Masked: Query(Doc ${idQ}) > Key(Doc ${idK})` : `Visible: Same Document or Key is Future (handled by Causal)`
                };
            });
            
            visualContent.appendChild(createLabel("Document Boundary Mask (Block if ID_Q > ID_K)"));
            visualContent.appendChild(grid);
        }
        else if (state === "final") {
            let docMap = []; let cid=0;
            for(let len of docLens) { for(let k=0; k<len; k++) docMap.push(cid); cid++; }
            while(docMap.length < seqLen) docMap.push(cid);
            docMap = docMap.slice(0, seqLen);

            const grid = createGrid(seqLen, totalLen);
            addGridHeaders(grid, seqLen, totalLen);
            
            fillGrid(grid, seqLen, totalLen, (i, j) => {
                const isPad = j >= (totalLen - padLen);
                
                let isCausal = false;
                let isDocDiff = false;
                
                if (j < pastLen) {
                    isCausal = false; 
                } else {
                    const kIdx = j - pastLen;
                    isCausal = kIdx > i;
                    if (kIdx < seqLen) {
                        isDocDiff = docMap[i] > docMap[kIdx];
                    }
                }

                let val = "0";
                let cls = "valid";
                let tooltip = "Visible";
                
                if (isPad) {
                    val = "PAD"; cls = "masked-pad"; tooltip="Masked by Padding Mask";
                } else if (isCausal) {
                    val = "CAU"; cls = "masked-causal"; tooltip="Masked by Causal Mask (Future)";
                } else if (isDocDiff) {
                    val = "DOC"; cls = "masked-doc"; tooltip="Masked by Doc Boundary (Cross-Doc)";
                } else if (j < pastLen) {
                    cls = "past"; tooltip="Visible Past KV";
                }
                
                return {text: val, cls: cls, tooltip: tooltip};
            });
            visualContent.appendChild(createLabel("Combined Attention Mask (Priority: Padding > Causal > Doc Boundary)"));
            visualContent.appendChild(grid);

            const legend = document.createElement('div');
            legend.style.marginTop = '15px'; legend.style.fontSize = '0.8em'; legend.style.display='flex'; legend.style.gap='10px';
            legend.innerHTML = `
                <div><span style="display:inline-block; width:12px; height:12px; background:#f8d7da; border:1px solid #f5c6cb;"></span> Causal</div>
                <div><span style="display:inline-block; width:12px; height:12px; background:#fff3cd; border:1px solid #ffeeba;"></span> Padding</div>
                <div><span style="display:inline-block; width:12px; height:12px; background:#e2d9f3; border:1px solid #d1c4e9;"></span> Doc Boundary</div>
            `;
            visualContent.appendChild(legend);
        }
    }

    function createGrid(rows, cols) {
        const grid = document.createElement('div');
        grid.style.display = 'grid'; grid.style.gap = '4px';
        grid.style.gridTemplateColumns = `30px repeat(${cols}, 45px)`;
        grid.style.background = 'white'; grid.style.padding='10px'; grid.style.borderRadius='12px'; grid.style.boxShadow='var(--shadow-md)';
        grid.style.overflowX = 'auto';
        return grid;
    }

    function addGridHeaders(grid, rows, cols) {
        grid.appendChild(createCell("", "header"));
        for(let j=0; j<cols; j++) {
            const c = createCell(j.toString(), "header");
            c.style.height = '30px';
            grid.appendChild(c);
        }
    }

    function fillGrid(grid, rows, cols, cellFn) {
        for(let i=0; i<rows; i++) {
            const rh = createCell(i.toString(), "header");
            rh.style.width = '30px';
            grid.appendChild(rh);
            
            for(let j=0; j<cols; j++) {
                const res = cellFn(i, j);
                grid.appendChild(createCell(res.text, res.cls, res.tooltip));
            }
        }
    }

    function createCell(text, cls, tooltip) {
        const cell = document.createElement('div');
        cell.style.width = '45px'; cell.style.height = '45px'; 
        cell.style.display = 'flex'; cell.style.alignItems = 'center'; cell.style.justifyContent = 'center';
        cell.style.borderRadius = '6px'; cell.style.fontSize = '0.75em'; cell.style.fontFamily='monospace';
        cell.innerText = text;
        cell.title = tooltip || "";
        cell.style.cursor = tooltip ? 'help' : 'default';
        
        if (cls === "valid") { cell.style.background = 'var(--primary-light)'; cell.style.color = 'var(--primary-color)'; }
        if (cls === "masked") { cell.style.background = '#4a5568'; cell.style.color = '#fff'; }
        if (cls === "masked-causal") { cell.style.background = '#f8d7da'; cell.style.color = '#721c24'; }
        if (cls === "masked-pad") { cell.style.background = '#fff3cd'; cell.style.color = '#856404'; }
        if (cls === "masked-doc") { cell.style.background = '#e2d9f3'; cell.style.color = '#5e35b1'; }
        if (cls === "past") { cell.style.background = '#f0fff4'; cell.style.color = '#38a169'; }
        if (cls === "header") { cell.style.background = 'transparent'; cell.style.color = '#a0aec0'; cell.style.fontWeight = 'bold'; }
        
        return cell;
    }

    function createLabel(txt) {
        const l = document.createElement('div'); l.style.margin = '15px 0'; l.style.fontWeight = '700'; l.style.color = 'var(--text-main)'; l.innerText = txt;
        return l;
    }

    // Bind events
    if(nextBtn) nextBtn.addEventListener('click', goNext);
    if(prevBtn) prevBtn.addEventListener('click', goPrev);
    if(resetBtn) resetBtn.addEventListener('click', init);
    if(seqLenInput) seqLenInput.addEventListener('change', init);
    if(padLenInput) padLenInput.addEventListener('change', init);
    if(pastLenInput) pastLenInput.addEventListener('change', init);
    if(docLensInput) docLensInput.addEventListener('change', init);

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });

    // Init
    init();
});
