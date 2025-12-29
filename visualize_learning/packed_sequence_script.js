document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');

    if (!visualContent) {
        console.error("Required elements not found in packed_sequence_script");
        return;
    }

    let currentStep = 0;

    const steps = [
        {
            title: "场景: 短文档浪费计算资源",
            description: "假设模型 Context Window = 8。如果我们直接训练两个短文档（长度分别为3和4），通常需要 Padding 到 8，导致大量无效计算（Padding 部分的 Loss 被 Mask，但 Attention 计算仍在进行）。",
            code: "Batch 1: [A, A, A, P, P, P, P, P]  -> 5/8 浪费\nBatch 2: [B, B, B, B, P, P, P, P]  -> 4/8 浪费",
            render: () => renderNaivePadding()
        },
        {
            title: "解决方案: Sequence Packing",
            description: "将多个文档拼接在同一个 Sequence 中，中间用 <code>EOT</code> (End Of Text) 分隔。这样可以填满 Context Window，极大提升吞吐量 (Throughput)。",
            code: "Packed: [A, A, A, EOT, B, B, B, B] -> 0 浪费\n# utils.py: pack_sequences",
            render: () => renderPacking()
        },
        {
            title: "问题 1: 跨文档注意力 (Cross-Document Attention)",
            description: "在标准 Self-Attention 中，后面的 Token 能看到前面所有 Token。这意味着文档 B 的 Token 会“偷看”到文档 A 的内容，这会污染上下文，导致模型混淆。",
            code: "Attention(B, A) != 0  =>  Error!",
            render: () => renderCrossDocIssue()
        },
        {
            title: "解决 1: Block-Diagonal Mask",
            description: "我们需要构建一个特殊的 Attention Mask。对于每个 Token，先计算它属于哪个文档 ID。只有当 Query 和 Key 属于同一个文档（且满足因果性）时，Mask 才为 0，否则为 -inf。<span class='step-badge'>utils.py:65-75</span>",
            code: `is_eot = (input_ids == tokenizer.end)\ndoc_ids = torch.cumsum(is_eot, dim=-1)\nmask = (doc_ids_q == doc_ids_k) & causal_mask`,
            render: () => renderBlockMask()
        },
        {
            title: "问题 2: 位置编码 (Position Encoding)",
            description: "如果直接使用全局索引 [0, 1, 2, 3, 4, 5, 6, 7]，文档 B 的第一个 Token 位置就是 4。这不对！对于文档 B 来说，它应该是开头 (Pos 0)。",
            code: "Global Pos: [0, 1, 2, 3, 4, 5, 6, 7]\nExpected:   [0, 1, 2, 3, 0, 1, 2, 3]",
            render: () => renderPosIssue()
        },
        {
            title: "解决 2: Reset Position IDs",
            description: "我们检测 EOT 标记。每当遇到 EOT，下一时刻的 Position ID 重置为 0。这样模型就能正确感知每个文档的相对位置结构。<span class='step-badge'>utils.py:100-115</span>",
            code: `is_reset = (input_ids == eot)\npos_ids = ... # reset accumulator`,
            render: () => renderPosReset()
        },
        {
            title: "实战模拟: 完整流程 (Simulation)",
            description: "让我们动态模拟一次打包过程。输入两个文档，系统将生成 Input IDs, Position IDs 和 Attention Mask。",
            code: "Input: Doc A (Len 2), Doc B (Len 3)\nTarget: Seq Len 8",
            render: () => renderSimulation()
        }
    ];

    function updateUI() {
        if (currentStep < 0) currentStep = 0;
        if (currentStep >= steps.length) currentStep = steps.length - 1;

        const step = steps[currentStep];
        if (infoBox) infoBox.innerHTML = `<strong>${step.title}</strong><p>${step.description}</p>`;
        if (codeSnippet) codeSnippet.textContent = step.code;

        visualContent.innerHTML = '';
        step.render();

        updateButtons();
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

    // --- Renderers ---

    function createGrid(rows, cols, cellCallback) {
        const grid = document.createElement('div');
        grid.style.display = 'grid';
        grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
        grid.style.gap = '5px';
        grid.style.marginBottom = '10px';
        
        for (let i = 0; i < rows * cols; i++) {
            const r = Math.floor(i / cols);
            const c = i % cols;
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.style.width = '30px';
            cell.style.height = '30px';
            cell.style.border = '1px solid #ccc';
            cell.style.display = 'flex';
            cell.style.alignItems = 'center';
            cell.style.justifyContent = 'center';
            cell.style.fontSize = '10px';
            cellCallback(cell, r, c, i);
            grid.appendChild(cell);
        }
        return grid;
    }

    function renderNaivePadding() {
        const container = document.createElement('div');
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.gap = '10px';

        const row1 = document.createElement('div');
        row1.innerHTML = '<div>Seq 1 (Pad=5)</div>';
        row1.appendChild(createGrid(1, 8, (cell, r, c) => {
            if (c < 3) { cell.innerText = 'A'; cell.style.background = '#d5f5e3'; }
            else { cell.innerText = 'P'; cell.style.background = '#eee'; cell.style.color = '#999'; }
        }));

        const row2 = document.createElement('div');
        row2.innerHTML = '<div>Seq 2 (Pad=4)</div>';
        row2.appendChild(createGrid(1, 8, (cell, r, c) => {
            if (c < 4) { cell.innerText = 'B'; cell.style.background = '#d6eaf8'; }
            else { cell.innerText = 'P'; cell.style.background = '#eee'; cell.style.color = '#999'; }
        }));

        container.appendChild(row1);
        container.appendChild(row2);
        visualContent.appendChild(container);
    }

    function renderPacking() {
        const container = document.createElement('div');
        container.innerHTML = '<div>Packed Sequence (Utilization=100%)</div>';
        
        // A=3, EOT=1, B=4
        const data = ['A','A','A','EOT','B','B','B','B'];
        
        container.appendChild(createGrid(1, 8, (cell, r, c) => {
            const val = data[c];
            cell.innerText = val;
            if (val === 'A') cell.style.background = '#d5f5e3';
            else if (val === 'B') cell.style.background = '#d6eaf8';
            else cell.style.background = '#fadbd8'; // EOT
        }));
        
        visualContent.appendChild(container);
    }

    function renderCrossDocIssue() {
        visualContent.innerHTML = `
            <div style="text-align:center;">
                <div style="font-size:24px; margin-bottom:10px;">Standard Causal Mask</div>
                <canvas id="bad-mask" width="200" height="200" style="border:1px solid #333;"></canvas>
                <div style="color:#e74c3c; margin-top:10px;">Red Zone: Doc B attending to Doc A (Data Leakage!)</div>
            </div>
        `;
        
        // Draw triangular mask
        setTimeout(() => {
            const canvas = document.getElementById('bad-mask');
            if(canvas) {
                const ctx = canvas.getContext('2d');
                const s = 200/8;
                for(let r=0; r<8; r++) {
                    for(let c=0; c<8; c++) {
                        // 0-3 is Doc A (incl EOT), 4-7 is Doc B
                        // If r >= 4 (Doc B) and c < 4 (Doc A), that's bad
                        ctx.strokeStyle="#fff";
                        if (c > r) {
                            ctx.fillStyle = "#eee"; // Future
                        } else {
                            if (r >= 4 && c < 4) {
                                ctx.fillStyle = "#e74c3c"; // BAD!
                            } else {
                                ctx.fillStyle = "#2ecc71"; // OK
                            }
                        }
                        ctx.fillRect(c*s, r*s, s, s);
                        ctx.strokeRect(c*s, r*s, s, s);
                    }
                }
            }
        }, 100);
    }

    function renderBlockMask() {
        visualContent.innerHTML = `
            <div style="text-align:center;">
                <div style="font-size:24px; margin-bottom:10px;">Block-Diagonal Mask</div>
                <canvas id="good-mask" width="200" height="200" style="border:1px solid #333;"></canvas>
                <div style="color:#27ae60; margin-top:10px;">Perfect Isolation</div>
            </div>
        `;
        
        setTimeout(() => {
            const canvas = document.getElementById('good-mask');
            if(canvas) {
                const ctx = canvas.getContext('2d');
                const s = 200/8;
                for(let r=0; r<8; r++) {
                    for(let c=0; c<8; c++) {
                        // 0-3: Group 0
                        // 4-7: Group 1
                        const groupR = r < 4 ? 0 : 1;
                        const groupC = c < 4 ? 0 : 1;
                        
                        ctx.strokeStyle="#fff";
                        if (c > r) {
                            ctx.fillStyle = "#eee"; // Future
                        } else if (groupR !== groupC) {
                            ctx.fillStyle = "#95a5a6"; // Masked by Doc ID
                        } else {
                            ctx.fillStyle = "#2ecc71"; // Visible
                        }
                        ctx.fillRect(c*s, r*s, s, s);
                        ctx.strokeRect(c*s, r*s, s, s);
                    }
                }
            }
        }, 100);
    }

    function renderPosIssue() {
        const container = document.createElement('div');
        container.innerHTML = '<div>Naive Global Position IDs</div>';
        
        const vals = [0,1,2,3,4,5,6,7];
        container.appendChild(createGrid(1, 8, (cell, r, c) => {
            cell.innerText = vals[c];
            // Highlight error
            if (c >= 4) {
                cell.style.color = '#c0392b';
                cell.style.fontWeight = 'bold';
                cell.style.border = '2px solid #e74c3c';
            }
        }));
        
        const msg = document.createElement('div');
        msg.style.color = '#e74c3c';
        msg.innerText = 'Doc B starts at Pos 4? Wrong! Should be 0.';
        container.appendChild(msg);
        
        visualContent.appendChild(container);
    }

    function renderPosReset() {
        const container = document.createElement('div');
        container.innerHTML = '<div>Reset Position IDs</div>';
        
        const vals = [0,1,2,3,0,1,2,3];
        container.appendChild(createGrid(1, 8, (cell, r, c) => {
            cell.innerText = vals[c];
            if (c >= 4) {
                cell.style.background = '#d6eaf8';
                cell.style.color = '#2980b9';
                cell.style.fontWeight = 'bold';
            } else {
                cell.style.background = '#d5f5e3';
            }
        }));
        
        visualContent.appendChild(container);
    }

    function renderSimulation() {
        visualContent.innerHTML = `
            <div style="display:flex; flex-direction:column; gap:20px; align-items:center; width:100%;">
                <div style="display:flex; gap:20px;">
                    <div class="doc-box" style="padding:10px; border:2px solid #2ecc71; background:#d5f5e3;">
                        Doc A (Len 2)
                    </div>
                    <div class="doc-box" style="padding:10px; border:2px solid #3498db; background:#d6eaf8;">
                        Doc B (Len 3)
                    </div>
                </div>
                
                <div class="arrow">⬇ Pack (Seq Len = 8) ⬇</div>
                
                <div class="seq-grid" style="display:grid; grid-template-columns:repeat(8, 40px); gap:2px;">
                    <!-- JS will fill -->
                </div>
                
                <div style="margin-top:10px;">
                    <strong>Legend:</strong> 
                    <span style="background:#d5f5e3; padding:2px 5px;">Doc A</span>
                    <span style="background:#d6eaf8; padding:2px 5px;">Doc B</span>
                    <span style="background:#fadbd8; padding:2px 5px;">EOT</span>
                    <span style="background:#ecf0f1; padding:2px 5px;">PAD</span>
                </div>
            </div>
        `;
        
        const grid = visualContent.querySelector('.seq-grid');
        
        // Simulation Data
        // A (2) -> A1, A2, EOT (idx 2)
        // B (3) -> B1, B2, B3, EOT (idx 6)
        // PAD -> idx 7
        
        const tokens = [
            {t:'A1', c:'#d5f5e3', p:0},
            {t:'A2', c:'#d5f5e3', p:1},
            {t:'EOT', c:'#fadbd8', p:2},
            {t:'B1', c:'#d6eaf8', p:0}, // Reset!
            {t:'B2', c:'#d6eaf8', p:1},
            {t:'B3', c:'#d6eaf8', p:2},
            {t:'EOT', c:'#fadbd8', p:3},
            {t:'PAD', c:'#ecf0f1', p:0},
        ];
        
        tokens.forEach(item => {
            const cell = document.createElement('div');
            cell.style.width = '40px';
            cell.style.height = '60px';
            cell.style.background = item.c;
            cell.style.border = '1px solid #aaa';
            cell.style.display = 'flex';
            cell.style.flexDirection = 'column';
            cell.style.alignItems = 'center';
            cell.style.justifyContent = 'center';
            cell.style.fontSize = '12px';
            
            cell.innerHTML = `
                <div style="font-weight:bold;">${item.t}</div>
                <div style="font-size:10px; margin-top:5px; color:#555;">P:${item.p}</div>
            `;
            
            grid.appendChild(cell);
        });
    }

    if (nextBtn) nextBtn.addEventListener('click', goNext);
    if (prevBtn) prevBtn.addEventListener('click', goPrev);
    if (resetBtn) resetBtn.addEventListener('click', () => { currentStep = 0; updateUI(); });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });

    updateUI();
});
