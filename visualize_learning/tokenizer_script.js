document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');
    
    // Guard
    if (!visualContent) {
        console.error("Required elements not found in tokenizer_script");
        return;
    }

    let currentStep = 0;
    let animationId = null;

    // Mock Data
    let inputJson = [
        { "role": "system", "content": "You are a helpful assistant." },
        { "role": "user", "content": "Hello!" },
        { "role": "assistant", "think": "User greets me.", "content": "Hi there!" }
    ];

    const steps = [
        {
            title: "Input: Raw Conversations (JSON)",
            description: "原始输入是一个包含角色 (role) 和内容 (content) 的字典列表。这是 SFT 和 RLHF 数据的标准格式。<br><span class='step-badge'>tokenizer.py:300-317</span>",
            code: "conversations: List[Dict[str, str]] = [\n  {'role': 'user', 'content': '...'},\n  {'role': 'assistant', 'content': '...'}\n]",
            render: () => renderJsonView()
        },
        {
            title: "Step 1: Apply Chat Template (Formatting)",
            description: "将结构化对话转换为线性文本。系统会遍历列表，添加角色标记。注意观察 <code>&lt;think&gt;</code> 和 <code>&lt;answer&gt;</code> 标签是如何自动插入到 Assistant 回复中的。<br><span class='step-badge'>tokenizer.py:330-356</span>",
            code: "if role == 'assistant':\n    if 'think' in conv:\n        content = f'<think>{conv['think']}</think>'\n    content += f'<answer>{conv['content']}</answer>'",
            render: () => renderTemplateAnimation()
        },
        {
            title: "Step 2: Tokenization (Encoding)",
            description: "文本被切分为 Token 并转换为 ID。特殊标记 (Special Tokens) 如 <code>&lt;user&gt;</code> 被映射为单一 ID，而普通文本被切分为子词 (Subwords)。<br><span class='step-badge'>tokenizer.py:103-126</span>",
            code: "ids = tokenizer.encode(text, add_special_tokens=False)\n# <user> -> 1001, Hello -> 1234",
            render: () => renderTokenizationAnimation()
        },
        {
            title: "Step 3: Special Tokens Map",
            description: "Tokenizer 维护了一个特殊 Token 到 ID 的映射字典。这些 ID 在整个模型训练中是固定的。<br><span class='step-badge'>tokenizer.py:24-82</span>",
            code: "special_tokens = {\n    '</s>': 2,\n    '<pad>': 0,\n    '<user>': 1001,\n    '<think>': 2001\n}",
            render: () => renderSpecialTokensView()
        },
        {
            title: "Step 4: Batch Encoding & Padding",
            description: "处理 Batch 时，不同长度的序列会被 Padding 到统一长度（通常是 Batch 内最长序列）。空余位置填充 <code>pad_token_id</code> (0)。<br><span class='step-badge'>tokenizer.py:150-175</span>",
            code: "batch = tokenizer(texts, padding='longest')\n# Returns: [B, SeqLen] tensor",
            render: () => renderPaddingAnimation()
        },
        {
            title: "Step 5: Decoding (Reconstruction)",
            description: "解码是将 ID 序列还原为文本的过程。<code>skip_special_tokens=True</code> 可以去除辅助标记，只保留人类可读内容。<br><span class='step-badge'>tokenizer.py:200-228</span>",
            code: "text = tokenizer.decode(ids, skip_special_tokens=True)",
            render: () => renderDecodingView()
        }
    ];

    function init() {
        try {
            if (!visualContent) return;
            updateUI();
        } catch (e) {
            console.error("Init failed", e);
        }
    }

    function updateUI() {
        // Clear previous
        if (visualContent) visualContent.innerHTML = '';
        if (animationId) cancelAnimationFrame(animationId);
        
        const step = steps[currentStep];
        if (infoBox) infoBox.innerHTML = `<strong>${step.title}</strong><br>${step.description}`;
        if (codeSnippet) {
            codeSnippet.textContent = step.code;
            if (window.hljs) hljs.highlightElement(codeSnippet);
        }
        
        try {
            step.render();
        } catch(e) {
            console.error("Render failed", e);
        }
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

    function renderJsonView() {
        const container = document.createElement('div');
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.gap = '10px';
        container.style.height = '100%';
        container.style.justifyContent = 'center';
        container.style.alignItems = 'center';

        const label = document.createElement('div');
        label.innerText = 'Editable Input JSON:';
        label.style.fontWeight = 'bold';
        label.style.color = '#333';
        
        const area = document.createElement('textarea');
        area.style.width = '400px';
        area.style.height = '200px';
        area.style.fontFamily = 'monospace';
        area.style.padding = '15px';
        area.style.border = '2px solid #bdc3c7';
        area.style.borderRadius = '8px';
        area.style.fontSize = '14px';
        area.value = JSON.stringify(inputJson, null, 4);
        
        area.oninput = (e) => {
            try {
                inputJson = JSON.parse(e.target.value);
                area.style.borderColor = '#2ecc71';
            } catch(err) {
                area.style.borderColor = '#e74c3c';
            }
        };
        
        container.appendChild(label);
        container.appendChild(area);
        visualContent.appendChild(container);
    }

    function renderTemplateAnimation() {
        visualContent.innerHTML = `
            <div style="display:flex; height:100%; gap:20px; padding:20px;">
                <div id="msg-container" style="flex:1; display:flex; flex-direction:column; gap:10px;"></div>
                <div style="width:2px; background:#eee;"></div>
                <div id="output-container" style="flex:1; font-family:monospace; white-space:pre-wrap; background:#f9f9f9; padding:15px; border-radius:8px; overflow-y:auto; color:#333; font-size:14px; border:1px solid #ddd;"></div>
            </div>
        `;

        const msgContainer = document.getElementById('msg-container');
        const outContainer = document.getElementById('output-container');
        
        // 1. Populate Messages
        inputJson.forEach((msg, i) => {
            const card = document.createElement('div');
            card.className = 'msg-card';
            card.style.border = '1px solid #bdc3c7';
            card.style.padding = '10px';
            card.style.borderRadius = '5px';
            card.style.background = '#fff';
            card.style.opacity = '0'; // Start hidden
            card.style.transform = 'translateX(-20px)';
            card.style.transition = 'all 0.5s ease';
            
            card.innerHTML = `
                <div style="font-weight:bold; color:#2c3e50; margin-bottom:5px;">${msg.role}</div>
                ${msg.think ? `<div style="font-size:12px; color:#e67e22; margin-bottom:2px;">Think: ${msg.think}</div>` : ''}
                <div style="font-size:13px; color:#555;">${msg.content}</div>
            `;
            msgContainer.appendChild(card);
            
            // Animation Sequence
            setTimeout(() => {
                if (!card.parentElement) return; // check if removed
                card.style.opacity = '1';
                card.style.transform = 'translateX(0)';
                
                // Generate Text Part
                setTimeout(() => {
                    if (!outContainer.parentElement) return;
                    const roleTag = `<span style="color:#2980b9; font-weight:bold;">&lt;${msg.role}&gt;</span>`;
                    let contentHtml = "";
                    
                    if (msg.role === 'assistant') {
                        if (msg.think) {
                            contentHtml += `<span style="color:#e67e22;">&lt;think&gt;${msg.think}&lt;/think&gt;</span>`;
                        }
                        contentHtml += `<span style="color:#27ae60;">&lt;answer&gt;${msg.content}&lt;/answer&gt;</span>`;
                    } else {
                        contentHtml += `<span>${msg.content}</span>`;
                    }
                    
                    contentHtml += `<span style="color:#c0392b;">&lt;/s&gt;</span>\n`;
                    
                    const line = document.createElement('div');
                    line.innerHTML = roleTag + contentHtml;
                    line.style.opacity = '0';
                    outContainer.appendChild(line);
                    
                    // Fade in line
                    requestAnimationFrame(() => {
                        line.style.transition = 'opacity 0.5s';
                        line.style.opacity = '1';
                    });
                    
                    // Highlight source card
                    card.style.borderColor = '#3498db';
                    card.style.background = '#ebf5fb';
                    
                }, 600);
            }, i * 1500); // Staggered
        });
    }

    function renderTokenizationAnimation() {
        visualContent.innerHTML = `
            <div style="display:flex; flex-direction:column; align-items:center; gap:30px; padding-top:40px;">
                <div id="raw-text" style="font-family:monospace; font-size:20px; display:flex; gap:2px;"></div>
                <div style="font-size:30px; color:#bdc3c7;">⬇</div>
                <div id="token-row" style="display:flex; gap:5px; flex-wrap:wrap; justify-content:center; max-width:600px;"></div>
            </div>
        `;
        
        const rawTextContainer = document.getElementById('raw-text');
        const tokenRow = document.getElementById('token-row');
        
        // Sample String construction
        const sample = [
            { t: "<user>", type: "special", id: 1001 },
            { t: "Hello", type: "text", id: 1234 },
            { t: "!", type: "text", id: 5 },
            { t: "</s>", type: "special", id: 2 }
        ];
        
        // Draw initial text
        sample.forEach((item, i) => {
            const span = document.createElement('div');
            span.innerText = item.t;
            span.style.padding = '5px 10px';
            span.style.border = '1px solid #ddd';
            span.style.borderRadius = '4px';
            span.style.background = '#f9f9f9';
            span.id = `raw-${i}`;
            rawTextContainer.appendChild(span);
        });
        
        // Animate conversion
        sample.forEach((item, i) => {
            setTimeout(() => {
                if (!rawTextContainer.parentElement) return;
                // Highlight source
                const src = document.getElementById(`raw-${i}`);
                if(src) {
                    src.style.background = '#f1c40f';
                    src.style.color = '#fff';
                    src.style.borderColor = '#f39c12';
                    src.style.transform = 'scale(1.1)';
                }
                
                // Create target token
                const tok = document.createElement('div');
                tok.className = 'token-card-anim';
                tok.style.width = '60px';
                tok.style.height = '60px';
                tok.style.background = item.type === 'special' ? '#fadbd8' : '#d6eaf8';
                tok.style.border = `2px solid ${item.type === 'special' ? '#e74c3c' : '#3498db'}`;
                tok.style.borderRadius = '8px';
                tok.style.display = 'flex';
                tok.style.flexDirection = 'column';
                tok.style.alignItems = 'center';
                tok.style.justifyContent = 'center';
                tok.style.opacity = '0';
                tok.style.transform = 'translateY(-20px)';
                
                tok.innerHTML = `
                    <div style="font-size:12px; font-weight:bold; color:#555;">${escapeHtml(item.t)}</div>
                    <div style="font-size:16px; font-weight:bold; color:#333;">${item.id}</div>
                `;
                
                if(tokenRow) tokenRow.appendChild(tok);
                
                // Animate in
                requestAnimationFrame(() => {
                    tok.style.transition = 'all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
                    tok.style.opacity = '1';
                    tok.style.transform = 'translateY(0)';
                });
                
            }, i * 800 + 500);
        });
    }

    function renderSpecialTokensView() {
        const tokens = [
            { text: '</s>', label: 'EOS', id: 2, color: '#e74c3c' },
            { text: '<pad>', label: 'PAD', id: 0, color: '#95a5a6' },
            { text: '<user>', label: 'USER', id: 1001, color: '#3498db' },
            { text: '<assistant>', label: 'ASSIST', id: 1002, color: '#9b59b6' },
            { text: '<system>', label: 'SYSTEM', id: 1000, color: '#16a085' },
            { text: '<think>', label: 'THINK', id: 2001, color: '#f39c12' },
            { text: '<answer>', label: 'ANSWER', id: 2003, color: '#27ae60' },
            { text: '<image>', label: 'IMG', id: 3000, color: '#d35400' }
        ];

        visualContent.innerHTML = '<div id="spec-grid" style="display:grid; grid-template-columns:repeat(auto-fill, minmax(120px, 1fr)); gap:15px; padding:30px; width:100%;"></div>';
        const grid = document.getElementById('spec-grid');

        tokens.forEach((t, i) => {
            const card = document.createElement('div');
            card.style.border = `2px solid ${t.color}`;
            card.style.borderRadius = '8px';
            card.style.padding = '15px';
            card.style.textAlign = 'center';
            card.style.background = '#fff';
            card.style.boxShadow = '0 2px 5px rgba(0,0,0,0.05)';
            card.style.cursor = 'pointer';
            card.style.transition = 'transform 0.2s';
            
            card.onmouseover = () => card.style.transform = 'scale(1.05)';
            card.onmouseout = () => card.style.transform = 'scale(1)';

            card.innerHTML = `
                <div style="color:${t.color}; font-weight:bold; font-size:14px; margin-bottom:5px;">${escapeHtml(t.text)}</div>
                <div style="font-family:monospace; font-size:24px; color:#333;">${t.id}</div>
                <div style="font-size:10px; color:#999; margin-top:5px; text-transform:uppercase;">${t.label}</div>
            `;
            
            grid.appendChild(card);
        });
    }

    function renderPaddingAnimation() {
        visualContent.innerHTML = '<canvas id="padCanvas"></canvas>';
        const canvas = document.getElementById('padCanvas');
        if (!canvas) return;
        
        canvas.width = visualContent.clientWidth || 800;
        canvas.height = 400;
        
        const ctx = canvas.getContext('2d');
        
        const batchData = [
            [101, 2034, 102], // Short
            [101, 56, 99, 1032, 843, 102], // Longest
            [101, 754, 102] // Short
        ];
        
        const maxLength = 6;
        const boxSize = 40;
        const gap = 5;
        const startX = canvas.width/2 - (maxLength * (boxSize+gap))/2;
        const startY = 100;
        
        let frame = 0;
        
        function animate() {
            if (!document.getElementById('padCanvas')) return; // Exit if switched
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            ctx.fillStyle = "#333";
            ctx.textAlign = "center";
            ctx.font = "16px Arial";
            ctx.fillText("Batch Padding (padding='longest')", canvas.width/2, 50);

            // Draw Rows
            batchData.forEach((row, r) => {
                const y = startY + r * 60;
                
                // Draw real tokens
                row.forEach((id, c) => {
                    const x = startX + c * (boxSize + gap);
                    ctx.fillStyle = "#3498db";
                    ctx.fillRect(x, y, boxSize, boxSize);
                    ctx.strokeStyle = "#2980b9";
                    ctx.strokeRect(x, y, boxSize, boxSize);
                    ctx.fillStyle = "#fff";
                    ctx.font = "12px monospace";
                    ctx.fillText(id, x + boxSize/2, y + boxSize/2);
                });
                
                // Animate Padding
                const needed = maxLength - row.length;
                if (needed > 0) {
                    for(let k=0; k<needed; k++) {
                        const c = row.length + k;
                        const x = startX + c * (boxSize + gap);
                        
                        // Delay based on index
                        const delay = 60 + k*10 + r*20;
                        const progress = Math.min(1, Math.max(0, (frame - delay)/30));
                        
                        if (progress > 0) {
                            ctx.globalAlpha = progress;
                            ctx.fillStyle = "#95a5a6";
                            ctx.fillRect(x, y, boxSize, boxSize);
                            ctx.strokeStyle = "#7f8c8d";
                            ctx.strokeRect(x, y, boxSize, boxSize);
                            ctx.fillStyle = "#fff";
                            ctx.font = "12px monospace";
                            ctx.fillText("0", x + boxSize/2, y + boxSize/2);
                            ctx.globalAlpha = 1.0;
                        }
                    }
                }
            });
            
            // Draw Max Length Line
            const lineX = startX + maxLength * (boxSize + gap) + 10;
            if (frame > 30) {
                ctx.strokeStyle = "#e74c3c";
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.moveTo(lineX, startY - 20);
                ctx.lineTo(lineX, startY + 200);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = "#e74c3c";
                ctx.fillText("Max Len", lineX, startY - 30);
            }

            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
    }

    function renderDecodingView() {
        visualContent.innerHTML = `
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; gap:20px;">
                <div id="id-seq" style="display:flex; gap:5px;"></div>
                <div style="font-size:24px; color:#bdc3c7;">⬇ decode()</div>
                <div id="text-out" style="font-family:monospace; font-size:20px; color:#2c3e50; min-height:30px;"></div>
            </div>
        `;
        
        const ids = [1001, 1234, 5, 2];
        const words = ["<user>", "Hello", "!", "</s>"];
        const idContainer = document.getElementById('id-seq');
        const textOut = document.getElementById('text-out');
        
        ids.forEach((id, i) => {
            const box = document.createElement('div');
            box.className = 'tok-box';
            box.innerText = id;
            box.style.background = (id === 1001 || id === 2) ? '#fadbd8' : '#d6eaf8';
            box.style.padding = '10px';
            box.style.borderRadius = '5px';
            box.style.border = '1px solid #ccc';
            idContainer.appendChild(box);
            
            setTimeout(() => {
                if (!box.parentElement) return;
                box.style.background = '#f1c40f';
                
                const isSpecial = (id === 1001 || id === 2);
                if (!isSpecial) {
                    textOut.innerText += words[i];
                } else {
                    const span = document.createElement('span');
                    span.innerText = words[i];
                    span.style.color = '#e74c3c';
                    span.style.fontSize = '12px';
                    span.style.opacity = '0.5';
                    span.style.margin = '0 5px';
                    textOut.appendChild(span);
                }
                
                setTimeout(() => {
                    if (box.parentElement) box.style.background = (id === 1001 || id === 2) ? '#fadbd8' : '#d6eaf8';
                }, 300);
                
            }, i * 600 + 500);
        });
    }

    function escapeHtml(text) {
        return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    // Bind events
    if (nextBtn) nextBtn.addEventListener('click', goNext);
    if (prevBtn) prevBtn.addEventListener('click', goPrev);
    if (resetBtn) resetBtn.addEventListener('click', init);

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });

    // Start
    init();
});
