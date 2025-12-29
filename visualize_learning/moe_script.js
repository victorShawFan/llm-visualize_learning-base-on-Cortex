document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');
    const topKInput = document.getElementById('topKInput');
    const nExpertsInput = document.getElementById('nExpertsInput');
    const modeSelect = document.getElementById('modeSelect');
    const auxTypeSelect = document.getElementById('auxType');

    // Guard: Ensure essential elements exist
    if (!visualContent) {
        console.error("Required elements not found in moe_script");
        return;
    }

    let currentStep = 0;
    let animationId = null;

    // Config state
    let topK = 2;
    let nExperts = 8;
    let mode = "train"; 

    // Data state
    let tokens = [];

    const steps = [
        {
            title: "步骤 0: 门控网络打分 (Gating/Scoring)",
            description: "每个 Token 通过线性层计算对所有专家的亲和度分数 (Logits -> Softmax)。<br><span class='step-badge'>sparse_moe.py:53-56</span>",
            code: "logits = F.linear(hidden_states, self.weight)\nscores = logits.softmax(dim=-1)",
            render: (ctx, canvas) => renderGating(ctx, canvas)
        },
        {
            title: "步骤 1: Top-K 稀疏选择 (Sparse Selection)",
            description: "每个 Token 只选择分数最高的 K 个专家。未被选中的专家权重被置为 0，实现稀疏计算。<br><span class='step-badge'>sparse_moe.py:58-60</span>",
            code: "topk_weight, topk_idx = torch.topk(scores, k=self.top_k)\n# Only K experts are active per token",
            render: (ctx, canvas) => renderTopKSelection(ctx, canvas)
        },
        {
            title: "步骤 2: 路由分发 (Routing)",
            description: () => mode === "train" 
                ? "<b>训练模式:</b> 使用 <code>repeat_interleave</code> 将 Token 复制 K 份，分别送往对应专家。这种方式内存开销大但易于并行和求导。<br><span class='step-badge'>sparse_moe.py:153</span>"
                : "<b>推理模式:</b> 使用 <code>argsort</code> 对 Token 进行排序和重组，避免复制数据，实现高效推理。<br><span class='step-badge'>sparse_moe.py:201</span>",
            code: () => mode === "train"
                ? "x = x.repeat_interleave(top_k, dim=0)\n# [Batch, Hidden] -> [Batch*K, Hidden]"
                : "idxs = topk_ids.view(-1).argsort()\n# Reorder tokens by expert ID",
            render: (ctx, canvas) => renderRouting(ctx, canvas)
        },
        {
            title: "步骤 3: 专家并行计算 (Expert Computation)",
            description: "各个专家网络 (FFN) 独立处理分配给它们的 Token。负载不均会导致某些专家过忙，而其他专家空闲。<br><span class='step-badge'>sparse_moe.py:206-214</span>",
            code: "for i, expert in enumerate(self.experts):\n    out = expert(tokens_for_expert_i)",
            render: (ctx, canvas) => renderExpertCompute(ctx, canvas)
        },
        {
            title: "步骤 4: 结果聚合 (Aggregation)",
            description: "将专家的输出乘路由权重，并加上<b>共享专家 (Shared Expert)</b> 的结果，得到最终的 Hidden States。<br><span class='step-badge'>sparse_moe.py:164-166</span>",
            code: "y = (y * weight).sum(dim=1)\noutput = y + self.shared_experts(identity)",
            render: (ctx, canvas) => renderAggregation(ctx, canvas)
        }
    ];

    function init() {
        try {
            if (!visualContent) return;
            visualContent.innerHTML = '<canvas id="moeCanvas"></canvas>';
            const canvas = document.getElementById('moeCanvas');
            resizeCanvas(canvas);
            window.addEventListener('resize', () => resizeCanvas(canvas));
            
            // Refresh inputs if they exist
            if (topKInput) topK = parseInt(topKInput.value) || 2;
            if (nExpertsInput) nExperts = parseInt(nExpertsInput.value) || 8;
            if (modeSelect) mode = modeSelect.value;
            
            generateMockData();
            updateUI();
        } catch (e) {
            console.error("Init failed:", e);
        }
    }

    function generateMockData() {
        tokens = [];
        const numTokens = 5;
        for (let i = 0; i < numTokens; i++) {
            const logits = Array.from({length: nExperts}, () => Math.random() * 10);
            const sum = logits.reduce((a, b) => a + b, 0);
            const scores = logits.map(v => v / sum);
            
            const withIdx = scores.map((s, idx) => ({s, idx}));
            withIdx.sort((a, b) => b.s - a.s);
            const top = withIdx.slice(0, topK);
            
            tokens.push({
                id: i,
                color: `hsl(${i * 60}, 70%, 50%)`,
                scores: scores,
                topk: top
            });
        }
    }

    function resizeCanvas(canvas) {
        if (!canvas || !visualContent) return;
        canvas.width = visualContent.clientWidth || 800;
        canvas.height = 400;
    }

    function updateUI() {
        const step = steps[currentStep];
        
        // Handle dynamic fields
        const desc = typeof step.description === 'function' ? step.description() : step.description;
        const code = typeof step.code === 'function' ? step.code() : step.code;

        if (infoBox) infoBox.innerHTML = `<strong>${step.title}</strong><br>${desc}`;
        if (codeSnippet) {
            codeSnippet.textContent = code;
            if (window.hljs) hljs.highlightElement(codeSnippet);
        }

        const canvas = document.getElementById('moeCanvas');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            if (animationId) cancelAnimationFrame(animationId);
            step.render(ctx, canvas);
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

    function drawExperts(ctx, x, y, h, activeIndices=[]) {
        const boxH = h / nExperts;
        const boxW = 60;
        const positions = [];
        
        for (let i=0; i<nExperts; i++) {
            const by = y + i * boxH + 5;
            const bh = boxH - 10;
            const isActive = activeIndices.includes(i);
            
            ctx.fillStyle = isActive ? "#d6eaf8" : "#f4f6f7";
            ctx.strokeStyle = isActive ? "#3498db" : "#bdc3c7";
            ctx.lineWidth = 2;
            
            ctx.fillRect(x, by, boxW, bh);
            ctx.strokeRect(x, by, boxW, bh);
            
            ctx.fillStyle = "#555";
            ctx.font = "12px Arial";
            ctx.textAlign = "center";
            ctx.fillText(`E${i}`, x + boxW/2, by + bh/2 + 4);
            
            positions.push({x: x, y: by + bh/2});
        }
        return positions;
    }

    function renderGating(ctx, canvas) {
        let frame = 0;
        function animate() {
            if (!canvas.offsetParent) return; // Stop if hidden
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const startX = 50;
            const startY = 50;
            
            tokens.forEach((t, i) => {
                const y = startY + i * 60;
                ctx.fillStyle = t.color;
                ctx.fillRect(startX, y, 40, 40);
                ctx.fillStyle = "#fff";
                ctx.font = "bold 14px Arial";
                ctx.textAlign = "center";
                ctx.fillText(`T${t.id}`, startX + 20, y + 25);
                
                const barX = startX + 60;
                const barW = 200;
                
                t.scores.forEach((s, idx) => {
                    const bx = barX + idx * (barW / nExperts);
                    const bw = (barW / nExperts) - 2;
                    const grow = Math.min(1, frame/60);
                    const height = s * 40 * grow;
                    
                    ctx.fillStyle = `rgba(52, 152, 219, ${0.5 + s*2})`;
                    ctx.fillRect(bx, y + 40 - height, bw, height);
                });
                ctx.fillStyle = "#333";
                ctx.fillText("Scores", barX + barW/2, y + 55);
            });
            
            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
    }

    function renderTopKSelection(ctx, canvas) {
        let frame = 0;
        function animate() {
            if (!canvas.offsetParent) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const startX = 50;
            const startY = 50;
            
            tokens.forEach((t, i) => {
                const y = startY + i * 60;
                ctx.fillStyle = t.color;
                ctx.fillRect(startX, y, 40, 40);
                
                const barX = startX + 60;
                const itemW = 50;
                
                t.topk.forEach((item, k) => {
                    const bx = barX + k * (itemW + 10);
                    const slide = Math.min(1, (frame - k*10)/30);
                    if (slide < 0) return;
                    
                    const curX = bx + (1-slide)*20;
                    ctx.fillStyle = "#fff";
                    ctx.strokeStyle = "#2ecc71";
                    ctx.lineWidth = 2;
                    ctx.fillRect(curX, y, itemW, 40);
                    ctx.strokeRect(curX, y, itemW, 40);
                    
                    ctx.fillStyle = "#333";
                    ctx.font = "12px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText(`E${item.idx}`, curX + itemW/2, y + 20);
                    ctx.font = "10px Arial";
                    ctx.fillText(item.s.toFixed(2), curX + itemW/2, y + 35);
                });
            });
            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
    }

    function renderRouting(ctx, canvas) {
        let frame = 0;
        function animate() {
            if (!canvas.offsetParent) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const w = canvas.width;
            const h = canvas.height;
            
            const gateX = 100;
            const expertX = w - 100;
            const expertPos = drawExperts(ctx, expertX, 50, h-100);
            
            tokens.forEach((t, i) => {
                const y = 80 + i * 50;
                ctx.fillStyle = t.color;
                ctx.beginPath();
                ctx.arc(gateX, y, 10, 0, Math.PI*2);
                ctx.fill();
                
                t.topk.forEach((item, k) => {
                    const dest = expertPos[item.idx];
                    const delay = i * 20;
                    const dur = 60;
                    let p = (frame - delay) / dur;
                    
                    if (p < 0) return;
                    if (p > 1) p = 1;
                    
                    const ease = 1 - Math.pow(1 - p, 3);
                    let curX, curY;
                    
                    if (mode === "train") {
                        curX = gateX + (dest.x - gateX) * ease;
                        curY = y + (dest.y - y) * ease;
                    } else {
                        const sortX = w/2;
                        const sortY = 50 + (item.idx * 30);
                        if (p < 0.5) {
                            const p1 = p * 2;
                            curX = gateX + (sortX - gateX) * p1;
                            curY = y + (sortY - y) * p1;
                        } else {
                            const p2 = (p - 0.5) * 2;
                            curX = sortX + (dest.x - sortX) * p2;
                            curY = sortY + (dest.y - sortY) * p2;
                        }
                    }
                    
                    ctx.fillStyle = t.color;
                    ctx.beginPath();
                    ctx.arc(curX, curY, 6, 0, Math.PI*2);
                    ctx.fill();
                });
            });
            ctx.fillStyle = "#333";
            ctx.textAlign = "center";
            ctx.fillText(mode === "train" ? "Token Copying (Train)" : "Sorting & Batching (Infer)", w/2, 30);
            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
    }

    function renderExpertCompute(ctx, canvas) {
        let frame = 0;
        function animate() {
            if (!canvas.offsetParent) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const w = canvas.width;
            const h = canvas.height;
            const expertX = w/2 - 30;
            
            const allActiveIndices = new Set();
            tokens.forEach(t => t.topk.forEach(k => allActiveIndices.add(k.idx)));
            
            const expertPos = drawExperts(ctx, expertX, 50, h-100, Array.from(allActiveIndices));
            
            expertPos.forEach((pos, i) => {
                if (allActiveIndices.has(i)) {
                    const pulse = Math.sin(frame * 0.2 + i);
                    if (pulse > 0) {
                        ctx.strokeStyle = "#f1c40f";
                        ctx.lineWidth = 3;
                        ctx.strokeRect(expertX - 5, pos.y - 15, 70, 30);
                    }
                    ctx.fillStyle = "#333";
                    ctx.font = "10px Arial";
                    ctx.fillText("Computing...", expertX + 30, pos.y + 25);
                }
            });
            ctx.fillStyle = "#333";
            ctx.textAlign = "center";
            ctx.fillText("Active Experts Processing", w/2, 30);
            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
    }

    function renderAggregation(ctx, canvas) {
        let frame = 0;
        function animate() {
            if (!canvas.offsetParent) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const w = canvas.width;
            const h = canvas.height;
            const expertX = 100;
            const finalX = w - 100;
            const expertPos = drawExperts(ctx, expertX, 50, h-100);
            
            tokens.forEach((t, i) => {
                const targetY = 80 + i * 50;
                ctx.fillStyle = "#ecf0f1";
                ctx.strokeStyle = "#333";
                ctx.fillRect(finalX, targetY - 15, 40, 30);
                ctx.strokeRect(finalX, targetY - 15, 40, 30);
                ctx.fillStyle = "#333";
                ctx.fillText(`H'${i}`, finalX + 20, targetY + 5);
                
                t.topk.forEach((item, k) => {
                    const start = expertPos[item.idx];
                    const delay = i * 20;
                    const p = Math.min(1, Math.max(0, (frame - delay) / 60));
                    const cx = expertX + 60 + (finalX - (expertX + 60)) * p;
                    const cy = start.y + (targetY - start.y) * p;
                    if (p > 0 && p < 1) {
                        ctx.fillStyle = t.color;
                        ctx.beginPath();
                        ctx.arc(cx, cy, 5, 0, Math.PI*2);
                        ctx.fill();
                    }
                });
                
                const sharedP = Math.min(1, Math.max(0, (frame - i*20 - 30)/30));
                if (sharedP > 0) {
                    const sx = w/2;
                    const sy = h - 30;
                    const cx = sx + (finalX - sx) * sharedP;
                    const cy = sy + (targetY - sy) * sharedP;
                    ctx.fillStyle = "#e67e22";
                    ctx.beginPath();
                    ctx.arc(cx, cy, 6, 0, Math.PI*2);
                    ctx.fill();
                }
            });
            
            ctx.fillStyle = "#f39c12";
            ctx.fillRect(w/2 - 50, h - 50, 100, 40);
            ctx.fillStyle = "#fff";
            ctx.fillText("Shared Expert", w/2, h - 25);
            ctx.fillStyle = "#333";
            ctx.textAlign = "center";
            ctx.fillText("Weighted Sum + Shared Expert", w/2, 30);
            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
    }

    // Bind events
    if (nextBtn) nextBtn.addEventListener('click', goNext);
    if (prevBtn) prevBtn.addEventListener('click', goPrev);
    if (resetBtn) resetBtn.addEventListener('click', init);
    if (topKInput) topKInput.addEventListener('change', init);
    if (nExpertsInput) nExpertsInput.addEventListener('change', init);
    if (modeSelect) modeSelect.addEventListener('change', init);
    if (auxTypeSelect) auxTypeSelect.addEventListener('change', init);

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });

    // Initial
    init();
});
