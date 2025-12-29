document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');
    
    // Guard: Check elements
    if (!visualContent) {
        console.error("Required elements not found in vlm_script");
        return;
    }

    let currentStep = 0;
    let animationId = null;

    // Configuration
    const config = {
        patch_size: 14,
        grid_size: 16, // 16x16 original grid
        pool_k: 2,     // 2x2 pooling
        vision_dim: 1024,
        llm_dim: 768
    };

    const steps = [
        {
            title: "步骤 1: 多模态输入构造 (Input Preparation)",
            description: "VLM 训练的第一步是准备混合序列。在 `SFTDataset` 中，单一个 `&lt;image&gt;` 占位符被展开为 $N$ 个连续的 tokens（例如 256 个），为即将注入的视觉特征预留空间。<br><span class='step-badge'>vlm_model.py:208-224</span>",
            code: "def get_input_embeddings(self, input_ids, ...):\n    # Expand <image> to multiple placeholders\n    # [Text, <img0>, <img1>, ..., <imgN>, Text]\n    inputs_embeds = self.embed_tokens(input_ids)",
            render: (ctx, canvas) => renderStep1(ctx, canvas)
        },
        {
            title: "步骤 2: 视觉塔特征提取 (Vision Tower)",
            description: "图像经过 Vision Transformer (如 SigLIP/CLIP)。图像被切分为 16x16 的 Patches，经过多层 Transformer Block 处理。我们通常提取**倒数第二层**的特征，因为它保留了更丰富的空间几何信息。<br><span class='step-badge'>vlm_model.py:187</span>",
            code: "vision_outputs = self.vision_tower(pixel_values)\n# Output shape: [Batch, 256, 1024]\n# (16x16 patches = 256 tokens)",
            render: (ctx, canvas) => renderStep2(ctx, canvas)
        },
        {
            title: "步骤 3: 空间重塑 (Spatial Reshape)",
            description: "ViT 输出的是扁平序列 `[B, N, D]`。为了利用卷积或池化操作，我们需要将其还原为 2D 空间网格结构 `[B, D, H, W]`。<br><span class='step-badge'>vlm_model.py:78-80</span>",
            code: "# [B, 256, 1024] -> [B, 1024, 16, 16]\nreshaped_vision_outputs = vision_outputs.transpose(1, 2).reshape(B, C, 16, 16)",
            render: (ctx, canvas) => renderStep3(ctx, canvas)
        },
        {
            title: "步骤 4: 自适应平均池化 (Avg Pooling)",
            description: "为了减少 LLM 的计算负担，我们使用 `AvgPool2d` 对视觉特征进行降采样。例如，将 2x2 的区域合并为 1 个 Token，序列长度减少 4 倍。<br><span class='step-badge'>vlm_model.py:87</span>",
            code: "# Pooling kernel=2, stride=2\npooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)\n# [B, 1024, 16, 16] -> [B, 1024, 8, 8]",
            render: (ctx, canvas) => renderStep4(ctx, canvas)
        },
        {
            title: "步骤 5: 多模态投影 (Multimodal Projection)",
            description: "视觉特征空间 (Dim=1024) 与 LLM 文本空间 (Dim=768) 不对齐。我们需要通过 `RMSNorm` 和线性投影层 (MLP) 将视觉特征“翻译”到 LLM 的语义空间。<br><span class='step-badge'>vlm_model.py:100-105</span>",
            code: "normed = self.vision_norm(pooled)\n# [B, 64, 1024] @ [1024, 768] -> [B, 64, 768]\nprojected = torch.matmul(normed, self.input_projection_weight)",
            render: (ctx, canvas) => renderStep5(ctx, canvas)
        },
        {
            title: "步骤 6: 特征注入 (Fusion / Injection)",
            description: "最后，利用 `masked_scatter` 将投影后的视觉 Embeddings 精确填充到步骤 1 预留的 `&lt;image&gt;` 占位符位置。文本和图像现在在同一个向量空间中，可以被 LLM 一起处理。<br><span class='step-badge'>vlm_model.py:239-251</span>",
            code: "mask = (input_ids == image_tok)\ninputs_embeds.masked_scatter_(mask, image_features)\n# Final sequence: [Text_Emb, Vis_Emb, Vis_Emb, ..., Text_Emb]",
            render: (ctx, canvas) => renderStep6(ctx, canvas)
        }
    ];

    function init() {
        try {
            if (!visualContent) return;
            visualContent.innerHTML = '<canvas id="vlmCanvas"></canvas>';
            const canvas = document.getElementById('vlmCanvas');
            resizeCanvas(canvas);
            window.addEventListener('resize', () => resizeCanvas(canvas));
            updateUI();
        } catch (e) {
            console.error("Init error", e);
        }
    }

    function resizeCanvas(canvas) {
        if (!canvas || !visualContent) return;
        canvas.width = visualContent.clientWidth || 800;
        canvas.height = 400; 
    }

    function updateUI() {
        const step = steps[currentStep];
        if (infoBox) infoBox.innerHTML = `<strong>${step.title}</strong><br>${step.description}`;
        if (codeSnippet) {
            codeSnippet.textContent = step.code;
            if (window.hljs) hljs.highlightElement(codeSnippet);
        }

        const canvas = document.getElementById('vlmCanvas');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            if (animationId) cancelAnimationFrame(animationId);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
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

    // --- Helpers ---

    function drawArrow(ctx, x1, y1, x2, y2, color="#34495e") {
        const headlen = 10;
        const dx = x2 - x1;
        const dy = y2 - y1;
        const angle = Math.atan2(dy, dx);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x2 - headlen * Math.cos(angle - Math.PI / 6), y2 - headlen * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(x2, y2);
        ctx.lineTo(x2 - headlen * Math.cos(angle + Math.PI / 6), y2 - headlen * Math.sin(angle + Math.PI / 6));
        ctx.fillStyle = color;
        ctx.fill();
    }

    function drawToken(ctx, x, y, width, height, text, color, borderColor="#bdc3c7") {
        ctx.fillStyle = color;
        ctx.fillRect(x, y, width, height);
        ctx.strokeStyle = borderColor;
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, width, height);
        
        ctx.fillStyle = "#2c3e50";
        ctx.font = "12px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(text, x + width/2, y + height/2);
    }

    // --- Renderers ---

    function renderStep1(ctx, canvas) {
        let frame = 0;
        
        function animate() {
            if (!canvas.offsetParent) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const centerX = canvas.width / 2;
            const startY = 100;
            
            // Initial state: User Input
            ctx.font = "16px Arial";
            ctx.fillStyle = "#333";
            ctx.textAlign = "center";
            ctx.fillText("Raw Input Text:", centerX, 50);
            
            const text = ["Describe", "this", "<image>", "."];
            const baseX = centerX - 120;
            
            // Draw initial tokens
            text.forEach((t, i) => {
                const x = baseX + i * 70;
                const isImg = t === "<image>";
                drawToken(ctx, x, 80, 60, 40, t, isImg ? "#ffeaa7" : "#ecf0f1", isImg ? "#fdcb6e" : "#bdc3c7");
            });

            // Expansion Animation
            const expansionProgress = Math.min(1, frame / 60);
            const expandedGap = 30 * expansionProgress;
            const numPlaceholders = 5;
            
            if (expansionProgress > 0) {
                ctx.fillStyle = "#333";
                ctx.fillText("Tokenizer Expansion:", centerX, 180);
                drawArrow(ctx, centerX, 130, centerX, 160);

                // Draw expanded sequence
                const startExpX = centerX - 200;
                let currentX = startExpX;
                
                // "Describe this"
                drawToken(ctx, currentX, 200, 60, 40, "Describe", "#ecf0f1");
                currentX += 65;
                drawToken(ctx, currentX, 200, 60, 40, "this", "#ecf0f1");
                currentX += 65;
                
                // Expanded Image Tokens
                for (let k = 0; k < numPlaceholders; k++) {
                    // Animate them popping out
                    const scale = Math.min(1, (frame - 20 - k*5) / 10);
                    if (scale > 0) {
                        ctx.save();
                        ctx.translate(currentX + 20, 220);
                        ctx.scale(scale, scale);
                        drawToken(ctx, -20, -20, 40, 40, `<img${k}>`, "#ffeaa7", "#fdcb6e");
                        ctx.restore();
                    }
                    currentX += 45;
                }
                
                // "."
                const dotDelay = 20 + numPlaceholders*5;
                if (frame > dotDelay) {
                     drawToken(ctx, currentX + 10, 200, 40, 40, ".", "#ecf0f1");
                }
            }

            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
    }

    function renderStep2(ctx, canvas) {
        let frame = 0;
        function animate() {
            if (!canvas.offsetParent) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const w = canvas.width;
            const imgSize = 160;
            const startX = w/2 - imgSize - 50;
            const startY = 100;
            
            // Draw "Input Image"
            const grd = ctx.createLinearGradient(startX, startY, startX+imgSize, startY+imgSize);
            grd.addColorStop(0, "#3498db");
            grd.addColorStop(1, "#e74c3c");
            ctx.fillStyle = grd;
            ctx.fillRect(startX, startY, imgSize, imgSize);
            
            // Draw Grid Lines
            ctx.strokeStyle = "rgba(255,255,255,0.5)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            for(let i=0; i<=8; i++) {
                ctx.moveTo(startX + i*20, startY);
                ctx.lineTo(startX + i*20, startY+imgSize);
                ctx.moveTo(startX, startY + i*20);
                ctx.lineTo(startX+imgSize, startY + i*20);
            }
            ctx.stroke();
            
            ctx.fillStyle = "#333";
            ctx.textAlign = "center";
            ctx.fillText("Input Image (224x224)", startX + imgSize/2, startY - 10);

            // Animation: Scanning and extracting patches
            const speed = 10; 
            const totalPatches = 64; // 8x8
            const currentPatchIdx = Math.floor(frame / speed) % totalPatches;
            
            const row = Math.floor(currentPatchIdx / 8);
            const col = currentPatchIdx % 8;
            
            // Highlight current patch on image
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 2;
            ctx.strokeRect(startX + col*20, startY + row*20, 20, 20);
            
            // Arrow
            drawArrow(ctx, w/2, startY + imgSize/2, w/2 + 40, startY + imgSize/2);

            // Draw Sequence on the right
            const seqX = w/2 + 60;
            const seqY = startY;
            
            ctx.fillText("Vision Transformer Layers", seqX + 80, startY - 10);
            
            // Draw stack of tokens
            for (let i=0; i<Math.min(currentPatchIdx+1, 64); i++) {
                const tx = seqX + (i % 8) * 22;
                const ty = seqY + Math.floor(i / 8) * 22;
                 
                ctx.fillStyle = "#2ecc71";
                ctx.fillRect(tx, ty, 18, 18);
            }

            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
    }

    function renderStep3(ctx, canvas) {
        let frame = 0;
        function animate() {
            if (!canvas.offsetParent) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const centerX = canvas.width / 2;
            
            // Phase 1: Flat Sequence
            const numTokens = 16;
            const tokenW = 20;
            const startX = centerX - (numTokens * tokenW) / 2;
            
            const cycle = 200;
            const t = (frame % cycle) / 100; // 0 to 2
            const progress = Math.min(1, Math.max(0, t > 1 ? 2 - t : t)); // Triangle wave
            
            ctx.textAlign = "center";
            ctx.fillStyle = "#333";
            ctx.fillText("Sequence [B, N, D]  ⟷  Grid [B, C, H, W]", centerX, 50);

            // Draw tokens
            for (let i = 0; i < numTokens; i++) {
                // Source pos (Line)
                const sx = startX + i * (tokenW + 2);
                const sy = 100;
                
                // Dest pos (Grid 4x4)
                const row = Math.floor(i / 4);
                const col = i % 4;
                const dx = centerX - 50 + col * 25;
                const dy = 200 + row * 25;
                
                // Lerp
                const cx = sx + (dx - sx) * progress;
                const cy = sy + (dy - sy) * progress;
                const colorVal = Math.floor(100 + i * 10);
                
                ctx.fillStyle = `rgb(52, 152, ${colorVal})`;
                ctx.fillRect(cx, cy, 20, 20);
                
                // Label indices
                if (progress < 0.1 || progress > 0.9) {
                    ctx.fillStyle = "#fff";
                    ctx.font = "10px Arial";
                    ctx.fillText(i, cx + 10, cy + 14);
                }
            }
            
            if (progress > 0.5) {
                ctx.fillStyle = "#333";
                ctx.fillText("Reshaped to 4x4 Grid", centerX, 330);
            } else {
                 ctx.fillText("Flat Sequence", centerX, 80);
            }

            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
    }

    function renderStep4(ctx, canvas) {
        let frame = 0;
        function animate() {
            if (!canvas.offsetParent) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const w = canvas.width;
            
            // 8x8 Grid (representing 16x16 logical)
            const gridSize = 8;
            const cellSize = 25;
            const startX = w/2 - (gridSize * cellSize) / 2 - 100;
            const startY = 100;
            
            ctx.textAlign = "center";
            ctx.fillStyle = "#333";
            ctx.fillText("Original Feature Map (H x W)", startX + (gridSize*cellSize)/2, startY - 20);

            // Draw 8x8 Grid
            for (let r=0; r<gridSize; r++) {
                for (let c=0; c<gridSize; c++) {
                    ctx.fillStyle = ((r+c)%2===0) ? "#ecf0f1" : "#bdc3c7";
                    ctx.fillRect(startX + c*cellSize, startY + r*cellSize, cellSize, cellSize);
                }
            }
            
            // Pooling Window Animation
            const speed = 60;
            const step = Math.floor(frame / speed);
            const maxSteps = (gridSize/2) * (gridSize/2);
            const currentStep = step % maxSteps;
            
            const poolRow = Math.floor(currentStep / (gridSize/2)) * 2;
            const poolCol = (currentStep % (gridSize/2)) * 2;
            
            // Highlight pooling window (2x2)
            ctx.strokeStyle = "#e74c3c";
            ctx.lineWidth = 3;
            ctx.strokeRect(startX + poolCol*cellSize, startY + poolRow*cellSize, cellSize*2, cellSize*2);
            ctx.fillStyle = "rgba(231, 76, 60, 0.3)";
            ctx.fillRect(startX + poolCol*cellSize, startY + poolRow*cellSize, cellSize*2, cellSize*2);
            
            // Arrow
            const arrowX = w/2;
            drawArrow(ctx, arrowX - 20, startY + 100, arrowX + 20, startY + 100);
            
            // Output Grid (4x4)
            const outSize = 4;
            const outCell = cellSize; // Same visual size
            const outX = w/2 + 100;
            const outY = startY + 50;
            
            ctx.fillStyle = "#333";
            ctx.fillText("Pooled Output (H/2 x W/2)", outX + (outSize*outCell)/2, outY - 20);

            for (let r=0; r<outSize; r++) {
                for (let c=0; c<outSize; c++) {
                    const myIdx = r * outSize + c;
                    if (myIdx < currentStep) {
                        ctx.fillStyle = "#27ae60"; // Processed
                    } else if (myIdx === currentStep) {
                         ctx.fillStyle = "#e74c3c"; // Active
                    } else {
                        ctx.fillStyle = "#eee"; // Waiting
                    }
                    ctx.fillRect(outX + c*outCell, outY + r*outCell, outCell-2, outCell-2);
                }
            }
            
            ctx.fillStyle = "#666";
            ctx.font = "14px Arial";
            ctx.fillText(`Pooling Window: [${poolRow}, ${poolCol}] -> [${poolRow+2}, ${poolCol+2}]`, w/2, 350);
            ctx.fillText("Average Pooling aggregates spatial features", w/2, 370);

            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
    }

    function renderStep5(ctx, canvas) {
        let frame = 0;
        function animate() {
            if (!canvas.offsetParent) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const cx = canvas.width / 2;
            const cy = canvas.height / 2;
            
            const numTokens = 8;
            const tokenW = 20;
            const tokenH = 60;
            
            // Left: Vision Tokens
            const leftX = cx - 200;
            ctx.fillStyle = "#333";
            ctx.textAlign = "center";
            ctx.fillText("Vision Features (1024d)", leftX + (numTokens*tokenW)/2, cy - 50);
            
            for (let i=0; i<numTokens; i++) {
                ctx.fillStyle = "#3498db";
                ctx.fillRect(leftX + i*22, cy - 30, tokenW, tokenH);
            }
            
            // Matrix Op
            const matX = cx;
            const pulse = Math.sin(frame * 0.1) * 0.5 + 0.5;
            
            ctx.save();
            ctx.translate(matX, cy);
            ctx.rotate(frame * 0.01);
            ctx.strokeStyle = `rgba(46, 204, 113, ${pulse + 0.2})`;
            ctx.lineWidth = 2;
            ctx.strokeRect(-30, -30, 60, 60);
            ctx.restore();
            
            ctx.fillStyle = "#2ecc71";
            ctx.fillText("Projection Matrix", matX, cy + 50);
            ctx.fillText("W [1024, 768]", matX, cy + 70);

            drawArrow(ctx, leftX + numTokens*22 + 10, cy, matX - 40, cy);
            drawArrow(ctx, matX + 40, cy, cx + 120, cy);
            
            // Right: LLM Tokens
            const rightX = cx + 130;
            ctx.fillText("LLM Embeddings (768d)", rightX + (numTokens*tokenW)/2, cy - 50);
            
            for (let i=0; i<numTokens; i++) {
                ctx.fillStyle = "#e67e22";
                ctx.fillRect(rightX + i*22, cy - 25, tokenW, 50); 
            }

            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
    }

    function renderStep6(ctx, canvas) {
        let frame = 0;
        function animate() {
            if (!canvas.offsetParent) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const w = canvas.width;
            
            const startY = 100;
            const gap = 10;
            const boxW = 50;
            const boxH = 50;
            
            const tokens = ["The", "image", "shows", "<img0>", "<img1>", "<img2>", "a", "cat"];
            const totalW = tokens.length * (boxW + gap);
            const startX = (w - totalW) / 2;
            
            ctx.fillStyle = "#333";
            ctx.textAlign = "center";
            ctx.fillText("Multimodal Sequence Construction", w/2, 50);
            
            // 1. Draw Template (Text + Holes)
            tokens.forEach((t, i) => {
                const x = startX + i * (boxW + gap);
                const isHole = t.startsWith("<img");
                
                ctx.strokeStyle = "#bdc3c7";
                ctx.lineWidth = 2;
                
                if (isHole) {
                    ctx.setLineDash([5, 5]);
                    ctx.strokeRect(x, startY, boxW, boxH);
                    ctx.setLineDash([]);
                    ctx.fillStyle = "#ecf0f1";
                    ctx.fillText("?", x + boxW/2, startY + boxH/2);
                } else {
                    ctx.fillStyle = "#fff";
                    ctx.fillRect(x, startY, boxW, boxH);
                    ctx.strokeRect(x, startY, boxW, boxH);
                    ctx.fillStyle = "#333";
                    ctx.fillText(t, x + boxW/2, startY + boxH/2);
                }
            });
            
            // 2. Incoming Vision Tokens
            const flyY = 250;
            const visTokens = ["<img0>", "<img1>", "<img2>"];
            
            visTokens.forEach((t, i) => {
                const targetIdx = 3 + i;
                const targetX = startX + targetIdx * (boxW + gap);
                const targetY = startY;
                
                const delay = i * 30;
                const progress = Math.min(1, Math.max(0, (frame - delay) / 60));
                const ease = 1 - Math.pow(1 - progress, 3);
                
                const currentX = (w/2 - 60 + i*60) * (1-ease) + targetX * ease;
                const currentY = flyY * (1-ease) + targetY * ease;
                
                ctx.save();
                ctx.shadowColor = "rgba(0,0,0,0.3)";
                ctx.shadowBlur = 10;
                ctx.fillStyle = "#e67e22";
                ctx.fillRect(currentX, currentY, boxW, boxH);
                ctx.fillStyle = "#fff";
                ctx.fillText("VIS", currentX + boxW/2, currentY + boxH/2);
                ctx.restore();
                
                if (progress === 1) {
                    ctx.strokeStyle = "#e67e22";
                    ctx.lineWidth = 3;
                    ctx.strokeRect(targetX - 2, targetY - 2, boxW + 4, boxH + 4);
                }
            });

            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
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
