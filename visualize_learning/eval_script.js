document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');
    
    // Guard
    if (!visualContent) {
        console.error("Required elements not found in eval_script");
        return;
    }

    let currentStep = 0;
    let animationId = null;

    const steps = [
        {
            title: "步骤 0: 梯度累积 (Accumulation Phase)",
            description: "为了节省显存，源码将大 Batch 切分为多个 Micro-batches。在累积阶段，梯度会被暂存而不触发同步。<br><span class='step-badge'>trainer.py:785-799</span>",
            code: "loss = model(batch)\nloss.backward()\n# No optimizer.step() yet\n# Gradient Accumulation: 1/4",
            render: (ctx, canvas) => renderAccumulation(ctx, canvas)
        },
        {
            title: "步骤 1: 参数更新 (Weight Update)",
            description: "达到指定的累积步数后，执行梯度裁剪并调用 <code>_apply_step()</code>。此时所有 RANK 的参数进行同步更新。<br><span class='step-badge'>trainer.py:855-857</span>",
            code: "self._apply_grad_clipping()\nself._apply_step()\n# optimizer.step() called",
            render: (ctx, canvas) => renderStepUpdate(ctx, canvas)
        },
        {
            title: "步骤 2: 指标聚合 (All-Reduce Loss)",
            description: "训练损失在各卡上独立计算。通过 <code>dist.all_reduce</code> 对每个 loss 标量做平均，确保日志中记录的是全局训练状态。<br><span class='step-badge'>trainer.py:736-755</span>",
            code: "avg_loss = loss.item()\ndist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)\n# All ranks get the global average",
            render: (ctx, canvas) => renderAllReduce(ctx, canvas)
        },
        {
            title: "步骤 3: 触发定期评估 (Eval Interval)",
            description: "每隔 <code>eval_batch_interval</code> 步，系统会自动保存 Checkpoint 并启动生成任务。<br><span class='step-badge'>trainer.py:890</span>",
            code: "if (batch - last_ckpt_batch) >= interval:\n    save_checkpoint()\n    self._eval()",
            render: (ctx, canvas) => renderTriggerEval(ctx, canvas)
        },
        {
            title: "步骤 4: 评估执行 (Single Process Generation)",
            description: "生成任务仅在 <b>Rank 0</b> 上执行。其他 Rank 会进入等待状态 (Barrier)，直到 Rank 0 完成生成并写入日志。<br><span class='step-badge'>trainer.py:667-690</span>",
            code: "if is_main_process:\n    generate(eval_model, prompt)\nelse:\n    dist.barrier() # Wait for Rank 0",
            render: (ctx, canvas) => renderGeneration(ctx, canvas)
        },
        {
            title: "步骤 5: 结果记录与恢复训练",
            description: "生成的文本被写入 `gen.txt`。之后模型切回 `train()` 模式，所有 Rank 同步继续训练。<br><span class='step-badge'>log.py:9-23</span>",
            code: "with open('gen.txt', 'a') as f:\n    f.write(f'{tag}, gen-> {result}')\n\nmodel.train()\n# Resume training loop",
            render: (ctx, canvas) => renderResume(ctx, canvas)
        }
    ];

    function updateUI() {
        // Cleanup previous animation
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }

        const step = steps[currentStep];
        if (infoBox) infoBox.innerHTML = `<strong>${step.title}</strong><br>${step.description}`;
        if (codeSnippet) {
            codeSnippet.textContent = step.code;
            if (window.hljs) hljs.highlightElement(codeSnippet);
        }

        // Re-create canvas to ensure clean state
        visualContent.innerHTML = '<canvas id="evalCanvas"></canvas>';
        const canvas = document.getElementById('evalCanvas');
        if (canvas) {
            resizeCanvas(canvas);
            const ctx = canvas.getContext('2d');
            step.render(ctx, canvas);
        }
        
        updateButtons();
    }
    
    function resizeCanvas(canvas) {
        if (!canvas || !visualContent) return;
        canvas.width = visualContent.clientWidth || 800;
        canvas.height = 400;
    }
    
    window.addEventListener('resize', () => {
        const canvas = document.getElementById('evalCanvas');
        if (canvas) resizeCanvas(canvas);
    });

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
    function drawRankBox(ctx, x, y, rankId, active=true, label="") {
        const w = 100;
        const h = 120;
        
        ctx.fillStyle = active ? "#fff" : "#ecf0f1";
        ctx.strokeStyle = active ? "#2c3e50" : "#bdc3c7";
        ctx.lineWidth = 2;
        
        // Shadow
        if (active) {
            ctx.shadowColor = "rgba(0,0,0,0.1)";
            ctx.shadowBlur = 10;
            ctx.shadowOffsetY = 5;
        } else {
            ctx.shadowColor = "transparent";
        }
        
        ctx.fillRect(x, y, w, h);
        ctx.strokeRect(x, y, w, h);
        
        // Header
        ctx.fillStyle = active ? "#34495e" : "#95a5a6";
        ctx.fillRect(x, y, w, 30);
        ctx.fillStyle = "#fff";
        ctx.font = "bold 14px Arial";
        ctx.textAlign = "center";
        ctx.fillText(`RANK ${rankId}`, x + w/2, y + 20);
        
        ctx.shadowColor = "transparent";
        
        // Content Label
        if (label) {
            ctx.fillStyle = "#333";
            ctx.font = "12px Arial";
            ctx.fillText(label, x + w/2, y + 60);
        }
        
        return {x, y, w, h, centerX: x+w/2, centerY: y+h/2};
    }

    // --- Renderers ---

    function renderAccumulation(ctx, canvas) {
        let frame = 0;
        const maxAccum = 4;
        
        function animate() {
            if (!canvas.offsetParent) return; // Stop if hidden
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const w = canvas.width;
            
            const spacing = 150;
            const startX = (w - 3*100 - 2*50) / 2;
            
            // Draw 3 Ranks
            for (let i=0; i<3; i++) {
                const bx = startX + i*spacing;
                const by = 100;
                const box = drawRankBox(ctx, bx, by, i, true);
                
                // Draw Accumulation Bar inside
                const barW = 80;
                const barH = 10;
                const barX = box.x + 10;
                const barY = box.y + 80;
                
                ctx.fillStyle = "#ecf0f1";
                ctx.fillRect(barX, barY, barW, barH);
                
                // Fill
                const cycle = 200;
                const progress = (frame % cycle) / cycle; // 0 to 1
                const accumStep = Math.floor(progress * maxAccum) + 1; // 1 to 4
                
                const fillW = (accumStep / maxAccum) * barW;
                ctx.fillStyle = "#e67e22";
                ctx.fillRect(barX, barY, fillW, barH);
                
                ctx.fillStyle = "#666";
                ctx.fillText(`Grads: ${accumStep}/${maxAccum}`, box.centerX, box.y + 70);
            }
            
            ctx.fillStyle = "#333";
            ctx.textAlign = "center";
            ctx.font = "16px Arial";
            ctx.fillText("Gradient Accumulation Phase", w/2, 50);

            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
    }

    function renderStepUpdate(ctx, canvas) {
        let frame = 0;
        
        function animate() {
            if (!canvas.offsetParent) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const w = canvas.width;
            
            const spacing = 150;
            const startX = (w - 3*100 - 2*50) / 2;
            
            // Pulse effect
            const pulse = Math.abs(Math.sin(frame * 0.1));
            
            for (let i=0; i<3; i++) {
                const bx = startX + i*spacing;
                const by = 100;
                
                ctx.save();
                if (pulse > 0.5) {
                    ctx.strokeStyle = `rgba(46, 204, 113, ${pulse})`;
                    ctx.lineWidth = 4;
                    ctx.strokeRect(bx-5, by-5, 110, 130);
                }
                ctx.restore();
                
                drawRankBox(ctx, bx, by, i, true, "Optimizer Step");
                
                // Flash text
                if (pulse > 0.5) {
                    ctx.fillStyle = "#27ae60";
                    ctx.font = "bold 16px Arial";
                    ctx.fillText("UPDATE!", bx + 50, by + 100);
                }
            }
            
            ctx.fillStyle = "#333";
            ctx.textAlign = "center";
            ctx.font = "16px Arial";
            ctx.fillText("Global Parameter Update", w/2, 50);

            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
    }

    function renderAllReduce(ctx, canvas) {
        let frame = 0;
        
        function animate() {
            if (!canvas.offsetParent) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const w = canvas.width;
            
            const spacing = 150;
            const startX = (w - 3*100 - 2*50) / 2;
            const startY = 100;
            
            const centers = [];
            
            // Draw Ranks
            for (let i=0; i<3; i++) {
                const bx = startX + i*spacing;
                const box = drawRankBox(ctx, bx, startY, i, true);
                centers.push({x: box.centerX, y: box.centerY});
                
                // Display local loss
                const localLoss = (2.5 + i*0.2).toFixed(2);
                ctx.fillStyle = "#333";
                ctx.fillText(`Loss: ${localLoss}`, box.centerX, box.y + 50);
            }
            
            // Animation Phase
            // 0-60: Particles move to center
            // 60-90: Merge
            // 90-150: Move back
            const cycle = 180;
            const t = frame % cycle;
            
            const globeX = w/2;
            const globeY = startY + 200;
            
            if (t < 60) {
                // Move to center
                const progress = t / 60; // 0->1
                const ease = progress * progress; // quad
                
                centers.forEach(c => {
                    const cx = c.x + (globeX - c.x) * ease;
                    const cy = c.y + (globeY - c.y) * ease;
                    
                    ctx.fillStyle = "#e74c3c";
                    ctx.beginPath();
                    ctx.arc(cx, cy, 5, 0, Math.PI*2);
                    ctx.fill();
                });
                
                ctx.fillStyle = "#333";
                ctx.fillText("Gathering...", globeX, globeY + 30);
                
            } else if (t < 90) {
                // Merging pulse
                const p = (t - 60) / 30;
                const r = 10 + Math.sin(p * Math.PI) * 10;
                
                ctx.fillStyle = "#2ecc71";
                ctx.beginPath();
                ctx.arc(globeX, globeY, r, 0, Math.PI*2);
                ctx.fill();
                
                ctx.fillStyle = "#333";
                ctx.fillText("Averaging...", globeX, globeY + 30);
                
            } else {
                // Move back
                const progress = (t - 90) / 60;
                 const ease = 1 - Math.pow(1 - progress, 3);
                
                centers.forEach(c => {
                    const cx = globeX + (c.x - globeX) * ease;
                    const cy = globeY + (c.y - globeY) * ease;
                    
                    ctx.fillStyle = "#2ecc71";
                    ctx.beginPath();
                    ctx.arc(cx, cy, 5, 0, Math.PI*2);
                    ctx.fill();
                });
                
                ctx.fillStyle = "#333";
                ctx.fillText("Broadcasting Avg", globeX, globeY + 30);
                
                if (progress > 0.8) {
                    centers.forEach(c => {
                         ctx.fillStyle = "#27ae60";
                         ctx.fillText("Avg: 2.70", c.x, startY + 90);
                    });
                }
            }

            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
    }

    function renderTriggerEval(ctx, canvas) {
        // Simple flash or icon showing Save -> Eval
        let frame = 0;
        function animate() {
            if (!canvas.offsetParent) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const w = canvas.width;
            const h = canvas.height;
            
            ctx.fillStyle = "#333";
            ctx.textAlign = "center";
            ctx.font = "20px Arial";
            ctx.fillText("Evaluation Triggered", w/2, h/2 - 50);
            
            // Floppy disk icon animation
            const scale = 1 + Math.sin(frame * 0.1) * 0.1;
            ctx.save();
            ctx.translate(w/2, h/2);
            ctx.scale(scale, scale);
            ctx.font = "40px Arial";
            ctx.fillText("💾 -> 🚀", 0, 0);
            ctx.restore();
            
            ctx.font = "14px Arial";
            ctx.fillText("Saving Checkpoint & Starting Eval...", w/2, h/2 + 50);
            
            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
    }

    function renderGeneration(ctx, canvas) {
        let frame = 0;
        
        function animate() {
            if (!canvas.offsetParent) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const w = canvas.width;
            const spacing = 150;
            const startX = (w - 3*100 - 2*50) / 2;
            const startY = 100;
            
            // Rank 0: Generating
            const r0 = drawRankBox(ctx, startX, startY, 0, true);
            
            // Rank 1 & 2: Waiting
            drawRankBox(ctx, startX + spacing, startY, 1, false);
            drawRankBox(ctx, startX + 2*spacing, startY, 2, false);
            
            // Rank 0 Text Animation
            const text = "Cortex is great";
            const charIndex = Math.floor(frame / 10) % (text.length + 5);
            const visibleText = text.substring(0, charIndex);
            
            ctx.fillStyle = "#2980b9";
            ctx.font = "12px monospace";
            ctx.fillText(visibleText, r0.centerX, r0.y + 80);
            
            // Blinking cursor
            if (Math.floor(frame / 20) % 2 === 0 && charIndex < text.length) {
                const metrics = ctx.measureText(visibleText);
                ctx.fillRect(r0.centerX + metrics.width/2 + 2, r0.y + 72, 2, 12);
            }
            
            // Zzz animation for others
            const zOffset = (frame % 60) / 60; // 0 to 1
            [1, 2].forEach(i => {
                const bx = startX + i*spacing;
                const by = startY;
                
                ctx.fillStyle = "#bdc3c7";
                ctx.font = "20px Arial";
                ctx.fillText("Wait...", bx + 50, by + 60);
                
                ctx.fillStyle = "#95a5a6";
                ctx.font = "14px Arial";
                ctx.globalAlpha = 1 - zOffset;
                ctx.fillText("z", bx + 70 + zOffset*10, by + 40 - zOffset*20);
                ctx.globalAlpha = 1.0;
            });

            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
    }

    function renderResume(ctx, canvas) {
        let frame = 0;
        function animate() {
            if (!canvas.offsetParent) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const w = canvas.width;
            
            ctx.fillStyle = "#333";
            ctx.textAlign = "center";
            ctx.font = "20px Arial";
            ctx.fillText("Resume Training", w/2, 100);
            
            // Log file
            ctx.fillStyle = "#2d3748";
            ctx.fillRect(w/2 - 150, 150, 300, 100);
            ctx.fillStyle = "#a0aec0";
            ctx.font = "12px monospace";
            ctx.textAlign = "left";
            ctx.fillText("gen.txt appended:", w/2 - 140, 170);
            ctx.fillStyle = "#48bb78";
            ctx.fillText("> Cortex is great", w/2 - 140, 190);
            
            // Progress bar resuming
            ctx.fillStyle = "#333";
            ctx.textAlign = "center";
            ctx.font = "14px Arial";
            ctx.fillText("All Ranks synced. Training continues...", w/2, 300);
            
            frame++;
            animationId = requestAnimationFrame(animate);
        }
        animate();
    }

    // Bind events
    if (nextBtn) nextBtn.addEventListener('click', goNext);
    if (prevBtn) prevBtn.addEventListener('click', goPrev);
    if (resetBtn) resetBtn.addEventListener('click', () => { currentStep = 0; updateUI(); });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });

    // Start
    updateUI();
});
