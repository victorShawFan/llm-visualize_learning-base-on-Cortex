document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const resetBtn = document.getElementById('reset-btn');
    const container = document.getElementById('log-container');

    // Guard
    if (!container) {
        console.error("Required elements not found in log_script");
        return;
    }

    const steps = [
        {
            title: "局部指标计算 (Local Metrics)",
            desc: "每个 GPU 基于本地数据计算 Loss、Throughput (TGS) 等。点击 'Refresh Data' 模拟不同 Rank 的计算结果。",
            badge: "trainer.py:520",
            state: "local"
        },
        {
            title: "全局同步 (All-Reduce AVG)",
            desc: "调用 dist.all_reduce，所有进程互相通信，计算全局平均 Loss。同步后，所有 Rank 持有的 avg_loss 变量值完全相等。",
            badge: "trainer.py:744",
            state: "sync"
        },
        {
            title: "主进程格式化 (Rank 0 Format)",
            desc: "仅 Rank 0 进入 _log 分支。拼接 Epoch, Step, LR 等标签，并附加平均后的 Loss 结果。",
            badge: "trainer.py:611",
            state: "format"
        },
        {
            title: "日志落盘 (Persistence)",
            desc: "Rank 0 调用 log() 函数。日志同时打印到标准输出（Console）并追加写入 LOG_DIR 下的 log.txt 文件。",
            badge: "log.py:26",
            state: "disk"
        },
        {
            title: "LOG_DIR 与日志目录 (_get_log_dir)",
            desc: "所有日志文件都会写入环境变量 <code>LOG_DIR</code> 指定的目录。<code>_get_log_dir()</code> 会在目录不存在时自动创建，并保证路径以 '/' 结尾。<span class='step-badge'>log.py:9-23</span>",
            badge: "log_dir = os.environ['LOG_DIR']\nif not os.path.exists(log_dir): os.mkdir(log_dir)",
            state: "logdir"
        },
        {
            title: "评估输出日志 (gen.txt)",
            desc: "eval.py 在每次评估生成结束后，将 <code>{tag}, gen-&gt;{文本}</code> 追加写入 <code>gen.txt</code>，用于离线人工阅读或自动评测。<span class='step-badge'>eval.py:51-52</span>",
            badge: "with open(f'{_get_log_dir()}gen.txt', 'a') as f:\n    f.write(f'{tag}, gen-> {gen_result}\\n')",
            state: "gen"
        },
        {
            title: "学习率日志 (lr.txt)",
            desc: "当 WarmupCosineAnnealingLRScheduler 的 <code>need_log=True</code> 时，每次更新学习率都会把 <code>step, lr</code> 写入 <code>lr.txt</code>，方便可视化 LR 曲线。<span class='step-badge'>scheduler.py:124-125</span>",
            badge: "if self.need_log:\n    log(f'step: {self.cur_steps}, lr: {lr}', 'lr.txt')",
            state: "lr"
        },
        {
            title: "异常日志 (exception.txt)",
            desc: "在训练过程中捕获到异常时，Trainer 会将 epoch、batch 以及异常位置写入 <code>exception.txt</code>，便于复现和排查问题。<span class='step-badge'>trainer.py:632-641</span>",
            badge: "log_msg = f'epoch: {epoch}, batch: {batch} -> {e} at {file} line {line}'\nlog(log_msg, 'exception.txt')",
            state: "exc"
        },
        {
            title: "控制台输出 vs 仅文件日志",
            desc: "<code>log(msg)</code> 只打印到控制台；<code>log(msg, 'log.txt')</code> 只写文件，不打印。Trainer 默认两者都调用一次，在终端和 log.txt 中保留相同的训练轨迹。<span class='step-badge'>log.py:26-43, trainer.py:611-621</span>",
            badge: "log(log_msg)          # console\nlog(f'{log_msg}\\n', 'log.txt')  # file",
            state: "console"
        }
    ];

    let currentStep = 0;
    let localLosses = [2.50, 2.70, 2.40, 2.80];
    let globalLoss = 2.60;

    function init() {
        currentStep = 0;
        generateData();
        render();
    }

    function generateData() {
        localLosses = Array.from({length: 4}, () => (2.0 + Math.random()).toFixed(2));
        const sum = localLosses.reduce((a, b) => parseFloat(a) + parseFloat(b), 0);
        globalLoss = (sum / 4).toFixed(4);
    }

    function render() {
        if (!container) return;
        const step = steps[currentStep];
        
        // Update labels if they exist
        const currSpan = document.getElementById('current-step');
        const totalSpan = document.getElementById('total-step');
        const titleEl = document.getElementById('step-title');
        const descEl = document.getElementById('step-desc');
        const badgeEl = document.querySelector('.step-badge');

        if (currSpan) currSpan.innerText = currentStep + 1;
        if (totalSpan) totalSpan.innerText = steps.length;
        if (titleEl) titleEl.innerText = step.title;
        if (descEl) descEl.innerHTML = step.desc;
        if (badgeEl) badgeEl.innerText = step.badge;

        container.innerHTML = '';
        
        try {
            if (step.state === 'local') renderLocal();
            else if (step.state === 'sync') renderSync();
            else if (step.state === 'format') renderFormat();
            else if (step.state === 'disk') renderDisk();
            else if (step.state === 'logdir') renderLogDir();
            else if (step.state === 'gen') renderGenLog();
            else if (step.state === 'lr') renderLrLog();
            else if (step.state === 'exc') renderExceptionLog();
            else if (step.state === 'console') renderConsoleVsFile();
        } catch (e) {
            console.error("Render error", e);
        }

        updateButtons();
    }

    function updateButtons() {
        if (prevBtn) prevBtn.disabled = currentStep === 0;
        if (nextBtn) nextBtn.innerText = currentStep === steps.length - 1 ? "完成" : "下一步";
    }

    function goNext() {
        if (currentStep < steps.length - 1) {
            currentStep++;
            render();
        }
    }

    function goPrev() {
        if (currentStep > 0) {
            currentStep--;
            render();
        }
    }

    // --- Renderers ---

    function renderLocal() {
        container.innerHTML = `
            <div style="margin-bottom: 20px;">
                <button id="refresh-data-btn" style="padding: 5px 15px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer;">Refresh Data</button>
            </div>
            <div class="metrics-row" style="display: flex; justify-content: space-around; gap: 10px;">
                ${localLosses.map((loss, i) => `
                    <div class="metric-card" style="border: 1px solid #ddd; padding: 15px; border-radius: 8px; background: #f9f9f9; text-align: center; width: 80px;">
                        <div style="font-weight: bold; margin-bottom: 5px;">Rank ${i}</div>
                        <div style="color: #e74c3c; font-family: monospace;">Loss: ${loss}</div>
                    </div>
                `).join('')}
            </div>
        `;
        
        const btn = document.getElementById('refresh-data-btn');
        if(btn) btn.addEventListener('click', () => {
            generateData();
            renderLocal();
        });
    }

    function renderSync() {
        container.innerHTML = `
            <div class="all-reduce-box" style="display: flex; flex-direction: column; align-items: center; gap: 20px;">
                <div style="display: flex; gap: 20px;">
                     ${localLosses.map((loss, i) => `
                        <div class="reduce-node" style="padding: 10px; border: 1px dashed #aaa; border-radius: 4px;">Rank ${i}: ${loss}</div>
                    `).join('')}
                </div>
                <div style="font-size: 24px; color: #aaa;">⬇️ dist.all_reduce(op=SUM) / world_size ⬇️</div>
                <div class="result-badge" style="background: #2ecc71; color: white; padding: 10px 30px; border-radius: 20px; font-size: 1.2em; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    Global Average Loss: ${globalLoss}
                </div>
                <div style="font-size: 0.9em; color: #666; margin-top: 10px;">
                    Formula: (${localLosses.join(' + ')}) / 4 = ${globalLoss}
                </div>
            </div>
        `;
    }

    function renderFormat() {
        container.innerHTML = `
            <div class="terminal-mock" style="background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; font-family: 'Menlo', monospace;">
                <div class="cursor">_</div>
                <div class="log-line">epoch: 1, step: 100 -> loss: ${globalLoss}, lr: 0.0001</div>
            </div>
            <div class="rank0-only" style="margin-top: 10px; font-style: italic; color: #7f8c8d;">* Only Rank 0 executes this logging block</div>
        `;
    }

    function renderDisk() {
        const curDate = new Date().toISOString().split('T')[0];
        const curTime = new Date().toLocaleTimeString();
        container.innerHTML = `
            <div class="file-system" style="border: 1px solid #ddd; border-radius: 5px; overflow: hidden;">
                <div class="file-header" style="background: #eee; padding: 5px 10px; border-bottom: 1px solid #ddd; display: flex; align-items: center; gap: 5px;">
                    <span>📄</span> <strong>log.txt</strong>
                </div>
                <div class="file-content" style="padding: 15px; font-family: monospace; font-size: 0.9em; background: #fff;">
                    [${curDate} ${curTime}] epoch: 1, step: 99 -> loss: ${(parseFloat(globalLoss)+0.05).toFixed(4)}...<br>
                    <span class="new-line" style="background: #fff3cd;">[${curDate} ${curTime}] epoch: 1, step: 100 -> loss: ${globalLoss}...</span>
                </div>
            </div>
        `;
    }

    function renderLogDir() {
        container.innerHTML = `
            <div class="file-system">
                <div class="file-icon" style="font-size: 3em; text-align: center;">📁</div>
                <div style="text-align: center; font-weight: bold;">LOG_DIR</div>
                <div class="file-content" style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 4px;">
                    LOG_DIR=/workspace/logs/exp1<br>
                    <hr style="margin: 5px 0; border: 0; border-top: 1px solid #ddd;">
                    若目录不存在，_get_log_dir() 会自动创建。<br>
                    所有日志文件 (log.txt, lr.txt, gen.txt, exception.txt) 都写在这里。
                </div>
            </div>
        `;
    }

    function renderGenLog() {
        container.innerHTML = `
            <div class="file-system">
                <div class="file-header" style="background: #eee; padding: 5px 10px;">📄 gen.txt</div>
                <div class="file-content" style="padding: 10px; font-family:monospace; background: #fff;">
                    sign:batch/epoch:0/batch:100, gen-> Cortex 是一个...<br>
                    sign:epoch/epoch:0, gen-> 在本轮训练中模型学到了...
                </div>
            </div>
        `;
    }

    function renderLrLog() {
        container.innerHTML = `
            <div class="file-system">
                <div class="file-header" style="background: #eee; padding: 5px 10px;">📄 lr.txt</div>
                <div class="file-content" style="padding: 10px; font-family:monospace; background: #fff;">
                    [2023-10-27 10:00:01] step: 0, lr: 1e-05<br>
                    [2023-10-27 10:05:01] step: 1000, lr: 8e-05<br>
                    [2023-10-27 10:10:01] step: 2000, lr: 3e-04
                </div>
            </div>
        `;
    }

    function renderExceptionLog() {
        container.innerHTML = `
            <div class="file-system">
                <div class="file-header" style="background: #fee; padding: 5px 10px; color: #c0392b;">📄 exception.txt</div>
                <div class="file-content" style="padding: 10px; font-family:monospace; background: #fff;">
                    [2023-10-27 11:00:00] epoch: 1, batch: 5 -> CUDA out of memory at trainer.py line 840
                </div>
            </div>
        `;
    }

    function renderConsoleVsFile() {
        container.innerHTML = `
            <div class="terminal-mock" style="background: #000; color: #0f0; padding: 10px; font-family: monospace; border-radius: 4px;">
                <div class="log-line">[2023-10-27 10:00:05] epoch: 1, batch: 100 -> loss: ${globalLoss}, lr: 0.0001</div>
            </div>
            <div class="arrow-down" style="text-align: center; font-size: 20px; margin: 10px 0;">⬇️ ALSO ⬇️</div>
            <div class="file-system" style="border: 1px solid #ddd;">
                <div class="file-header" style="background: #eee; padding: 5px;">📄 log.txt</div>
                <div class="file-content" style="padding: 10px;">
                    同一条 log_msg 也会被追加写入到 log.txt 中，便于离线分析与可视化。
                </div>
            </div>
        `;
    }

    // Bind events
    if (nextBtn) nextBtn.addEventListener('click', goNext);
    if (prevBtn) prevBtn.addEventListener('click', goPrev);
    if (resetBtn) resetBtn.addEventListener('click', init);

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });

    // Init
    init();
});
