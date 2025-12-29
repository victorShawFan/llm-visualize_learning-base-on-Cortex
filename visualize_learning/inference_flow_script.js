document.addEventListener('DOMContentLoaded', () => {
    const visualContent = document.getElementById('visualContent');
    const infoBox = document.getElementById('infoBox');
    const codeSnippet = document.getElementById('codeSnippet');
    const nextBtn = document.getElementById('nextStep');
    const prevBtn = document.getElementById('prevStep');
    const resetBtn = document.getElementById('reset');
    const currentStepSpan = document.getElementById('current-step');

    if (!visualContent || !infoBox || !nextBtn || !prevBtn || !resetBtn || !currentStepSpan) {
        console.error("Critical DOM elements missing");
        return;
    }

    let currentStep = 0;

    // Simulation State
    const prompt = ["The", " sky", " is"];
    const generated = []; 
    const cacheState = []; // Array of strings stored in KV

    const steps = [
        {
            title: "Phase 1: Prefill - Input Processing",
            desc: "推理的第一阶段是 'Prefill'。我们将完整的 Prompt 输入模型。虽然是一次性输入，但模型并行计算所有位置的 Embedding。",
            code: "input_ids = tokenizer('The sky is')",
            render: () => {
                return `<div class='chat-container anim-fade'>
                    <div class='message user-msg'>The sky is</div>
                </div>
                <div class='layer-box active'>
                    <h3>Parallel Embedding</h3>
                    <div class='token-stream'>
                        <div class='token-chip new anim-fly-in' style='animation-delay:0s'>The</div>
                        <div class='token-chip new anim-fly-in' style='animation-delay:0.1s'>sky</div>
                        <div class='token-chip new anim-fly-in' style='animation-delay:0.2s'>is</div>
                    </div>
                </div>`;
            }
        },
        {
            title: "Phase 1: Prefill - KV Cache Population",
            desc: "在 Attention 层，我们将 Prompt 中每个 Token 的 Key 和 Value 向量存入 KV Cache，供后续生成使用。",
            code: "kv_cache.update(k, v, layer_idx)",
            render: () => {
                return `<div class='layer-box'>
                    <h3>KV Cache Update</h3>
                    <div class='kv-status anim-fade' style='animation-delay:0s'>
                        <span>Slot 0: "The"</span> <div class='cache-bar'><div class='cache-fill anim-pulse-cache' style='width:100%'></div></div>
                    </div>
                    <div class='kv-status anim-fade' style='animation-delay:0.1s'>
                        <span>Slot 1: "sky"</span> <div class='cache-bar'><div class='cache-fill anim-pulse-cache' style='width:100%'></div></div>
                    </div>
                    <div class='kv-status anim-fade' style='animation-delay:0.2s'>
                        <span>Slot 2: "is"</span> <div class='cache-bar'><div class='cache-fill anim-pulse-cache' style='width:100%'></div></div>
                    </div>
                </div>`;
            }
        },
        {
            title: "Phase 1: Prefill - First Generation",
            desc: "基于最后一个 Token 'is' 的输出 Logits，我们进行采样（Temperature/Top-P），选出下一个词。",
            code: "next_token = sample(logits[-1, :]) # ' blue'",
            render: () => {
                return `<div class='layer-box active'>
                    <h3>Sampling (Top-P)</h3>
                    <div class='prob-container'>
                        <div class='prob-row'>
                            <div class='prob-label'> blue</div>
                            <div class='prob-bar-bg'><div class='prob-bar-fill anim-grow' style='--width: 85%'></div></div>
                            <div style='margin-left:5px'>0.85</div>
                        </div>
                        <div class='prob-row'>
                            <div class='prob-label'> clear</div>
                            <div class='prob-bar-bg'><div class='prob-bar-fill anim-grow' style='--width: 10%; background:#bdc3c7'></div></div>
                            <div style='margin-left:5px'>0.10</div>
                        </div>
                        <div class='prob-row'>
                            <div class='prob-label'> dark</div>
                            <div class='prob-bar-bg'><div class='prob-bar-fill anim-grow' style='--width: 5%; background:#bdc3c7'></div></div>
                            <div style='margin-left:5px'>0.05</div>
                        </div>
                    </div>
                    <div style='margin-top:10px; font-weight:bold; color:#2e7d32'>Selected: " blue"</div>
                </div>
                <div class='chat-container'>
                    <div class='message user-msg'>The sky is</div>
                    <div class='message bot-msg typing'> blue</div>
                </div>`;
            }
        },
        {
            title: "Phase 2: Decode Step 1 - Input",
            desc: "进入 'Decode' 阶段。我们只把刚刚生成的 ' blue' 作为输入喂给模型，而不是整个句子。",
            code: "input_ids = [' blue']",
            render: () => {
                return `<div class='layer-box active'>
                    <h3>Single Token Input</h3>
                    <div class='token-chip new anim-fly-in'> blue</div>
                </div>`;
            }
        },
        {
            title: "Phase 2: Decode - Attention with Cache",
            desc: "Attention 层计算时，Query 来自 ' blue'，Key/Value 来自 KV Cache（'The', 'sky', 'is'）以及当前的 ' blue'。",
            code: "attn_output = attention(q, k_cache + k_curr, v_cache + v_curr)",
            render: () => {
                return `<div class='layer-box'>
                    <h3>Attention Mechanism</h3>
                    <p>Query: " blue"</p>
                    <div class='arrow-down'>⬇️ attends to ⬇️</div>
                    <div class='token-stream' style='justify-content:center; opacity:0.8'>
                        <div class='token-chip anim-fade' style='animation-delay:0s'>The</div>
                        <div class='token-chip anim-fade' style='animation-delay:0.1s'>sky</div>
                        <div class='token-chip anim-fade' style='animation-delay:0.2s'>is</div>
                        <div class='token-chip new anim-fly-in' style='animation-delay:0.3s; border-color:#2ecc71'> blue</div>
                    </div>
                </div>`;
            }
        },
        {
            title: "Phase 2: Decode - Cache Update",
            desc: "计算完成后，将 ' blue' 的 K/V 也存入 Cache，供下一轮使用。",
            code: "kv_cache.append(k_blue, v_blue)",
            render: () => {
                return `<div class='layer-box'>
                    <h3>KV Cache Status</h3>
                    <p>Cache Size: 3 -> 4</p>
                    <div class='kv-status'>
                        <span>Slots [0-3] Filled</span> 
                        <div class='cache-bar'><div class='cache-fill anim-grow' style='--width: 100%'></div></div>
                    </div>
                    <div style='font-size:12px; color:#888; margin-top:5px'>Appending new K/V tensor...</div>
                </div>`;
            }
        },
        {
            title: "Phase 2: Decode - Generation",
            desc: "再次采样，生成下一个 Token。",
            code: "next_token = sample(logits) # '.'",
            render: () => {
                return `<div class='layer-box active'>
                    <h3>Sampling</h3>
                     <div class='prob-container'>
                        <div class='prob-row'>
                            <div class='prob-label'> .</div>
                            <div class='prob-bar-bg'><div class='prob-bar-fill anim-grow' style='--width: 90%'></div></div>
                            <div style='margin-left:5px'>0.90</div>
                        </div>
                        <div class='prob-row'>
                            <div class='prob-label'> ,</div>
                            <div class='prob-bar-bg'><div class='prob-bar-fill anim-grow' style='--width: 8%; background:#bdc3c7'></div></div>
                            <div style='margin-left:5px'>0.08</div>
                        </div>
                    </div>
                    <div style='margin-top:10px; font-weight:bold; color:#2e7d32'>Selected: "."</div>
                </div>
                <div class='chat-container'>
                    <div class='message user-msg'>The sky is</div>
                    <div class='message bot-msg typing'> blue.</div>
                </div>`;
            }
        },
        {
            title: "Phase 3: Completion",
            desc: "重复上述过程，直到生成停止符（EOS）。",
            code: "if next_token == EOS_ID: break",
            render: () => {
                return `<div class='chat-container'>
                    <div class='message user-msg'>The sky is</div>
                    <div class='message bot-msg'> blue.</div>
                </div>
                <div class='layer-box anim-fade' style='background:#d5f5e3'>
                    <h3>Inference Complete</h3>
                    <p>Total Tokens: 5</p>
                    <p style='font-size:12px; color:#666'>Performance: 45 tok/s</p>
                </div>`;
            }
        }
    ];

    function renderStep(index) {
        if (index < 0) index = 0;
        if (index >= steps.length) index = steps.length - 1;
        
        currentStep = index;
        const stepData = steps[index];

        // Update UI
        currentStepSpan.textContent = currentStep + 1;
        infoBox.innerHTML = `<h3>${stepData.title}</h3><p>${stepData.desc}</p>`;
        codeSnippet.textContent = stepData.code || "";
        visualContent.innerHTML = stepData.render();

        // Update buttons
        prevBtn.disabled = currentStep === 0;
        nextBtn.disabled = currentStep === steps.length - 1;
    }

    // Event Listeners
    nextBtn.addEventListener('click', () => renderStep(currentStep + 1));
    prevBtn.addEventListener('click', () => renderStep(currentStep - 1));
    resetBtn.addEventListener('click', () => renderStep(0));

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') renderStep(currentStep - 1);
        if (e.key === 'ArrowRight') renderStep(currentStep + 1);
    });

    // Initial render
    renderStep(0);
});
