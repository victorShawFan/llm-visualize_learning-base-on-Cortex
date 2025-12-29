document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');
    
    // Guard
    if (!visualContent) {
        console.error("Required elements not found in online_pipeline_script");
        return;
    }

    let currentStep = 0;

    // 在线推理总览：Browser → Bottle /api/chat → search.py → streaming_generate(_generate) → SSE
    const steps = [
      {
        title: 'Phase 0: 浏览器前端与 /api/chat 请求',
        description:
          '前端聊天页 <code>static/index.html</code> 中，点击“发送”后会读取输入框内容、思考模式、深度搜索、采样参数等，构造 JSON payload 并通过 <code>fetch(\'/api/chat\')</code> 以 POST 方式发送到后端。payload 中包含 <code>history</code>（多轮对话）、<code>thinking</code> 开关、<code>deep_search</code> 标志、<code>temperature</code>/<code>top_p</code>、思考预算以及用户 UUID。<span class="step-badge">static/index.html:189-395</span>',
        code: `await fetch('/api/chat', {\n  method: 'POST',\n  headers: { 'Content-Type': 'application/json' },\n  body: JSON.stringify({\n    history,\n    thinking: isThinkingEnabled,\n    deep_search: isDeepSearchEnabled,\n    temperature,\n    top_p: topP,\n    think_budget_enable: isThinkingBudgetEnabled,\n    think_budget: thinkingBudgetValue,\n    uuid: user_uuid\n  })\n})`,
        render: () => renderPhase0(),
      },
      {
        title: 'Phase 1: Bottle 后端入口与请求解析',
        description:
          '<code>sse_chat</code> 路由函数将响应头设为 <code>text/event-stream</code>，以 SSE 形式向前端持续推送 JSON 行。首先从 <code>request.json</code> 中解析 <code>history/thinking/uuid/temperature/top_p/deep_search/think_budget_enable/think_budget</code> 等字段，并根据 <code>deep_search</code> 决定是否创建搜索 API 与是否截断历史轮数。若未提供思考预算，则自动关闭预算逻辑。<span class="step-badge">app.py:71-103</span>',
        code: `payload = request.json\nchat_history = payload.get('history')\nthinking = payload.get('thinking')\ndeep_search = payload.get('deep_search')\n...\nif deep_search:\n    search_api = get_search_api()\n    chat_history = chat_history[-1:]  # 只保留当前轮\nelse:\n    search_api = None\n    chat_history = chat_history[-3:]  # 保留最近三轮`,
        render: () => renderPhase1(),
      },
      {
        title: 'Phase 2: 系统提示词与搜索结果注入',
        description:
          '根据是否使用搜索 API，后端会构造不同的 system 提示词：<br>• deep_search=True 时，system 说明“需要根据搜索结果回答问题”；<br>• 否则 system 内容为一个占位空串。随后调用 <code>TrainerTools().tokenizer.apply_chat_template</code> 将多轮对话拼成模板，并在末尾追加 <code>&lt;assistant&gt;</code>，作为生成的起点。<span class="step-badge">app.py:111-119</span>',
        code: `if search_api:\n    chat_history = [{'role': 'system', 'content': '我需要根据用户的问题和搜索到的结果给出用户答案，回答的方式要简明扼要'}, *chat_history]\nelse:\n    chat_history = [{'role': 'system', 'content': ' '}, *chat_history]\nchat_template = TrainerTools().tokenizer.apply_chat_template(chat_history, tokenizer=False)\nchat_template = f'{chat_template}<assistant>'`,
        render: () => renderPhase2(),
      },
      {
        title: 'Phase 3: 调用 search.py 并在 &lt;think&gt; 中注入搜索摘要',
        description:
          '当 <code>deep_search=True</code> 时，<code>get_search_api</code> 会返回封装好的 Brave/BochaAI 搜索函数。后端提取用户 query（去掉 /think 或 /no think 指令），向搜索 API 发起请求并取前若干条 Summary。若有结果，则立即向 SSE 流发送一个 <code>thinking_chunk</code> 事件展示“搜索到的内容”，同时把同样的文本作为 <code>&lt;think&gt;...&lt;/think&gt;</code> 注入到 <code>chat_template</code>，成为模型思考的一部分。<span class="step-badge">search.py:32-60, app.py:120-127</span>',
        code: `user_prompt = chat_history[-1]['content'].replace('/think', '').replace('/no think', '')\nsearch_result = search_api(user_prompt)\nif search_result:\n    search_msg = f"根据用户的问题，我搜索到了如下内容：\n{search_result}"\n    yield fmt_msg('thinking_chunk', search_msg)\n    chat_template = f"{chat_template}<think>{search_msg}</think>"`,
        render: () => renderPhase3(),
      },
      {
        title: 'Phase 4: Token 化与思考预算 (Think Budget)',
        description:
          '将 <code>chat_template</code> 用训练时同一套 Tokenizer 编码为 <code>prompt_token</code>，并根据目标上下文长度 2048 计算剩余可生成 token 数。若 <code>thinking</code> 与 <code>think_budget_enable</code> 同时开启，则先预留一个文本片段 <code>think_budget_content</code>，计算其编码长度并从 <code>think_budget</code> 中扣除，确保思考阶段不会超出预算；若前端未提供预算值，则在后端自动关闭预算功能。<span class="step-badge">app.py:87-92, 128-136</span>',
        code: `prompt_token = TrainerTools().tokenizer.encode(chat_template, unsqueeze=True)\noutput_token_count = max(2048 - prompt_token.shape[-1], 0)\nif think_budget_enable:\n    think_budget_content = '。考虑到用户的时间限制，我现在必须根据思考直接给出解决方案\n'\n    think_budget_encoded = TrainerTools().tokenizer.encode(f'{think_budget_content}</think>')\n    think_budget = think_budget - len(think_budget_encoded)\n    output_token_count = min(think_budget, output_token_count)`,
        render: () => renderPhase4(),
      },
      {
        title: 'Phase 5: streaming_generate 第一次调用——思考阶段',
        description:
          '当启用思考预算时，后端会先以当前 <code>prompt_token</code> 和精简后的 max_new_tokens 调用 <code>streaming_generate</code>。在流中累积字符串 <code>think_content</code>，遇到 <code>&lt;/think&gt;</code> 立即停止思考阶段，将每个 chunk 以 <code>thinking_chunk</code> 事件推送给前端；若在预算内未生成 <code>&lt;/think&gt;</code>，则强制追加 <code>think_budget_content</code> 和结束标签。<span class="step-badge">app.py:137-156, llm_trainer/generate_utils.py:204-260</span>',
        code: `generator = streaming_generate(model=model, prompt=prompt_token, max_new_tokens=output_token_count, ...)\nthink_content = ''\nfor chunk in generator:\n    think_content += chunk\n    if chunk == '</think>': break\n    yield fmt_msg('thinking_chunk', chunk)\nif '</think>' not in think_content:\n    think_content += f'{think_budget_content}</think>'\n    yield fmt_msg('thinking_chunk', think_budget_content)`,
        render: () => renderPhase5(),
      },
      {
        title: 'Phase 6: streaming_generate 第二次调用——回答阶段',
        description:
          '思考阶段结束后，将 <code>think_content</code> 再次编码并与原始 <code>prompt_token</code> 拼接，重新计算可用 token 数，然后再次调用 <code>streaming_generate</code> 进行回答阶段生成。此时输出 token 流中会交替出现 <code>&lt;think&gt;/&lt;/think&gt;/&lt;answer&gt;/&lt;/answer&gt;/&lt;/s&gt;</code> 等标签，后端会用一个简单状态机进行过滤。<span class="step-badge">app.py:157-168, 170-187</span>',
        code: `prompt_token = torch.concat([prompt_token, TrainerTools().tokenizer.encode(think_content, unsqueeze=True)], dim=-1)\noutput_token_count = max(2048 - prompt_token.shape[-1], 0)\ngenerator = streaming_generate(model=model, prompt=prompt_token, max_new_tokens=output_token_count, ...)`,
        render: () => renderPhase6(),
      },
      {
        title: 'Phase 7: 标记流过滤与 SSE 事件类型',
        description:
          '后端遍历第二次生成的 token 流，通过状态变量 <code>msg_type</code> 区分“思考片段”与“答案片段”：<br>• 遇到 <code>&lt;think&gt;</code> → <code>msg_type = "thinking_chunk"</code>；<br>• 遇到 <code>&lt;answer&gt;</code> → <code>msg_type = "answer_chunk"</code>；<br>• 遇到 <code>&lt;/s&gt;</code> 终止；<br>• 所有标签本身（<code>&lt;assistant&gt;</code>、结束标签等）都被跳过，只把真正的内容 token 通过 <code>fmt_msg</code> 编码为 SSE JSON 行返回。<span class="step-badge">app.py:170-188</span>',
        code: `msg_type = None\nfor chunk in generator:\n    if chunk == '</s>': break\n    if chunk == '<assistant>' or chunk == '</assistant>': continue\n    if chunk == '</think>' or chunk == '</answer>': continue\n    if chunk == '<think>': msg_type = 'thinking_chunk'; continue\n    elif chunk == '<answer>': msg_type = 'answer_chunk'; continue\n    if msg_type: yield fmt_msg(msg_type, chunk)`,
        render: () => renderPhase7(),
      },
      {
        title: 'Phase 8: 核心生成内核 _generate 与采样策略',
        description:
          '<code>streaming_generate</code> 内部调用 <code>_generate</code> 完成逐 token 生成：<br>• 初始化 <code>attention_mask</code> 为 1；如是 VlmModel 则通过 <code>batch_repeat_image_tok</code> 展开图像 token；<br>• 每步调用 <code>model(...)</code>，只取最后一个位置的 logits，并按顺序应用 suppress_tokens、temperature、top-k、top-p；<br>• 对选出的 <code>next_token</code> 进行 <code>yield</code>，同时更新 KVCache、tokens 与 attention_mask。<span class="step-badge">generate_utils.py:204-260, 359-421</span>',
        code: `attention_mask = torch.ones_like(tokens, device=device, dtype=torch.long)\nif isinstance(model, VlmModel):\n    tokens = batch_repeat_image_tok(tokens, tokens_per_image)\n...\nfor _ in range(max_new_tokens):\n    result = model(input_ids=tokens, attention_mask=attention_mask, ... )\n    logits = result['logits'][:, -1, :]\n    # temperature / top-k / top-p / suppress_tokens 依次应用\n    yield next_token`,
        render: () => renderPhase8(),
      },
      {
        title: 'Phase 9: SSE 前端消费与多 lane 可视化',
        description:
          '前端通过 <code>EventSource</code> 订阅 <code>/api/chat</code> SSE 流，根据 event 字段区分 <code>thinking_chunk</code> 与 <code>answer_chunk</code>：<br>• 将 thinking 片段渲染到“思考 Lane”；<br>• 将 answer 片段渲染到“回答 Lane”；<br>• 遇到 error event 则在 UI 提示异常。该总览页面抽象出 Browser / Server / Model / Search 四条 Lane，帮助读者把所有步骤串联起来理解。',
        code: `const es = new EventSource('/api/chat');\nes.onmessage = (evt) => {\n  const msg = JSON.parse(evt.data);\n  if (msg.event === 'thinking_chunk') appendToThinkingLane(msg.data);\n  if (msg.event === 'answer_chunk') appendToAnswerLane(msg.data);\n}`,
        render: () => renderPhase9(),
      },
    ];

    function updateUI() {
      const step = steps[currentStep];
      if (infoBox) infoBox.innerHTML = `<div class="step-badge">Online Inference</div><strong>${step.title}</strong><br>${step.description}`;
      if (codeSnippet) {
          codeSnippet.textContent = step.code;
          if(window.hljs) hljs.highlightElement(codeSnippet);
      }
      visualContent.innerHTML = '';
      visualContent.className = 'fade-in';
      try {
          step.render();
      } catch(e) {
          console.error("Render failed", e);
      }
      updateButtons();
    }

    function renderPhase0() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:20px; align-items:center; width:100%;">
          <div class="seq-viz">
            <div class="token prompt">Browser</div>
            <div class="arrow">→</div>
            <div class="token gen">POST /api/chat</div>
          </div>
          <div class="dict-entry" style="max-width:520px;">payload: { history, thinking, deep_search, temperature, top_p, uuid }</div>
        </div>`;
    }

    function renderPhase1() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:15px; align-items:center; width:100%;">
          <div class="dict-entry" style="max-width:520px;">Bottle 路由 <code>@app.route('/api/chat', method=['POST'])</code> 解析 JSON，设置 SSE 响应头。</div>
          <div class="code-block-dark" style="max-width:520px;">response.content_type = 'text/event-stream'\nresponse.set_header('Cache-Control', 'no-cache')</div>
        </div>`;
    }

    function renderPhase2() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:15px; align-items:center; width:100%;">
          <div class="dict-entry" style="max-width:520px;">根据 deep_search 插入不同 system 提示词，然后通过 apply_chat_template + '&lt;assistant&gt;' 构造最终 chat_template。</div>
          <div class="code-block-dark" style="max-width:520px;">&lt;system&gt;...&lt;/s&gt;\n&lt;user&gt;...&lt;/s&gt;\n&lt;assistant&gt;</div>
        </div>`;
    }

    function renderPhase3() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:15px; align-items:center; width:100%;">
          <div class="dict-entry" style="max-width:520px;">search.py 中封装 Brave/BochaAI 搜索 API，返回的 search_result 作为一段自然语言摘要。</div>
          <div class="code-block-dark" style="max-width:520px;">search_result = get_search_api()(user_prompt)\nsearch_msg = "根据用户的问题，我搜索到了如下内容：\n..."\nyield thinking_chunk(search_msg)\nchat_template += &lt;think&gt;search_msg&lt;/think&gt;</div>
        </div>`;
    }

    function renderPhase4() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:15px; align-items:center; width:100%;">
          <div class="dict-entry" style="max-width:520px;">将 chat_template 编码为 prompt_token，并用 2048 上限减去 prompt 长度得到 max_new_tokens，启用思考预算时再预留 think_budget_content。</div>
        </div>`;
    }

    function renderPhase5() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:20px; align-items:center; width:100%;">
          <div class="dict-entry" style="max-width:520px;">第一次 streaming_generate 只负责思考阶段，循环读取 chunk：遇到 &lt;/think&gt; 提前停止，或者用预算补齐。</div>
          <div class="code-block-dark" style="max-width:520px;">for chunk in generator:\n    think_content += chunk\n    if chunk == '</think>': break\n    yield thinking_chunk(chunk)</div>
        </div>`;
    }

    function renderPhase6() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:15px; align-items:center; width:100%;">
          <div class="dict-entry" style="max-width:520px;">将 think_content 重新 encode 并拼接到 prompt_token 末尾，重新调用 streaming_generate，进入回答阶段。</div>
        </div>`;
    }

    function renderPhase7() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:20px; align-items:center; width:100%;">
          <div class="dict-entry" style="max-width:520px;">状态机根据 &lt;think&gt;/&lt;answer&gt;/&lt;/s&gt; 等标签切换当前 msg_type，并跳过标签本身，只把正文以 SSE JSON 行返回。</div>
          <div class="code-block-dark" style="max-width:520px;">if chunk == '<think>': msg_type = 'thinking_chunk'\nelif chunk == '<answer>': msg_type = 'answer_chunk'\nelif chunk == '</s>': break</div>
        </div>`;
    }

    function renderPhase8() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:15px; align-items:center; width:100%;">
          <div class="dict-entry" style="max-width:520px;">_generate 实现了 KV Cache + temperature/top-k/top-p/suppress_tokens 组合逻辑，每步仅取最后一个 token 的 logits。</div>
          <div class="code-block-dark" style="max-width:520px;">result = model(input_ids=tokens, attention_mask=attention_mask, ...)\nlogits = result['logits'][:, -1, :]\n# 采样逻辑 + yield next_token</div>
        </div>`;
    }

    function renderPhase9() {
      visualContent.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:20px; align-items:center; width:100%;">
          <div class="seq-viz">
            <div class="token prompt">Browser</div>
            <div class="arrow">⇄</div>
            <div class="token gen">EventSource('/api/chat')</div>
          </div>
          <div class="dict-entry" style="max-width:520px;">前端根据 event 类型将内容分别渲染到“思考 Lane”与“回答 Lane”，形成清晰的多通道可视化。</div>
        </div>`;
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

    if (nextBtn) nextBtn.addEventListener('click', goNext);
    if (prevBtn) prevBtn.addEventListener('click', goPrev);
    if (resetBtn) resetBtn.addEventListener('click', () => {
      currentStep = 0;
      updateUI();
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft') goPrev();
      if (e.key === 'ArrowRight') goNext();
    });

    updateUI();
});
