/* =========================================================================
   StringLLM Playground — Client-Side Application
   ========================================================================= */

(function () {
    "use strict";

    /* -----------------------------------------------------------------------
       State
       ----------------------------------------------------------------------- */
    let nodes = [];
    let providerStatuses = [];
    let templates = [];

    /* -----------------------------------------------------------------------
       DOM references
       ----------------------------------------------------------------------- */
    const nodesContainer = document.getElementById("nodes-container");
    const addNodeBtn = document.getElementById("add-node-btn");
    const providerSelect = document.getElementById("provider-select");
    const inputText = document.getElementById("input-text");
    const runBtn = document.getElementById("run-btn");
    const resultsSection = document.getElementById("results-section");
    const errorBox = document.getElementById("error-box");
    const providerDotsContainer = document.getElementById("provider-dots");
    const templatesList = document.getElementById("templates-list");

    /* -----------------------------------------------------------------------
       Example Chains
       ----------------------------------------------------------------------- */
    const EXAMPLES = {
        "summarize_analyze": {
            name: "Summarize & Analyze",
            desc: "Summarize text then analyze the sentiment of the summary.",
            input: "Artificial intelligence is transforming industries worldwide. From healthcare diagnostics to autonomous vehicles, AI systems are becoming increasingly sophisticated. However, concerns about job displacement, privacy, and algorithmic bias continue to spark heated debates among policymakers, technologists, and the public.",
            nodes: [
                {
                    name: "Summarizer",
                    prompt: "Summarize the following text into 3 concise bullet points:\n\n{text}",
                    output_key: "summary"
                },
                {
                    name: "Sentiment Analyzer",
                    prompt: "Analyze the sentiment of the following summary. Classify it as positive, negative, or neutral and explain why:\n\n{summary}",
                    output_key: "sentiment"
                }
            ]
        },
        "translate_pipeline": {
            name: "Translate Pipeline",
            desc: "Translate text to French, then back to English.",
            input: "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet and has been used as a typing exercise for over a century.",
            nodes: [
                {
                    name: "Translate to French",
                    prompt: "Translate the following text to French. Output only the translation:\n\n{text}",
                    output_key: "french"
                },
                {
                    name: "Translate back to English",
                    prompt: "Translate the following French text back to English. Output only the translation:\n\n{french}",
                    output_key: "back_to_english"
                },
                {
                    name: "Compare",
                    prompt: "Compare these two English texts and note any differences in meaning or nuance:\n\nOriginal: {text}\n\nRound-trip translation: {back_to_english}",
                    output_key: "comparison"
                }
            ]
        },
        "code_review": {
            name: "Code Review",
            desc: "Review code, then generate improved version.",
            input: "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)",
            nodes: [
                {
                    name: "Code Reviewer",
                    prompt: "Review the following code for bugs, performance issues, and best practices. Provide actionable feedback:\n\n{text}",
                    output_key: "review"
                },
                {
                    name: "Code Improver",
                    prompt: "Based on this code review feedback:\n\n{review}\n\nRewrite the original code with all suggested improvements applied:\n\n{text}",
                    output_key: "improved_code"
                }
            ]
        }
    };

    /* -----------------------------------------------------------------------
       Node Management
       ----------------------------------------------------------------------- */
    function createDefaultNode() {
        return {
            id: Date.now() + Math.random(),
            name: "",
            prompt: "",
            output_key: ""
        };
    }

    function addNode() {
        nodes.push(createDefaultNode());
        renderNodes();
    }

    function removeNode(index) {
        nodes.splice(index, 1);
        renderNodes();
    }

    function updateNode(index, field, value) {
        if (nodes[index]) {
            nodes[index][field] = value;
        }
    }

    /* -----------------------------------------------------------------------
       Render Nodes
       ----------------------------------------------------------------------- */
    function renderNodes() {
        nodesContainer.innerHTML = "";

        if (nodes.length === 0) {
            nodesContainer.innerHTML =
                '<div class="empty-state">' +
                '<div class="empty-icon">&#9734;</div>' +
                "<p>No nodes yet. Add a node to start building your chain.</p>" +
                "</div>";
            return;
        }

        nodes.forEach(function (node, i) {
            var wrapper = document.createElement("div");
            wrapper.className = "node-wrapper";

            // Connector arrow (between nodes, not before the first)
            if (i > 0) {
                var connector = document.createElement("div");
                connector.className = "node-connector";
                connector.innerHTML = '<div class="arrow"></div>';
                wrapper.appendChild(connector);
            }

            var card = document.createElement("div");
            card.className = "node-card";
            card.innerHTML =
                '<div class="node-header">' +
                '  <span class="node-number">' + (i + 1) + "</span>" +
                '  <button class="btn btn-danger" data-index="' + i + '">Remove</button>' +
                "</div>" +
                '<div class="node-fields">' +
                "  <div>" +
                "    <label>Node Name</label>" +
                '    <input type="text" class="node-name" placeholder="e.g. Summarizer" value="' + escapeAttr(node.name) + '" data-index="' + i + '" />' +
                "  </div>" +
                "  <div>" +
                "    <label>Output Key</label>" +
                '    <input type="text" class="node-output-key" placeholder="e.g. summary" value="' + escapeAttr(node.output_key) + '" data-index="' + i + '" />' +
                "  </div>" +
                '  <div class="field-full">' +
                "    <label>Prompt Template</label>" +
                '    <textarea class="node-prompt" placeholder="Use {variable} for inputs from previous steps..." rows="3" data-index="' + i + '">' + escapeHtml(node.prompt) + "</textarea>" +
                "  </div>" +
                "</div>";

            wrapper.appendChild(card);
            nodesContainer.appendChild(wrapper);
        });

        // Attach events
        nodesContainer.querySelectorAll(".btn-danger").forEach(function (btn) {
            btn.addEventListener("click", function () {
                removeNode(parseInt(this.getAttribute("data-index"), 10));
            });
        });
        nodesContainer.querySelectorAll(".node-name").forEach(function (el) {
            el.addEventListener("input", function () {
                updateNode(parseInt(this.getAttribute("data-index"), 10), "name", this.value);
            });
        });
        nodesContainer.querySelectorAll(".node-output-key").forEach(function (el) {
            el.addEventListener("input", function () {
                updateNode(parseInt(this.getAttribute("data-index"), 10), "output_key", this.value);
            });
        });
        nodesContainer.querySelectorAll(".node-prompt").forEach(function (el) {
            el.addEventListener("input", function () {
                updateNode(parseInt(this.getAttribute("data-index"), 10), "prompt", this.value);
            });
        });
    }

    /* -----------------------------------------------------------------------
       Load Example
       ----------------------------------------------------------------------- */
    function loadExample(key) {
        var ex = EXAMPLES[key];
        if (!ex) return;

        nodes = ex.nodes.map(function (n) {
            return {
                id: Date.now() + Math.random(),
                name: n.name,
                prompt: n.prompt,
                output_key: n.output_key
            };
        });

        inputText.value = ex.input;
        renderNodes();
        clearResults();
        clearError();
    }

    /* -----------------------------------------------------------------------
       Run Chain
       ----------------------------------------------------------------------- */
    async function runChain() {
        clearError();
        clearResults();

        // Validate
        if (nodes.length === 0) {
            showError("Add at least one node to the chain before running.");
            return;
        }

        for (var i = 0; i < nodes.length; i++) {
            if (!nodes[i].name.trim()) {
                showError("Node " + (i + 1) + " is missing a name.");
                return;
            }
            if (!nodes[i].prompt.trim()) {
                showError("Node " + (i + 1) + " is missing a prompt.");
                return;
            }
            if (!nodes[i].output_key.trim()) {
                showError("Node " + (i + 1) + " is missing an output key.");
                return;
            }
        }

        var body = {
            nodes: nodes.map(function (n) {
                return { name: n.name.trim(), prompt: n.prompt, output_key: n.output_key.trim() };
            }),
            input: { text: inputText.value },
            provider: providerSelect.value
        };

        setLoading(true);

        try {
            var resp = await fetch("/api/chain/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body)
            });

            var data = await resp.json();

            if (!resp.ok) {
                showError(data.detail || "Server error (" + resp.status + ")");
                return;
            }

            renderResults(data);
        } catch (err) {
            showError("Network error: " + err.message);
        } finally {
            setLoading(false);
        }
    }

    /* -----------------------------------------------------------------------
       Render Results
       ----------------------------------------------------------------------- */
    function renderResults(data) {
        resultsSection.innerHTML = "";

        // Summary stats
        var summary = document.createElement("div");
        summary.className = "result-summary";
        summary.innerHTML =
            '<div class="result-stat">' +
            '  <div class="stat-value">' + data.total_tokens + "</div>" +
            '  <div class="stat-label">Total Tokens</div>' +
            "</div>" +
            '<div class="result-stat">' +
            '  <div class="stat-value">' + data.total_time_ms.toFixed(0) + "ms</div>" +
            '  <div class="stat-label">Total Time</div>' +
            "</div>" +
            '<div class="result-stat">' +
            '  <div class="stat-value">' + escapeHtml(data.provider_used) + "</div>" +
            '  <div class="stat-label">Provider</div>' +
            "</div>";
        resultsSection.appendChild(summary);

        // Step accordion
        if (data.steps && data.steps.length > 0) {
            data.steps.forEach(function (step, idx) {
                var item = document.createElement("div");
                item.className = "accordion-item" + (idx === data.steps.length - 1 ? " open" : "");

                item.innerHTML =
                    '<div class="accordion-header">' +
                    '  <span class="step-name">' + escapeHtml(step.node_name) + "</span>" +
                    '  <span class="step-meta">' +
                    "    <span>" + step.tokens + " tokens</span>" +
                    "    <span>" + step.time_ms.toFixed(0) + "ms</span>" +
                    "    <span>" + escapeHtml(step.provider) + "</span>" +
                    "  </span>" +
                    '  <span class="chevron">&#9660;</span>' +
                    "</div>" +
                    '<div class="accordion-body">' +
                    '  <div class="accordion-content">' + escapeHtml(step.output) + "</div>" +
                    "</div>";

                item.querySelector(".accordion-header").addEventListener("click", function () {
                    item.classList.toggle("open");
                });

                resultsSection.appendChild(item);
            });
        }
    }

    function clearResults() {
        resultsSection.innerHTML = "";
    }

    /* -----------------------------------------------------------------------
       Provider Status
       ----------------------------------------------------------------------- */
    async function checkProviders() {
        try {
            var resp = await fetch("/api/providers/status");
            var data = await resp.json();
            providerStatuses = data.providers || [];
            renderProviderDots();
        } catch (err) {
            // silently degrade — dots stay grey
        }
    }

    function renderProviderDots() {
        providerDotsContainer.innerHTML = "";
        providerStatuses.forEach(function (p) {
            var dotClass = "status-dot";
            if (p.healthy) {
                dotClass += " green";
            } else if (p.available) {
                dotClass += " yellow";
            } else {
                dotClass += " red";
            }

            var el = document.createElement("div");
            el.className = "provider-dot";
            el.innerHTML = '<span class="' + dotClass + '"></span>' + escapeHtml(capitalize(p.name));
            providerDotsContainer.appendChild(el);
        });
    }

    /* -----------------------------------------------------------------------
       Templates
       ----------------------------------------------------------------------- */
    async function loadTemplates() {
        try {
            var resp = await fetch("/api/templates");
            var data = await resp.json();
            templates = data.templates || [];
            renderTemplates();
        } catch (err) {
            // silently ignore
        }
    }

    function renderTemplates() {
        if (!templatesList) return;
        templatesList.innerHTML = "";

        if (templates.length === 0) {
            templatesList.innerHTML = '<span class="template-chip">No templates loaded</span>';
            return;
        }

        templates.forEach(function (t) {
            var chip = document.createElement("span");
            chip.className = "template-chip";
            chip.textContent = t.name.replace(/_/g, " ");
            chip.title = t.template;
            templatesList.appendChild(chip);
        });
    }

    /* -----------------------------------------------------------------------
       UI Helpers
       ----------------------------------------------------------------------- */
    function setLoading(on) {
        if (on) {
            runBtn.classList.add("loading");
            runBtn.disabled = true;
        } else {
            runBtn.classList.remove("loading");
            runBtn.disabled = false;
        }
    }

    function showError(msg) {
        errorBox.textContent = msg;
        errorBox.classList.add("visible");
    }

    function clearError() {
        errorBox.textContent = "";
        errorBox.classList.remove("visible");
    }

    function escapeHtml(str) {
        if (!str) return "";
        return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    function escapeAttr(str) {
        if (!str) return "";
        return str.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function capitalize(s) {
        if (!s) return "";
        return s.charAt(0).toUpperCase() + s.slice(1);
    }

    /* -----------------------------------------------------------------------
       Example Buttons
       ----------------------------------------------------------------------- */
    function initExampleButtons() {
        document.querySelectorAll(".example-btn").forEach(function (btn) {
            btn.addEventListener("click", function () {
                loadExample(this.getAttribute("data-example"));
            });
        });
    }

    /* -----------------------------------------------------------------------
       Init
       ----------------------------------------------------------------------- */
    document.addEventListener("DOMContentLoaded", function () {
        addNodeBtn.addEventListener("click", addNode);
        runBtn.addEventListener("click", runChain);

        initExampleButtons();
        renderNodes();
        checkProviders();
        loadTemplates();

        // Refresh provider status every 60 seconds
        setInterval(checkProviders, 60000);
    });
})();
