"use strict";

const $ = (id) => document.getElementById(id);
const form = $("search-form");
const question = $("question");
let result = null;
let selected = null;
let loading = false;
let audience = "all";

function renderExamples() {
  $("examples").replaceChildren();
  document.querySelectorAll("[data-audience]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.audience === audience));
  });
  for (const example of schoolGuide.examplesFor(audience)) {
    const fragment = $("example-template").content.cloneNode(true);
    const button = fragment.querySelector("button");
    button.dataset.example = example.id;
    button.disabled = loading;
    button.querySelector(".example-topic").textContent = example.topic;
    button.querySelector(".example-question").textContent = example.label;
    button.addEventListener("click", () => {
      question.value = example.question;
      form.requestSubmit();
    });
    $("examples").append(fragment);
  }
}

function setLoading(value) {
  loading = value;
  $("search-button").disabled = value;
  $("top-k").disabled = value;
  $("search-label").textContent = value ? "찾고 있어요…" : "찾아보기";
  form.setAttribute("aria-busy", String(value));
  document.querySelectorAll("[data-example], [data-audience]").forEach((button) => { button.disabled = value; });
}

function selectSource(source) {
  selected = source;
  document.querySelectorAll(".result-card").forEach((card) => {
    card.setAttribute("aria-pressed", String(card.dataset.sourceId === source.source_id));
  });
  $("reader-topic").textContent = schoolGuide.topicFor(source.doc_id);
  $("reader-title").textContent = schoolGuide.displayTitle(source.doc_id);
  $("reader-original-title").textContent = source.doc_id;
  $("reader-path").textContent = source.path || "이 자료에는 위치 정보가 없어요.";
  $("reader-body").textContent = source.body;
  $("reader-body").scrollTop = 0;
  $("copy-button").textContent = "내용 복사";
  const chips = $("condition-list");
  chips.replaceChildren();
  $("condition-section").hidden = result.condition_audit.length === 0;
  for (const audit of result.condition_audit) {
    const fields = audit.sources.find((entry) => entry.source_id === source.source_id)?.fields || [];
    const label = schoolGuide.conditionLabel(fields);
    const chip = document.createElement("span");
    chip.className = `condition-chip ${label.tone}`;
    chip.textContent = `${audit.condition} · ${label.text}`;
    chips.append(chip);
  }
}

function renderResults(data) {
  result = data;
  selected = null;
  const packet = data.context;
  $("empty-state").hidden = true;
  $("results-section").hidden = false;
  $("result-count").textContent = `${packet.sources.length}개`;
  $("result-question").textContent = `궁금한 점: ${packet.original_question}`;
  $("date-warning").hidden = data.missing_date_conditions.length === 0;
  $("date-warning").textContent = `${data.missing_date_conditions.join(", ")}에 적용되는 내용인지 확인이 필요해요. 찾은 내용과 항목 이름에 이 날짜가 적혀 있지 않아요. 다른 방식으로 날짜가 쓰여 있거나 별도의 안내가 있을 수 있어요.`;
  $("budget-warning").hidden = packet.omitted_chunk_ids.length === 0;
  $("budget-warning").textContent = "내용이 길어 일부 자료를 이번 화면에 모두 담지 못했어요. 질문을 조금 더 구체적으로 적어 다시 찾아보세요.";
  $("result-list").replaceChildren();
  for (const source of packet.sources) {
    const fragment = $("result-template").content.cloneNode(true);
    const card = fragment.querySelector(".result-card");
    card.dataset.sourceId = source.source_id;
    card.setAttribute("aria-label", `${schoolGuide.displayTitle(source.doc_id)} 내용 보기`);
    card.querySelector(".topic-badge").textContent = schoolGuide.topicFor(source.doc_id);
    card.querySelector(".card-title").textContent = schoolGuide.displayTitle(source.doc_id);
    card.querySelector(".card-preview").textContent = source.body.slice(0, 300);
    card.addEventListener("click", () => selectSource(source));
    $("result-list").append(fragment);
  }
  $("reader").hidden = packet.sources.length === 0;
  $("export-button").disabled = packet.sources.length === 0;
  $("status").textContent = packet.sources.length ? `관련 내용 ${packet.sources.length}개를 찾았어요. 자료를 선택해 살펴보세요.` : "표시할 내용을 찾지 못했어요. 다른 말로 질문하거나 한 번에 보는 개수를 바꿔보세요.";
  if (packet.sources.length) selectSource(packet.sources[0]);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (loading) return;
  if (!question.value.trim()) {
    $("error").textContent = "궁금한 내용을 먼저 적어주세요.";
    $("error").hidden = false;
    question.focus();
    return;
  }
  setLoading(true);
  $("error").hidden = true;
  $("results-section").hidden = true;
  $("empty-state").hidden = true;
  $("status").textContent = "궁금한 내용과 관련된 교육 자료를 찾고 있어요. 잠시만 기다려 주세요.";
  try {
    const response = await fetch("/api/search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      // Audience selection changes suggestions only, never the submitted question or filters.
      body: JSON.stringify({ question: question.value, top_k: Number($("top-k").value) })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "자료를 찾지 못했어요. 잠시 후 다시 시도해 주세요.");
    renderResults(data);
  } catch (error) {
    $("status").textContent = "";
    $("error").textContent = error instanceof TypeError ? "지금은 자료를 불러올 수 없어요. 화면을 새로고침하거나 잠시 후 다시 시도해 주세요." : error.message;
    $("error").hidden = false;
    $("empty-state").hidden = false;
  } finally {
    setLoading(false);
  }
});

question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-audience]").forEach((button) => {
  button.addEventListener("click", () => {
    audience = button.dataset.audience;
    renderExamples();
  });
});

$("copy-button").addEventListener("click", async () => {
  if (!selected) return;
  try {
    await navigator.clipboard.writeText(schoolGuide.sourceText(selected, result.context.sources.indexOf(selected)+1));
    $("copy-button").textContent = "복사했어요 ✓";
    $("status").textContent = "선택한 내용과 자료 이름을 복사했어요. 필요한 곳에 붙여넣어 보세요.";
  } catch {
    $("status").textContent = "자동으로 복사하지 못했어요. 필요한 내용을 직접 선택해서 복사해 주세요.";
  }
});

$("export-button").addEventListener("click", () => {
  if (!result) return;
  const url = URL.createObjectURL(new Blob(["\ufeff", schoolGuide.saveText(result)], { type: "text/plain;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `학교생활안내-${new Date().toISOString().slice(0, 10)}.txt`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  $("status").textContent = "찾은 내용을 저장하도록 요청했어요. 내려받은 파일에 질문과 자료 내용이 들어 있으니 공유 전에 확인해 주세요.";
});

renderExamples();
fetch("/api/info").then((response) => {
  if (!response.ok) throw new Error("service unavailable");
  return response.json();
}).then((info) => {
  $("index-info").textContent = `현재 ${info.document_count}개의 교육 자료에서 학교생활기록부와 교육과정 내용을 찾아드려요.`;
}).catch(() => {
  $("index-info").textContent = "지금은 자료를 불러올 수 없어요. 잠시 후 다시 방문해 주세요.";
});
