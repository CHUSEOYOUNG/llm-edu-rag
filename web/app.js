"use strict";

const $ = (id) => document.getElementById(id);
const form = $("search-form");
const question = $("question");
let result = null;
let selected = null;
let loading = false;
let audience = "all";
let schoolLevel = "all";
let schoolLevelManuallySet = false;
let sourceGroups = [];
let searchTerms = [];

const schoolLabels = {all: "전체", elementary: "초등학교", middle: "중학교", high: "고등학교"};

function renderSchoolFilter() {
  document.querySelectorAll("[data-school-level]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.schoolLevel === schoolLevel));
  });
  $("school-filter-help").textContent = schoolLevel === "all"
    ? "전체 자료에서 찾아요."
    : `${schoolLabels[schoolLevel]} 관련 자료만 찾아요.`;
}

function appendHighlighted(element, value, terms = searchTerms) {
  const text = String(value || "");
  element.replaceChildren();
  if (!terms.length) {
    element.textContent = text;
    return;
  }
  const lowered = text.toLocaleLowerCase();
  let offset = 0;
  while (offset < text.length) {
    let matchIndex = -1;
    let matchTerm = "";
    for (const term of terms) {
      const found = lowered.indexOf(term.toLocaleLowerCase(), offset);
      if (found >= 0 && (matchIndex < 0 || found < matchIndex || (found === matchIndex && term.length > matchTerm.length))) {
        matchIndex = found;
        matchTerm = term;
      }
    }
    if (matchIndex < 0) {
      element.append(document.createTextNode(text.slice(offset)));
      break;
    }
    if (matchIndex > offset) element.append(document.createTextNode(text.slice(offset, matchIndex)));
    const mark = document.createElement("mark");
    mark.textContent = text.slice(matchIndex, matchIndex + matchTerm.length);
    element.append(mark);
    offset = matchIndex + matchTerm.length;
  }
}

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
  document.querySelectorAll("[data-example], [data-audience], [data-school-level]").forEach((button) => { button.disabled = value; });
}

function renderReaderBody(raw) {
  const container = $("reader-body");
  container.replaceChildren();
  for (const block of schoolGuide.readableBlocks(raw)) {
    if (block.type === "text") {
      const text = document.createElement("div");
      text.className = "reader-text-block";
      appendHighlighted(text, block.text);
      container.append(text);
      continue;
    }
    const wrapper = document.createElement("div");
    wrapper.className = "reader-table-wrap";
    const table = document.createElement("table");
    table.className = "reader-table";
    table.setAttribute("aria-label", "자료에 포함된 표");
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    for (const value of block.headers) {
      const cell = document.createElement("th");
      cell.scope = "col";
      appendHighlighted(cell, value);
      headRow.append(cell);
    }
    head.append(headRow);
    table.append(head);
    const body = document.createElement("tbody");
    for (const row of block.rows) {
      const tableRow = document.createElement("tr");
      for (const value of row) {
        const cell = document.createElement("td");
        appendHighlighted(cell, value);
        tableRow.append(cell);
      }
      body.append(tableRow);
    }
    table.append(body);
    wrapper.append(table);
    container.append(wrapper);
  }
}

function selectSource(source, group) {
  selected = source;
  document.querySelectorAll(".result-card").forEach((card) => {
    card.setAttribute("aria-pressed", String(Number(card.dataset.groupIndex) === sourceGroups.indexOf(group)));
  });
  $("reader-topic").textContent = schoolGuide.topicFor(source.doc_id);
  appendHighlighted($("reader-title"), schoolGuide.sectionTitle(source.path));
  $("reader-original-title").textContent = source.doc_id;
  $("reader-path").textContent = source.path || "이 자료에는 위치 정보가 없어요.";
  $("reader-page").textContent = schoolGuide.pageLabel(source);
  const sourceLink = $("source-link");
  sourceLink.hidden = !source.source_url;
  if (source.source_url) sourceLink.href = source.source_url;
  else sourceLink.removeAttribute("href");
  renderReaderBody(source.body);
  $("reader-body").scrollTop = 0;
  $("copy-button").textContent = "내용 복사";
  const sourceChoices = $("source-choices");
  const choiceList = $("source-choice-list");
  choiceList.replaceChildren();
  sourceChoices.hidden = group.sources.length < 2;
  for (const [index, choice] of group.sources.entries()) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "source-choice";
    button.setAttribute("aria-pressed", String(choice.source_id === source.source_id));
    button.textContent = `${index + 1}. ${schoolGuide.pageLabel(choice)}`;
    button.addEventListener("click", () => selectSource(choice, group));
    choiceList.append(button);
  }
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
  searchTerms = schoolGuide.highlightTerms(packet.original_question);
  sourceGroups = schoolGuide.groupSources(packet.sources);
  $("empty-state").hidden = true;
  $("results-section").hidden = false;
  $("result-count").textContent = sourceGroups.length === packet.sources.length
    ? `${sourceGroups.length}개`
    : `${sourceGroups.length}개 항목 · 내용 ${packet.sources.length}개`;
  $("result-question").textContent = `궁금한 점: ${packet.original_question}`;
  $("active-filter").hidden = data.school_level === "all";
  $("active-filter").textContent = data.school_level === "all" ? "" : `${schoolLabels[data.school_level]} 자료만 모아봤어요. ‘전체’를 누르면 다른 학교급 자료도 함께 볼 수 있어요.`;
  $("date-warning").hidden = data.missing_date_conditions.length === 0;
  $("date-warning").textContent = `${data.missing_date_conditions.join(", ")}에 적용되는 내용인지 확인이 필요해요. 찾은 내용과 항목 이름에 이 날짜가 적혀 있지 않아요. 다른 방식으로 날짜가 쓰여 있거나 별도의 안내가 있을 수 있어요.`;
  $("budget-warning").hidden = packet.omitted_chunk_ids.length === 0;
  $("budget-warning").textContent = "내용이 길어 일부 자료를 이번 화면에 모두 담지 못했어요. 질문을 조금 더 구체적으로 적어 다시 찾아보세요.";
  $("result-list").replaceChildren();
  for (const [groupIndex, group] of sourceGroups.entries()) {
    const source = group.sources[0];
    const fragment = $("result-template").content.cloneNode(true);
    const card = fragment.querySelector(".result-card");
    card.dataset.groupIndex = String(groupIndex);
    card.setAttribute("aria-label", `${schoolGuide.sectionTitle(source.path)} 내용 보기`);
    card.querySelector(".topic-badge").textContent = schoolGuide.topicFor(source.doc_id);
    appendHighlighted(card.querySelector(".card-title"), schoolGuide.sectionTitle(source.path));
    const level = schoolGuide.schoolLevelFor(source.doc_id);
    card.querySelector(".card-document").textContent = `${level.label} · ${schoolGuide.displayTitle(source.doc_id)}`;
    card.querySelector(".card-page").textContent = schoolGuide.pageLabel(source);
    appendHighlighted(card.querySelector(".card-preview"), schoolGuide.readablePreview(source.body).slice(0, 300));
    card.querySelector(".card-related").textContent = group.sources.length > 1 ? `관련 내용 ${group.sources.length}개 모아보기` : "내용 살펴보기";
    card.addEventListener("click", () => selectSource(source, group));
    $("result-list").append(fragment);
  }
  $("reader").hidden = packet.sources.length === 0;
  $("export-button").disabled = packet.sources.length === 0;
  $("status").textContent = packet.sources.length ? `관련 내용 ${packet.sources.length}개를 ${sourceGroups.length}개 항목으로 정리했어요.` : `${schoolLabels[data.school_level]} 자료에서는 관련 내용을 찾지 못했어요. 학교급을 ‘전체’로 바꾸거나 다른 말로 찾아보세요.`;
  if (packet.sources.length) selectSource(sourceGroups[0].sources[0], sourceGroups[0]);
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
  if (!schoolLevelManuallySet) {
    schoolLevel = schoolGuide.detectedSchoolLevel(question.value);
    renderSchoolFilter();
  }
  setLoading(true);
  $("error").hidden = true;
  $("results-section").hidden = true;
  $("empty-state").hidden = true;
  $("status").textContent = "궁금한 내용과 관련된 교육 자료를 찾고 있어요. 잠시만 기다려 주세요.";
  try {
    const response = await fetch("/api/search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      // Audience selection changes suggestions only. School level is an explicit, visible filter.
      body: JSON.stringify({ question: question.value, top_k: Number($("top-k").value), school_level: schoolLevel })
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

document.querySelectorAll("[data-school-level]").forEach((button) => {
  button.addEventListener("click", () => {
    schoolLevel = button.dataset.schoolLevel;
    schoolLevelManuallySet = true;
    renderSchoolFilter();
    if (result && question.value.trim()) form.requestSubmit();
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
renderSchoolFilter();
fetch("/api/info").then((response) => {
  if (!response.ok) throw new Error("service unavailable");
  return response.json();
}).then((info) => {
  $("index-info").textContent = `현재 ${info.document_count}개의 교육 자료에서 학교생활기록부와 교육과정 내용을 찾아드려요.`;
}).catch(() => {
  $("index-info").textContent = "지금은 자료를 불러올 수 없어요. 잠시 후 다시 방문해 주세요.";
});
