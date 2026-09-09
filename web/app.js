"use strict";

const $ = (id) => document.getElementById(id);
const form = $("search-form");
const question = $("question");
const HISTORY_KEY = "school-life-guide.search-history.v1";
let result = null;
let selected = null;
let loading = false;
let generating = false;
let generationEnabled = false;
let lastSearchPayload = null;
let audience = "all";
let schoolLevel = "all";
let schoolLevelManuallySet = false;
let sourceGroups = [];
let searchTerms = [];
let previousQuestion = null;
let searchHistory = [];

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

function renderHistory() {
  const section = $("recent-searches");
  const list = $("history-list");
  list.replaceChildren();
  section.hidden = searchHistory.length === 0;
  for (const item of searchHistory) {
    const fragment = $("history-template").content.cloneNode(true);
    const button = fragment.querySelector("button");
    button.querySelector(".history-question").textContent = item.question;
    button.querySelector(".history-level").textContent = schoolLabels[item.schoolLevel];
    button.addEventListener("click", () => {
      question.value = item.searchQuery;
      schoolLevel = item.schoolLevel;
      schoolLevelManuallySet = true;
      previousQuestion = null;
      renderSchoolFilter();
      form.requestSubmit();
      $("search-area").scrollIntoView({behavior: "smooth", block: "start"});
    });
    list.append(fragment);
  }
}

function loadHistory() {
  try {
    searchHistory = schoolGuide.normalizeHistory(JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"));
  } catch {
    searchHistory = [];
  }
  renderHistory();
}

function rememberSearch(data) {
  searchHistory = schoolGuide.addHistory(searchHistory, {
    question: data.context.original_question,
    searchQuery: data.search_query,
    schoolLevel: data.school_level
  });
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(searchHistory));
  } catch {
    // The list still works for this page when browser storage is unavailable.
  }
  renderHistory();
}

function refreshBusyState() {
  const busy = loading || generating;
  $("search-button").disabled = busy;
  $("top-k").disabled = busy;
  question.disabled = busy;
  $("search-label").textContent = loading ? "찾고 있어요…" : "찾아보기";
  form.setAttribute("aria-busy", String(busy));
  document.querySelectorAll("[data-example], [data-audience], [data-school-level]").forEach((button) => { button.disabled = busy; });
  $("answer-button").disabled = busy || !result?.context?.sources?.length;
}

function setLoading(value) {
  loading = value;
  refreshBusyState();
}

function setGenerating(value) {
  generating = value;
  $("answer-loading").hidden = !value;
  if (value) $("answer-offer").hidden = true;
  refreshBusyState();
}

function resetAnswerPanel(hasSources) {
  $("answer-panel").hidden = !generationEnabled || !hasSources;
  $("answer-offer").hidden = false;
  $("answer-loading").hidden = true;
  $("answer-result").hidden = true;
  $("answer-error").hidden = true;
  $("answer-claims").replaceChildren();
  $("answer-reason").hidden = true;
  $("answer-reason").textContent = "";
  $("answer-button-label").textContent = "자료로 답변 만들기";
  refreshBusyState();
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

function openAnswerSource(answerSource, label) {
  const source = result.context.sources.find((item) => item.chunk_id === answerSource.chunk_id);
  if (!source) return;
  const group = sourceGroups.find((item) => item.sources.some((choice) => choice.chunk_id === source.chunk_id));
  if (!group) return;
  selectSource(source, group);
  $("reader").scrollIntoView({behavior: "smooth", block: "start"});
  $("status").textContent = `${label}에 사용한 자료를 열었어요.`;
}

function renderAnswer(data) {
  $("answer-offer").hidden = true;
  $("answer-loading").hidden = true;
  $("answer-error").hidden = true;
  const container = $("answer-claims");
  const reason = $("answer-reason");
  container.replaceChildren();
  reason.hidden = true;
  reason.textContent = "";

  if (data.status === "draft_answer") {
    $("answer-result-title").textContent = "이렇게 확인했어요";
    $("answer-result-title").classList.remove("answer-result-title-muted");
    const sources = Object.fromEntries(data.context.sources.map((source) => [source.source_id, source]));
    for (const claim of data.claims) {
      const paragraph = document.createElement("p");
      paragraph.className = "answer-claim";
      paragraph.append(document.createTextNode(claim.text));
      const ids = [...new Set(claim.evidence.map((item) => item.source_id))];
      for (const id of ids) {
        const source = sources[id];
        const button = document.createElement("button");
        button.type = "button";
        button.className = "answer-citation";
        button.textContent = id.replace(/^S/, "자료 ");
        button.setAttribute("aria-label", `${id.replace(/^S/, "자료 ")} 확인하기`);
        if (source) button.addEventListener("click", () => openAnswerSource(source, id.replace(/^S/, "자료 ")));
        else button.disabled = true;
        paragraph.append(button);
      }
      container.append(paragraph);
    }
    $("status").textContent = "찾은 자료의 원문 인용을 확인한 뒤 답변을 표시했어요.";
  } else {
    $("answer-result-title").textContent = "자료만으로는 답하기 어려워요";
    $("answer-result-title").classList.add("answer-result-title-muted");
    reason.textContent = data.reason || "질문을 조금 더 구체적으로 적거나 아래 원문을 직접 확인해 주세요.";
    reason.hidden = false;
    $("status").textContent = "확인할 수 있는 내용만 보여드렸어요. 아래 관련 자료도 살펴보세요.";
  }
  $("answer-result").hidden = false;
}

function renderResults(data) {
  result = data;
  selected = null;
  const packet = data.context;
  searchTerms = schoolGuide.highlightTerms(data.search_query || packet.original_question);
  sourceGroups = schoolGuide.groupSources(packet.sources);
  $("empty-state").hidden = true;
  $("results-section").hidden = false;
  $("result-count").textContent = sourceGroups.length === packet.sources.length
    ? `${sourceGroups.length}개`
    : `${sourceGroups.length}개 항목 · 내용 ${packet.sources.length}개`;
  $("result-question").textContent = `궁금한 점: ${packet.original_question}`;
  $("follow-up-notice").hidden = !data.follow_up_applied;
  $("follow-up-notice").textContent = data.follow_up_applied
    ? `이전 질문과 이어서 “${data.search_query}”로 찾아봤어요.`
    : "";
  $("active-filter").hidden = data.school_level === "all";
  $("active-filter").textContent = data.school_level === "all" ? "" : `${schoolLabels[data.school_level]} 자료만 모아봤어요. ‘전체’를 누르면 다른 학교급 자료도 함께 볼 수 있어요.`;
  const reviewRecommended = data.result_assessment.level === "review_recommended";
  $("match-warning").hidden = !reviewRecommended;
  $("match-warning").textContent = !reviewRecommended ? ""
    : data.result_assessment.reason === "local_information"
      ? "현재 자료에는 학교별 급식, 행사, 신청 마감일 같은 최신 정보가 없어요. 아래 일반 자료와 함께 학교 홈페이지나 가정통신문을 확인해 주세요."
      : "가까운 자료를 보여드리지만 질문에 바로 답하는 내용인지는 확실하지 않아요. 학교급, 학년, 연도를 더해 다시 찾아보거나 원문을 직접 확인해 주세요.";
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
  resetAnswerPanel(packet.sources.length > 0);
  $("status").textContent = packet.sources.length ? `관련 내용 ${packet.sources.length}개를 ${sourceGroups.length}개 항목으로 정리했어요.` : `${schoolLabels[data.school_level]} 자료에서는 관련 내용을 찾지 못했어요. 학교급을 ‘전체’로 바꾸거나 다른 말로 찾아보세요.`;
  if (packet.sources.length) selectSource(sourceGroups[0].sources[0], sourceGroups[0]);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (loading || generating) return;
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
  const currentQuestion = question.value.trim();
  setLoading(true);
  $("error").hidden = true;
  $("results-section").hidden = true;
  $("empty-state").hidden = true;
  $("status").textContent = "궁금한 내용과 관련된 교육 자료를 찾고 있어요. 잠시만 기다려 주세요.";
  try {
    const payload = {question: currentQuestion, top_k: Number($("top-k").value), school_level: schoolLevel};
    if (previousQuestion) payload.previous_question = previousQuestion;
    const response = await fetch("/api/search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      // Audience selection changes suggestions only. School level is an explicit, visible filter.
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "자료를 찾지 못했어요. 잠시 후 다시 시도해 주세요.");
    lastSearchPayload = payload;
    renderResults(data);
    previousQuestion = data.search_query;
    rememberSearch(data);
  } catch (error) {
    $("status").textContent = "";
    $("error").textContent = error instanceof TypeError ? "지금은 자료를 불러올 수 없어요. 화면을 새로고침하거나 잠시 후 다시 시도해 주세요." : error.message;
    $("error").hidden = false;
    $("empty-state").hidden = false;
  } finally {
    setLoading(false);
  }
});

$("answer-button").addEventListener("click", async () => {
  if (loading || generating || !lastSearchPayload || !result?.context?.sources?.length) return;
  setGenerating(true);
  $("answer-result").hidden = true;
  $("answer-error").hidden = true;
  $("status").textContent = "내 Mac에서 찾은 자료를 읽고 답변을 정리하고 있어요.";
  try {
    const payload = {...lastSearchPayload, top_k: Math.min(Number(lastSearchPayload.top_k || 5), 3)};
    const response = await fetch("/api/answer", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "답변을 만들지 못했어요. 잠시 후 다시 시도해 주세요.");
    if (data.status === "generation_error" || data.status === "validation_failed") {
      throw new Error(data.reason || "답변을 안전하게 확인하지 못해 표시하지 않았어요.");
    }
    renderAnswer(data);
  } catch (error) {
    $("answer-offer").hidden = false;
    $("answer-button-label").textContent = "다시 만들어 보기";
    $("answer-error").textContent = error instanceof TypeError
      ? "로컬 답변 기능에 연결할 수 없어요. Ollama가 실행 중인지 확인해 주세요."
      : error.message;
    $("answer-error").hidden = false;
    $("status").textContent = "관련 자료는 그대로 볼 수 있어요.";
  } finally {
    setGenerating(false);
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

$("clear-history").addEventListener("click", () => {
  searchHistory = [];
  try {
    localStorage.removeItem(HISTORY_KEY);
  } catch {
    // Nothing else is required when browser storage is unavailable.
  }
  renderHistory();
  $("status").textContent = "최근 찾아본 질문을 모두 지웠어요.";
});

renderExamples();
renderSchoolFilter();
loadHistory();
fetch("/api/info").then((response) => {
  if (!response.ok) throw new Error("service unavailable");
  return response.json();
}).then((info) => {
  generationEnabled = Boolean(info.generation_enabled);
  if (result) resetAnswerPanel(result.context.sources.length > 0);
  $("index-info").textContent = `현재 ${info.document_count}개의 교육 자료에서 학교생활기록부와 교육과정 내용을 찾아드려요.`;
}).catch(() => {
  $("index-info").textContent = "지금은 자료를 불러올 수 없어요. 잠시 후 다시 방문해 주세요.";
});
