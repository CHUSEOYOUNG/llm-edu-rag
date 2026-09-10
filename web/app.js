"use strict";

const $ = (id) => document.getElementById(id);
const form = $("search-form");
const question = $("question");
const HISTORY_KEY = "school-life-guide.search-history.v1";
const SESSION_CHAT_KEY = "school-life-guide.session-chat.v1";
const CLIENT_ID_KEY = "school-life-guide.client-id.v1";
const BACKEND_ORIGIN = location.hostname === "localhost" ? "http://localhost:8080" : "http://127.0.0.1:8080";
const BACKEND_API = `${BACKEND_ORIGIN}/api/v1`;
let result = null;
let selected = null;
let loading = false;
let generating = false;
let generationEnabled = false;
let generationInfoReady = Promise.resolve();
let lastSearchPayload = null;
let audience = "all";
let schoolLevel = "all";
let schoolLevelManuallySet = false;
let sourceGroups = [];
let searchTerms = [];
let previousQuestion = null;
let searchHistory = [];
let conversationHistory = [];
let currentConversation = null;
let currentHistoryId = null;

function clientId() {
  try {
    const saved = localStorage.getItem(CLIENT_ID_KEY);
    if (/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(saved || "")) return saved;
    const created = crypto.randomUUID();
    localStorage.setItem(CLIENT_ID_KEY, created);
    return created;
  } catch {
    return crypto.randomUUID();
  }
}

const browserClientId = clientId();

async function postApplication(path, payload) {
  const options = {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  };
  try {
    return await fetch(`${BACKEND_API}/${path}`, options);
  } catch (error) {
    if (!(error instanceof TypeError)) throw error;
    const directPayload = {...payload};
    delete directPayload.client_id;
    return fetch(`/api/${path}`, {...options, body: JSON.stringify(directPayload)});
  }
}

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
    const openButton = fragment.querySelector(".history-open");
    const deleteButton = fragment.querySelector(".history-delete");
    openButton.querySelector(".history-question").textContent = item.question;
    openButton.querySelector(".history-level").textContent = schoolLabels[item.schoolLevel];
    openButton.addEventListener("click", () => {
      question.value = item.searchQuery;
      schoolLevel = item.schoolLevel;
      schoolLevelManuallySet = true;
      previousQuestion = null;
      renderSchoolFilter();
      form.requestSubmit();
      $("search-area").scrollIntoView({behavior: "smooth", block: "start"});
    });
    deleteButton.setAttribute("aria-label", `“${item.question}” 질문 삭제`);
    deleteButton.addEventListener("click", () => deleteHistoryItem(item));
    list.append(fragment);
  }
}

function sameHistoryItem(left, right) {
  if (left.id && right.id) return left.id === right.id;
  return left.searchQuery === right.searchQuery && left.schoolLevel === right.schoolLevel;
}

function saveHistoryLocally() {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(searchHistory));
  } catch {
    // The current page still keeps the list when browser storage is unavailable.
  }
}

async function deleteHistoryItem(item) {
  searchHistory = searchHistory.filter((entry) => !sameHistoryItem(entry, item));
  conversationHistory = conversationHistory.filter((entry) => !sameHistoryItem(entry, item));
  saveHistoryLocally();
  saveConversationHistory();
  renderHistory();
  renderConversationHistory();
  $("status").textContent = "선택한 질문을 지웠어요.";
  if (!item.id) return;
  try {
    const response = await fetch(`${BACKEND_API}/search-history/${encodeURIComponent(item.id)}?clientId=${encodeURIComponent(browserClientId)}`, {method: "DELETE"});
    if (!response.ok) throw new Error("history delete failed");
  } catch {
    $("status").textContent = "이 화면에서는 질문을 지웠어요. 서버 기록은 연결될 때 다시 지워주세요.";
  }
}

function normalizedConversationHistory(value) {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const questionText = typeof item.question === "string" ? item.question.trim() : "";
    const answer = typeof item.answer === "string" ? item.answer.trim() : "";
    if (!questionText || !answer || questionText.length > 4000 || answer.length > 12000) return [];
    const validLevels = new Set(Object.keys(schoolLabels));
    const normalized = {question: questionText, answer, searchQuery: item.searchQuery || questionText,
      schoolLevel: validLevels.has(item.schoolLevel) ? item.schoolLevel : "all"};
    if (/^[0-9a-f-]{36}$/i.test(item.id || "")) normalized.id = item.id;
    return [normalized];
  }).slice(-10);
}

function saveConversationHistory() {
  try {
    sessionStorage.setItem(SESSION_CHAT_KEY, JSON.stringify(conversationHistory));
  } catch {
    // Session storage is optional.
  }
}

function renderConversationHistory() {
  const container = $("conversation-history");
  container.replaceChildren();
  container.hidden = conversationHistory.length === 0;
  for (const item of conversationHistory) {
    const exchange = document.createElement("article");
    exchange.className = "archived-exchange";

    const userMessage = document.createElement("div");
    userMessage.className = "message user-message";
    const userAvatar = document.createElement("span");
    userAvatar.className = "message-avatar";
    userAvatar.textContent = "나";
    const userText = document.createElement("p");
    userText.className = "result-question";
    userText.textContent = item.question;
    userMessage.append(userAvatar, userText);

    const assistantMessage = document.createElement("div");
    assistantMessage.className = "message assistant-message archived-assistant-message";
    const assistantAvatar = document.createElement("span");
    assistantAvatar.className = "message-avatar assistant-avatar";
    assistantAvatar.textContent = "✦";
    const answer = document.createElement("p");
    answer.className = "archived-answer";
    answer.textContent = item.answer;
    assistantMessage.append(assistantAvatar, answer);

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "conversation-delete";
    deleteButton.textContent = "질문 삭제";
    deleteButton.setAttribute("aria-label", `“${item.question}” 대화 삭제`);
    deleteButton.addEventListener("click", () => deleteHistoryItem(item));
    exchange.append(userMessage, assistantMessage, deleteButton);
    container.append(exchange);
  }
}

function archiveCurrentConversation() {
  if (!currentConversation?.answer) return;
  conversationHistory = [...conversationHistory, currentConversation].slice(-10);
  currentConversation = null;
  saveConversationHistory();
  renderConversationHistory();
}

function loadConversationHistory() {
  try {
    conversationHistory = normalizedConversationHistory(JSON.parse(sessionStorage.getItem(SESSION_CHAT_KEY) || "[]"));
  } catch {
    conversationHistory = [];
  }
  renderConversationHistory();
}

function loadHistory() {
  try {
    searchHistory = schoolGuide.normalizeHistory(JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"));
  } catch {
    searchHistory = [];
  }
  renderHistory();
  fetch(`${BACKEND_API}/search-history?clientId=${encodeURIComponent(browserClientId)}`)
    .then((response) => {
      if (!response.ok) throw new Error("history unavailable");
      return response.json();
    })
    .then((items) => {
      searchHistory = schoolGuide.normalizeHistory(items);
      try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(searchHistory));
      } catch {
        // The server copy is still available when browser storage is unavailable.
      }
      renderHistory();
    })
    .catch(() => {
      // Keep the browser copy when the application backend is not running.
    });
}

function rememberSearch(data, historyId) {
  searchHistory = schoolGuide.addHistory(searchHistory, {
    id: historyId,
    question: data.context.original_question,
    searchQuery: data.search_query,
    schoolLevel: data.school_level
  });
  saveHistoryLocally();
  renderHistory();
}

function refreshBusyState() {
  const busy = loading || generating;
  $("search-button").disabled = busy;
  $("top-k").disabled = busy;
  question.disabled = busy;
  $("search-label").textContent = loading ? "찾고 있어요…" : "질문하기";
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
  $("answer-button-label").textContent = "답변 다시 만들기";
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
  $("sources-panel").open = true;
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
    $("answer-result-title").textContent = "답변";
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
    $("status").textContent = "";
    if (currentConversation) {
      currentConversation.answer = data.claims.map((claim) => claim.text).join("\n\n");
    }
  } else {
    $("answer-result-title").textContent = "자료만으로는 답하기 어려워요";
    $("answer-result-title").classList.add("answer-result-title-muted");
    reason.textContent = data.reason || "질문을 조금 더 구체적으로 적거나 아래 원문을 직접 확인해 주세요.";
    reason.hidden = false;
    $("status").textContent = "확인할 수 있는 내용만 보여드렸어요. 아래 관련 자료도 살펴보세요.";
    if (currentConversation) currentConversation.answer = reason.textContent;
  }
  $("answer-result").hidden = false;
}

function renderResults(data) {
  result = data;
  document.body.classList.add("has-results");
  selected = null;
  const packet = data.context;
  currentConversation = {
    id: currentHistoryId || undefined,
    question: packet.original_question,
    answer: packet.sources.length
      ? `관련 교육 자료 ${packet.sources.length}개를 찾았어요.`
      : "관련 교육 자료를 찾지 못했어요.",
    searchQuery: data.search_query || packet.original_question,
    schoolLevel: data.school_level || "all"
  };
  searchTerms = schoolGuide.highlightTerms(data.search_query || packet.original_question);
  sourceGroups = schoolGuide.groupSources(packet.sources);
  $("empty-state").hidden = true;
  $("results-section").hidden = false;
  $("result-count").textContent = `${packet.sources.length}개`;
  $("result-question").textContent = packet.original_question;
  $("follow-up-notice").hidden = !data.follow_up_applied;
  $("follow-up-notice").textContent = data.follow_up_applied
    ? `이전 질문과 이어서 “${data.search_query}”로 찾아봤어요.`
    : "";
  $("active-filter").hidden = data.school_level === "all";
  $("active-filter").textContent = data.school_level === "all" ? "" : `${schoolLabels[data.school_level]} 자료를 기준으로 찾았어요.`;
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

async function generateAnswer() {
  if (loading || generating || !lastSearchPayload || !result?.context?.sources?.length) return;
  setGenerating(true);
  $("answer-result").hidden = true;
  $("answer-error").hidden = true;
  $("status").textContent = "";
  try {
    const payload = {...lastSearchPayload, top_k: Math.min(Number(lastSearchPayload.top_k || 5), 3)};
    const response = await postApplication("answer", payload);
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || data.error || "답변을 만들지 못했어요. 잠시 후 다시 시도해 주세요.");
    if (data.status === "generation_error" || data.status === "validation_failed") {
      throw new Error(data.reason || "답변을 안전하게 확인하지 못해 표시하지 않았어요.");
    }
    renderAnswer(data);
  } catch (error) {
    $("answer-offer").hidden = false;
    $("answer-button-label").textContent = "답변 다시 만들기";
    $("answer-error").textContent = error instanceof TypeError
      ? "로컬 답변 기능에 연결할 수 없어요. Ollama가 실행 중인지 확인해 주세요."
      : error.message;
    $("answer-error").hidden = false;
    $("status").textContent = "관련 자료는 그대로 볼 수 있어요.";
    if (currentConversation) currentConversation.answer = $("answer-error").textContent;
  } finally {
    setGenerating(false);
  }
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
  archiveCurrentConversation();
  setLoading(true);
  $("error").hidden = true;
  $("results-section").hidden = true;
  $("empty-state").hidden = true;
  $("status").textContent = "궁금한 내용과 관련된 교육 자료를 찾고 있어요. 잠시만 기다려 주세요.";
  let shouldGenerateAnswer = false;
  try {
    const payload = {client_id: browserClientId, question: currentQuestion, top_k: Number($("top-k").value), school_level: schoolLevel};
    if (previousQuestion) payload.previous_question = previousQuestion;
    // Audience selection changes suggestions only. School level is an explicit, visible filter.
    const response = await postApplication("search", payload);
    const envelope = await response.json();
    if (!response.ok) throw new Error(envelope.message || envelope.error || "자료를 찾지 못했어요. 잠시 후 다시 시도해 주세요.");
    const data = envelope.search || envelope;
    if (!data?.context) throw new Error("검색 결과 형식을 확인하지 못했어요.");
    lastSearchPayload = payload;
    currentHistoryId = envelope.history_id || null;
    renderResults(data);
    previousQuestion = data.search_query;
    rememberSearch(data, currentHistoryId);
    question.value = "";
    await generationInfoReady;
    shouldGenerateAnswer = generationEnabled && data.context.sources.length > 0;
  } catch (error) {
    currentHistoryId = null;
    $("status").textContent = "";
    $("error").textContent = error instanceof TypeError ? "지금은 자료를 불러올 수 없어요. 화면을 새로고침하거나 잠시 후 다시 시도해 주세요." : error.message;
    $("error").hidden = false;
    $("empty-state").hidden = false;
  } finally {
    setLoading(false);
  }
  if (shouldGenerateAnswer) await generateAnswer();
});

$("answer-button").addEventListener("click", generateAnswer);

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

$("clear-history").addEventListener("click", async () => {
  searchHistory = [];
  conversationHistory = [];
  currentConversation = null;
  try {
    localStorage.removeItem(HISTORY_KEY);
  } catch {
    // Nothing else is required when browser storage is unavailable.
  }
  try {
    sessionStorage.removeItem(SESSION_CHAT_KEY);
  } catch {
    // Nothing else is required when session storage is unavailable.
  }
  renderHistory();
  renderConversationHistory();
  $("status").textContent = "최근 찾아본 질문을 모두 지웠어요.";
  try {
    const response = await fetch(`${BACKEND_API}/search-history?clientId=${encodeURIComponent(browserClientId)}`, {method: "DELETE"});
    if (!response.ok) throw new Error("history delete failed");
  } catch {
    $("status").textContent = "이 화면의 기록은 지웠어요. 서버 기록은 연결될 때 다시 지워주세요.";
  }
});

renderExamples();
renderSchoolFilter();
loadConversationHistory();
loadHistory();
generationInfoReady = fetch("/api/info").then((response) => {
  if (!response.ok) throw new Error("service unavailable");
  return response.json();
}).then((info) => {
  generationEnabled = Boolean(info.generation_enabled);
  if (result) resetAnswerPanel(result.context.sources.length > 0);
  $("index-info").textContent = `현재 ${info.document_count}개의 교육 자료에서 학교생활기록부와 교육과정 내용을 찾아드려요.`;
}).catch(() => {
  $("index-info").textContent = "지금은 자료를 불러올 수 없어요. 잠시 후 다시 방문해 주세요.";
});
